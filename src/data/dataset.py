import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import pickle
from pathlib import Path
from sqlmodel import Session, select, func
from scipy.signal import butter, sosfilt
from typing import List, Optional
import torchaudio.functional as F_audio
import random
# from torchcodec.decoders import AudioDecoder

from src.data.models import SpeechFile, NoiseFile, RIRFile
from src.db.engine import create_db_engine

class SpatialMixingDataset(Dataset):
    """
    고속 공간 합성(Spatial Mixing) 학습을 위한 데이터셋 클래스.
    SQLite 기반 메타데이터 조회와 .npy 메모리 맵핑(mmap)을 통한 초고속 데이터 로딩을 지원합니다.
    """
    def __init__(self, db_path: str, target_sr: int = 16000, split: str = "train", snr_range: tuple = (-5, 20), chunk_size: int = 48000):
        """
        Args:
            db_path: SQLite 메타데이터 데이터베이스 경로
            target_sr: 목표 샘플 레이트 (기본값: 16kHz)
            split: 데이터 분할 종류 ('train', 'val', 'test')
            snr_range: 학습 시 무작위로 적용될 SNR 범위 (dB)
            chunk_size: 학습용 오디오 샘플 길이 (샘플 수 기준, 48000 = 3초)
        """
        self.engine = create_db_engine(db_path)
        self.target_sr = target_sr
        self.split = split
        self.snr_range = snr_range
        self.chunk_size = chunk_size

        # 데이터베이스 커넥션 병목 현상을 방지하기 위해 초기화 시점에 전체 경로를 메모리에 캐싱
        with Session(self.engine) as session:
            # 1. Speech
            stmt = select(SpeechFile.id, SpeechFile.path).where(SpeechFile.split == split)
            self.speech_data = session.exec(stmt).all() # List of (id, path)
            
            # 2. Noise
            stmt = select(NoiseFile.id, NoiseFile.path).where(NoiseFile.split == split)
            self.noise_data = session.exec(stmt).all() # List of (id, path)
            
            # 3. RIR
            stmt = select(RIRFile.id, RIRFile.path).where(RIRFile.split == split)
            self.rir_data = session.exec(stmt).all() # List of (id, path)
            
        if not self.speech_data:
            raise ValueError(f"No speech files found for {split} in DB.")

        # RIR 캐시 관리 설정 (메모리 사용량 최적화용)
        self.max_sources_supported = 8
        self.rir_cache = {} # path -> tensor_data
        self.max_cache_size = 100 # 시스템 메모리에 따라 조정 가능
        
    def _get_rir_tensor(self, rir_path: str):
        """
        RIR 파일을 로드하고 텐서 형태로 변환합니다. 빈번한 파일 I/O를 방지하기 위해 캐싱을 지원합니다.
        
        Args:
            rir_path: .pkl 포맷의 RIR 파일 경로
        Returns:
            rir_item: 텐서 데이터 및 메타정보를 포함한 딕셔너리
        """
        if rir_path in self.rir_cache:
            return self.rir_cache[rir_path]
        
        with open(rir_path, 'rb') as f:
            data = pickle.load(f)
        
        num_mics = len(data['rirs'])
        num_available_sources = len(data['rirs'][0])
        # 각 소스별 RIR 중 최대 길이를 기준으로 텐서 크기 결정
        rir_len = max(data['rirs'][m][s].shape[0] for m in range(num_mics) for s in range(num_available_sources))
        
        # 데이터 구조 정의: (Mic_Channels, Max_Sources, Impulse_Response_Len)
        tensor = torch.zeros((num_mics, self.max_sources_supported, rir_len), dtype=torch.float32)
        for m in range(num_mics):
            for s in range(min(num_available_sources, self.max_sources_supported)):
                r = data['rirs'][m][s]
                tensor[m, s, :r.shape[0]] = torch.from_numpy(r).float()
        
        rir_item = {
            'tensor': tensor,
            'meta': data['meta'],
            'num_sources': num_available_sources,
            'path': rir_path
        }
        
        # 캐시 크기 초과 시 가장 오래된 항목(FIFO 방식)을 제거하여 메모리 점유율 관리
        if len(self.rir_cache) >= self.max_cache_size:
            self.rir_cache.pop(next(iter(self.rir_cache)))
            
        self.rir_cache[rir_path] = rir_item
        return rir_item

    def __len__(self):
        return len(self.speech_data)

    def _load_audio(self, path: str, start_frame: int = 0, num_frames: int = -1):
        """
        오디오 데이터를 로드합니다. .npy 포맷은 메모리 맵핑을 통해 초고속으로 읽어옵니다.
        """
        # Case 1: .npy 포맷 (고속 학습 최적화)
        if str(path).endswith(".npy"):
            try:
                # mmap_mode='r'을 통해 파일 전체를 메모리에 올리지 않고 필요한 세그먼트만 디스크에서 직접 읽음
                data = np.load(path, mmap_mode='r')
                
                if num_frames == -1:
                    waveform = data[start_frame:]
                else:
                    waveform = data[start_frame : start_frame + num_frames]
                
                # 원본 mmap 버퍼 보호를 위해 복사본 생성 후 정규화(float32, -1.0 ~ 1.0) 수행
                waveform = torch.from_numpy(waveform.copy()).float() / 32768.0
                return waveform
            except Exception as e:
                print(f"Error: .npy 로드 실패 {path}: {e}")
                return torch.zeros(num_frames if num_frames > 0 else self.target_sr)

        # Case 2: 일반 오디오 포맷 (WAV 등, fallback용)
        try:
            waveform, sr = torchaudio.load(
                path, 
                frame_offset=start_frame, 
                num_frames=num_frames
            )
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform[0]
            else:
                waveform = waveform.squeeze(0)

            if sr != self.target_sr:
                # Calculate new number of frames if num_frames was specified
                resampled_num_frames = -1
                if num_frames != -1:
                    resampled_num_frames = int(num_frames * self.target_sr / sr)
                
                # Reload or resample? Since we already loaded, let's resample
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
                
            return waveform
        except Exception as e:
            # Fallback for corrupted files or torchcodec bugs
            print(f"Warning: Failed to load {path}: {e}. Returning zeros.")
            total_frames = num_frames if num_frames > 0 else self.target_sr
            return torch.zeros(total_frames)

    def _get_noise_long_enough(self, target_samples: int):
        """
        지정된 길이(target_samples) 이상의 노이즈 세그먼트를 추출합니다.
        파일이 짧은 경우 순환(Looping)을 통해 길이를 확보합니다.
        """
        noise_id, noise_path = random.choice(self.noise_data)
        
        if str(noise_path).endswith(".npy"):
            data = np.load(noise_path, mmap_mode='r')
            num_frames = data.shape[0]
        else:
            # torchaudio.info를 사용하여 메타데이터 추출 (torchcodec 대체)
            info = torchaudio.info(noise_path)
            num_frames = info.num_frames
        
        if num_frames > target_samples:
            # 여유 길이가 있는 경우 무작위 구간 추출
            max_start = num_frames - target_samples
            start = random.randint(0, max_start)
            return self._load_audio(noise_path, start_frame=start, num_frames=target_samples)
        else:
            # If noise is too short, fall back to loading whole and looping (rare for this dataset)
            noise_wav = self._load_audio(noise_path)
            while noise_wav.shape[0] < target_samples:
                # Append another random noise
                extra_id, extra_path = random.choice(self.noise_data)
                extra_wav = self._load_audio(extra_path)
                noise_wav = torch.cat([noise_wav, extra_wav], dim=0)
            return noise_wav[:target_samples]
    
    def _apply_rir(self, audio, rirs, target_len): pass # Deprecated for GPU
    def _get_aligned_dry(self, audio, rirs, target_len): pass # Deprecated for GPU
    def _apply_bcm_modeling(self, audio, mic_config): pass # Deprecated for GPU

    def __getitem__(self, idx: int):
        """
        학습용 데이터 샘플(음성, 노이즈, RIR 메타데이터)을 생성합니다.
        실제 믹싱 연산은 GPU 효율성을 위해 SEModule._apply_gpu_synthesis에서 수행됩니다.
        """
        # 1. 스피치 소스 선정 및 전처리 (Random Crop & Padding)
        id, path = self.speech_data[idx]
        clean_mono = self._load_audio(path) 
        
        if self.chunk_size:
            L = clean_mono.shape[0]
            if L >= self.chunk_size:
                # 3초보다 긴 경우 무작위 시작 지점 선정 (Random Cropping)
                start = random.randint(0, L - self.chunk_size)
                clean_mono = clean_mono[start : start + self.chunk_size]
            else:
                # 3초보다 짧은 경우 부족한 뒷부분을 0으로 채움 (Zero Padding)
                clean_mono = F.pad(clean_mono, (0, self.chunk_size - L))
        
        target_len = clean_mono.shape[0]

        # 2. RIR(공간 정보) 무작위 추출 및 캐시 로드
        rir_id, rir_path = random.choice(self.rir_data)
        rir_item = self._get_rir_tensor(rir_path)
        
        rir_tensor = rir_item['tensor']
        num_available_sources = rir_item['num_sources']
        meta = rir_item['meta']
        
        # 3. 노이즈 원본 수집 (최대 소스 개수만큼 수집하며, GPU에서 합성 수행)
        noise_waveforms = torch.zeros((self.max_sources_supported - 1, target_len), dtype=torch.float32)
        for s in range(1, min(num_available_sources, self.max_sources_supported)):
            noise_waveforms[s-1] = self._get_noise_long_enough(target_len)
            
        # 4. Clean mic_config
        clean_mic_config = {k: v for k, v in meta['mic_config'].items() if v is not None}
            
        return {
            "raw_speech": clean_mono,
            "raw_noises": noise_waveforms,
            "rir_tensor": rir_tensor,
            "num_sources": num_available_sources, 
            "snr": random.uniform(*self.snr_range),
            "mic_config": clean_mic_config,
            "rir_id": 0, # Placeholder
            "rir_path": rir_item['path'],
        }

# --- Quick Test Logic ---
if __name__ == "__main__":
    db_path = "data/metadata.db"
    dataset = SpatialMixingDataset(db_path, split="train")
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample Speech Shape: {sample['raw_speech'].shape}") # Expected (target_len,)
    print(f"Sample SNR: {sample['snr']:.2f} dB")
