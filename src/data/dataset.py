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
from torchcodec.decoders import AudioDecoder

from src.data.models import SpeechFile, NoiseFile, RIRFile
from src.db.engine import create_db_engine

class SpatialMixingDataset(Dataset):
    def __init__(self, db_path, target_sr=16000, is_eval=False, snr_range=(-5, 20), chunk_size=48000): # Default 3 sec
        self.engine = create_db_engine(db_path)
        self.target_sr = target_sr
        self.is_eval = is_eval
        self.snr_range = snr_range
        self.chunk_size = chunk_size

        # Pre-load all paths to memory to avoid DB access in __getitem__ (Contention Fix)
        with Session(self.engine) as session:
            # 1. Speech
            stmt = select(SpeechFile.id, SpeechFile.path).where(SpeechFile.is_eval == is_eval)
            self.speech_data = session.exec(stmt).all() # List of (id, path)
            
            # 2. Noise
            stmt = select(NoiseFile.id, NoiseFile.path)
            self.noise_data = session.exec(stmt).all() # List of (id, path)
            
            # 3. RIR
            stmt = select(RIRFile.id, RIRFile.path)
            self.rir_data = session.exec(stmt).all() # List of (id, path)
            
        if not self.speech_data:
            raise ValueError(f"No speech files found for {'eval' if is_eval else 'train'} in DB.")

        # RIR Management (Scalable Cache)
        self.max_sources_supported = 8
        self.rir_cache = {} # path -> tensor_data
        self.max_cache_size = 100 # Adjust based on RAM (e.g., 100-200 is safe)
        
    def _get_rir_tensor(self, rir_path):
        """Scalable RIR Loader with soft-caching."""
        if rir_path in self.rir_cache:
            return self.rir_cache[rir_path]
        
        with open(rir_path, 'rb') as f:
            data = pickle.load(f)
        
        num_mics = len(data['rirs'])
        num_available_sources = len(data['rirs'][0])
        rir_len = max(data['rirs'][m][s].shape[0] for m in range(num_mics) for s in range(num_available_sources))
        
        # Structure: (Mics, Max_Sources, RIR_Len)
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
        
        # Simple cache management: If too many, clear half (could be better LRU, but this is fast)
        if len(self.rir_cache) >= self.max_cache_size:
            # Pop a random item or just clear (RIR files are small enough that re-reading isn't fatal)
            self.rir_cache.pop(next(iter(self.rir_cache)))
            
        self.rir_cache[rir_path] = rir_item
        return rir_item

    def __len__(self):
        return len(self.speech_data)

    def _load_audio(self, path, start_frame=0, num_frames=-1):
        # 1. High-speed .npy + memmap loading
        if str(path).endswith(".npy"):
            try:
                # mmap_mode='r' keeps the file on disk and reads into memory on-demand
                data = np.load(path, mmap_mode='r')
                
                if num_frames == -1:
                    waveform = data[start_frame:]
                else:
                    waveform = data[start_frame : start_frame + num_frames]
                
                # Copy to avoid issues with read-only memmap buffer and divide to normalize
                waveform = torch.from_numpy(waveform.copy()).float() / 32768.0
                return waveform
            except Exception as e:
                print(f"Warning: Failed to load npy {path}: {e}. Returning zeros.")
                return torch.zeros(num_frames if num_frames > 0 else 16000)

        # 2. torchaudio fallback (Standard method)
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
                pass
                
            return waveform
        except Exception as e:
            # Fallback for corrupted files or torchcodec bugs
            print(f"Warning: Failed to load {path}: {e}. Returning zeros.")
            total_frames = num_frames if num_frames > 0 else 16000
            return torch.zeros(total_frames)

    def _get_noise_long_enough(self, target_samples):
        # Pick a random noise file
        noise_id, noise_path = random.choice(self.noise_data)
        
        if str(noise_path).endswith(".npy"):
            data = np.load(noise_path, mmap_mode='r')
            num_frames = data.shape[0]
        else:
            # In torchaudio 2.10, .info() is missing/integrated into torchcodec
            decoder = AudioDecoder(noise_path)
            num_frames = int(decoder.metadata.duration_seconds * decoder.metadata.sample_rate)
        
        if num_frames > target_samples:
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

    def __getitem__(self, idx):
        # 1. Pick Speech
        id, path = self.speech_data[idx]
        clean_mono = self._load_audio(path) 
        
        if self.chunk_size:
            L = clean_mono.shape[0]
            if L >= self.chunk_size:
                start = random.randint(0, L - self.chunk_size)
                clean_mono = clean_mono[start : start + self.chunk_size]
            else:
                clean_mono = F.pad(clean_mono, (0, self.chunk_size - L))
        
        target_len = clean_mono.shape[0]

        # 2. Pick Random RIR (Scalable Loading)
        rir_id, rir_path = random.choice(self.rir_data)
        rir_item = self._get_rir_tensor(rir_path)
        
        rir_tensor = rir_item['tensor']
        num_available_sources = rir_item['num_sources']
        meta = rir_item['meta']
        
        # 3. Collect Raw Noises (GPU will handle mixing)
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
    dataset = SpatialMixingDataset(db_path, is_eval=False)
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample Speech Shape: {sample['raw_speech'].shape}") # Expected (target_len,)
    print(f"Sample SNR: {sample['snr']:.2f} dB")
