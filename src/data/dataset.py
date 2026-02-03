import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import pickle
from pathlib import Path
from sqlmodel import Session, select, func
from scipy.signal import convolve, butter, sosfilt
import random

from src.data.models import SpeechFile, NoiseFile, RIRFile
from src.db.engine import create_db_engine

class SpatialMixingDataset(Dataset):
    def __init__(self, db_path, target_sr=16000, is_eval=False, snr_range=(-5, 20)):
        self.engine = create_db_engine(db_path)
        self.target_sr = target_sr
        self.is_eval = is_eval
        self.snr_range = snr_range

        # Fetch all IDs once to avoid repeated queries
        with Session(self.engine) as session:
            self.speech_ids = session.exec(select(SpeechFile.id).where(SpeechFile.is_eval == is_eval)).all()
            self.noise_ids = session.exec(select(NoiseFile.id)).all()
            self.rir_ids = session.exec(select(RIRFile.id)).all()
            
        if not self.speech_ids:
            raise ValueError(f"No speech files found for {'eval' if is_eval else 'train'} in DB.")

    def __len__(self):
        return len(self.speech_ids)

    def _load_audio(self, path):
        # Use soundfile instead of torchaudio to avoid FFmpeg dependencies
        import soundfile as sf
        waveform, sr = sf.read(path)
        
        # soundfile returns (Samples, Channels) by default
        if len(waveform.shape) > 1:
            waveform = waveform[:, 0] # Take first channel if stereo
            
        if sr != self.target_sr:
            # Simple resampling using numpy/scipy if needed, 
            # but assuming raw data is already 16kHz for now
            pass
            
        return torch.from_numpy(waveform).float() # (Samples,)

    def _get_noise_long_enough(self, target_samples, session):
        """Fetches and concatenates random noises until the length is satisfied."""
        combined_noise = []
        current_samples = 0
        
        while current_samples < target_samples:
            noise_id = random.choice(self.noise_ids)
            noise_meta = session.get(NoiseFile, noise_id)
            noise_wav = self._load_audio(noise_meta.path)
            
            combined_noise.append(noise_wav)
            current_samples += noise_wav.shape[0]
            
        combined_noise = torch.cat(combined_noise, dim=0)
        return combined_noise[:target_samples]

    def _apply_rir(self, audio_np, rir_list, target_len=None):
        """Applies multi-channel RIRs to a mono audio signal."""
        # rir_list is a list of numpy arrays (channels)
        channels = []
        for rir_chan in rir_list:
            # Full convolution
            output = convolve(audio_np, rir_chan, mode='full')
            channels.append(output)
        
        # Consistent length padding for channels
        max_out_len = max(len(c) for c in channels)
        padded_channels = []
        for c in channels:
            if len(c) < max_out_len:
                c = np.pad(c, (0, max_out_len - len(c)))
            padded_channels.append(c)
            
        result = np.stack(padded_channels, axis=0)
        
        # Truncate or pad to match the original input length (speech mono length)
        if target_len is not None:
            if result.shape[1] > target_len:
                result = result[:, :target_len]
            elif result.shape[1] < target_len:
                result = np.pad(result, ((0, 0), (0, target_len - result.shape[1])))
                
        return result

    def _get_aligned_dry(self, clean_mono, rir_list):
        """Extracts direct path (peak) from RIRs and applies to mono speech for alignment."""
        channels = []
        for rir_chan in rir_list:
            # Find the peak (direct path)
            peak_idx = np.argmax(np.abs(rir_chan))
            peak_val = rir_chan[peak_idx]
            
            # Create a delta RIR (only peak remains)
            delta_rir = np.zeros_like(rir_chan)
            delta_rir[peak_idx] = peak_val
            
            # Convolve to get aligned dry
            dry_aligned = convolve(clean_mono, delta_rir, mode='full')
            channels.append(dry_aligned)
            
        # Shape match (truncate to same target_len will be handled by calling logic or padding here)
        result = np.stack(channels, axis=0)
        return result

    def _apply_bcm_modeling(self, audio_mc, mic_config):
        """Applies BCM-specific physics (LPF + Noise Attenuation) to the last channel."""
        if not mic_config.get('use_bcm'):
            return audio_mc
            
        bcm_ch = -1
        fs = self.target_sr
        cutoff = mic_config.get('bcm_cutoff_hz', 500.0)
        
        # 1. Low Pass Filter (BCM only senses low-freq vibrations)
        sos = butter(4, cutoff, btype='low', fs=fs, output='sos')
        audio_mc[bcm_ch] = sosfilt(sos, audio_mc[bcm_ch])
        
        return audio_mc

    def __getitem__(self, idx):
        with Session(self.engine) as session:
            # 1. Pick Speech
            speech_id = self.speech_ids[idx]
            speech_meta = session.get(SpeechFile, speech_id)
            clean_mono = self._load_audio(speech_meta.path).numpy()
            target_len = len(clean_mono)

            # 2. Pick Random RIR and parse N
            rir_id = random.choice(self.rir_ids)
            rir_meta = session.get(RIRFile, rir_id)
            with open(rir_meta.path, 'rb') as f:
                rir_data = pickle.load(f)
            
            meta = rir_data['meta']
            mic_config = meta['mic_config']
            num_mics = len(rir_data['rirs'])
            num_available_sources = len(rir_data['rirs'][0])
            
            # 3. Process Speech with RIR Source 0 (Target position)
            speech_rir_mics = [rir_data['rirs'][m][0] for m in range(num_mics)]
            
            # Spatialized (Reverberant)
            speech_mc = self._apply_rir(clean_mono, speech_rir_mics, target_len=target_len)
            speech_mc = self._apply_bcm_modeling(speech_mc, mic_config)
            
            # Aligned Dry (Direct path only)
            dry_mc = self._get_aligned_dry(clean_mono, speech_rir_mics)
            # Truncate/Pad dry_mc to target_len
            if dry_mc.shape[1] > target_len:
                dry_mc = dry_mc[:, :target_len]
            elif dry_mc.shape[1] < target_len:
                dry_mc = np.pad(dry_mc, ((0, 0), (0, target_len - dry_mc.shape[1])))
                
            dry_mc = self._apply_bcm_modeling(dry_mc, mic_config)
            
            # 4. Process Noises with RIR Sources 1..N
            noise_mc_total = np.zeros_like(speech_mc)
            noise_components = []
            
            # BCM Noise Attenuation Param
            bcm_atten_db = mic_config.get('bcm_noise_attenuation_db', 20.0)
            bcm_atten_factor = 10 ** (-bcm_atten_db / 20.0)

            for k in range(1, num_available_sources):
                noise_mono = self._get_noise_long_enough(target_len, session).numpy()
                noise_rir_mics = [rir_data['rirs'][m][k] for m in range(num_mics)]
                noise_spatialized = self._apply_rir(noise_mono, noise_rir_mics, target_len=target_len)
                
                # Apply BCM modeling (Filtering + Isolation)
                noise_spatialized = self._apply_bcm_modeling(noise_spatialized, mic_config)
                if mic_config.get('use_bcm'):
                    noise_spatialized[-1] *= bcm_atten_factor # Isolation attenuation

                noise_mc_total += noise_spatialized
                noise_components.append(noise_spatialized)

            # 5. SNR Scaling (Calculated based on Air Microphones for realism)
            air_ch_idx = slice(0, num_mics-1) if mic_config.get('use_bcm') else slice(0, num_mics)
            clean_rms = np.sqrt(np.mean(speech_mc[air_ch_idx]**2))
            noise_rms = np.sqrt(np.mean(noise_mc_total[air_ch_idx]**2))
            
            snr = random.uniform(*self.snr_range)
            if noise_rms > 0:
                target_noise_rms = clean_rms / (10**(snr/20))
                noise_mc_total *= (target_noise_rms / noise_rms)
                # Apply same scaling to components for verification consistency
                noise_components = [nc * (target_noise_rms / noise_rms) for nc in noise_components]

            # 6. Final Mix
            noisy_mc = speech_mc + noise_mc_total

            return {
                "noisy": torch.from_numpy(noisy_mc).float(),
                "clean": torch.from_numpy(speech_mc).float(), 
                "aligned_dry": torch.from_numpy(dry_mc).float(),
                "snr": snr,
                "rir_id": rir_id,
                "rir_path": rir_meta.path,
                "speech_only": torch.from_numpy(speech_mc).float(),
                "noise_only": torch.from_numpy(noise_mc_total).float(),
                "noise_components": [torch.from_numpy(nc).float() for nc in noise_components]
            }

# --- Quick Test Logic ---
if __name__ == "__main__":
    db_path = "data/metadata.db"
    dataset = SpatialMixingDataset(db_path, is_eval=False)
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample Noisy Shape: {sample['noisy'].shape}") # Expected (5, target_len)
    print(f"Sample SNR: {sample['snr']:.2f} dB")
