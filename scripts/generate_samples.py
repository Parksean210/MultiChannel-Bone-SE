#!/usr/bin/env python
import os
import sys
import shutil
import pickle
import random
import argparse
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())
from src.data.dataset import SpatialMixingDataset

TARGET_SR = 16000

def save_audio(path, tensor, sr=TARGET_SR):
    """Saves a (Channels, Samples) tensor to WAV."""
    # soundfile expects (Samples, Channels)
    sf.write(path, tensor.T.numpy(), sr)

def export_rir_wavs(rir_pkl_path, output_dir, sample_idx, sr=TARGET_SR):
    """Loads RIR pickle and exports Target/Noise RIRs as WAVs."""
    with open(rir_pkl_path, 'rb') as f:
        rir_data = pickle.load(f)
        
    rirs_by_channel = rir_data['rirs']
    num_mics = len(rirs_by_channel)
    num_sources = len(rirs_by_channel[0])
    
    for src_idx in range(num_sources):
        # Extract RIR for this source across all mics
        src_rirs = [rirs_by_channel[m][src_idx] for m in range(num_mics)]
        
        # Pad to consistent length
        max_len = max(len(r) for r in src_rirs)
        padded_rirs = [np.pad(r, (0, max_len - len(r))) if len(r) < max_len else r for r in src_rirs]
        
        # Stack: (Channels, Samples) -> Transpose: (Samples, Channels)
        rir_tensor = np.stack(padded_rirs).T
        
        if src_idx == 0:
            name = f"sample_{sample_idx:02d}_rir_target.wav"
        else:
            name = f"sample_{sample_idx:02d}_rir_noise_source_{src_idx}.wav"
            
        sf.write(os.path.join(output_dir, name), rir_tensor, sr)

def copy_visualization(rir_path, output_dir, sample_idx):
    """Copies the visualization image for the RIR if it exists."""
    rir_basename = os.path.basename(rir_path).replace('.pkl', '')
    # Assume: data/rirs/viz/rir_XXXXX.png
    viz_src = os.path.join(os.path.dirname(os.path.dirname(rir_path)), 'viz', f"{rir_basename}.png")
    
    if os.path.exists(viz_src):
        dst = os.path.join(output_dir, f"sample_{sample_idx:02d}_rir_viz.png")
        shutil.copy(viz_src, dst)

def apply_synthesis(sample):
    """Performs CPU-based spatial synthesis following the GPU logic in SEModule."""
    raw_speech = sample['raw_speech']   # (T,)
    raw_noises = sample['raw_noises']   # (S-1, T)
    rir_tensor = sample['rir_tensor']   # (M, S, L)
    snr = sample['snr']
    mic_config = sample['mic_config']
    
    T = raw_speech.shape[-1]
    M, S, L = rir_tensor.shape

    # 1. Speech Spatialization
    speech_rir = rir_tensor[:, 0, :] # (M, L)
    import torchaudio.functional as F_audio
    speech_mc = F_audio.fftconvolve(raw_speech.unsqueeze(0), speech_rir, mode="full")
    speech_mc = speech_mc[:, :T]

    # 2. Noise Spatialization
    num_available_sources = sample['num_sources']
    noise_mc_total = torch.zeros_like(speech_mc)
    noise_components = []
    
    for k in range(1, min(num_available_sources, S)):
        noise_rir = rir_tensor[:, k, :]
        noise_wav = raw_noises[k-1, :]
        
        noise_spatialized = F_audio.fftconvolve(noise_wav.unsqueeze(0), noise_rir, mode="full")
        noise_spatialized = noise_spatialized[:, :T]
        noise_mc_total += noise_spatialized
        noise_components.append(noise_spatialized)

    # 3. BCM Modeling
    use_bcm = mic_config.get('use_bcm', False)
    if use_bcm:
        cutoff = mic_config.get('bcm_cutoff_hz', 400)
        box_len = int(16000 / cutoff)
        kernel = torch.ones((1, 1, box_len)) / box_len
        
        # Target
        bcm_speech = speech_mc[-1:] # (1, T)
        bcm_speech_padded = torch.nn.functional.pad(bcm_speech.unsqueeze(0), (box_len // 2, box_len // 2), mode='reflect')
        speech_mc[-1:] = torch.nn.functional.conv1d(bcm_speech_padded, kernel).squeeze(0)[:, :T]
        
        # Total Noise
        bcm_noise = noise_mc_total[-1:]
        bcm_noise_padded = torch.nn.functional.pad(bcm_noise.unsqueeze(0), (box_len // 2, box_len // 2), mode='reflect')
        noise_mc_total[-1:] = torch.nn.functional.conv1d(bcm_noise_padded, kernel).squeeze(0)[:, :T]
        
        # Noise attenuation
        atten_db = mic_config.get('bcm_noise_attenuation_db', 20)
        atten_factor = 10 ** (-atten_db / 20.0)
        noise_mc_total[-1] *= atten_factor
        
        # Update components for logging
        for nc in noise_components:
            nc_bcm_padded = torch.nn.functional.pad(nc[-1:].unsqueeze(0), (box_len // 2, box_len // 2), mode='reflect')
            nc[-1:] = torch.nn.functional.conv1d(nc_bcm_padded, kernel).squeeze(0)[:, :T]
            nc[-1] *= atten_factor

    # 4. SNR Scaling
    air_idx = slice(0, M-1) if use_bcm else slice(0, M)
    clean_rms = torch.sqrt(torch.mean(speech_mc[air_idx, :]**2) + 1e-8)
    noise_rms = torch.sqrt(torch.mean(noise_mc_total[air_idx, :]**2) + 1e-8)
    
    target_factor = (clean_rms / (10**(snr/20))) / (noise_rms + 1e-8)
    noise_mc_total *= target_factor
    for nc in noise_components:
        nc *= target_factor

    # 5. Aligned Dry (Target alignment)
    # reference mic (0)의 RIR 피크 위치를 찾아 propagation delay 파악
    peak_sample = torch.argmax(torch.abs(speech_rir[0]))
    aligned_dry = torch.zeros_like(speech_mc)
    
    # 각 채널별로 raw_speech를 배치 (단순 딜레이만 적용하거나, 참조 채널만 생성)
    # 여기서는 '지표 계산용'으로 참조 채널(0) 기준으로 정렬된 드라이 신호 생성
    shifted_speech = torch.zeros(T)
    if peak_sample < T:
        shifted_speech[peak_sample:] = raw_speech[:T-peak_sample]
    
    # 모든 채널에 동일한 딜레이를 적용한 Dry 신호 (SI-SDR 등 계산용)
    aligned_dry = shifted_speech.unsqueeze(0).expand(M, -1)

    return {
        'noisy': speech_mc + noise_mc_total,
        'speech_only': speech_mc,
        'noise_only': noise_mc_total,
        'noise_components': noise_components,
        'aligned_dry': aligned_dry,
        'clean': raw_speech.unsqueeze(0), # Monoral original
        'rir_path': sample['rir_path'],
        'snr': snr
    }

def generate_samples(num_samples, output_dir, db_path, sr, split):
    os.makedirs(output_dir, exist_ok=True)
    dataset = SpatialMixingDataset(db_path, target_sr=sr, split=split)
    
    print(f"Generating {num_samples} samples to '{output_dir}' (SR={sr}, Split={split})...")
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            idx = random.randint(0, len(dataset) - 1)
            raw_sample = dataset[idx]
            
            # Synthesis on CPU
            sample = apply_synthesis(raw_sample)
            
            # Unpack Data
            noisy = sample['noisy']
            speech_only = sample['speech_only']
            noise_only = sample['noise_only']
            noise_components = sample['noise_components']
            rir_path = sample['rir_path']
            snr = sample['snr']

            # 1. Main Output Files
            save_audio(os.path.join(output_dir, f"sample_{i:02d}_clean_mono.wav"), sample['clean'], sr)
            save_audio(os.path.join(output_dir, f"sample_{i:02d}_aligned_dry.wav"), sample['aligned_dry'], sr)
            save_audio(os.path.join(output_dir, f"sample_{i:02d}_noisy_snr{snr:.1f}.wav"), noisy, sr)
            save_audio(os.path.join(output_dir, f"sample_{i:02d}_target_reverb.wav"), speech_only, sr)
            save_audio(os.path.join(output_dir, f"sample_{i:02d}_total_noise.wav"), noise_only, sr)

            # 2. Individual Noise Sources
            for k, nc in enumerate(noise_components):
                save_audio(os.path.join(output_dir, f"sample_{i:02d}_noise_source_{k+1}.wav"), nc, sr)

            # 3. Channel-wise Breakdown (Debugging)
            for ch in range(noisy.shape[0]):
                base = f"sample_{i:02d}"
                save_audio(os.path.join(output_dir, f"{base}_noisy_ch{ch}.wav"), noisy[ch:ch+1], sr)
                save_audio(os.path.join(output_dir, f"{base}_target_ch{ch}.wav"), speech_only[ch:ch+1], sr)

            # 4. Export RIRs and Autosave Viz
            export_rir_wavs(rir_path, output_dir, i, sr)
            copy_visualization(rir_path, output_dir, i)

def main():
    parser = argparse.ArgumentParser(description="Generate verification audio samples.")
    parser.add_argument("--num", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="data/samples", help="Output directory")
    parser.add_argument("--db", type=str, default="data/metadata.db", help="Path to metadata DB")
    parser.add_argument("--sr", type=int, default=16000, help="Sampling rate")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Dataset split")
    
    args = parser.parse_args()
    generate_samples(args.num, args.out, args.db, args.sr, args.split)

if __name__ == "__main__":
    main()
