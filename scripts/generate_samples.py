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

def generate_samples(num_samples, output_dir, db_path, sr):
    os.makedirs(output_dir, exist_ok=True)
    dataset = SpatialMixingDataset(db_path, target_sr=sr, is_eval=False)
    
    print(f"Generating {num_samples} samples to '{output_dir}' (SR={sr})...")
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        
        # Unpack Data
        noisy = sample['noisy']
        speech_only = sample['speech_only']
        noise_only = sample['noise_only']
        noise_components = sample['noise_components']
        rir_path = sample['rir_path']
        snr = sample['snr']

        # 1. Main Output Files
        save_audio(os.path.join(output_dir, f"sample_{i:02d}_clean.wav"), sample['clean'], sr)
        save_audio(os.path.join(output_dir, f"sample_{i:02d}_aligned_dry.wav"), sample['aligned_dry'], sr)
        save_audio(os.path.join(output_dir, f"sample_{i:02d}_noisy_snr{snr:.1f}.wav"), noisy, sr)
        save_audio(os.path.join(output_dir, f"sample_{i:02d}_component_speech.wav"), speech_only, sr)
        save_audio(os.path.join(output_dir, f"sample_{i:02d}_component_noise_total.wav"), noise_only, sr)

        # 2. Individual Noise Sources
        for k, nc in enumerate(noise_components):
            save_audio(os.path.join(output_dir, f"sample_{i:02d}_noise_source_{k+1}.wav"), nc, sr)

        # 3. Channel-wise Breakdown (Debugging)
        for ch in range(noisy.shape[0]):
            base = f"sample_{i:02d}"
            save_audio(os.path.join(output_dir, f"{base}_noisy_ch{ch}.wav"), noisy[ch:ch+1], sr)
            save_audio(os.path.join(output_dir, f"{base}_speech_ch{ch}.wav"), speech_only[ch:ch+1], sr)
            save_audio(os.path.join(output_dir, f"{base}_noise_ch{ch}.wav"), noise_only[ch:ch+1], sr)

        # 4. Export RIRs and Autosave Viz
        export_rir_wavs(rir_path, output_dir, i, sr)
        copy_visualization(rir_path, output_dir, i)

    print(f"Successfully generated {num_samples} samples.")

def main():
    parser = argparse.ArgumentParser(description="Generate verification audio samples.")
    parser.add_argument("--num", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="data/samples", help="Output directory")
    parser.add_argument("--db", type=str, default="data/metadata.db", help="Path to metadata DB")
    parser.add_argument("--sr", type=int, default=TARGET_SR, help="Sampling rate")
    
    args = parser.parse_args()
    generate_samples(args.num, args.out, args.db, args.sr)

if __name__ == "__main__":
    main()
