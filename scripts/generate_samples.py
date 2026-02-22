#!/usr/bin/env python
"""
검증용 오디오 샘플 생성 스크립트.
src.utils.synthesis의 통합 파이프라인을 사용하여 SEModule과 동일한 합성 로직을 적용합니다.
"""
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

sys.path.append(os.getcwd())
from src.data.dataset import SpatialMixingDataset
from src.utils.synthesis import apply_spatial_synthesis, create_bcm_kernel

TARGET_SR = 16000


def save_multichannel_audio(path, tensor, sr=TARGET_SR):
    """(Channels, Samples) 텐서를 멀티채널 WAV로 저장합니다."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    sf.write(path, tensor.T.numpy(), sr)


def export_rir_wavs(rir_pkl_path, output_dir, sample_idx, sr=TARGET_SR):
    """RIR pickle을 로드하여 Target/Noise RIR을 WAV로 내보냅니다."""
    with open(rir_pkl_path, 'rb') as f:
        rir_data = pickle.load(f)

    rirs_by_channel = rir_data['rirs']
    num_mics = len(rirs_by_channel)
    num_sources = len(rirs_by_channel[0])

    for src_idx in range(num_sources):
        src_rirs = [rirs_by_channel[m][src_idx] for m in range(num_mics)]
        max_len = max(len(r) for r in src_rirs)
        padded_rirs = [np.pad(r, (0, max_len - len(r))) if len(r) < max_len else r for r in src_rirs]
        rir_tensor = np.stack(padded_rirs).T

        if src_idx == 0:
            name = f"sample_{sample_idx:02d}_rir_target.wav"
        else:
            name = f"sample_{sample_idx:02d}_rir_noise_source_{src_idx}.wav"

        sf.write(os.path.join(output_dir, name), rir_tensor, sr)


def copy_visualization(rir_path, output_dir, sample_idx):
    """RIR에 대한 시각화 이미지가 존재하면 복사합니다."""
    rir_basename = os.path.basename(rir_path).replace('.pkl', '')
    viz_src = os.path.join(os.path.dirname(os.path.dirname(rir_path)), 'viz', f"{rir_basename}.png")

    if os.path.exists(viz_src):
        dst = os.path.join(output_dir, f"sample_{sample_idx:02d}_rir_viz.png")
        shutil.copy(viz_src, dst)


def synthesize_sample(sample, bcm_kernel=None, sample_rate=TARGET_SR):
    """
    단일 샘플에 대해 공간 합성을 수행합니다.
    apply_spatial_synthesis에 배치 차원을 추가하여 위임합니다.
    """
    batch = {
        'raw_speech': sample['raw_speech'].unsqueeze(0),
        'raw_noises': sample['raw_noises'].unsqueeze(0),
        'rir_tensor': sample['rir_tensor'].unsqueeze(0),
        'snr': sample['snr'].unsqueeze(0) if isinstance(sample['snr'], torch.Tensor)
               else torch.tensor([sample['snr']]),
        'mic_config': sample['mic_config'],
    }

    batch = apply_spatial_synthesis(
        batch, bcm_kernel=bcm_kernel, sample_rate=sample_rate,
        return_individual_noise=True,
    )

    # 배치 차원 제거
    return {
        'noisy': batch['noisy'][0],
        'speech_only': batch['clean'][0],
        'noise_only': batch['noise_only'][0],
        'noise_components': [nc[0] for nc in batch['noise_components']],
        'aligned_dry': batch['aligned_dry'][0],
        'clean': sample['raw_speech'].unsqueeze(0),
        'rir_path': sample['rir_path'],
        'snr': sample['snr'],
    }


def generate_samples(num_samples, output_dir, db_path, sr, split):
    os.makedirs(output_dir, exist_ok=True)
    dataset = SpatialMixingDataset(db_path, target_sr=sr, split=split)

    # BCM 커널 사전 생성 (sinc+hann LPF, SEModule과 동일)
    bcm_kernel = create_bcm_kernel(cutoff_hz=500.0, sample_rate=sr, num_taps=101)

    print(f"Generating {num_samples} samples to '{output_dir}' (SR={sr}, Split={split})...")

    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            idx = random.randint(0, len(dataset) - 1)
            raw_sample = dataset[idx]

            sample = synthesize_sample(raw_sample, bcm_kernel=bcm_kernel, sample_rate=sr)

            noisy = sample['noisy']
            speech_only = sample['speech_only']
            noise_only = sample['noise_only']
            noise_components = sample['noise_components']
            rir_path = sample['rir_path']
            snr = sample['snr']

            # 1. Main Output Files
            save_multichannel_audio(os.path.join(output_dir, f"sample_{i:02d}_clean_mono.wav"), sample['clean'], sr)
            save_multichannel_audio(os.path.join(output_dir, f"sample_{i:02d}_aligned_dry.wav"), sample['aligned_dry'], sr)
            save_multichannel_audio(os.path.join(output_dir, f"sample_{i:02d}_noisy_snr{snr:.1f}.wav"), noisy, sr)
            save_multichannel_audio(os.path.join(output_dir, f"sample_{i:02d}_target_reverb.wav"), speech_only, sr)
            save_multichannel_audio(os.path.join(output_dir, f"sample_{i:02d}_total_noise.wav"), noise_only, sr)

            # 2. Individual Noise Sources
            for k, nc in enumerate(noise_components):
                save_multichannel_audio(os.path.join(output_dir, f"sample_{i:02d}_noise_source_{k+1}.wav"), nc, sr)

            # 3. Channel-wise Breakdown
            for ch in range(noisy.shape[0]):
                base = f"sample_{i:02d}"
                save_multichannel_audio(os.path.join(output_dir, f"{base}_noisy_ch{ch}.wav"), noisy[ch:ch+1], sr)
                save_multichannel_audio(os.path.join(output_dir, f"{base}_target_ch{ch}.wav"), speech_only[ch:ch+1], sr)

            # 4. Export RIRs and Visualization
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
