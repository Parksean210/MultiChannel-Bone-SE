"""
파이프라인 검증 스크립트 (scripts/verify_pipeline.py)

데이터가 원하는 대로 들어오고 있는지 직접 눈으로 확인합니다.

검증 항목:
  1. DB split 분리 확인  (train/val/test 겹침 없는지)
  2. 배치 shape 확인      (텐서 차원이 예상과 맞는지)
  3. SNR 정확도 확인      (목표 SNR vs 실제 SNR)
  4. BCM 채널 LPF 확인   (고주파 에너지가 낮은지)
  5. 오디오 저장          (직접 청취 가능)
  6. 스펙트로그램 저장    (시각적 확인)

실행:
  uv run python scripts/verify_pipeline.py
  uv run python scripts/verify_pipeline.py --split val --n_batches 3
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서 PNG 저장
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from sqlmodel import Session, select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import SpatialMixingDataset
from src.data.datamodule import collate_gpu_synthesis
from src.data.models import SpeechFile, NoiseFile, RIRFile
from src.db.engine import create_db_engine
from src.utils.synthesis import create_bcm_kernel, apply_spatial_synthesis

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DB_PATH    = "data/metadata.db"
SR         = 16000
OUTPUT_DIR = "results/verify_pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


# ─────────────────────────────────────────────
# 1. DB split 분리 확인
# ─────────────────────────────────────────────
def check_db_splits(db_path: str):
    print("\n" + "="*55)
    print("  검증 1: DB split 분리 확인")
    print("="*55)

    engine = create_db_engine(db_path)
    with Session(engine) as session:
        for Model, label in [(SpeechFile, "Speech"), (NoiseFile, "Noise"), (RIRFile, "RIR")]:
            counts = {}
            ids    = {}
            for split in ["train", "val", "test"]:
                rows = session.exec(
                    select(Model.id).where(Model.split == split)
                ).all()
                counts[split] = len(rows)
                ids[split]    = set(rows)

            print(f"\n[{label}]")
            for split, cnt in counts.items():
                print(f"  {split:5s}: {cnt:>7,} 개")

            # 겹침 검사
            tv = ids["train"] & ids["val"]
            tt = ids["train"] & ids["test"]
            vt = ids["val"]   & ids["test"]
            ok = (len(tv) == 0 and len(tt) == 0 and len(vt) == 0)
            total = sum(counts.values())
            print(f"  total: {total:>7,} 개   겹침: {PASS if ok else FAIL}")
            if not ok:
                print(f"    train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}")


# ─────────────────────────────────────────────
# 2. 배치 shape 확인
# ─────────────────────────────────────────────
def check_batch_shapes(batch: dict, split: str):
    print("\n" + "="*55)
    print(f"  검증 2: 배치 shape 확인  (split={split})")
    print("="*55)

    B = batch["raw_speech"].shape[0]
    checks = {
        "raw_speech  (B, T)":       (batch["raw_speech"].shape,  (B, 48000)),
        "raw_noises  (B, S-1, T)":  (batch["raw_noises"].shape,  (B, 7, 48000)),
        "rir_tensor  (B, M, S, L)": (batch["rir_tensor"].shape[:3], (B, 5, 8)),
        "noise_ids   (B, S-1)":     (batch["noise_ids"].shape,   (B, 7)),
        "snr         (B,)":         (batch["snr"].shape,          (B,)),
    }
    for name, (got, expected) in checks.items():
        ok = (got[:len(expected)] == torch.Size(expected))
        tag = PASS if ok else FAIL
        print(f"  {tag} {name:30s}  got={tuple(got)}")

    # dtype 확인 (snr이 float64면 AMP 충돌 위험)
    snr_dtype = batch["snr"].dtype
    dtype_ok = (snr_dtype == torch.float32)
    print(f"  {'[INFO]'} snr dtype: {snr_dtype}  {'(ok)' if dtype_ok else '(WARNING: float64 → AMP 충돌 위험)'}")


# ─────────────────────────────────────────────
# 3. GPU 합성 후 검증
# ─────────────────────────────────────────────
def check_synthesis(batch: dict, device: torch.device):
    print("\n" + "="*55)
    print("  검증 3: GPU 합성 결과 확인")
    print("="*55)

    # GPU로 이동
    gpu_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    bcm_kernel = create_bcm_kernel(cutoff_hz=500.0, sample_rate=SR, num_taps=101).to(device)
    out = apply_spatial_synthesis(gpu_batch, bcm_kernel=bcm_kernel, sample_rate=SR)

    noisy       = out["noisy"]        # (B, M, T)
    clean       = out["clean"]        # (B, M, T)
    aligned_dry = out["aligned_dry"]  # (B, 1, T)

    print(f"  {INFO} noisy       shape: {tuple(noisy.shape)}")
    print(f"  {INFO} clean       shape: {tuple(clean.shape)}")
    print(f"  {INFO} aligned_dry shape: {tuple(aligned_dry.shape)}")

    # ── 3-a. NaN / Inf 없는지 ──
    for name, t in [("noisy", noisy), ("clean", clean), ("aligned_dry", aligned_dry)]:
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        ok = (not has_nan and not has_inf)
        print(f"  {PASS if ok else FAIL} {name:12s}  NaN={has_nan}  Inf={has_inf}")

    # ── 3-b. 실제 SNR vs 목표 SNR ──
    print(f"\n  [SNR 정확도]")
    target_snrs = gpu_batch["snr"].float()
    eps = 1e-8
    for b in range(min(noisy.shape[0], 4)):
        speech_rms = torch.sqrt(torch.mean(clean[b, 0, :] ** 2) + eps)
        noise_sig  = noisy[b, 0, :] - clean[b, 0, :]
        noise_rms  = torch.sqrt(torch.mean(noise_sig ** 2) + eps)
        actual_snr = 20 * torch.log10(speech_rms / (noise_rms + eps))
        target_snr = target_snrs[b]
        diff = abs(actual_snr.item() - target_snr.item())
        ok = diff < 2.0  # 2dB 이내면 정상
        print(f"    sample {b}: target={target_snr:.1f}dB  actual={actual_snr:.1f}dB  diff={diff:.2f}dB  {PASS if ok else FAIL}")

    # ── 3-c. BCM 채널 LPF 확인 ──
    M = noisy.shape[1]
    if M >= 5:
        print(f"\n  [BCM 채널 LPF 확인 (채널 {M-1}이 BCM)]")
        b = 0
        fft_size = 2048
        air_ch  = noisy[b, 0, :fft_size].cpu().float()
        bcm_ch  = noisy[b, M-1, :fft_size].cpu().float()

        air_spec = torch.abs(torch.fft.rfft(air_ch))
        bcm_spec = torch.abs(torch.fft.rfft(bcm_ch))

        freqs      = torch.fft.rfftfreq(fft_size, d=1.0/SR)
        lo_mask    = freqs < 500
        hi_mask    = freqs > 2000

        air_hi  = air_spec[hi_mask].mean().item()
        bcm_hi  = bcm_spec[hi_mask].mean().item()

        # BCM 고주파 에너지가 공기전도 마이크보다 낮아야 함
        ok = (bcm_hi < air_hi * 0.5) if air_hi > 1e-8 else True
        ratio = bcm_hi / (air_hi + eps)
        print(f"    air_mic  고주파(>2kHz) 에너지: {air_hi:.4f}")
        print(f"    bcm_ch   고주파(>2kHz) 에너지: {bcm_hi:.4f}  (비율={ratio:.3f})  {PASS if ok else FAIL}")

    return out


# ─────────────────────────────────────────────
# 4. 오디오 저장 (직접 청취)
# ─────────────────────────────────────────────
def save_audio_samples(out: dict, n_samples: int = 2):
    print("\n" + "="*55)
    print("  검증 4: 오디오 저장 (직접 청취)")
    print("="*55)

    noisy       = out["noisy"].cpu().float()
    clean       = out["clean"].cpu().float()
    aligned_dry = out["aligned_dry"].cpu().float()

    for i in range(min(n_samples, noisy.shape[0])):
        for ch in range(min(noisy.shape[1], 3)):  # 최대 3채널만
            path = os.path.join(OUTPUT_DIR, f"sample{i}_ch{ch}_noisy.wav")
            sf.write(path, noisy[i, ch, :].numpy(), SR)

        path = os.path.join(OUTPUT_DIR, f"sample{i}_ch0_clean.wav")
        sf.write(path, clean[i, 0, :].numpy(), SR)

        path = os.path.join(OUTPUT_DIR, f"sample{i}_aligned_dry.wav")
        sf.write(path, aligned_dry[i, 0, :].numpy(), SR)

        print(f"  {PASS} sample {i} → {OUTPUT_DIR}/sample{i}_*.wav")


# ─────────────────────────────────────────────
# 5. 스펙트로그램 저장
# ─────────────────────────────────────────────
def save_spectrograms(out: dict, n_samples: int = 1):
    print("\n" + "="*55)
    print("  검증 5: 스펙트로그램 저장")
    print("="*55)

    noisy       = out["noisy"].cpu().float()
    clean       = out["clean"].cpu().float()
    aligned_dry = out["aligned_dry"].cpu().float()
    M = noisy.shape[1]

    for i in range(min(n_samples, noisy.shape[0])):
        n_plots = M + 2  # 채널별 noisy + clean ch0 + aligned_dry
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))

        def plot_spec(ax, sig, title):
            sig_np = sig.numpy()
            ax.specgram(sig_np, Fs=SR, NFFT=512, noverlap=256, cmap="magma")
            ax.set_title(title, fontsize=9)
            ax.set_ylabel("Hz")
            ax.set_ylim(0, SR // 2)

        for ch in range(M):
            label = f"Noisy ch{ch}" + (" (BCM)" if ch == M-1 and M >= 5 else "")
            plot_spec(axes[ch], noisy[i, ch, :], label)

        plot_spec(axes[M],   clean[i, 0, :],       "Clean ch0 (spatialized)")
        plot_spec(axes[M+1], aligned_dry[i, 0, :], "Aligned Dry ch0 (target)")

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"sample{i}_spectrogram.png")
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"  {PASS} sample {i} → {path}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",     default="train", choices=["train", "val", "test"])
    parser.add_argument("--n_batches", type=int, default=2, help="검증할 배치 수")
    parser.add_argument("--batch_size",type=int, default=4)
    parser.add_argument("--db_path",   default=DB_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{INFO} device: {device}")
    print(f"{INFO} split:  {args.split}")
    print(f"{INFO} output: {OUTPUT_DIR}/")

    # ── 검증 1: DB split ──
    check_db_splits(args.db_path)

    # ── DataLoader 구성 ──
    dataset = SpatialMixingDataset(
        db_path=args.db_path,
        split=args.split,
        snr_range=(-5, 20),
    )
    print(f"\n{INFO} {args.split} dataset 크기: {len(dataset):,} 개")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,          # 디버깅 시 0이 편함
        collate_fn=collate_gpu_synthesis,
    )

    out = None
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            # ── 검증 2: 배치 shape ──
            check_batch_shapes(batch, args.split)

            # ── 검증 3 ~ 5: 합성 결과 ──
            out = check_synthesis(batch, device)
            save_audio_samples(out, n_samples=min(args.batch_size, 2))
            save_spectrograms(out, n_samples=1)

        if batch_idx + 1 >= args.n_batches:
            break

    print("\n" + "="*55)
    print(f"  완료! 결과물 위치: {OUTPUT_DIR}/")
    print(f"  - *.wav  : 직접 재생하여 음질 확인")
    print(f"  - *.png  : 스펙트로그램으로 채널별 특성 확인")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
