"""
test_model/LABNet (ABNet) vs src/models/LABNet 인아웃 동일성 검증.

동일한 가중치를 복사 후 같은 입력을 넣어 출력이 일치하는지 확인한다.
"""
import sys, os

# -- path setup --
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
# ABNet은 generator 패키지만 필요 (model/__init__.py 통해 fast_bss_eval 로드 방지)
ABNET_PKG = os.path.join(ROOT, 'test_model', 'LABNet', 'model', 'generator')
sys.path.insert(0, os.path.join(ROOT, 'test_model', 'LABNet', 'model'))
sys.path.insert(0, os.path.join(ROOT, 'test_model', 'LABNet'))

import torch
import numpy as np

# ── 원본 ABNet (model/__init__ 우회해서 직접 임포트) ─────────────────────────
from generator.abnet import ABNet

# ── 우리 LABNet ──────────────────────────────────────────────────────────────
from src.models.labnet import LABNet


def copy_weights(src_model: torch.nn.Module, dst_model: torch.nn.Module):
    """src_model의 state_dict를 dst_model로 복사 (key 매핑 포함)."""
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    # dst key → src key 매핑 (LABNet은 BaseSEModel STFT buffer가 추가로 있음)
    not_found = []
    for dst_key in dst_sd:
        if dst_key in src_sd and src_sd[dst_key].shape == dst_sd[dst_key].shape:
            dst_sd[dst_key] = src_sd[dst_key].clone()
        else:
            not_found.append(dst_key)

    dst_model.load_state_dict(dst_sd)
    return not_found


def run_abnet(abnet: ABNet, x: torch.Tensor):
    """ABNet forward; returns (est, est_mag, est_spec)."""
    with torch.no_grad():
        res = abnet(x)
    # est: (B, T)  -> (B, 1, T)
    return (res['est'].unsqueeze(1),
            res['est_mag'],
            res['est_spec'])


def run_labnet(labnet: LABNet, x: torch.Tensor):
    """LABNet forward_with_intermediates; returns (est_wav, est_mag, est_spec)."""
    with torch.no_grad():
        res = labnet.forward_with_intermediates(x)
    return res['est_wav'], res['est_mag'], res['est_spec']


def check(name, a, b, atol=1e-5):
    match = torch.allclose(a.float(), b.float(), atol=atol)
    diff = (a.float() - b.float()).abs()
    print(f"  [{name}]  shape={tuple(a.shape)}  match={match}  "
          f"max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}")
    return match


def main():
    torch.manual_seed(42)
    device = 'cpu'

    # ── 모델 생성 ─────────────────────────────────────────────────────────────
    num_channels  = 16
    n_fft         = 512
    hop_length    = 256
    compress      = 0.3

    abnet  = ABNet(num_channels=num_channels, n_fft=n_fft,
                   hop_length=hop_length, compress_factor=compress).to(device).eval()
    labnet = LABNet(in_channels=5, num_channels=num_channels,
                    n_fft=n_fft, hop_length=hop_length,
                    win_length=n_fft, compress_factor=compress).to(device).eval()

    # ── 가중치 복사 ────────────────────────────────────────────────────────────
    not_found = copy_weights(abnet, labnet)
    print("=== Weight copy ===")
    print(f"  Keys not copied (expected: STFT buffers): {not_found}")

    # ── 입력 생성 (B=2, C=5, T=32000) ────────────────────────────────────────
    B, C, T = 2, 5, 32000
    x = torch.randn(B, C, T, device=device)

    # ── ABNet forward ─────────────────────────────────────────────────────────
    # ABNet은 내부에서 모든 채널 처리
    est_ab, mag_ab, spec_ab = run_abnet(abnet, x)

    # ── LABNet forward ────────────────────────────────────────────────────────
    est_lb, mag_lb, spec_lb = run_labnet(labnet, x)

    # ── 비교 ──────────────────────────────────────────────────────────────────
    print("\n=== Output parity check ===")
    # est_mag, est_wav: float32 단순 곱셈 수준 → atol=1e-5
    ok_mag  = check("est_mag  (B,T,F)", mag_ab,  mag_lb,  atol=1e-5)
    ok_wav  = check("est_wav  (B,1,T)", est_ab,  est_lb,  atol=1e-5)
    # est_spec: Griffin-Lim iSTFT→STFT→cos/sin 누적 float32 오차 → atol=1e-4
    ok_spec_r = check("est_spec.real   ", spec_ab.real, spec_lb.real, atol=1e-4)
    ok_spec_i = check("est_spec.imag   ", spec_ab.imag, spec_lb.imag, atol=1e-4)

    print()
    if all([ok_mag, ok_wav, ok_spec_r, ok_spec_i]):
        print("✅ 모든 출력이 일치합니다.")
        print("   (est_mag/wav: atol=1e-5, est_spec: atol=1e-4 — GL float32 오차)")
    else:
        print("❌ 출력 불일치 있음. 위 diff 값 확인 필요.")

    # ── 추가: est_spec 형태 확인 ───────────────────────────────────────────────
    print("\n=== Shape summary ===")
    print(f"  ABNet  est:      {est_ab.shape}")
    print(f"  LABNet est_wav:  {est_lb.shape}")
    print(f"  ABNet  est_mag:  {mag_ab.shape}")
    print(f"  LABNet est_mag:  {mag_lb.shape}")


if __name__ == '__main__':
    main()
