import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'test_model', 'LABNet', 'model'))
sys.path.insert(0, os.path.join(ROOT, 'test_model', 'LABNet'))

import torch
from generator.abnet import ABNet
from src.models.labnet import LABNet

torch.manual_seed(0)
B, C, T = 1, 5, 16000
x = torch.randn(B, C, T)

abnet = ABNet(num_channels=16).eval()
labnet = LABNet(in_channels=5, num_channels=16).eval()

# Raw STFT complex (T-first vs F-first 정렬 확인)
src_flat = x.reshape(B*C, T)
ab_raw = abnet.apply_stft(src_flat)  # (B*C, T, F)

lb_raw_4d = labnet.stft(x)                       # (B, C, F, T)
lb_raw = lb_raw_4d.permute(0,1,3,2).reshape(B*C, lb_raw_4d.shape[3], lb_raw_4d.shape[2])

print("=== Raw STFT complex ===")
print(f"  ABNet  shape: {ab_raw.shape}")
print(f"  LABNet shape: {lb_raw.shape}")
print(f"  real match:   {torch.allclose(ab_raw.real, lb_raw.real)}")
print(f"  imag match:   {torch.allclose(ab_raw.imag, lb_raw.imag)}")
print(f"  max diff real: {(ab_raw.real - lb_raw.real).abs().max():.2e}")
print(f"  max diff imag: {(ab_raw.imag - lb_raw.imag).abs().max():.2e}")

# Phase via power_compress vs direct
ab_pc = abnet.power_compress(ab_raw)
ab_pha_direct = ab_raw.angle()
ab_pha_via_pc = ab_pc.angle()
diff_pha = (ab_pha_direct - ab_pha_via_pc).abs()
print(f"\n=== ABNet: direct pha vs power_compress → pha ===")
print(f"  max diff:             {diff_pha.max():.4e}")
print(f"  #bins diff > 0.001:   {(diff_pha > 0.001).sum()}")
print(f"  (= 2π diff bins):     {(diff_pha > 6.28).sum()}")

# LABNet uses spec.angle() directly — compare to ABNet's src_pha
lb_pha_direct = lb_raw.angle()
diff2 = (ab_pha_via_pc - lb_pha_direct).abs()
print(f"\n=== ABNet power_compress.angle() vs LABNet direct.angle() ===")
print(f"  max diff:             {diff2.max():.4e}")
print(f"  #bins diff > 0.001:   {(diff2 > 0.001).sum()}")

# Fix: use power_compress.angle() in LABNet as well
lb_pha_fixed = ab_pc.angle()  # same window, same STFT → should match
diff3 = (ab_pha_via_pc - lb_pha_fixed).abs()
print(f"\n=== Both using power_compress.angle() ===")
print(f"  max diff:             {diff3.max():.4e}")
