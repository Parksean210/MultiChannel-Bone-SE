"""
ABNet vs LABNet 단계별 중간값 비교 디버그 스크립트.
어느 단계에서 값이 갈라지는지 pinpoint.
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'test_model', 'LABNet', 'model'))
sys.path.insert(0, os.path.join(ROOT, 'test_model', 'LABNet'))

import torch
from generator.abnet import ABNet
from src.models.labnet import LABNet


def copy_weights(src_model, dst_model):
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()
    for k in dst_sd:
        if k in src_sd and src_sd[k].shape == dst_sd[k].shape:
            dst_sd[k] = src_sd[k].clone()
    dst_model.load_state_dict(dst_sd)


def close(name, a, b, atol=1e-5):
    ok = torch.allclose(a.float(), b.float(), atol=atol)
    diff = (a.float() - b.float()).abs()
    print(f"  {name:30s} match={ok}  max={diff.max():.3e}  mean={diff.mean():.3e}")
    return ok


torch.manual_seed(0)
device = 'cpu'

abnet  = ABNet(num_channels=16, n_fft=512, hop_length=256, compress_factor=0.3).eval()
labnet = LABNet(in_channels=5, num_channels=16, n_fft=512, hop_length=256,
                win_length=512, compress_factor=0.3).eval()
copy_weights(abnet, labnet)

B, C, T = 1, 5, 16000
x = torch.randn(B, C, T)

print("=" * 60)
print("Step 1: STFT")
# ABNet STFT
src_flat = x.reshape(B*C, T)
ab_spec = abnet.apply_stft(src_flat)          # (B*C, T, F) complex
ab_spec_pc = abnet.power_compress(ab_spec)    # (B*C, T, F) power-compressed complex
ab_mag = ab_spec_pc.abs()                     # (B*C, T, F)
ab_pha = ab_spec_pc.angle()                   # (B*C, T, F)

# LABNet STFT
lb_spec_4d = labnet.stft(x)                   # (B, C, F, T)
lb_spec = lb_spec_4d.permute(0, 1, 3, 2)      # (B, C, T, F)
lb_spec_flat = lb_spec.reshape(B*C, *lb_spec.shape[2:])  # (B*C, T, F)
lb_mag = lb_spec_flat.abs().clamp(min=1e-8) ** labnet.compress_factor
lb_pha = lb_spec_flat.angle()

close("STFT mag (raw, before compress)", ab_spec.abs(), lb_spec_flat.abs())
close("mag (power-compressed)         ", ab_mag,        lb_mag)
close("pha                            ", ab_pha,         lb_pha)

print()
print("Step 2: GD / IFD features")
ab_gd  = ABNet.cal_gd(ab_pha)
ab_ifd = abnet.cal_ifd(ab_pha)
lb_gd  = labnet._cal_gd(lb_pha)
lb_ifd = labnet._cal_ifd(lb_pha)
close("GD                             ", ab_gd,  lb_gd)
close("IFD                            ", ab_ifd, lb_ifd)

print()
print("Step 3: Encoder input features")
ab_feat = torch.stack([ab_mag, ab_gd/torch.pi, ab_ifd/torch.pi], dim=1)  # (B*C,3,T,F)
lb_feat = torch.stack([lb_mag, lb_gd/torch.pi, lb_ifd/torch.pi], dim=1)
close("encoder input                  ", ab_feat, lb_feat)

print()
print("Step 4: Encoder output")
with torch.no_grad():
    ab_enc = abnet.encoder(ab_feat)  # list of 3
    lb_enc = labnet.encoder(lb_feat)
for i, (a, b) in enumerate(zip(ab_enc, lb_enc)):
    close(f"encoder_out[{i}]               ", a, b)

print()
print("Step 5: est_mag (mask decoder output)")
with torch.no_grad():
    ab_res = abnet(x)                    # full forward
    lb_res = labnet.forward_with_intermediates(x)

ab_est_mag  = ab_res['est_mag']
lb_est_mag  = lb_res['est_mag']
ab_est_spec = ab_res['est_spec']
lb_est_spec = lb_res['est_spec']
ab_est      = ab_res['est'].unsqueeze(1)  # (B, 1, T)
lb_est_wav  = lb_res['est_wav']

close("est_mag                        ", ab_est_mag,       lb_est_mag)
close("est_spec.real                  ", ab_est_spec.real, lb_est_spec.real)
close("est_spec.imag                  ", ab_est_spec.imag, lb_est_spec.imag)
close("est_wav                        ", ab_est,           lb_est_wav)
