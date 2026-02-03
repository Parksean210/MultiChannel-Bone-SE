import torch
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base import BaseSEModel

def test_reconstruction():
    print("🚀 BaseSEModel 검증 시작 (Perfect Reconstruction Test)\n")
    
    # 설정
    B, C, T = 2, 5, 16000 # 배치 2, 채널 5, 1초 분량
    x = torch.randn(B, C, T)
    
    # 1. STFT / iSTFT 검증 (기본 50% Overlap)
    model_stft = BaseSEModel(n_fft=1024, hop_length=256, win_length=1024, window_type="hann")
    spec = model_stft.stft(x)
    recon_stft = model_stft.istft(spec, length=T)
    
    diff_stft = torch.abs(x - recon_stft).max().item()
    print(f"[STFT -> iSTFT] Max Difference: {diff_stft:.2e}")
    assert diff_stft < 1e-6, "STFT Reconstruction Failed!"

    # 2. Time-domain Framing 검증 (50% Overlap)
    model_time_overlap = BaseSEModel(win_length=400, hop_length=200, window_type="hann")
    frames = model_time_overlap.to_frames(x)
    print(f"Frames shape: {frames.shape}")
    recon_time = model_time_overlap.from_frames(frames, length=T)
    print(f"Recon Time shape: {recon_time.shape}")
    
    diff_time = torch.abs(x - recon_time).max().item()
    print(f"[to_frames -> from_frames (50% overlap)] Max Difference: {diff_time:.2e}")
    
    if diff_time >= 1e-6:
        print("\nDEBUG: First 10 samples comparison")
        print("Original:", x[0, 0, :10])
        print("Reconstructed:", recon_time[0, 0, :10])
        print("Diff:", (x - recon_time)[0, 0, :10])
        
    assert diff_time < 1e-6, "Time-domain Overlay Reconstruction Failed!"

    # 3. Time-domain Framing 검증 (0% Overlap / Non-overlapping)
    model_no_overlap = BaseSEModel(win_length=320, hop_length=320, window_type="rect")
    frames_no = model_no_overlap.to_frames(x)
    recon_no = model_no_overlap.from_frames(frames_no, length=T)
    
    diff_no = torch.abs(x - recon_no).max().item()
    print(f"[to_frames -> from_frames (0% overlap)] Max Difference: {diff_no:.2e}")
    assert diff_no < 1e-6, "Non-overlapping Reconstruction Failed!"

    print("\n✅ 모든 검증 통과! 수학적으로 완벽하게 복원됩니다.")

if __name__ == "__main__":
    test_reconstruction()
