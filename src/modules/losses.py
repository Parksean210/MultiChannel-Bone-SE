import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) 손실 함수.
    음성 향상 및 분리 분야에서 표준적으로 사용되는 시간 영역(Time-domain) 평가지표이자 손실 함수입니다.
    참조 논문: https://arxiv.org/abs/1811.02508
    """
    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: 수치적 안정성을 위한 미소값 (Divide by zero 방지)
        """
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: 모델이 추정한 음성 파형 (B, C, T) 또는 (B, T)
            targets: 참조용 클린 음성 파형 (B, C, T) 또는 (B, T)
        Returns:
            Negative SI-SDR (최소화를 위해 음수 처리된 값)
        """
        # Ensure shapes match
        if preds.shape != targets.shape:
             raise RuntimeError(f"Shape mismatch: preds {preds.shape} != targets {targets.shape}")

        # Flatten if multi-channel: (B, C, T) -> (B*C, T)
        if preds.dim() == 3:
            B, C, T = preds.shape
            preds = preds.view(B * C, T)
            targets = targets.view(B * C, T)

        # Remove mean (Zero-centering)
        preds_mean = preds - torch.mean(preds, dim=-1, keepdim=True)
        targets_mean = targets - torch.mean(targets, dim=-1, keepdim=True)

        # Calculate Scale-Invariant SNR
        # alpha = <x, s> / <s, s>
        target_energy = torch.sum(targets_mean ** 2, dim=-1, keepdim=True) + self.eps
        alpha = torch.sum(preds_mean * targets_mean, dim=-1, keepdim=True) / target_energy
        
        target_scaled = alpha * targets_mean
        
        noise = preds_mean - target_scaled
        
        si_sdr = 10 * torch.log10(
            (torch.sum(target_scaled ** 2, dim=-1) + self.eps) / 
            (torch.sum(noise ** 2, dim=-1) + self.eps)
        )
        
        # We want to maximize SI-SDR, so minimize negative SI-SDR
        return -torch.mean(si_sdr)

class STFTLoss(nn.Module):
    """
    단일 해상도 STFT 손실 함수.
    Spectral Convergence Loss와 Log Magnitude Loss의 합으로 구성됩니다.
    """
    def __init__(self, 
                 fft_size: int = 1024, 
                 hop_size: int = 120, 
                 win_length: int = 600, 
                 window: str = "hann_window"):
        """
        Args:
            fft_size: FFT 포인트 크기
            hop_size: 프레임 간격 (Hop size)
            win_length: 윈도우 길이
            window: 사용할 윈도우 함수 명칭
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move window to correct device if needed
        if self.window.device != preds.device:
            self.window = self.window.to(preds.device)

        # Flatten (B, C, T) -> (B*C, T)
        if preds.dim() == 3:
            B, C, T = preds.shape
            preds = preds.view(B * C, T)
            targets = targets.view(B * C, T)

        # Compute STFT
        preds_stft = torch.stft(
            preds, self.fft_size, self.hop_size, self.win_length, self.window,
            return_complex=True
        )
        targets_stft = torch.stft(
            targets, self.fft_size, self.hop_size, self.win_length, self.window,
            return_complex=True
        )
        
        preds_mag = torch.abs(preds_stft) + 1e-8
        targets_mag = torch.abs(targets_stft) + 1e-8

        # 1. Spectral Convergence Loss
        # || |Y| - |X| ||_F / || |Y| ||_F
        sc_loss = torch.norm(targets_mag - preds_mag, p="fro") / torch.norm(targets_mag, p="fro")

        # 2. Log Magnitude Loss
        # || log(|Y|) - log(|X|) ||_1
        mag_loss = F.l1_loss(torch.log(targets_mag), torch.log(preds_mag))

        return sc_loss + mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    """
    다중 해상도(Multi-Resolution) STFT 손실 함수.
    다양한 FFT 해상도에서 손실을 계산하여 주파수 영역의 세밀한 특징과 거시적인 특징을 동시에 학습합니다.
    """
    def __init__(self,
                 fft_sizes: List[int] = [1024, 2048, 512],
                 hop_sizes: List[int] = [120, 240, 50],
                 win_lengths: List[int] = [600, 1200, 240],
                 window: str = "hann_window",
                 factor_sc: float = 0.5,
                 factor_mag: float = 0.5):
        """
        Args:
            fft_sizes: 각 STFT 계층의 FFT 크기 리스트
            hop_sizes: 각 STFT 계층의 Hop 크기 리스트
            win_lengths: 각 STFT 계층의 윈도우 길이 리스트
            factor_sc: Spectral Convergence 가중치
            factor_mag: Log Magnitude 가중치
        """
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.stft_losses = nn.ModuleList([
            STFTLoss(fft_size, hop_size, win_len, window)
            for fft_size, hop_size, win_len in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for f in self.stft_losses:
            loss += f(preds, targets)
        
        return loss / len(self.stft_losses)

class CompositeLoss(nn.Module):
    """
    복합 손실 함수 (Hybrid Loss).
    시간 영역의 SI-SDR 손실과 주파수 영역의 Multi-Resolution STFT 손실을 결합합니다.
    """
    def __init__(self, alpha: float = 0.1): 
        """
        Args:
            alpha: 주파수 영역 손실(MR-STFT)에 적용할 가중치
        """
        super().__init__()
        self.sisdr = SISDRLoss()
        self.mrstft = MultiResolutionSTFTLoss()
        self.alpha = alpha
        
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_time = self.sisdr(preds, targets)
        loss_freq = self.mrstft(preds, targets)
        
        # Usually STFT loss is much larger than SI-SDR, 
        # so we might need alpha to balance them or just sum them.
        # This is a simple weighted sum.
        return loss_time + self.alpha * loss_freq
