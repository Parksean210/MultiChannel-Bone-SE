import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


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

        # alpha = <x, s> / <s, s>
        target_energy = torch.sum(targets_mean ** 2, dim=-1, keepdim=True) + self.eps
        alpha = torch.sum(preds_mean * targets_mean, dim=-1, keepdim=True) / target_energy

        target_scaled = alpha * targets_mean
        noise = preds_mean - target_scaled

        si_sdr = 10 * torch.log10(
            (torch.sum(target_scaled ** 2, dim=-1) + self.eps) /
            (torch.sum(noise ** 2, dim=-1) + self.eps)
        )

        return -torch.mean(si_sdr)


class SDRLoss(nn.Module):
    """
    Signal-to-Distortion Ratio (SDR) 손실 함수.
    스케일을 고정한 상태에서 신호 대 왜곡 비율을 계산합니다.
    SI-SDR과 달리 스케일 불변 조정(alpha projection)을 수행하지 않습니다.
    SDR = 10 * log10(||target||^2 / ||target - pred||^2)
    """
    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: 수치적 안정성을 위한 미소값
        """
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: 모델 출력 파형 (B, C, T) 또는 (B, T)
            targets: 참조 클린 파형 (B, C, T) 또는 (B, T)
        Returns:
            Negative SDR
        """
        if preds.shape != targets.shape:
            raise RuntimeError(f"Shape mismatch: preds {preds.shape} != targets {targets.shape}")

        if preds.dim() == 3:
            B, C, T = preds.shape
            preds = preds.view(B * C, T)
            targets = targets.view(B * C, T)

        noise = preds - targets
        sdr = 10 * torch.log10(
            (torch.sum(targets ** 2, dim=-1) + self.eps) /
            (torch.sum(noise ** 2, dim=-1) + self.eps)
        )
        return -torch.mean(sdr)


class WaveformLoss(nn.Module):
    """
    파형(Waveform) 도메인 손실 함수.
    시간 영역에서 직접 L1 또는 L2 거리를 계산합니다.
    구현이 단순하여 다른 손실과 보조적으로 사용합니다.
    """
    def __init__(self, loss_type: str = "l1"):
        """
        Args:
            loss_type: 'l1' (MAE) 또는 'l2' (MSE)
        """
        super().__init__()
        assert loss_type in ("l1", "l2"), "loss_type must be 'l1' or 'l2'"
        self.loss_fn = F.l1_loss if loss_type == "l1" else F.mse_loss

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(preds, targets)


class STFTLoss(nn.Module):
    """
    단일 해상도 STFT 손실 함수.
    Spectral Convergence Loss와 Log Magnitude Loss의 합으로 구성됩니다.
    'f_weight' 옵션을 통해 고주파 영역에 가중치를 줄 수 있습니다.
    """
    def __init__(self,
                 fft_size: int = 1024,
                 hop_size: int = 120,
                 win_length: int = 600,
                 window: str = "hann_window",
                 use_f_weight: bool = False):
        """
        Args:
            fft_size: FFT 포인트 크기
            hop_size: 프레임 간격 (Hop size)
            win_length: 윈도우 길이
            window: 사용할 윈도우 함수 명칭
            use_f_weight: 고주파 빈(Bin)으로 갈수록 높은 가중치를 부여할지 여부
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer('window', getattr(torch, window)(win_length))

        if use_f_weight:
            f_weight = self._build_f_weight(fft_size // 2 + 1)
            self.register_buffer('f_weight',
                                 torch.from_numpy(f_weight).float().view(1, -1, 1))
        else:
            self.f_weight = None

    @staticmethod
    def _build_f_weight(n_freqs: int,
                        weight_min: float = 1.0,
                        weight_max: float = 4.0) -> np.ndarray:
        """저주파→고주파 선형 증가 가중치 [weight_min, ..., weight_max] 생성."""
        return np.linspace(weight_min, weight_max, n_freqs, dtype=np.float32)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        window = self.window.to(preds.device)

        if preds.dim() == 3:
            B, C, T = preds.shape
            preds = preds.view(B * C, T)
            targets = targets.view(B * C, T)

        preds_stft = torch.stft(
            preds, self.fft_size, self.hop_size, self.win_length, window,
            return_complex=True, center=True
        )
        targets_stft = torch.stft(
            targets, self.fft_size, self.hop_size, self.win_length, window,
            return_complex=True, center=True
        )

        preds_mag = torch.abs(preds_stft) + 1e-8
        targets_mag = torch.abs(targets_stft) + 1e-8
        error = targets_mag - preds_mag

        if self.f_weight is not None:
            w = self.f_weight.to(preds.device)
            sc_loss  = torch.norm(error * w, p="fro") / torch.norm(targets_mag * w, p="fro")
            mag_loss = F.l1_loss(torch.log(preds_mag) * w, torch.log(targets_mag) * w)
        else:
            sc_loss  = torch.norm(error, p="fro") / torch.norm(targets_mag, p="fro")
            mag_loss = F.l1_loss(torch.log(preds_mag), torch.log(targets_mag))

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
                 factor_mag: float = 0.5,
                 use_f_weight: bool = False):
        """
        Args:
            fft_sizes: 각 STFT 계층의 FFT 크기 리스트
            hop_sizes: 각 STFT 계층의 Hop 크기 리스트
            win_lengths: 각 STFT 계층의 윈도우 길이 리스트
            window: 사용할 윈도우 함수 명칭
            factor_sc: Spectral Convergence 가중치
            factor_mag: Log Magnitude 가중치
            use_f_weight: 고주파 가중 손실 사용 여부
        """
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.stft_losses = nn.ModuleList([
            STFTLoss(fft_size, hop_size, win_len, window, use_f_weight=use_f_weight)
            for fft_size, hop_size, win_len in zip(fft_sizes, hop_sizes, win_lengths)
        ])

        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for f in self.stft_losses:
            loss += f(preds, targets)

        return loss / len(self.stft_losses)


class ComplexSTFTLoss(nn.Module):
    """
    복소수 STFT 손실 함수.
    실수부(Real), 허수부(Imag), 크기(Magnitude) 모두를 손실 계산에 활용합니다.
    위상(Phase) 정보를 직접 학습하여 위상 일관성을 개선합니다.
    DCCRN, DPCRN 등 복소수 도메인 모델에서 표준적으로 사용됩니다.

    loss = complex_weight * (L1(real) + L1(imag)) + mag_weight * L1(log|mag|)
    """
    def __init__(self,
                 fft_size: int = 512,
                 hop_size: int = 128,
                 win_length: int = 512,
                 window: str = "hann_window",
                 mag_weight: float = 1.0,
                 complex_weight: float = 1.0):
        """
        Args:
            fft_size: FFT 포인트 크기
            hop_size: 프레임 간격
            win_length: 윈도우 길이
            window: 윈도우 함수 명칭
            mag_weight: 크기 손실 가중치
            complex_weight: 복소수(실수부+허수부) 손실 가중치
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.mag_weight = mag_weight
        self.complex_weight = complex_weight
        self.register_buffer('window', getattr(torch, window)(win_length))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        window = self.window.to(preds.device)

        if preds.dim() == 3:
            B, C, T = preds.shape
            preds = preds.view(B * C, T)
            targets = targets.view(B * C, T)

        preds_stft = torch.stft(
            preds, self.fft_size, self.hop_size, self.win_length, window,
            return_complex=True, center=True
        )
        targets_stft = torch.stft(
            targets, self.fft_size, self.hop_size, self.win_length, window,
            return_complex=True, center=True
        )

        # Complex domain: L1 on real + imag separately
        complex_loss = (F.l1_loss(preds_stft.real, targets_stft.real) +
                        F.l1_loss(preds_stft.imag, targets_stft.imag))

        # Log magnitude loss
        preds_mag = torch.abs(preds_stft) + 1e-8
        targets_mag = torch.abs(targets_stft) + 1e-8
        mag_loss = F.l1_loss(torch.log(preds_mag), torch.log(targets_mag))

        return self.complex_weight * complex_loss + self.mag_weight * mag_loss


class MelSpectrogramLoss(nn.Module):
    """
    멜 스펙트로그램 L1 손실 함수.
    인간 청각의 비선형 주파수 인지를 반영한 멜 스케일에서 손실을 계산합니다.
    HiFi-GAN, Encodec, DAC(Descript Audio Codec) 등 고품질 오디오 모델에서 표준적으로 사용됩니다.

    멜 필터뱅크는 __init__ 에서 1회 생성 후 register_buffer로 캐싱됩니다.
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 fft_size: int = 1024,
                 hop_size: int = 256,
                 win_length: int = 1024,
                 window: str = "hann_window",
                 fmin: float = 0.0,
                 fmax: Optional[float] = None,
                 log_scale: bool = True):
        """
        Args:
            sample_rate: 오디오 샘플레이트 (Hz)
            n_mels: 멜 필터 수
            fft_size: FFT 포인트 크기
            hop_size: 프레임 간격
            win_length: 윈도우 길이
            window: 윈도우 함수 명칭
            fmin: 멜 필터 최저 주파수 (Hz)
            fmax: 멜 필터 최고 주파수 (Hz), None이면 sample_rate/2
            log_scale: True이면 log 멜 스펙트로그램 사용
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.log_scale = log_scale

        self.register_buffer('window', getattr(torch, window)(win_length))

        fmax = fmax or (sample_rate / 2.0)
        mel_fb = self._build_mel_filterbank(n_mels, fft_size, sample_rate, fmin, fmax)
        self.register_buffer('mel_fb', torch.from_numpy(mel_fb).float())  # (n_mels, F)

    @staticmethod
    def _build_mel_filterbank(n_mels: int, n_fft: int, sample_rate: int,
                               fmin: float, fmax: float) -> np.ndarray:
        def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
        def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        bin_points = np.clip(bin_points, 0, n_fft // 2)

        n_freqs = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_left   = bin_points[m - 1]
            f_center = bin_points[m]
            f_right  = bin_points[m + 1]
            if f_center > f_left:
                for k in range(f_left, f_center):
                    filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
            if f_right > f_center:
                for k in range(f_center, f_right):
                    filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
        return filterbank

    def _compute_mel(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T) → mel: (N, n_mels, frames)"""
        window = self.window.to(x.device)
        stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, window,
                          return_complex=True, center=True)
        power = torch.abs(stft) ** 2                        # (N, F, T)
        mel = torch.matmul(self.mel_fb.to(x.device), power) # (N, n_mels, T)
        if self.log_scale:
            mel = torch.log(mel.clamp(min=1e-8))
        return mel

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.dim() == 3:
            B, C, T = preds.shape
            preds = preds.view(B * C, T)
            targets = targets.view(B * C, T)

        return F.l1_loss(self._compute_mel(preds), self._compute_mel(targets))


class MultiScaleMelLoss(nn.Module):
    """
    다중 해상도 멜 스펙트로그램 손실 함수.
    다양한 FFT 해상도에서 멜 스펙트로그램을 계산하여 단기/장기 주파수 특성을 모두 학습합니다.
    Encodec, DAC(Descript Audio Codec)에서 채택한 방식입니다.
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 fft_sizes: List[int] = [512, 1024, 2048],
                 hop_sizes: List[int] = [128, 256, 512],
                 win_lengths: List[int] = [512, 1024, 2048],
                 window: str = "hann_window",
                 fmin: float = 0.0,
                 fmax: Optional[float] = None):
        """
        Args:
            sample_rate: 오디오 샘플레이트 (Hz)
            n_mels: 멜 필터 수
            fft_sizes: 각 해상도의 FFT 크기
            hop_sizes: 각 해상도의 Hop 크기
            win_lengths: 각 해상도의 윈도우 길이
            window: 윈도우 함수 명칭
            fmin: 멜 필터 최저 주파수 (Hz)
            fmax: 멜 필터 최고 주파수 (Hz)
        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.mel_losses = nn.ModuleList([
            MelSpectrogramLoss(
                sample_rate=sample_rate, n_mels=n_mels,
                fft_size=fft, hop_size=hop, win_length=win,
                window=window, fmin=fmin, fmax=fmax
            )
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = sum(f(preds, targets) for f in self.mel_losses)
        return loss / len(self.mel_losses)


class AWeightedSTFTLoss(nn.Module):
    """
    A-가중 STFT 손실 함수 (Perceptual Loss).
    인간 청각계의 주파수별 민감도를 모사한 A-가중치(IEC 61672)를 STFT 손실에 적용합니다.
    저주파/고주파보다 1~4 kHz 대역(말소리 핵심 대역)에 더 높은 가중치를 부여합니다.

    A(f) = 12194² * f⁴ / ((f²+20.6²) * sqrt((f²+107.7²)(f²+737.9²)) * (f²+12194²))
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 fft_size: int = 1024,
                 hop_size: int = 256,
                 win_length: int = 1024,
                 window: str = "hann_window"):
        """
        Args:
            sample_rate: 오디오 샘플레이트 (Hz) — A-가중치 주파수 축 계산에 필요
            fft_size: FFT 포인트 크기
            hop_size: 프레임 간격
            win_length: 윈도우 길이
            window: 윈도우 함수 명칭
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer('window', getattr(torch, window)(win_length))

        # A-가중치 커브를 FFT 빈별로 미리 계산하여 캐싱
        n_freqs = fft_size // 2 + 1
        freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
        freqs[0] = 1e-6  # DC 빈 division-by-zero 방지
        a_weights = self._a_weighting(freqs)
        a_weights = (a_weights / a_weights.max()).astype(np.float32)
        self.register_buffer('a_weights',
                             torch.from_numpy(a_weights).view(1, -1, 1))  # (1, F, 1)

    @staticmethod
    def _a_weighting(freqs: np.ndarray) -> np.ndarray:
        """IEC 61672 A-가중치 (선형 스케일)"""
        f2 = freqs ** 2
        f4 = f2 ** 2
        num = (12194.0 ** 2) * f4
        den = (
            (f2 + 20.6 ** 2) *
            np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) *
            (f2 + 12194.0 ** 2)
        )
        return num / (den + 1e-10)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        window = self.window.to(preds.device)

        if preds.dim() == 3:
            B, C, T = preds.shape
            preds = preds.view(B * C, T)
            targets = targets.view(B * C, T)

        preds_stft = torch.stft(
            preds, self.fft_size, self.hop_size, self.win_length, window,
            return_complex=True, center=True
        )
        targets_stft = torch.stft(
            targets, self.fft_size, self.hop_size, self.win_length, window,
            return_complex=True, center=True
        )

        preds_mag = torch.abs(preds_stft) + 1e-8
        targets_mag = torch.abs(targets_stft) + 1e-8

        w = self.a_weights.to(preds.device)  # (1, F, 1) broadcast over (N, F, T)
        sc_loss = (torch.norm((targets_mag - preds_mag) * w, p="fro") /
                   torch.norm(targets_mag * w, p="fro"))
        mag_loss = F.l1_loss(torch.log(preds_mag) * w, torch.log(targets_mag) * w)

        return sc_loss + mag_loss


class CompositeLoss(nn.Module):
    """
    복합 손실 함수 (Hybrid Loss).
    시간 영역의 SI-SDR 손실과 주파수 영역의 Multi-Resolution STFT 손실을 결합합니다.
    """
    def __init__(self, alpha: float = 0.1, use_f_weight: bool = False):
        """
        Args:
            alpha: 주파수 영역 손실(MR-STFT)에 적용할 가중치
            use_f_weight: 고주파 가중 손실 사용 여부
        """
        super().__init__()
        self.sisdr = SISDRLoss()
        self.mrstft = MultiResolutionSTFTLoss(use_f_weight=use_f_weight)
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_time = self.sisdr(preds, targets)
        loss_freq = self.mrstft(preds, targets)
        return loss_time + self.alpha * loss_freq
