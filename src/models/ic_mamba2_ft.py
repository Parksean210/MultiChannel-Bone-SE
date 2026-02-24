"""
ICMamba2FT: TF-Domain Inter-Channel Mamba2 for Multichannel Speech Enhancement
Uses STFT/iSTFT and Complex Ratio Masking (CRM) to improve high-frequency performance.
Causal: left-only padding in STFT (no future context).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from mamba_ssm import Mamba2
from .base import BaseSEModel

class Mamba2Layer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 128, d_conv: int = 4, expand: int = 2, headdim: int = 64):
        super().__init__()
        assert (d_model * expand) % headdim == 0, "d_model * expand must be divisible by headdim"
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))

class ICMamba2FT(BaseSEModel):
    """
    TF-Domain Mamba2 Model (Causal).
    - Input: raw waveform (B, M, T)
    - Encoding: causal STFT (left-pad only) -> (B, M, F, T_spec)
    - Separation: Mamba2 on (B*M, T_spec, Dim)
    - Masking: Complex Ratio Mask (CRM)
    - Decoding: causal iSTFT -> waveform (B, M, T)
    """
    def __init__(self,
                 in_channels: int = 5,
                 n_fft: int = 512,
                 hop_length: int = 128,
                 bot_num_feats: int = 128,
                 out_channels: int = 64,
                 d_state: int = 128,
                 headdim: int = 64,
                 num_layers: int = 8,
                 use_checkpoint: bool = False):
        
        # Hamming window: boundary값 0.08 > 0 → center=False iSTFT NOLA 조건 만족
        super().__init__(in_channels=in_channels, n_fft=n_fft, hop_length=hop_length, window_type="hamming")
        
        self.freq_bins = n_fft // 2 + 1
        self.bot_num_feats = bot_num_feats
        self.out_channels = out_channels
        
        # Encoder: Complex (Real+Imag) 2*F -> N
        self.bottleneck = nn.Conv1d(2 * self.freq_bins, bot_num_feats, 1)
        
        # Channel Projection: M -> C
        self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Mamba2 Stack
        self.layers = nn.ModuleList([
            Mamba2Layer(bot_num_feats, d_state=d_state, headdim=headdim)
            for _ in range(num_layers)
        ])
        
        # Mask Generation: N -> 2*F (Real/Imag Mask for Complex Ratio Mask)
        self.mask_proj = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1) # 2 for Real/Imag
        )
        
        # Final projection to map bot_num_feats back to spectral features
        self.mask_feat_proj = nn.Linear(bot_num_feats, self.freq_bins)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Causal STFT: left-pad (win_length - hop_length) only, center=False.
        Hamming window 사용으로 NOLA 조건 만족 (경계값 0.08 > 0).
        """
        B, C, T = x.shape
        pad_l = self.win_length - self.hop_length   # 384 (for win=512, hop=128)
        x_flat = x.view(B * C, T)
        x_padded = F.pad(x_flat, (pad_l, 0))
        spec = torch.stft(
            x_padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=False,
        )
        return spec.view(B, C, spec.shape[1], spec.shape[2])

    def istft(self, x_spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """Causal iSTFT: center=False, 왼쪽 패딩 제거 후 원본 길이로 crop."""
        B, C, F, T_spec = x_spec.shape
        pad_l = self.win_length - self.hop_length
        x_flat = x_spec.view(B * C, F, T_spec)
        # padded 신호 길이 기준으로 요청 (max_output = pad_l + T)
        padded_length = (length + pad_l) if length is not None else None
        wav = torch.istft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            length=padded_length,
        )
        wav = wav[:, pad_l:]
        if length is not None:
            wav = wav[:, :length]
        return wav.view(B, C, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, T = x.shape

        # 1. Causal STFT: (B, M, T) -> (B, M, F, L) Complex
        spec = self.stft(x)
        L = spec.shape[-1]
        
        # 2. Extract Real and Imaginary parts: (B, M, 2*F, L)
        # We concatenate along the frequency dimension for the bottleneck
        spec_ri = torch.cat([spec.real, spec.imag], dim=2) # (B, M, 2*F, L)
        
        # 3. Bottleneck: (B*M, 2*F, L) -> (B*M, N, L)
        w_bot = self.bottleneck(spec_ri.view(B * M, 2 * self.freq_bins, L))
        
        # 4. Channel Projection: (B, M, N, L) -> (B, C, N, L)
        y = w_bot.view(B, M, self.bot_num_feats, L)
        y = self.channel_proj(y)
        C = y.shape[1]
        
        # 5. Mamba2 Separation: (B*C, L, N)
        y = y.permute(0, 1, 3, 2).reshape(B * C, L, self.bot_num_feats)
        for layer in self.layers:
             y = layer(y)
             
        # (B*C, L, N) -> (B, C, N, L)
        y = y.reshape(B, C, L, self.bot_num_feats).permute(0, 1, 3, 2)
        
        # 6. Mask Estimation (CRM)
        # N -> spectral mapping
        y = y.permute(0, 1, 3, 2) # (B, C, L, N)
        y = self.mask_feat_proj(y) # (B, C, L, F)
        y = y.permute(0, 1, 3, 2) # (B, C, F, L)
        
        mask_raw = self.mask_proj(y) # (B, 2*M, F, L)
        mask_raw = mask_raw.view(B, 2, M, self.freq_bins, L)
        
        mask_real = mask_raw[:, 0] # (B, M, F, L)
        mask_imag = mask_raw[:, 1] # (B, M, F, L)
        
        # 7. Apply Complex Ratio Mask
        # Enhanced = Noisy * Mask (Complex multiplication)
        # (A+Bi) * (C+Di) = (AC-BD) + (AD+BC)i
        est_real = spec.real * mask_real - spec.imag * mask_imag
        est_imag = spec.real * mask_imag + spec.imag * mask_real
        
        est_spec = torch.complex(est_real, est_imag)
        
        # 8. iSTFT: (B, M, F, L) -> (B, M, T)
        est_wav = self.istft(est_spec, length=T)
        return est_wav
