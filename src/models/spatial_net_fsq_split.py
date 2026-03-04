import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .base import BaseSEModel
from .spatial_net import CrossBandBlock


# ============================================================
# Causal Modules (for <100ms latency)
# ============================================================

class CausalMHSAModule(nn.Module):
    """Causal Multi-Head Self-Attention: 미래 프레임 참조를 차단하는 MHSA."""

    def __init__(self, C, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.mha = nn.MultiheadAttention(
            embed_dim=C, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # h: (B*F, T, C)
        res = h
        h = self.norm(h)
        T = h.shape[1]
        causal_mask = torch.triu(
            torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1
        )
        attn_out, _ = self.mha(h, h, h, attn_mask=causal_mask)
        return res + self.dropout(attn_out)


class CausalTConvFFNModule(nn.Module):
    """Temporal Conv FFN with causal (left-only) padding."""

    def __init__(self, C, C_prime, num_groups=8, kernel_size=5, dropout=0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(C)
        self.lin1 = nn.Linear(C, C_prime)
        self.silu1 = nn.SiLU()

        # Causal Conv1d: padding=0, left-pad manually
        self.conv1 = nn.Conv1d(C_prime, C_prime, kernel_size, padding=0, groups=num_groups)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv1d(C_prime, C_prime, kernel_size, padding=0, groups=num_groups)
        self.gn = nn.GroupNorm(num_groups, C_prime)
        self.silu3 = nn.SiLU()
        self.conv3 = nn.Conv1d(C_prime, C_prime, kernel_size, padding=0, groups=num_groups)

        self.silu4 = nn.SiLU()
        self.lin2 = nn.Linear(C_prime, C)
        self.dropout = nn.Dropout(dropout)

    def _causal_pad(self, h):
        """Left-only padding for causal convolution."""
        return F.pad(h, (self.kernel_size - 1, 0))

    def forward(self, h):
        res = h
        h = self.silu1(self.lin1(self.norm(h)))

        h = h.transpose(1, 2)  # (B*F, C', T)
        h = self.silu2(self.conv1(self._causal_pad(h)))
        h = self.silu3(self.gn(self.conv2(self._causal_pad(h))))
        h = self.silu4(self.conv3(self._causal_pad(h)))

        h = h.transpose(1, 2)  # (B*F, T, C')
        return res + self.dropout(self.lin2(h))


class CausalNarrowBandBlock(nn.Module):
    """NarrowBandBlock with causal temporal processing."""

    def __init__(self, C, C_prime, num_heads=4, num_groups=8, kernel_size=5, dropout=0.1):
        super().__init__()
        self.mhsa = CausalMHSAModule(C, num_heads, dropout)
        self.t_conv_ffn = CausalTConvFFNModule(C, C_prime, num_groups, kernel_size, dropout)

    def forward(self, h):
        # h: (B, F, T, C)
        B, F, T, C = h.shape
        h = h.reshape(B * F, T, C)

        h = self.mhsa(h)
        h = self.t_conv_ffn(h)

        return h.reshape(B, F, T, C)


# ============================================================
# FSQ (Finite Scalar Quantization)
# ============================================================

class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantization: 각 차원을 L개 레벨로 독립 양자화.
    Codebook 없음, auxiliary loss 불필요.

    [8] * 512 → 512차원 × 3bits = 1536 bits/frame = 96kbps @ 62.5fps

    학습 시: additive uniform noise (smooth gradient, Ballé et al.)
    추론 시: hard rounding (실제 양자화)
    """

    def __init__(self, levels: List[int]):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.long)
        self.register_buffer("levels", _levels)
        self.dim = len(levels)
        # 양자화 스텝 크기: [-1, 1] 범위에서 L레벨 간격
        self.register_buffer(
            "step_sizes", 2.0 / (_levels.float() - 1.0)
        )

    def forward(self, z):
        """
        Args:
            z: (..., D) in [-1, 1] (Tanh 출력)
        Returns:
            quantized: (..., D) quantized values in [-1, 1]
            indices: (..., D) integer in [0, L-1]
        """
        half_levels = (self.levels.float() - 1.0) / 2.0

        if self.training:
            # Uniform noise: 양자화와 동일한 분포의 노이즈 주입
            # gradient가 smooth하게 흐르면서 양자화 효과를 시뮬레이션
            noise = (torch.rand_like(z) - 0.5) * self.step_sizes
            noisy = (z + noise).clamp(-1.0, 1.0)

            # indices (logging용)
            scaled = (noisy + 1.0) / 2.0 * (self.levels.float() - 1.0)
            indices = scaled.round().long().clamp(min=0)
            indices = torch.min(indices, self.levels - 1)
            return noisy, indices
        else:
            # Hard quantization (추론 시)
            scaled = (z + 1.0) / 2.0 * (self.levels.float() - 1.0)
            indices = scaled.round().long().clamp(min=0)
            indices = torch.min(indices, self.levels - 1)
            quantized = indices.float() / half_levels - 1.0
            return quantized, indices

    def indices_to_codes(self, indices):
        """폰 측: 수신된 인덱스를 연속값으로 복원."""
        half_levels = (self.levels.float() - 1.0) / 2.0
        return indices.float() / half_levels - 1.0


# ============================================================
# Glass Encoder / Phone Decoder
# ============================================================

class GlassFSQEncoder(nn.Module):
    """Glass: (B*T, F, C_pp) → (B*T, fsq_dim) in [-1, 1]"""

    def __init__(self, F, C_pp, fsq_dim):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(F * C_pp, fsq_dim),
            nn.LayerNorm(fsq_dim),
            nn.Tanh(),
        )

    def forward(self, h_bottleneck):
        B_T = h_bottleneck.shape[0]
        return self.compress(h_bottleneck.reshape(B_T, -1))


class PhoneFSQDecoder(nn.Module):
    """Phone: (B*T, fsq_dim) → (B*T, F, C)"""

    def __init__(self, fsq_dim, F, C):
        super().__init__()
        self.F = F
        self.C = C
        self.expand = nn.Sequential(
            nn.Linear(fsq_dim, F * C),
            nn.SiLU(),
        )
        # NarrowBandBlock 진입 전 feature 스케일 안정화
        self.norm = nn.LayerNorm(C)

    def forward(self, quantized):
        h = self.expand(quantized)
        h = h.reshape(-1, self.F, self.C)
        return self.norm(h)


# ============================================================
# Main Model
# ============================================================

class SpatialNetFSQSplit(BaseSEModel):
    """
    SpatialNet with FSQ bottleneck for AR Glass ↔ Mobile Phone split computing.

    Glass side:  STFT → Causal InputConv → CrossBandBlock[0] → Bottleneck → FSQ encode
    BT (96kbps): 512 integer indices (3 bits each) = 1536 bits/frame
    Phone side:  FSQ decode → CausalNarrowBandBlock[0] → Blocks[1:] → Output → iSTFT

    forward() returns a single tensor for SEModule compatibility (no auxiliary loss needed).
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: Optional[int] = 512,
        window_type: str = "hann",
        num_blocks: int = 8,
        C: int = 96,
        C_prime: int = 192,
        C_pp: int = 8,
        fsq_dim: int = 512,
        fsq_levels: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_type=window_type,
        )

        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.F = n_fft // 2 + 1  # 257
        self.C = C
        self.C_pp = C_pp
        self.fsq_dim = fsq_dim

        if fsq_levels is None:
            fsq_levels = [8] * fsq_dim  # 3 bits/dim × 512 = 1536 bits/frame

        # Causal Input Conv: kernel (1,5), causal left-pad applied in _causal_input_conv
        self.input_conv = nn.Conv2d(
            in_channels * 2, C, kernel_size=(1, 5), padding=(0, 0)
        )

        # CrossBandBlock: 프레임 독립 처리 → 이미 causal
        # CausalNarrowBandBlock: causal attention + causal convolution
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossBandBlock(C=C, C_pp=C_pp, F=self.F),
                        CausalNarrowBandBlock(
                            C=C, C_prime=C_prime, dropout=dropout
                        ),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

        # --- FSQ Split 모듈 ---
        self.bottleneck_reduce = nn.Sequential(
            nn.LayerNorm(C),
            nn.Linear(C, C_pp),
            nn.SiLU(),
        )
        self.glass_encoder = GlassFSQEncoder(self.F, C_pp, fsq_dim)
        self.fsq = FiniteScalarQuantizer(fsq_levels)
        self.phone_decoder = PhoneFSQDecoder(fsq_dim, self.F, C)

        # Output
        self.output_linear = nn.Linear(C, out_channels * 2)

    def _causal_input_conv(self, x_feat):
        """Causal input convolution: left-pad time axis by kernel_size-1=4."""
        x_feat = F.pad(x_feat, (4, 0, 0, 0))  # (W_left, W_right, H_top, H_bottom)
        return self.input_conv(x_feat)

    def _glass_encode(self, x):
        """Glass 측 공통 로직: audio → FSQ latent + indices."""
        spec = self.stft(x)
        B, M, F_dim, T = spec.shape
        x_feat = torch.cat([spec.real, spec.imag], dim=1)

        h = self._causal_input_conv(x_feat).permute(0, 2, 3, 1)  # (B, F, T, C)

        # CrossBandBlock[0]
        h = self.blocks[0][0](h)

        # Bottleneck: C → C_pp
        B_orig, F_orig, T_orig, C_orig = h.shape
        h_flat = h.transpose(1, 2).reshape(B_orig * T_orig, F_orig, C_orig)
        h_bottleneck = self.bottleneck_reduce(h_flat)  # (B*T, F, C_pp)

        # FSQ encode
        latent = self.glass_encoder(h_bottleneck)  # (B*T, fsq_dim)
        quantized, indices = self.fsq(latent)

        return quantized, indices, B_orig, T_orig

    def _phone_decode(self, quantized, B, T, length):
        """Phone 측 공통 로직: FSQ codes → enhanced waveform."""
        h_recovered = self.phone_decoder(quantized)  # (B*T, F, C)
        h_phone = h_recovered.reshape(B, T, self.F, self.C).transpose(1, 2)

        # CausalNarrowBandBlock[0] + remaining blocks
        h = self.blocks[0][1](h_phone)
        for i in range(1, self.num_blocks):
            cb, nb = self.blocks[i]
            h = cb(h)
            h = nb(h)

        out = self.output_linear(h).permute(0, 3, 1, 2)
        out_spec = torch.complex(
            out[:, : self.out_channels], out[:, self.out_channels :]
        )
        return self.istft(out_spec, length=length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end forward for training. Returns single tensor (SEModule compatible)."""
        quantized, _, B, T = self._glass_encode(x)
        return self._phone_decode(quantized, B, T, length=x.shape[-1])

    def glass_forward(self, x: torch.Tensor) -> torch.Tensor:
        """AR Glass inference: returns FSQ indices for BT transmission.

        Args:
            x: (B, M, T_samples) raw waveform
        Returns:
            indices: (B*T_frames, fsq_dim) integer tensor, each in [0, 7] (3 bits)
        """
        _, indices, _, _ = self._glass_encode(x)
        return indices

    def phone_forward(
        self, indices: torch.Tensor, num_frames: int, length: int
    ) -> torch.Tensor:
        """Mobile Phone inference: receives BT indices, produces enhanced audio.

        Args:
            indices: (B*T_frames, fsq_dim) integer tensor from glass_forward
            num_frames: T (number of STFT frames per utterance)
            length: target waveform length in samples
        Returns:
            Enhanced waveform (B, out_channels, length)
        """
        B = indices.shape[0] // num_frames
        quantized = self.fsq.indices_to_codes(indices)
        return self._phone_decode(quantized, B, num_frames, length)
