"""
ICMamba2BCMGuide: BCM-Guided Inter-Channel Mamba2 for Multichannel Speech Enhancement

BCM (Bone Conduction Microphone, last input channel) is used as a soft mask guide
before the Mamba2 separation stage. Since BCM ≈ clean speech (bone-conducted,
low-frequency, time-aligned), its bottleneck features serve as a learned
speech-activity gate that suppresses noise-dominant regions in the feature map.

Architecture Flow:
  (B, M, T) → Encoder → Bottleneck → (B, M, N, L)
    ├── BCM branch: (B, 1, N, L) → BCMGuide → mask (B, C, N, L)
    └── Main:       channel_proj(M→C) → (B, C, N, L)  ×  mask
        → (B*C, L, N) → Mamba2 × num_layers → (B, C, N, L)
        → MaskGen → Decoder → (B, M, T)

Mamba2 constraints:
  d_inner = d_model × expand  must be divisible by headdim
  e.g. d_model=128, expand=2, headdim=64 → d_inner=256, 256/64=4 heads ✓

Requirements:
    pip install mamba-ssm causal-conv1d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from mamba_ssm import Mamba2
from .base import BaseSEModel


class Mamba2Layer(nn.Module):
    """
    Pre-LayerNorm + Mamba2 (SSD) + Residual.

    Mamba2 uses the State Space Duality (SSD) algorithm:
      - Multi-head SSM with headdim
      - Built-in RMSNorm inside Mamba2 (rmsnorm=True by default)
      - Causal by design (left-to-right chunk scan)
    """
    def __init__(self,
                 d_model: int,
                 d_state: int = 128,   # SSM 상태 차원 (Mamba2 기본값)
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64):   # d_model * expand must be divisible by headdim
        super().__init__()
        assert (d_model * expand) % headdim == 0, (
            f"d_model({d_model}) * expand({expand}) = {d_model * expand} "
            f"must be divisible by headdim({headdim})"
        )
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        return x + self.mamba(self.norm(x))


class ICMamba2BCMGuide(BaseSEModel):
    """
    BCM-guided causal Mamba2 for multichannel speech enhancement.

    The last input channel (index in_channels-1) is treated as the BCM
    (bone conduction mic) signal, which is mostly clean speech.

    BCM Guide mechanism:
      - BCM bottleneck features: (B, 1, N, L)
      - BCMGuide network projects to soft mask: (B, C, N, L) ∈ [0, 1]
      - Mask is multiplied into the C-channel feature map before Mamba2
      - Acts as a speech-activity gate:
          BCM active  → mask ≈ 1 → pass features through
          BCM silent  → mask ≈ 0 → suppress noise-dominant regions

    Default parameters:
      in_channels=5, out_channels=64, enc_kernel=256, enc_num_feats=512,
      bot_num_feats=128, d_state=128, expand=2, headdim=64, num_layers=8
    """
    def __init__(self,
                 in_channels: int = 5,        # M: 마이크 수 (마지막 채널 = BCM)
                 out_channels: int = 64,      # C: 내부 채널 수
                 enc_kernel: int = 256,       # K: 인코더 윈도우 길이
                 enc_num_feats: int = 512,    # F: 인코더 특징 수
                 bot_num_feats: int = 128,    # N: 보틀넥 특징 수 (= Mamba2 d_model)
                 d_state: int = 128,          # Mamba2 SSM 상태 차원
                 d_conv: int = 4,             # Mamba2 causal conv 커널 크기
                 expand: int = 2,             # d_inner = expand × bot_num_feats
                 headdim: int = 64,           # Mamba2 멀티헤드 헤드 차원
                 num_layers: int = 8,         # Mamba2 레이어 수
                 use_checkpoint: bool = False):

        super().__init__(in_channels=in_channels, n_fft=enc_kernel, hop_length=enc_kernel // 2)

        self.enc_kernel    = enc_kernel
        self.enc_stride    = enc_kernel // 2
        self.enc_num_feats = enc_num_feats
        self.bot_num_feats = bot_num_feats
        self.out_channels  = out_channels
        self.use_checkpoint = use_checkpoint

        # 1. Shared Encoder
        self.encoder = nn.Conv1d(1, enc_num_feats, enc_kernel, stride=self.enc_stride, bias=False)

        # 2. Bottleneck: F → N
        self.bottleneck = nn.Conv1d(enc_num_feats, bot_num_feats, 1, bias=False)
        self.bot_norm   = nn.GroupNorm(1, bot_num_feats, eps=1e-5)

        # 3. Channel Projection: M → C  (ICConvTasNet/ICMamba와 동일)
        self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 4. BCM Guide: BCM 보틀넥 (B, 1, N, L) → soft mask (B, C, N, L)
        #    2-layer 1×1 Conv: 1 → C//4 → C, sigmoid
        bcm_hidden = max(out_channels // 4, 8)
        self.bcm_guide = nn.Sequential(
            nn.Conv2d(1, bcm_hidden, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(bcm_hidden, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # 5. Mamba2 Separation Stack (Causal)
        self.layers = nn.ModuleList([
            Mamba2Layer(bot_num_feats, d_state, d_conv, expand, headdim)
            for _ in range(num_layers)
        ])

        # 6. Mask Generation  (ICMamba와 동일)
        self.mask_prelu = nn.PReLU()
        self.mask_conv  = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.mask_proj  = nn.Sequential(
            nn.Linear(bot_num_feats, enc_num_feats),
            nn.Sigmoid(),
        )

        # 7. Shared Decoder
        self.decoder = nn.ConvTranspose1d(
            enc_num_feats, 1, enc_kernel, stride=self.enc_stride, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, T) — 마지막 채널이 BCM
        returns: (B, M, T) — 향상된 다채널 파형
        """
        B, M, T = x.shape

        # --- 1. Encode (shared) ---
        w = F.relu(self.encoder(x.view(B * M, 1, T)))   # (B*M, F, L)
        L = w.shape[-1]

        # --- 2. Bottleneck ---
        w_bot = self.bot_norm(self.bottleneck(w))        # (B*M, N, L)

        # --- 3. Reshape to (B, M, N, L) ---
        w_bot_4d = w_bot.view(B, M, self.bot_num_feats, L)

        # --- 4. BCM Guide Mask ---
        # BCM = 마지막 채널 (index M-1)
        bcm_feat = w_bot_4d[:, -1:, :, :]               # (B, 1, N, L)
        bcm_mask = self.bcm_guide(bcm_feat)              # (B, C, N, L), values ∈ [0, 1]

        # --- 5. Channel Projection (M → C) ---
        y = self.channel_proj(w_bot_4d)                  # (B, C, N, L)

        # --- 6. Apply BCM mask (speech-activity gating) ---
        y = y * bcm_mask                                 # (B, C, N, L)

        # --- 7. Causal Mamba2 Separation ---
        # (B, C, N, L) → (B*C, L, N): L frames, each with N features
        C = y.shape[1]
        y = y.permute(0, 1, 3, 2).reshape(B * C, L, self.bot_num_feats)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                y = torch.utils.checkpoint.checkpoint(layer, y, use_reentrant=False)
            else:
                y = layer(y)

        # (B*C, L, N) → (B, C, N, L)
        y = y.reshape(B, C, L, self.bot_num_feats).permute(0, 1, 3, 2)

        # --- 8. Mask Estimation ---
        y = self.mask_prelu(y)
        y = self.mask_conv(y)                    # (B, M, N, L)
        y = y.permute(0, 1, 3, 2)               # (B, M, L, N)
        mask = self.mask_proj(y)                 # (B, M, L, F)
        mask = mask.permute(0, 1, 3, 2)         # (B, M, F, L)

        # --- 9. Masking ---
        w_reshaped = w.view(B, M, self.enc_num_feats, L)
        masked_w   = w_reshaped * mask

        # --- 10. Decode (shared) ---
        est = self.decoder(masked_w.view(B * M, self.enc_num_feats, L))
        return est.view(B, M, -1)
