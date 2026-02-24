"""
ICMamba2BCMGuideV2: BCM-Guided Inter-Channel Mamba2 with FiLM Conditioning

Improvements over V1:
  1. BCM Guide: simple gate (×sigmoid) → FiLM (scale×sigmoid + shift×tanh)
     - scale: speech-activity gate ∈ [0, 1]  (suppresses noise-dominant regions)
     - shift: additive bias from BCM          (boosts speech-active regions)
     - Together: y = y * scale + shift
  2. Final mask: Sigmoid removed → ReLU
     - V1 mask ∈ [0, 1]: can only suppress encoder features
     - V2 mask ∈ [0, ∞): allows amplification → better speech recovery

Architecture Flow (same as V1 except marked [V2]):
  (B, M, T) → Encoder → Bottleneck → (B, M, N, L)
    ├── BCM branch: (B, 1, N, L) → BCMGuide → scale/shift (B, C, N, L)  [V2: FiLM]
    └── Main:       channel_proj(M→C) → (B, C, N, L)  ×scale +shift
        → (B*C, L, N) → Mamba2 × num_layers → (B, C, N, L)
        → MaskGen → Decoder → (B, M, T)                                  [V2: ReLU mask]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from mamba_ssm import Mamba2
from .base import BaseSEModel


class Mamba2Layer(nn.Module):
    """Pre-LayerNorm + Mamba2 (SSD) + Residual."""
    def __init__(self,
                 d_model: int,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64):
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


class ICMamba2BCMGuideV2(BaseSEModel):
    """
    BCM-guided causal Mamba2 with FiLM conditioning and unbounded mask.

    [V2 changes]
    BCM Guide — FiLM instead of simple gate:
      - bcm_guide outputs 2*C channels → split into scale, shift
      - scale = sigmoid(·)  ∈ [0, 1] : speech-activity gate
      - shift = tanh(·)*0.1 : small additive push from BCM signal
      - Applied as: y = y * scale + shift
      (old V1: y = y * sigmoid(·))

    Final mask — ReLU instead of Sigmoid:
      - mask ∈ [0, ∞): encoder features can be amplified, not just suppressed
      (old V1: mask ∈ [0, 1] via Sigmoid)
    """
    def __init__(self,
                 in_channels: int = 5,
                 out_channels: int = 64,
                 enc_kernel: int = 256,
                 enc_num_feats: int = 512,
                 bot_num_feats: int = 128,
                 d_state: int = 128,
                 d_conv: int = 4,
                 expand: int = 2,
                 headdim: int = 64,
                 num_layers: int = 8,
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

        # 3. Channel Projection: M → C
        self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 4. [V2] BCM Guide: FiLM conditioning
        #    1 → C//4 → 2*C  (scale + shift)
        bcm_hidden = max(out_channels // 4, 8)
        self.bcm_guide = nn.Sequential(
            nn.Conv2d(1, bcm_hidden, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(bcm_hidden, out_channels * 2, kernel_size=1),  # 2C: scale + shift
        )

        # 5. Mamba2 Separation Stack (Causal)
        self.layers = nn.ModuleList([
            Mamba2Layer(bot_num_feats, d_state, d_conv, expand, headdim)
            for _ in range(num_layers)
        ])

        # 6. [V2] Mask Generation: Sigmoid → ReLU
        self.mask_prelu = nn.PReLU()
        self.mask_conv  = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.mask_proj  = nn.Sequential(
            nn.Linear(bot_num_feats, enc_num_feats),
            nn.ReLU(),  # unbounded: mask ∈ [0, ∞)
        )

        # 7. Shared Decoder
        self.decoder = nn.ConvTranspose1d(
            enc_num_feats, 1, enc_kernel, stride=self.enc_stride, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, T) — 마지막 채널이 BCM
        returns: (B, M, T)
        """
        B, M, T = x.shape

        # --- 1. Encode ---
        w = F.relu(self.encoder(x.view(B * M, 1, T)))   # (B*M, F, L)
        L = w.shape[-1]

        # --- 2. Bottleneck ---
        w_bot = self.bot_norm(self.bottleneck(w))        # (B*M, N, L)

        # --- 3. Reshape ---
        w_bot_4d = w_bot.view(B, M, self.bot_num_feats, L)

        # --- 4. [V2] BCM FiLM Conditioning ---
        bcm_feat = w_bot_4d[:, -1:, :, :]               # (B, 1, N, L)
        film_out = self.bcm_guide(bcm_feat)              # (B, 2*C, N, L)
        scale_raw, shift_raw = film_out.chunk(2, dim=1)  # (B, C, N, L) each
        scale = torch.sigmoid(scale_raw)                 # ∈ [0, 1]: gate
        shift = torch.tanh(shift_raw) * 0.1             # small additive push

        # --- 5. Channel Projection ---
        y = self.channel_proj(w_bot_4d)                  # (B, C, N, L)

        # --- 6. [V2] Apply FiLM: scale × y + shift ---
        y = y * scale + shift                            # (B, C, N, L)

        # --- 7. Causal Mamba2 Separation ---
        C = y.shape[1]
        y = y.permute(0, 1, 3, 2).reshape(B * C, L, self.bot_num_feats)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                y = torch.utils.checkpoint.checkpoint(layer, y, use_reentrant=False)
            else:
                y = layer(y)

        y = y.reshape(B, C, L, self.bot_num_feats).permute(0, 1, 3, 2)

        # --- 8. [V2] Mask Estimation (ReLU, unbounded) ---
        y = self.mask_prelu(y)
        y = self.mask_conv(y)                    # (B, M, N, L)
        y = y.permute(0, 1, 3, 2)               # (B, M, L, N)
        mask = self.mask_proj(y)                 # (B, M, L, F)
        mask = mask.permute(0, 1, 3, 2)         # (B, M, F, L)

        # --- 9. Masking ---
        w_reshaped = w.view(B, M, self.enc_num_feats, L)
        masked_w   = w_reshaped * mask

        # --- 10. Decode ---
        est = self.decoder(masked_w.view(B * M, self.enc_num_feats, L))
        return est.view(B, M, -1)
