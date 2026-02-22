"""
IC-Mamba (Causal): Inter-Channel Mamba for Multichannel Speech Enhancement

Causal (unidirectional) Mamba using the official mamba-ssm CUDA kernel.
Suitable for low-latency / real-time speech enhancement.

Requirements:
    pip install mamba-ssm causal-conv1d

Architecture Flow:
  (B, M, T) → Encoder → Bottleneck → ChannelProj(M→C)
  → (B*C, L, N) → CausalMamba × num_layers → (B, C, N, L)
  → MaskGen → Decoder → (B, M, T)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from mamba_ssm import Mamba
from .base import BaseSEModel


class MambaLayer(nn.Module):
    """
    Causal Mamba layer: Pre-LayerNorm + Mamba (causal SSM) + Residual.
    Mamba is causal by design (left-to-right scan, causal conv1d).
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        return x + self.mamba(self.norm(x))


class ICMamba(BaseSEModel):
    """
    Inter-Channel Causal Mamba for multichannel speech enhancement.

    Identical encoder / bottleneck / channel-projection / mask / decoder to ICConvTasNet.
    Separation module: stacked causal MambaLayers (mamba-ssm CUDA kernel).

    Causal design:
      - Frame t depends only on frames 0..t  (no future context)
      - Enables streaming / low-latency inference
      - mamba-ssm supports stateful inference via InferenceParams for true streaming

    Default parameters (~2.3 M params):
      in_channels=5, out_channels=64, enc_kernel=256, enc_num_feats=512,
      bot_num_feats=128, d_state=16, expand=2, num_layers=8
    """
    def __init__(self,
                 in_channels: int = 5,       # M: 입력 마이크 수
                 out_channels: int = 64,     # C: 내부 채널 수
                 enc_kernel: int = 256,      # K: 인코더 윈도우 길이
                 enc_num_feats: int = 512,   # F: 인코더 특징 수
                 bot_num_feats: int = 128,   # N: 보틀넥 (= Mamba d_model)
                 d_state: int = 16,          # Mamba SSM 상태 차원
                 d_conv: int = 4,            # Mamba causal conv 커널 크기
                 expand: int = 2,            # d_inner = expand × bot_num_feats
                 num_layers: int = 8,        # Mamba 레이어 수
                 use_checkpoint: bool = False):

        super().__init__(in_channels=in_channels, n_fft=enc_kernel, hop_length=enc_kernel // 2)

        self.enc_kernel    = enc_kernel
        self.enc_stride    = enc_kernel // 2
        self.enc_num_feats = enc_num_feats
        self.bot_num_feats = bot_num_feats
        self.out_channels  = out_channels
        self.use_checkpoint = use_checkpoint

        # 1. Encoder (Shared)
        self.encoder = nn.Conv1d(1, enc_num_feats, enc_kernel, stride=self.enc_stride, bias=False)

        # 2. Bottleneck: F → N
        self.bottleneck = nn.Conv1d(enc_num_feats, bot_num_feats, 1, bias=False)
        self.bot_norm   = nn.GroupNorm(1, bot_num_feats, eps=1e-5)

        # 3. Channel Projection: M → C
        self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 4. Separation: Causal Mamba stack
        self.layers = nn.ModuleList([
            MambaLayer(bot_num_feats, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])

        # 5. Mask Generation
        self.mask_prelu = nn.PReLU()
        self.mask_conv  = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.mask_proj  = nn.Sequential(
            nn.Linear(bot_num_feats, enc_num_feats),
            nn.Sigmoid()
        )

        # 6. Decoder (Shared)
        self.decoder = nn.ConvTranspose1d(
            enc_num_feats, 1, enc_kernel, stride=self.enc_stride, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, M, T) — Raw multichannel waveform
        returns: (B, M, T) — Enhanced waveform
        """
        B, M, T = x.shape

        # --- 1. Encode (shared) ---
        w = F.relu(self.encoder(x.view(B * M, 1, T)))   # (B*M, F, L)
        L = w.shape[-1]

        # --- 2. Bottleneck ---
        w_bot = self.bot_norm(self.bottleneck(w))        # (B*M, N, L)

        # --- 3. Channel Projection ---
        y = w_bot.view(B, M, self.bot_num_feats, L)     # (B, M, N, L)
        y = self.channel_proj(y)                          # (B, C, N, L)
        C = y.shape[1]

        # --- 4. Causal Mamba Separation ---
        # (B, C, N, L) → (B*C, L, N): L frames, each with N features
        y = y.permute(0, 1, 3, 2).reshape(B * C, L, self.bot_num_feats)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                y = torch.utils.checkpoint.checkpoint(layer, y, use_reentrant=False)
            else:
                y = layer(y)

        # (B*C, L, N) → (B, C, N, L)
        y = y.reshape(B, C, L, self.bot_num_feats).permute(0, 1, 3, 2)

        # --- 5. Mask Estimation ---
        y = self.mask_prelu(y)
        y = self.mask_conv(y)                    # (B, M, N, L)
        y = y.permute(0, 1, 3, 2)               # (B, M, L, N)
        mask = self.mask_proj(y)                 # (B, M, L, F)
        mask = mask.permute(0, 1, 3, 2)         # (B, M, F, L)

        # --- 6. Masking ---
        w_reshaped = w.view(B, M, self.enc_num_feats, L)
        masked_w   = w_reshaped * mask

        # --- 7. Decode (shared) ---
        est = self.decoder(masked_w.view(B * M, self.enc_num_feats, L))
        return est.view(B, M, -1)
