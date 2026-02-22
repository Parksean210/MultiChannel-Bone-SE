import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from .base import BaseSEModel

class GlobalLayerNorm(nn.Module):
    """
    Conv-TasNet에서 사용하는 Global Layer Normalization.
    (Batch, Channels, Time) 또는 (Batch, Channels, Freq, Time) 모두 지원.
    """
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.GroupNorm(1, channels, eps=eps)

    def forward(self, x):
        return self.norm(x)

class ICConvBlock(nn.Module):
    """
    [논문 Figure 2(c)] Inter-Channel Conv Block
    - Depthwise Conv: Feature(N)와 Time(L) 축에 대해 2D로 작동
    - 1x1 Conv: Channel(C) 정보를 섞어줌
    - Residual Path & Skip Connection Path 분리 구현 (논문 파라미터 수 재현 핵심)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, norm_type="gLN"):
        super().__init__()
        
        # 1. Expansion (C -> H)
        self.conv1x1_exp = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLayerNorm(out_channels)
        
        # 2. Depthwise Conv (2-D D-Conv)
        self.dconv = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            dilation=dilation, 
            groups=out_channels # Depthwise
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLayerNorm(out_channels)
        
        # 3. Projection (H -> C) : Residual Path (다음 블록 입력용)
        self.conv1x1_proj = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        
        # 4. Skip Connection (H -> C) : Skip Path (최종 출력 합산용)
        # * 이 레이어가 있어야 논문의 1.67M 파라미터가 완성됩니다.
        self.conv1x1_skip = nn.Conv2d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        Returns:
            res_out: 다음 블록으로 전달되는 값 (Input + Residual)
            skip_out: 최종 출력을 위해 누적되는 값 (Skip)
        """
        residual = x
        
        x = self.conv1x1_exp(x)
        x = self.prelu1(x)
        x = self.norm1(x)
        
        x = self.dconv(x)
        x = self.prelu2(x)
        x = self.norm2(x)
        
        # 두 갈래 길 (Residual / Skip)
        res_out = self.conv1x1_proj(x) + residual
        skip_out = self.conv1x1_skip(x)
        
        return res_out, skip_out

class ICConvTasNet(BaseSEModel):
    """
    Inter-Channel Conv-TasNet implementation based on Lee et al. (2021)
    * Best Performance Model Configuration:
      - Channel Projection (M -> C)
      - Large Bottleneck (N=128)
      - Skip Connections Accumulation
    """
    def __init__(self,
                 in_channels: int = 5,       # M: 입력 마이크 개수
                 out_channels_tcn: int = 64, # C: 논문 내부 채널 (Best=64)
                 enc_kernel: int = 256,      # K: 윈도우 길이
                 enc_num_feats: int = 512,   # F: 인코더 특징 수
                 bot_num_feats: int = 128,   # N: 보틀넥 특징 수 (Best=128)
                 tcn_hidden: int = 256,      # H: TCN 히든 채널
                 num_layers: int = 8,        # D: 스택 당 레이어 수
                 num_stacks: int = 3,        # S: 스택 수
                 kernel_size: int = 3,       # P: 커널 크기
                 use_checkpoint: bool = False): # 메모리 최적화 옵션 (A100/H100 등 대용량 GPU에서는 False 권장)
        
        super().__init__(in_channels=in_channels, n_fft=enc_kernel, hop_length=enc_kernel//2)
        
        self.enc_kernel = enc_kernel
        self.enc_stride = enc_kernel // 2
        self.enc_num_feats = enc_num_feats
        self.bot_num_feats = bot_num_feats
        self.out_channels_tcn = out_channels_tcn
        self.use_checkpoint = use_checkpoint
        
        # 1. Encoder (Shared)
        self.encoder = nn.Conv1d(
            1, enc_num_feats, kernel_size=enc_kernel, stride=self.enc_stride, bias=False
        )
        
        # 2. Feature Bottleneck (Shared): F -> N
        self.bottleneck = nn.Conv1d(enc_num_feats, bot_num_feats, 1, bias=False)
        self.bot_norm = GlobalLayerNorm(bot_num_feats)

        # 3. Channel Projection (M -> C)
        # 입력 마이크 M개를 논문의 내부 채널 C로 확장
        self.channel_proj = nn.Conv2d(in_channels, out_channels_tcn, kernel_size=1)

        # 4. Separation Module (IC-TCN)
        self.tcn = nn.ModuleList()
        for s in range(num_stacks):
            for i in range(num_layers):
                dilation = 2**i
                padding = (1, dilation) # (Feature, Time) padding
                
                self.tcn.append(
                    ICConvBlock(
                        in_channels=out_channels_tcn, # C (64)
                        out_channels=tcn_hidden,      # H (256)
                        kernel_size=(kernel_size, kernel_size),
                        stride=1,
                        padding=padding,
                        dilation=(1, dilation)
                    )
                )

        # 5. Mask Generation
        self.mask_prelu = nn.PReLU()
        
        # C(64) -> M(5) : 채널 복구
        self.mask_conv = nn.Conv2d(out_channels_tcn, in_channels, 1)
        
        # N(128) -> F(512) : 특징 차원 복구 (Linear Projection)
        self.mask_proj = nn.Sequential(
            nn.Linear(bot_num_feats, enc_num_feats),
            nn.Sigmoid()
        )

        # 6. Decoder (Shared)
        self.decoder = nn.ConvTranspose1d(
            enc_num_feats, 1, kernel_size=enc_kernel, stride=self.enc_stride, bias=False
        )

    def forward(self, x):
        """
        x: (Batch, M, T) - Raw Waveform
        """
        B, M, T = x.shape
        
        # --- 1. Encoding (Shared) ---
        # (B, M, T) -> (B*M, 1, T) -> (B*M, F, L)
        x_flat = x.view(B * M, 1, T)
        w = self.encoder(x_flat) 
        w = F.relu(w)

        # --- 2. Bottleneck (Shared) ---
        # (B*M, F, L) -> (B*M, N, L)
        w_bot = self.bottleneck(w)
        w_bot = self.bot_norm(w_bot)
        
        # --- 3. Channel Projection ---
        # (B*M, N, L) -> (B, M, N, L)
        y = w_bot.view(B, M, w_bot.shape[1], w_bot.shape[2])
        
        # (B, M, N, L) -> (B, C, N, L) : 5채널 -> 64채널 확장
        y = self.channel_proj(y) 
        
        # --- 4. Separation (TCN with Skip Accumulation) ---
        skip_connection_sum = 0
        
        for block in self.tcn:
            if self.use_checkpoint and self.training:
                # Gradient Checkpointing으로 메모리 절약
                y, skip = torch.utils.checkpoint.checkpoint(block, y, use_reentrant=False)
            else:
                y, skip = block(y)
            
            # Skip Output 누적
            skip_connection_sum = skip_connection_sum + skip
            
        # 최종 출력은 마지막 블록 출력이 아니라 Skip들의 합
        y = skip_connection_sum
        
        # --- 5. Mask Estimation ---
        y = self.mask_prelu(y)
        
        # (B, C, N, L) -> (B, M, N, L) : 64채널 -> 5채널 복구
        y = self.mask_conv(y) 
        
        # (B, M, N, L) -> (B, M, L, N) -> (B, M, L, F) -> (B, M, F, L)
        y = y.permute(0, 1, 3, 2) 
        mask = self.mask_proj(y)
        mask = mask.permute(0, 1, 3, 2)
        
        # --- 6. Masking ---
        # w: (B*M, F, L) -> (B, M, F, L)
        w_reshaped = w.view(B, M, w.shape[1], w.shape[2])
        masked_w = w_reshaped * mask
        
        # --- 7. Decoding (Shared) ---
        # (B, M, F, L) -> (B*M, F, L)
        masked_w_flat = masked_w.view(B * M, -1, masked_w.shape[-1])
        est_source = self.decoder(masked_w_flat) # (B*M, 1, T)
        
        # (B*M, 1, T) -> (B, M, T)
        est_source = est_source.view(B, M, -1)
        
        return est_source