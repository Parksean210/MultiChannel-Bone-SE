import torch
import torch.nn as nn
from typing import Optional

from .base import BaseSEModel

class FrequencyConvolutionalModule(nn.Module):
    def __init__(self, C, num_groups=8, kernel_size=3):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.f_gconv1d = nn.Conv1d(
            in_channels=C, out_channels=C, 
            kernel_size=kernel_size, padding=kernel_size // 2, groups=num_groups
        )
        self.prelu = nn.PReLU()

    def forward(self, h):
        # h: (B*T, F, C)
        res = h
        h = self.norm(h).transpose(1, 2) # (B*T, C, F)
        h = self.f_gconv1d(h)
        h = h.transpose(1, 2)            # (B*T, F, C)
        return res + self.prelu(h)


class FullBandLinearModule(nn.Module):
    def __init__(self, C, C_pp, F):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.reduce_linear = nn.Linear(C, C_pp)
        self.reduce_silu = nn.SiLU()
        
        # F-Linear: 채널마다 주파수 전역(F)을 매핑 (논문의 Eq. 10)
        self.f_linears = nn.ModuleList([nn.Linear(F, F) for _ in range(C_pp)])
        
        self.expand_linear = nn.Linear(C_pp, C)
        self.expand_silu = nn.SiLU()

    def forward(self, h):
        # h: (B*T, F, C)
        res = h
        h = self.norm(h)
        h_prime = self.reduce_silu(self.reduce_linear(h)) # (B*T, F, C'')
        
        h_prime = h_prime.transpose(1, 2) # (B*T, C'', F)
        h_prime_out = torch.empty_like(h_prime)
        for c, f_linear in enumerate(self.f_linears):
            h_prime_out[:, c, :] = f_linear(h_prime[:, c, :])
            
        h_prime = h_prime_out.transpose(1, 2) # (B*T, F, C'')
        h_out = self.expand_silu(self.expand_linear(h_prime))
        return res + h_out


class CrossBandBlock(nn.Module):
    def __init__(self, C, C_pp, F, num_groups=8, kernel_size=3):
        super().__init__()
        self.freq_conv1 = FrequencyConvolutionalModule(C, num_groups, kernel_size)
        self.full_band_linear = FullBandLinearModule(C, C_pp, F)
        self.freq_conv2 = FrequencyConvolutionalModule(C, num_groups, kernel_size)

    def forward(self, h):
        # h: (B, F, T, C)
        B, F, T, C = h.shape
        h = h.transpose(1, 2).reshape(B * T, F, C) # 프레임(T) 독립 처리
        
        h = self.freq_conv1(h)
        h = self.full_band_linear(h)
        h = self.freq_conv2(h)
        
        return h.reshape(B, T, F, C).transpose(1, 2) # (B, F, T, C)


class MHSAModule(nn.Module):
    def __init__(self, C, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.mha = nn.MultiheadAttention(embed_dim=C, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # h: (B*F, T, C)
        res = h
        h = self.norm(h)
        attn_out, _ = self.mha(h, h, h) # Self-Attention
        return res + self.dropout(attn_out)


class TConvFFNModule(nn.Module):
    def __init__(self, C, C_prime, num_groups=8, kernel_size=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.lin1 = nn.Linear(C, C_prime)
        self.silu1 = nn.SiLU()
        
        # T-Convs
        self.conv1 = nn.Conv1d(C_prime, C_prime, kernel_size, padding=kernel_size//2, groups=num_groups)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv1d(C_prime, C_prime, kernel_size, padding=kernel_size//2, groups=num_groups)
        self.gn = nn.GroupNorm(num_groups, C_prime)
        self.silu3 = nn.SiLU()
        self.conv3 = nn.Conv1d(C_prime, C_prime, kernel_size, padding=kernel_size//2, groups=num_groups)
        
        # 추가되어야 할 비선형 및 드롭아웃
        self.silu4 = nn.SiLU() # conv3 이후 SiLU
        self.lin2 = nn.Linear(C_prime, C)
        self.dropout = nn.Dropout(dropout) # 최종 Dropout

    def forward(self, h):
        res = h
        h = self.silu1(self.lin1(self.norm(h)))
        
        h = h.transpose(1, 2) # (B*F, C', T)
        h = self.silu2(self.conv1(h))
        h = self.silu3(self.gn(self.conv2(h)))
        h = self.silu4(self.conv3(h)) # 논문 규격: SiLU 추가
        
        h = h.transpose(1, 2) # (B*F, T, C')
        # 논문 규격: Linear 이후 Dropout 적용
        return res + self.dropout(self.lin2(h))

class NarrowBandBlock(nn.Module):
    def __init__(self, C, C_prime, num_heads=4, num_groups=8, kernel_size=5):
        super().__init__()
        self.mhsa = MHSAModule(C, num_heads)
        self.t_conv_ffn = TConvFFNModule(C, C_prime, num_groups, kernel_size)

    def forward(self, h):
        # h: (B, F, T, C)
        B, F, T, C = h.shape
        h = h.reshape(B * F, T, C) # 주파수(F) 독립 처리
        
        h = self.mhsa(h)
        h = self.t_conv_ffn(h)
        
        return h.reshape(B, F, T, C)


class SpatialNet(BaseSEModel):
    """
    BaseSEModel을 상속받아 구현된 SpatialNet.
    in_channels 크기의 파형을 입력받아 타겟 음성의 파형을 출력합니다.
    """
    def __init__(self,
                 in_channels: int = 5,
                 out_channels: int = 1, # 통상적으로 타겟 화자 1명
                 n_fft: int = 512,
                 hop_length: int = 256,
                 win_length: Optional[int] = 512,
                 window_type: str = "hann",
                 num_blocks: int = 8,  # L (Small: 8, Large: 12)
                 C: int = 96,          # Hidden dimension
                 C_prime: int = 192,   # T-ConvFFN dimension
                 C_pp: int = 8,       # FullBand dimension
                 dropout: float = 0.1):
        
        # 1. BaseSEModel 초기화 (STFT 파라미터 셋업)
        super().__init__(in_channels=in_channels, n_fft=n_fft, 
                         hop_length=hop_length, win_length=win_length, window_type=window_type)
        
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        
        # F 차원 계산 (n_fft 기준)
        self.F = n_fft // 2 + 1
        
        # 2. Input Layer (T-Conv1d)
        # 2M 차원의 입력을 C 차원으로 임베딩. 주파수 축 커널 1, 시간 축 커널 5 (논문 스펙)
        self.input_conv = nn.Conv2d(
            in_channels=in_channels * 2, # Real + Imag
            out_channels=C,
            kernel_size=(1, 5),
            padding=(0, 2)
        )
        
        # 3. Interleaved Blocks (Cross-band -> Narrow-band 반복)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                CrossBandBlock(C=C, C_pp=C_pp, F=self.F),
                NarrowBandBlock(C=C, C_prime=C_prime)
            ) for _ in range(num_blocks)
        ])
        
        # 4. Output Layer
        # 타겟 화자(P)의 실수부/허수부 예측 (2P 차원)
        self.output_linear = nn.Linear(C, out_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, M, T_time) 형태의 Raw Waveform
        Returns:
            y: (B, P, T_time) 형태의 Enhanced Waveform
        """
        # [1] STFT 수행 (BaseSEModel 메서드 사용)
        # spec shape: (B, M, F, T)
        spec = self.stft(x) 
        
        # [2] Feature 생성: 실수부와 허수부를 채널 축(M)으로 결합
        B, M, F, T = spec.shape
        real = spec.real
        imag = spec.imag
        
        # (B, M, F, T) -> (B, 2M, F, T)
        x_feat = torch.cat([real, imag], dim=1)
        
        # [3] Input Convolution
        # h shape: (B, C, F, T)
        h = self.input_conv(x_feat)
        
        # 블록 처리를 위해 축 변환: (B, F, T, C)
        h = h.permute(0, 2, 3, 1)
        
        # [4] Interleaved Blocks
        for block in self.blocks:
            h = block(h)
            
        # [5] Output Linear Layer
        # out shape: (B, F, T, 2P)
        out = self.output_linear(h)
        
        # STFT 복원을 위해 축 변환: (B, 2P, F, T)
        out = out.permute(0, 3, 1, 2)
        
        # [6] 복소수 스펙트로그램 조립
        out_real = out[:, :self.out_channels, :, :]
        out_imag = out[:, self.out_channels:, :, :]
        out_spec = torch.complex(out_real, out_imag)
        
        # [7] iSTFT 수행 및 길이 보정 (BaseSEModel 메서드 사용)
        out_wav = self.istft(out_spec, length=x.shape[-1])
        
        return out_wav