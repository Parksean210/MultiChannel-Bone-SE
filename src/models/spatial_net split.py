import torch
import torch.nn as nn
from typing import Optional

from .base import BaseSEModel

class VectorQuantizer(nn.Module):
    """
    코드북(Codebook) 기반의 양자화 모듈.
    연속적인 벡터를 가장 가까운 코드북 벡터로 치환하고 인덱스를 반환.
    """
    def __init__(self, num_embeddings=1024, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 코드북: (1024개의 128차원 벡터)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: (B*T, embedding_dim)
        
        # 1. 입력 벡터와 코드북 벡터 간의 거리 계산
        # (A-B)^2 = A^2 + B^2 - 2AB
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))

        # 2. 가장 가까운 코드북 벡터의 인덱스 찾기
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # (B*T, 1)

        # 3. 인덱스를 이용해 코드북에서 벡터 가져오기
        quantized = self.embedding(encoding_indices).squeeze(1) # (B*T, embedding_dim)

        # 4. Loss 계산 (VQ-VAE의 핵심)
        # q_loss: 코드북이 입력 데이터를 따라가게 함
        # e_loss: 입력 데이터가 코드북을 따라가게 함 (Commitment)
        q_loss = F.mse_loss(quantized, inputs.detach())
        e_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = q_loss + self.commitment_cost * e_loss

        # 5. Straight-Through Estimator (STE)
        # 역전파 시에는 미분 불가능한 argmin을 무시하고 입력을 그대로 통과시킴
        quantized = inputs + (quantized - inputs).detach()

        # 인덱스(encoding_indices)가 바로 블루투스로 전송될 "데이터"입니다.
        return quantized, vq_loss, encoding_indices

class GlassVQEncoder(nn.Module):
    def __init__(self, F, C_pp, latent_dim=128):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(F * C_pp, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh()
        )

    def forward(self, h_prime):
        B_T = h_prime.shape[0]
        h_flat = h_prime.reshape(B_T, -1)
        return self.compress(h_flat) 

class PhoneVQDecoder(nn.Module):
    def __init__(self, latent_dim, F, C):
        super().__init__()
        self.F = F
        self.C = C
        # 128차원에서 96차원(C) 전체로 확장하여 정보 손실 방지
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, F * C),
            nn.SiLU()
        )

    def forward(self, quantized):
        h_flat = self.expand(quantized)
        return h_flat.reshape(-1, self.F, self.C)

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


class SpatialNet_VQ_Split(BaseSEModel):
    def __init__(self,
                 in_channels: int = 5, out_channels: int = 1,
                 n_fft: int = 512, hop_length: int = 256,
                 win_length: Optional[int] = 512, window_type: str = "hann",
                 num_blocks: int = 8, C: int = 96, C_prime: int = 192, C_pp: int = 8,
                 latent_dim: int = 128, num_embeddings: int = 1024, # VQ 파라미터
                 dropout: float = 0.1):
        
        super().__init__(in_channels=in_channels, n_fft=n_fft, 
                         hop_length=hop_length, win_length=win_length, window_type=window_type)
        
        self.out_channels, self.num_blocks = out_channels, num_blocks
        self.F, self.C_pp = n_fft // 2 + 1, C_pp

        self.input_conv = nn.Conv2d(in_channels * 2, C, kernel_size=(1, 5), padding=(0, 2))
        
        # --- VQ Split 모듈 ---
        self.glass_encoder = GlassVQEncoder(self.F, C_pp, latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim)
        self.phone_decoder = PhoneVQDecoder(latent_dim, self.F, C) # C 전체 복원
        # ---------------------

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                CrossBandBlock(C=C, C_pp=C_pp, F=self.F),
                NarrowBandBlock(C=C, C_prime=C_prime)
            ]) for _ in range(num_blocks)
        ])
        self.output_linear = nn.Linear(C, out_channels * 2)

    def forward(self, x: torch.Tensor):
        spec = self.stft(x) 
        B, M, F, T = spec.shape
        x_feat = torch.cat([spec.real, spec.imag], dim=1)
        h = self.input_conv(x_feat).permute(0, 2, 3, 1) # (B, F, T, C)
        
        cross_block, narrow_block = self.blocks[0]
        h_cross = cross_block(h) 
        
        # [Glass] 압축
        h_for_phone = h_cross.transpose(1, 2).reshape(B*T, F, -1)
        latent = self.glass_encoder(h_for_phone[:, :, :self.C_pp]) 
        
        # [분산 컴퓨팅] 코드북 양자화 (Glass에서 수행)
        quantized, vq_loss, encoding_indices = self.vq_layer(latent)
        
        # ---> 여기서 'encoding_indices' 만 블루투스로 전송합니다! <---
        # ---> 폰에서는 'encoding_indices'를 받아 코드북을 참조해 'quantized' 벡터를 꺼냅니다. <---

        # [Phone] 복원
        # 모델은 96채널(C) 전체를 복원하여 NarrowBandBlock이 정상 작동하도록 유도
        h_recovered = self.phone_decoder(quantized) 
        h_phone_input = h_recovered.reshape(B, T, F, -1).transpose(1, 2)
        
        # [Phone] 나머지 블록 처리
        h = narrow_block(h_phone_input)
        for i in range(1, self.num_blocks):
            cb, nb = self.blocks[i]
            h = cb(h); h = nb(h)

        out = self.output_linear(h).permute(0, 3, 1, 2)
        out_spec = torch.complex(out[:, :self.out_channels], out[:, self.out_channels:])
        out_wav = self.istft(out_spec, length=x.shape[-1])
        
        # 주의: 학습 시에 vq_loss도 전체 Loss에 더해줘야 합니다.
        return out_wav, vq_loss