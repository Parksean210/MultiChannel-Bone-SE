import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .base import BaseSEModel


# ==========================================
# 1. LABNet Core Utility Layers
# ==========================================

class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta  = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps       = eps
        self.stat_dims = stat_dims
        self.num_dims  = num_dims

    def forward(self, x):
        mu_  = x.mean(dim=self.stat_dims, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps)
        return ((x - mu_) / std_) * self.gamma + self.beta


class RNN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout_p=0.1, bidirectional=False):
        super().__init__()
        self.rnn = nn.GRU(emb_dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
        self.dense = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, emb_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.dense(x)


class DualPathRNN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_freqs=32, dropout_p=0.1):
        super().__init__()
        self.intra_norm     = nn.LayerNorm((n_freqs, emb_dim))
        self.intra_rnn_attn = RNN(emb_dim, hidden_dim // 2, dropout_p, bidirectional=True)
        self.inter_norm     = nn.LayerNorm((n_freqs, emb_dim))
        self.inter_rnn_attn = RNN(emb_dim, hidden_dim, dropout_p, bidirectional=False)

    def forward(self, x):
        # x: (B, D, T, F)
        B, D, T, F = x.size()
        x = x.permute(0, 2, 3, 1)      # (B, T, F, D)

        # Intra (frequency)
        x_res = x
        x = self.intra_norm(x)
        x = x.reshape(B * T, F, D)
        x = self.intra_rnn_attn(x)
        x = x.reshape(B, T, F, D) + x_res

        # Inter (time)
        x_res = x
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1, 3).reshape(B * F, T, D)
        x = self.inter_rnn_attn(x)
        x = x.reshape(B, F, T, D).permute(0, 2, 1, 3) + x_res

        return x.permute(0, 3, 1, 2)   # (B, D, T, F)


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim  = int(emb_dim * expansion_factor)
        self.norm   = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1    = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), value=0.0),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, groups=hidden_dim),
        )
        self.act     = nn.Mish()
        self.fc2     = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x: (B, D, T, F)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x) + res
        return x


class DPR(nn.Module):
    def __init__(self, emb_dim=16, hidden_dim=24, n_freqs=32, dropout_p=0.1):
        super().__init__()
        self.dp_rnn_attn = DualPathRNN(emb_dim, hidden_dim, n_freqs, dropout_p)
        self.conv_glu    = ConvolutionalGLU(emb_dim, n_freqs=n_freqs, expansion_factor=2,
                                             dropout_p=dropout_p)

    def forward(self, x):
        return self.conv_glu(self.dp_rnn_attn(x))


class AttentionBlock(nn.Module):
    """Cross-channel attention. Input: (B,C,D,T,F) → Output: (B,1,D,T,F)"""
    def __init__(self, emb_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.norm_q  = nn.LayerNorm([emb_dim])
        self.norm_kv = nn.LayerNorm([emb_dim])
        self.Wq = nn.Linear(emb_dim, hidden_dim)
        self.Wk = nn.Linear(emb_dim, hidden_dim)
        self.Wv = nn.Linear(emb_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, emb_dim)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale    = self.head_dim ** -0.5

    def forward(self, x):
        # x: (B, C, D, T, F)
        B, C, D, T, F = x.size()
        x = x.permute(0, 3, 4, 1, 2)           # (B, T, F, C, D)
        x_ref = x[..., :1, :]                   # (B, T, F, 1, D)

        q = self.Wq(self.norm_q(x_ref)).reshape(B, T, F, 1, self.n_heads, self.head_dim).transpose(-2, -3)
        x_norm = self.norm_kv(x)
        k = self.Wk(x_norm).reshape(B, T, F, C, self.n_heads, self.head_dim).transpose(-2, -3)
        v = self.Wv(x_norm).reshape(B, T, F, C, self.n_heads, self.head_dim).transpose(-2, -3)

        attn = torch.matmul(q, k.transpose(-2, -1)).mul(self.scale).softmax(dim=-1)
        out  = torch.matmul(attn, v).transpose(-2, -3).flatten(-2, -1)
        out  = self.Wo(out) + x_ref             # (B, T, F, 1, D)

        return out.permute(0, 3, 4, 1, 2)       # (B, 1, D, T, F)


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta  = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


# ==========================================
# 2. Convolution Blocks (DSConv / USConv)
# ==========================================

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 4
        self.low_conv  = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3)),
        )
        self.high_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 5), stride=(1, 3)),
        )
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act  = nn.PReLU(out_channels)

    def forward(self, x):
        x_low  = self.low_conv(x[..., :self.low_freqs])
        x_high = self.high_conv(x[..., self.low_freqs:])
        return self.act(self.norm(torch.cat([x_low, x_high], dim=-1)))


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad  = nn.ConstantPad2d(
            (kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] - 1, 0), value=0.0)
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size)
        self.r    = r

    def forward(self, x):
        x   = self.pad(x)
        out = self.conv(x)
        B, nch, H, W = out.shape
        return (out.view(B, self.r, nch // self.r, H, W)
                   .permute(0, 2, 3, 4, 1)
                   .contiguous()
                   .view(B, nch // self.r, H, -1))


class USConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 2
        self.low_conv  = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3)),
        )
        self.high_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(1, 3), r=3)

    def forward(self, x):
        x_low  = self.low_conv(x[..., :self.low_freqs])
        x_high = self.high_conv(x[..., self.low_freqs:])
        return torch.cat([x_low, x_high], dim=-1)


# ==========================================
# 3. Encoder & Decoder
# ==========================================

class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=16):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels // 4, (1, 1)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )
        self.conv_2 = DSConv(num_channels // 4,     num_channels // 2,     n_freqs=257)
        self.conv_3 = DSConv(num_channels // 2,     num_channels // 4 * 3, n_freqs=128)
        self.conv_4 = DSConv(num_channels // 4 * 3, num_channels,          n_freqs=64)

    def forward(self, x):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x); out_list.append(x)   # F=128
        x = self.conv_3(x); out_list.append(x)   # F=64
        x = self.conv_4(x); out_list.append(x)   # F=32
        return out_list


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channels=64, out_channel=2, beta=1):
        super().__init__()
        self.up1 = USConv(num_channels * 2,          num_channels // 4 * 3, n_freqs=32)
        self.up2 = USConv(num_channels // 4 * 3 * 2, num_channels // 2,     n_freqs=64)
        self.up3 = USConv(num_channels // 2 * 2,     num_channels // 4,     n_freqs=128)
        self.mask_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(num_channels // 4, out_channel, (2, 2)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(num_features, beta=beta)

    def forward(self, x, encoder_out_list):
        x = self.up1(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.up2(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.up3(torch.cat([x, encoder_out_list.pop()], dim=1))
        x = self.mask_conv(x)                               # (B, 2, T, F)
        x = self.lsigmoid(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return x


# ==========================================
# 4. Main LABNet Model
# ==========================================

class LABNet(BaseSEModel):
    """
    LABNet: Lightweight Attentive Beamforming Network
    BaseSEModel 인터페이스를 준수하는 다채널 음성 향상 모델.

    Input : (B, C, T) raw waveform  (C = in_channels)
    Output: (B, 1, T) enhanced reference channel waveform
    """

    def __init__(
        self,
        in_channels:     int   = 5,
        n_fft:           int   = 512,
        hop_length:      int   = 256,
        win_length:      int   = 512,
        window_type:     str   = "hann",
        num_channels:    int   = 16,
        compress_factor: float = 0.3,
        gl_iters:        int   = 1,
    ):
        super().__init__(
            in_channels=in_channels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_type=window_type,
        )

        if n_fft != 512:
            import warnings
            warnings.warn("LABNet은 n_fft=512(257 bins)에 최적화되어 있습니다. "
                          "다른 n_fft 사용 시 Sub-band Conv shape mismatch가 발생할 수 있습니다.")

        self.n_freqs        = n_fft // 2 + 1  # 257
        self.compress_factor = compress_factor
        self.gl_iters       = gl_iters

        # Encoder: 3ch feature (mag + GD + IFD) → bottleneck F=32
        self.encoder = Encoder(in_channels=3, num_channels=num_channels)

        # Bottleneck freq = 257 // 8 = 32 (DSConv x3: 257→128→64→32)
        n_bottle = self.n_freqs // 8  # 32

        # Stage 1: 채널별 공유 DPR + CCA
        self.all_dpr    = DPR(emb_dim=num_channels, hidden_dim=num_channels // 2 * 3,
                               n_freqs=n_bottle)
        self.attn_block1 = AttentionBlock(emb_dim=num_channels,
                                           hidden_dim=num_channels // 2 * 3, n_heads=4)

        # Stage 2: Pair-wise alignment
        self.align_dpr = nn.Sequential(
            nn.Conv2d(2 * num_channels, num_channels, 1),
            DPR(emb_dim=num_channels, hidden_dim=num_channels // 2 * 3, n_freqs=n_bottle),
        )
        self.attn_block2 = AttentionBlock(emb_dim=num_channels,
                                           hidden_dim=num_channels // 2 * 3, n_heads=4)

        # Stage 3: Post refinement
        self.final_dpr = DPR(emb_dim=num_channels, hidden_dim=num_channels // 2 * 3,
                              n_freqs=n_bottle)

        # Mask Decoder
        self.decoder = MaskDecoder(self.n_freqs, num_channels=num_channels,
                                    out_channel=2, beta=1)

    # ── Feature helpers ─────────────────────────────────────

    def _cal_gd(self, x_pha: torch.Tensor) -> torch.Tensor:
        """Group Delay: d(phase)/d(freq).  (B,T,F) → (B,T,F)"""
        b, t, f = x_pha.size()
        gd = torch.diff(x_pha, dim=2,
                        prepend=torch.zeros(b, t, 1, device=x_pha.device))
        return torch.atan2(gd.sin(), gd.cos())

    def _cal_ifd(self, x_pha: torch.Tensor) -> torch.Tensor:
        """Instantaneous Frequency Deviation.  (B,T,F) → (B,T,F)"""
        b, t, f = x_pha.size()
        x_if = torch.diff(x_pha, dim=1,
                          prepend=torch.zeros(b, 1, f, device=x_pha.device))
        expected = (2 * torch.pi * self.hop_length / self.n_fft *
                    torch.arange(f, device=x_pha.device)[None, None, :])
        return torch.atan2((x_if - expected).sin(), (x_if - expected).cos())

    def _griffinlim(self, mag: torch.Tensor, pha: torch.Tensor,
                     length: int) -> torch.Tensor:
        """
        Griffin-Lim phase refinement.

        수정 사항 (원본 버그 fix):
          tprev를 매 iter 시작 전 scalar 0으로 초기화하지 않고,
          이전 iter rebuilt을 별도 변수(tprev)에 저장하되
          in-place mul 대신 일반 곱셈을 사용 → gl_iters > 1 시 안전.

        Args:
            mag:    power-compressed magnitude (B, T, F), grad 불필요
            pha:    initial phase (B, T, F), noisy ref phase
            length: original waveform length
        Returns:
            refined phase (B, T, F)
        """
        mag = mag.detach() ** (1.0 / self.compress_factor)  # uncompress
        momentum = 0.99 / (1.0 + 0.99)                      # ≈ 0.497

        tprev = None   # 첫 iter는 momentum term 없음

        for _ in range(self.gl_iters):
            # iSTFT: (B, T, F) → (B, 1, T_wav)
            spec = torch.complex(mag * pha.cos(), mag * pha.sin())
            spec_4d = spec.permute(0, 2, 1).unsqueeze(1)        # (B,1,F,T)
            inverse = self.istft(spec_4d, length=length)         # (B,1,T_wav)

            # STFT: (B,1,T_wav) → (B,1,F,T) → (B,T,F) complex
            rebuilt = self.stft(inverse).squeeze(1).permute(0, 2, 1)

            # Momentum correction (non-inplace)
            if tprev is not None:
                pha_new = rebuilt - tprev * momentum
            else:
                pha_new = rebuilt

            pha   = pha_new.angle()
            tprev = rebuilt          # 다음 iter용 저장 (non-inplace 참조)

        return pha

    # ── Forward ─────────────────────────────────────────────

    def _forward_full(self, x: torch.Tensor) -> dict:
        """
        전체 forward 연산. 최종 파형과 함께 손실 계산에 필요한
        중간 STFT 표현(power-compressed magnitude/spectrum)을 함께 반환.

        Args:
            x: (B, C, T_wav)
        Returns:
            dict with:
              'est_wav'  : (B, 1, T_wav)   최종 복원 파형
              'est_mag'  : (B, T, F)        power-compressed magnitude (마스크 출력)
              'est_spec' : (B, T, F) complex power-compressed complex (GL phase 적용)
        """
        B, C, T_wav = x.shape

        # 1. STFT: (B, C, T) → (B, C, F, T_spec)
        spec = self.stft(x)
        spec = spec.permute(0, 1, 3, 2)        # (B, C, T_spec, F)

        mag = spec.abs().clamp(min=1e-8) ** self.compress_factor  # power compression

        # 원본 ABNet과 동일하게 power_compress 복원 후 .angle()로 phase 유도.
        # stft.angle()을 직접 쓰면 ±π 경계 bin에서 float32 cos/sin 반올림으로
        # 2π 차이가 발생해 GD/IFD 계산이 달라짐.
        pha_raw = spec.angle()
        spec_pc = torch.complex(mag * pha_raw.cos(), mag * pha_raw.sin())
        pha = spec_pc.angle()   # matches ABNet's src_pha = power_compress(stft).angle()

        # 2. Feature extraction: (B*C, T, F) → (B*C, 3, T, F)
        BC     = B * C
        T_spec = mag.shape[2]
        F_     = mag.shape[3]

        mag_flat = mag.reshape(BC, T_spec, F_)
        pha_flat = pha.reshape(BC, T_spec, F_)

        gd_flat  = self._cal_gd(pha_flat)
        ifd_flat = self._cal_ifd(pha_flat)

        feat = torch.stack(
            [mag_flat, gd_flat / torch.pi, ifd_flat / torch.pi], dim=1
        )  # (B*C, 3, T, F)

        # 3. Encoder
        encoder_out_list = self.encoder(feat)   # [F=128, F=64, F=32] each (B*C,D,T,F)
        h = encoder_out_list[-1]                # (B*C, D, T, F_bottle)

        # 4. Stage 1: 공유 DPR (B*C 한번에 처리)
        h = self.all_dpr(h)                     # (B*C, D, T, F_bottle)
        h = h.reshape(B, C, h.size(1), h.size(2), h.size(3))  # (B, C, D, T, F)

        # 5. CCA 1
        h_ref = self.attn_block1(h)             # (B, 1, D, T, F)

        # 6. Stage 2: Pair-wise alignment (B*C 한번에 처리)
        h_concat = torch.cat(
            [h_ref.expand_as(h), h], dim=2
        ).flatten(0, 1)                         # (B*C, 2D, T, F)
        h_align = self.align_dpr(h_concat)      # (B*C, D, T, F)
        h_align = h_align.reshape(B, C, h_align.size(1),
                                   h_align.size(2), h_align.size(3))

        # 7. CCA 2
        h_ref2 = self.attn_block2(h_align)      # (B, 1, D, T, F)

        # 8. Stage 3: final refinement
        h_final = self.final_dpr(
            h_ref2.flatten(0, 1)                # (B, D, T, F)  — B*1=B
        )

        # 9. Decoder skip-connection: reference 채널(0번)만 사용
        for idx in range(len(encoder_out_list)):
            enc = encoder_out_list[idx]         # (B*C, D, T, F)
            encoder_out_list[idx] = enc.reshape(B, C, *enc.shape[1:])[:, 0]

        mask = self.decoder(h_final, encoder_out_list)   # (B, 2, T, F)

        # 10. Mask 적용
        ref_mag = mag[:, 0]                     # (B, T, F)
        # 원본 ABNet과 동일: (mask0 + mask1) * ref_mag
        est_mag = (mask[:, 0] + 1e-8) * ref_mag + (mask[:, 1] + 1e-8) * ref_mag

        # 11. Griffin-Lim phase refinement
        ref_pha = pha[:, 0]                     # (B, T, F)
        est_pha = self._griffinlim(est_mag, ref_pha, length=T_wav)

        # 12. power-compressed complex spectrum (손실 계산용, iSTFT 이전)
        # 원본 ABNet의 results['est_spec']과 동일한 표현
        est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())

        # 13. Reconstruct & iSTFT
        est_mag_unc = est_mag ** (1.0 / self.compress_factor)
        est_spec_unc = torch.complex(est_mag_unc * est_pha.cos(),
                                      est_mag_unc * est_pha.sin())
        est_spec_4d = est_spec_unc.unsqueeze(1).permute(0, 1, 3, 2)  # (B,1,F,T)
        est_wav = self.istft(est_spec_4d, length=T_wav)               # (B,1,T)

        return {
            'est_wav':  est_wav,   # (B, 1, T_wav)
            'est_mag':  est_mag,   # (B, T, F)  power-compressed magnitude
            'est_spec': est_spec,  # (B, T, F)  power-compressed complex (GL phase)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T_wav)  — noisy multichannel waveform
        Returns:
            (B, 1, T_wav) — enhanced reference channel
        """
        return self._forward_full(x)['est_wav']

    def forward_with_intermediates(self, x: torch.Tensor) -> dict:
        """
        중간 STFT 표현을 포함한 전체 결과 반환. MetricGAN 학습에 사용.

        Returns:
            'est_wav'  : (B, 1, T_wav)
            'est_mag'  : (B, T, F)  power-compressed magnitude (마스크 직접 출력)
            'est_spec' : (B, T, F)  power-compressed complex spectrum (GL phase)
        """
        return self._forward_full(x)
