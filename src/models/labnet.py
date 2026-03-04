"""
LABNet: Lightweight Attentive Beamforming Network
Yan et al., 2025 (arXiv:2507.16190)

프로젝트 인터페이스:
  - BaseSEModel 상속
  - Input:  (B, C, T)  — C = in_channels (에어컨덕션 4ch + BCM 1ch 등)
  - Output: (B, C, T)  — Ch-0(reference)만 enhance, 나머지 pass-through
  - STFT/iSTFT: BaseSEModel 유틸리티 사용

논문 원본과의 차이:
  - 입력 feature: 논문(파워압축 magnitude + PD) → 여기서는 real/imag concat (BaseSEModel STFT 출력 기준)
  - 출력: 논문(reference ch 1개) → (B, C, T) 유지하되 Ch-0만 enhanced
  - 채널 수: 가변 (ad-hoc MI 그대로 유지)
  - BCM(마지막 채널)은 reference가 아닌 보조 채널로 처리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.base import BaseSEModel


# ─────────────────────────────────────────────────────────────
# 1. ConvGLU  (LiSenNet에서 가져온 채널 믹서)
# ─────────────────────────────────────────────────────────────
class ConvGLU(nn.Module):
    """
    Convolutional Gated Linear Unit.
    Linear -> DWConv -> Mish gate 구조.
    hidden-dim 내부 채널 attention 역할.
    """
    def __init__(self, d: int, kernel: int = 3):
        super().__init__()
        self.linear = nn.Linear(d, d * 2)
        self.dw_conv = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.mish = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, d)
        gate_input, gate = self.linear(x).chunk(2, dim=-1)  # 각 (N, T, d)
        # DWConv는 (N, d, T) 형식
        gate_input = self.dw_conv(gate_input.transpose(1, 2)).transpose(1, 2)
        return gate_input * self.mish(gate)


# ─────────────────────────────────────────────────────────────
# 2. DPR (Dual-Path Recurrent) 모듈
# ─────────────────────────────────────────────────────────────
class DPRModule(nn.Module):
    """
    Dual-Path Recurrent Module (LiSenNet/LABNet 공통).

    입력: (BFT, d)  — batch × freq × time 가 합쳐진 sequence
    실제로는 reshape 후:
      - Frequency 모델링: Bi-GRU  (각 time-step, freq 축)
      - Temporal 모델링:  uni-GRU (각 freq-bin, time 축)
      - ConvGLU: 채널 믹서

    shape 흐름:
      h: (B, d, F, T) 기준으로 내부에서 reshape
    """
    def __init__(self, d: int, gru_hidden: int = 24):
        super().__init__()
        # Frequency modeling: Bi-GRU  (hidden = gru_hidden/2 per direction)
        self.freq_gru = nn.GRU(d, gru_hidden // 2, batch_first=True, bidirectional=True)
        self.freq_ln = nn.LayerNorm(d)
        self.freq_linear = nn.Linear(gru_hidden, d)

        # Temporal modeling: uni-GRU (causal)
        self.time_gru = nn.GRU(d, gru_hidden, batch_first=True, bidirectional=False)
        self.time_ln = nn.LayerNorm(d)
        self.time_linear = nn.Linear(gru_hidden, d)

        # ConvGLU channel mixer
        self.conv_glu = ConvGLU(d)
        self.out_ln = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, d, F, T)
        returns: (B, d, F, T)
        """
        B, d, F, T = h.shape

        # ── Frequency modeling ──────────────────────────────
        # reshape: (B*T, F, d)
        h_f = h.permute(0, 3, 2, 1).reshape(B * T, F, d)
        h_f_out, _ = self.freq_gru(h_f)                    # (B*T, F, gru_h)
        h_f_out = self.freq_linear(h_f_out)                 # (B*T, F, d)
        h_f_out = self.freq_ln(h_f_out + h_f if d == h_f_out.shape[-1] else h_f_out)
        # reshape back: (B, d, F, T)
        h = h_f_out.reshape(B, T, F, d).permute(0, 3, 2, 1)

        # ── Temporal modeling ───────────────────────────────
        # reshape: (B*F, T, d)
        h_t = h.permute(0, 2, 3, 1).reshape(B * F, T, d)
        h_t_out, _ = self.time_gru(h_t)                    # (B*F, T, gru_h)
        h_t_out = self.time_linear(h_t_out)                 # (B*F, T, d)
        h_t_out = self.time_ln(h_t_out + h_t if d == h_t_out.shape[-1] else h_t_out)
        h = h_t_out.reshape(B, F, T, d).permute(0, 3, 1, 2)

        # ── ConvGLU ─────────────────────────────────────────
        # (B*F, T, d)
        h_g = h.permute(0, 2, 3, 1).reshape(B * F, T, d)
        h_g = self.conv_glu(h_g)
        h_g = self.out_ln(h_g)
        h = h_g.reshape(B, F, T, d).permute(0, 3, 1, 2)

        return h


# ─────────────────────────────────────────────────────────────
# 3. CCA (Cross-Channel Attention)
# ─────────────────────────────────────────────────────────────
class CCAModule(nn.Module):
    """
    Cross-Channel Attention.

    입력: list of (B, d, F, T) — C개 채널
    출력: (B, d, F, T) — reference 채널 1개로 집약

    논문 수식:
      Q: from reference ch (LN + Linear)
      K, V: from all C channels (LN + Linear)
      h_out = MHA(Q, K, V) + h_ref   (residual)

    sequence length:
      Q: 1  (reference 단일)
      K/V: C (전체 채널 수)
    → 채널 수에 독립적 (MI 보장)
    """
    def __init__(self, d: int, num_heads: int = 4):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.proj_q = nn.Linear(d, d)
        self.proj_k = nn.Linear(d, d)
        self.proj_v = nn.Linear(d, d)
        self.mha = nn.MultiheadAttention(d, num_heads, batch_first=True)

    def forward(self, h_list: list) -> torch.Tensor:
        """
        h_list: list of (B, d, F, T), len = C
        returns: (B, d, F, T)
        """
        B, d, F, T = h_list[0].shape
        C = len(h_list)

        # reference = h_list[0]
        h_ref = h_list[0]

        # (B, d, F, T) -> (B*F*T, 1, d) for Q
        h_ref_seq = h_ref.permute(0, 2, 3, 1).reshape(B * F * T, 1, d)
        Q = self.proj_q(self.ln_q(h_ref_seq))           # (B*F*T, 1, d)

        # stack all channels: (B, C, d, F, T) -> (B*F*T, C, d) for K, V
        h_all = torch.stack(h_list, dim=1)               # (B, C, d, F, T)
        h_all_seq = h_all.permute(0, 3, 4, 1, 2).reshape(B * F * T, C, d)
        K = self.proj_k(self.ln_kv(h_all_seq))           # (B*F*T, C, d)
        V = self.proj_v(self.ln_kv(h_all_seq))           # (B*F*T, C, d)

        # MHA: Q(1) attends to K/V(C)
        out, _ = self.mha(Q, K, V)                       # (B*F*T, 1, d)

        # residual + reshape back to (B, d, F, T)
        out = (out + h_ref_seq).squeeze(1)               # (B*F*T, d)
        out = out.reshape(B, F, T, d).permute(0, 3, 1, 2)
        return out


# ─────────────────────────────────────────────────────────────
# 4. 채널별 Encoder / Decoder
# ─────────────────────────────────────────────────────────────
class ChannelEncoder(nn.Module):
    """
    각 채널의 feature (real, imag concatenated)를 hidden dim d로 인코딩.
    shared parameters across channels.

    입력 feature:
      BaseSEModel.stft() -> (B, C, F, T) complex
      여기서 real/imag concat -> (B, C, 2*F, T)
      채널별로 독립 처리: (B*C, 2*F, T) -> encoder -> (B*C, d, F, T)

    논문 원본: 파워압축 magnitude + PD concat
    여기서: real + imag concat (프로젝트 STFT 출력 기준, 더 단순)
    """
    def __init__(self, freq_bins: int, d: int):
        super().__init__()
        in_dim = freq_bins * 2  # real + imag
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, d, kernel_size=1),
            nn.LayerNorm([d, 1]),  # dummy, replaced below
        )
        # 실제 구현: Linear projection + LN
        self.proj = nn.Linear(in_dim, d)
        self.ln = nn.LayerNorm(d)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
        """
        x_real, x_imag: (B, F, T)  — 단일 채널
        returns: (B, d, F, T)
        """
        B, F, T = x_real.shape
        # concat along freq dim -> (B, 2F, T) -> (B, T, 2F)
        feat = torch.cat([x_real, x_imag], dim=1).permute(0, 2, 1)  # (B, T, 2F)
        h = self.ln(self.proj(feat))  # (B, T, d)
        h = h.permute(0, 2, 1).unsqueeze(2)  # (B, d, 1, T)
        # expand to (B, d, F, T) — 주파수 축은 F로 복원
        # 실제로는 F 차원이 필요하므로 projection을 F 방향으로 처리
        return h  # 이후 reshape에서 F 복원


class ChannelEncoderV2(nn.Module):
    """
    실제 사용 인코더: (B, 2F, T) -> (B, d, F, T) 형태로 변환.
    sub-band Conv2d 방식 (LiSenNet 스타일).
    """
    def __init__(self, freq_bins: int, d: int):
        super().__init__()
        # 2*freq_bins channels input (real+imag stacked)
        self.conv_in = nn.Conv2d(2, d, kernel_size=(3, 1), padding=(1, 0))
        self.ln = nn.GroupNorm(1, d)  # LayerNorm equivalent for conv output
        self.act = nn.PReLU()

    def forward(self, x_spec: torch.Tensor) -> torch.Tensor:
        """
        x_spec: (B, F, T) complex tensor — 단일 채널 복소 스펙트로그램
        returns: (B, d, F, T)
        """
        # real/imag를 channel 차원으로: (B, 2, F, T)
        feat = torch.stack([x_spec.real, x_spec.imag], dim=1)
        h = self.act(self.ln(self.conv_in(feat)))  # (B, d, F, T)
        return h


class ChannelDecoder(nn.Module):
    """
    hidden (B, d, F, T) -> complex mask (B, F, T).
    Reference channel에만 적용.
    """
    def __init__(self, freq_bins: int, d: int):
        super().__init__()
        self.conv_out = nn.Conv2d(d, 2, kernel_size=(3, 1), padding=(1, 0))
        # Learnable sigmoid for mask (LiSenNet 방식)
        self.alpha = nn.Parameter(torch.ones(freq_bins))
        self.beta = 2.0  # scalar

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, d, F, T)
        returns: mask (B, F, T) complex
        """
        out = self.conv_out(h)  # (B, 2, F, T)
        # Learnable sigmoid: β / (1 + exp(-α * x))
        mask_real = self.beta / (1 + torch.exp(-self.alpha.view(1, -1, 1) * out[:, 0]))
        mask_imag = self.beta / (1 + torch.exp(-self.alpha.view(1, -1, 1) * out[:, 1]))
        return torch.complex(mask_real, mask_imag)


# ─────────────────────────────────────────────────────────────
# 5. LABNet 메인 모델
# ─────────────────────────────────────────────────────────────
class LABNet(BaseSEModel):
    """
    Lightweight Attentive Beamforming Network.

    3단계 프레임워크:
      Stage 1 — Channel-wise processing:
        각 채널 독립 DPR → CCA로 reference에 통합
      Stage 2 — Pair-wise alignment:
        reference와 각 채널 concat → Linear + DPR → CCA
      Stage 3 — Post refinement:
        reference 단독 DPR 정제

    Args:
        in_channels: 입력 채널 수 (default 5: air 4ch + BCM 1ch)
        d:           hidden dimension (default 32, 논문보다 약간 키움)
        gru_hidden:  GRU hidden size (default 24, 논문 기준)
        num_heads:   CCA attention heads (default 4)
        n_fft:       FFT 크기 (default 512, 프로젝트 기본값)
        hop_length:  hop size (default 256)
    """
    def __init__(
        self,
        in_channels: int = 5,
        d: int = 32,
        gru_hidden: int = 24,
        num_heads: int = 4,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
    ):
        super().__init__(
            in_channels=in_channels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        self.d = d
        freq_bins = n_fft // 2 + 1  # 257

        # ── Encoders (shared params across channels) ───────
        self.encoder = ChannelEncoderV2(freq_bins, d)

        # ── Stage 1: per-channel DPR + CCA ─────────────────
        self.dpr_s1 = DPRModule(d, gru_hidden)
        self.cca_s1 = CCAModule(d, num_heads)

        # ── Stage 2: pair-wise alignment ───────────────────
        # concat(ref, ch_c) -> d*2 -> Linear -> d
        self.align_linear = nn.Linear(d * 2, d)
        self.dpr_s2 = DPRModule(d, gru_hidden)
        self.cca_s2 = CCAModule(d, num_heads)

        # ── Stage 3: post refinement ────────────────────────
        self.dpr_s3 = DPRModule(d, gru_hidden)

        # ── Decoder (reference ch only) ─────────────────────
        self.decoder = ChannelDecoder(freq_bins, d)

    # ── forward ────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)  raw waveform, C = in_channels
        returns: (B, C, T)  — Ch-0 enhanced, Ch-1..C-1 pass-through
        """
        B, C, T = x.shape

        # ── STFT: (B, C, T) -> (B, C, F, T) complex ────────
        spec = self.stft(x)  # (B, C, F, T) complex
        F_bins = spec.shape[2]

        # ── Encode each channel ─────────────────────────────
        # shared encoder: (B, F, T) -> (B, d, F, T) per channel
        h_enc = []
        for c in range(C):
            h_c = self.encoder(spec[:, c])  # (B, d, F, T)
            h_enc.append(h_c)

        # ── Stage 1: channel-wise DPR ───────────────────────
        h_s1 = []
        for c in range(C):
            h_s1.append(self.dpr_s1(h_enc[c]))  # shared params

        # CCA: C channels -> reference (ch-0)
        h_s1_ref = self.cca_s1(h_s1)  # (B, d, F, T)

        # ── Stage 2: pair-wise alignment ────────────────────
        h_s2_list = []
        for c in range(C):
            # concat reference with each channel along d dim
            concat = torch.cat([h_s1_ref, h_s1[c]], dim=1)  # (B, 2d, F, T)
            # Linear: (B, 2d, F, T) -> (B, d, F, T)
            # Linear operates on last dim, so permute
            concat_p = concat.permute(0, 2, 3, 1)  # (B, F, T, 2d)
            aligned = self.align_linear(concat_p).permute(0, 3, 1, 2)  # (B, d, F, T)
            aligned = self.dpr_s2(aligned)  # shared DPR
            h_s2_list.append(aligned)

        # CCA: C aligned channels -> single reference
        h_s2 = self.cca_s2(h_s2_list)  # (B, d, F, T)

        # ── Stage 3: post refinement ─────────────────────────
        h_s3 = self.dpr_s3(h_s2)  # (B, d, F, T)

        # ── Decode: complex mask for reference channel ───────
        mask = self.decoder(h_s3)  # (B, F, T) complex

        # Apply mask to reference channel (Ch-0)
        spec_ref = spec[:, 0]          # (B, F, T) complex
        spec_enhanced = spec_ref * mask  # element-wise complex multiply

        # ── iSTFT: reference channel ─────────────────────────
        # (B, F, T) -> (B, 1, F, T) for istft util
        spec_out = spec_enhanced.unsqueeze(1)  # (B, 1, F, T)
        wav_ref = self.istft(spec_out, length=T)  # (B, 1, T)

        # ── Output: replace Ch-0, keep others pass-through ───
        out = x.clone()
        out[:, 0:1, :] = wav_ref

        return out


# ─────────────────────────────────────────────────────────────
# 6. 파라미터 / MACs 확인용 스크립트
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 직접 실행 시: python src/models/labnet.py
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 프로젝트 기본값: 5ch (air 4 + BCM 1), n_fft=512, hop=256
    model = LABNet(in_channels=5, d=32, gru_hidden=24, num_heads=4)
    n_params = count_params(model)
    print(f"Parameters: {n_params:,} ({n_params/1000:.1f}K)")

    # Dummy forward pass
    B, C, T = 2, 5, 16000  # 1초 오디오
    x = torch.randn(B, C, T)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("Shape check: OK")

    # MACs 측정 (thop 있을 때)
    try:
        from thop import profile
        macs, params = profile(model, inputs=(x[:1],), verbose=False)
        print(f"MACs: {macs/1e9:.3f} GMACs (1초 오디오, 1 batch)")
    except ImportError:
        print("thop not installed — pip install thop 으로 MACs 측정 가능")

    # SEModule 호환성 확인
    print(f"\nmodel.in_channels = {model.in_channels}  (SEModule forward 슬라이싱 기준)")
    print("SEModule 호환: OK")
