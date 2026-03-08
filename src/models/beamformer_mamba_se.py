"""
BeamformerMambaSE — LCMV Beamforming + oSpatialNet-Mamba Speech Enhancement.

Architecture follows the oSpatialNet-Mamba paper (Quan & Li, 2024):
  - Input Conv2d (freq_kernel=1, time_kernel=5) → hidden features
  - L interleaved blocks: CrossBandBlock + NarrowBandBlock(Mamba×2)
    - CrossBandBlock: FreqConv + FullBandLinear + FreqConv (time-frame independent)
    - NarrowBandBlock: Two Mamba blocks per frequency (frequency independent)
  - Output: cRM mask on mouth beam → iSTFT

Extensions over vanilla oSpatialNet:
  - LCMV beamforming front-end: 4ch air-mics → 5 directional beams
  - BCM FiLM conditioning: bone-conduction magnitude → (gamma, beta)

Input : (B, 5, T)  — 4 air-conduction mics + 1 BCM
Output: (B, 1, T)  — enhanced speech in Mouth beam direction

Requires mamba_ssm (Mamba2). Falls back to GRU if not installed.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from .base import BaseSEModel
from src.utils.beamforming import (
    compute_lcmv_weights,
    lcmv_weights_to_tensor,
    apply_beamforming,
)

try:
    from mamba_ssm import Mamba2
    _MAMBA2_AVAILABLE = True
except ImportError:
    _MAMBA2_AVAILABLE = False


# ── CrossBandBlock sub-modules (process time frames independently) ──────────


class _FrequencyConvModule(nn.Module):
    """Grouped Conv1d along frequency axis with residual. (N, F, C) → (N, F, C)"""

    def __init__(self, C: int, num_groups: int = 8, kernel_size: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.conv = nn.Conv1d(C, C, kernel_size, padding=kernel_size // 2,
                              groups=min(num_groups, C))
        self.act = nn.PReLU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        res = h
        h = self.norm(h).transpose(1, 2)   # (N, C, F)
        h = self.conv(h).transpose(1, 2)   # (N, F, C)
        return res + self.act(h)


class _FullBandLinearModule(nn.Module):
    """Channel-wise full-band frequency linear (Eq. 10 in SpatialNet paper).
    (N, F, C) → (N, F, C)"""

    def __init__(self, C: int, C_pp: int, F: int):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.reduce = nn.Linear(C, C_pp)
        self.reduce_act = nn.SiLU()
        self.f_linears = nn.ModuleList([nn.Linear(F, F) for _ in range(C_pp)])
        self.expand = nn.Linear(C_pp, C)
        self.expand_act = nn.SiLU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        res = h
        h = self.reduce_act(self.reduce(self.norm(h)))  # (N, F, C'')
        h = h.transpose(1, 2)                            # (N, C'', F)
        h_out = torch.empty_like(h)
        for c, f_lin in enumerate(self.f_linears):
            h_out[:, c, :] = f_lin(h[:, c, :])
        h = h_out.transpose(1, 2)                        # (N, F, C'')
        return res + self.expand_act(self.expand(h))


class _CrossBandBlock(nn.Module):
    """FreqConv → FullBandLinear → FreqConv. Processes time frames independently."""

    def __init__(self, C: int, C_pp: int, F: int, num_groups: int = 8):
        super().__init__()
        self.fconv1 = _FrequencyConvModule(C, num_groups)
        self.fb_linear = _FullBandLinearModule(C, C_pp, F)
        self.fconv2 = _FrequencyConvModule(C, num_groups)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, F, T, C)"""
        B, F, T, C = h.shape
        h = h.transpose(1, 2).reshape(B * T, F, C)  # time-frame independent
        h = self.fconv1(h)
        h = self.fb_linear(h)
        h = self.fconv2(h)
        return h.reshape(B, T, F, C).transpose(1, 2)  # (B, F, T, C)


# ── NarrowBandBlock: Two Mamba blocks (process frequencies independently) ───


class _MambaBlock(nn.Module):
    """Single Mamba block: LayerNorm → Mamba2 → residual.
    Processes along time axis. (N, T, C) → (N, T, C)"""

    def __init__(self, C: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        if _MAMBA2_AVAILABLE:
            assert (C * expand) % headdim == 0, (
                f"C({C}) * expand({expand}) = {C * expand} must be divisible by headdim({headdim})"
            )
            self.ssm = Mamba2(d_model=C, d_state=d_state, d_conv=d_conv,
                              expand=expand, headdim=headdim)
        else:
            self.ssm = nn.GRU(C, C, batch_first=True)
        self._use_mamba2 = _MAMBA2_AVAILABLE

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        res = h
        h = self.norm(h)
        if self._use_mamba2:
            h = self.ssm(h.float()).to(res.dtype)
        else:
            h, _ = self.ssm(h)
        return res + h


class _NarrowBandBlock(nn.Module):
    """Two Mamba blocks (oSpatialNet-Mamba style). Processes frequencies independently."""

    def __init__(self, C: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, headdim: int = 64):
        super().__init__()
        self.mamba1 = _MambaBlock(C, d_state, d_conv, expand, headdim)
        self.mamba2 = _MambaBlock(C, d_state, d_conv, expand, headdim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, F, T, C)"""
        B, F, T, C = h.shape
        h = h.reshape(B * F, T, C)  # frequency-independent
        h = self.mamba1(h)
        h = self.mamba2(h)
        return h.reshape(B, F, T, C)


# ── BCM FiLM conditioning ──────────────────────────────────────────────────


class _BoneFilm(nn.Module):
    """BCM magnitude spectrum → per-block FiLM (gamma, beta) parameters."""

    def __init__(self, n_freq: int, C: int, n_blocks: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.C = C
        self.proj = nn.Sequential(
            nn.Linear(n_freq, C),
            nn.GELU(),
            nn.Linear(C, 2 * C * n_blocks),
        )

    def forward(self, bcm_mag: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            bcm_mag: (B, n_freq, T_frames) float
        Returns:
            list of (gamma, beta) tuples, each (B, T_frames, C)
        """
        B, n_freq, T = bcm_mag.shape
        film = self.proj(bcm_mag.permute(0, 2, 1))          # (B, T, 2*C*n_blocks)
        film = film.view(B, T, self.n_blocks, 2, self.C)
        film = film.permute(2, 3, 0, 1, 4)                  # (n_blocks, 2, B, T, C)
        return [(film[k, 0], film[k, 1]) for k in range(self.n_blocks)]


# ── Main Model ────────────────────────────────────────────────────────────────


class BeamformerMambaSE(BaseSEModel):
    """
    LCMV Beamforming + oSpatialNet-Mamba Speech Enhancement.

    Architecture (oSpatialNet-Mamba style):
      1. LCMV beamforming: 4ch air-mics → 5 directional beams
      2. Input Conv2d (freq_kernel=1, time_kernel=5) → H hidden features
      3. L interleaved blocks:
         - CrossBandBlock: FreqConv + FullBandLinear + FreqConv (time-frame independent)
         - NarrowBandBlock: Mamba×2 per frequency (frequency independent)
         - BCM FiLM conditioning after each block
      4. cRM mask → Mouth beam → iSTFT

    Args:
        in_channels:  total input channels (4 air + 1 BCM = 5)
        n_air:        number of air-conduction mics feeding LCMV (default 4)
        hidden_dim:   internal feature dimension H (paper: 96)
        n_blocks:     number of interleaved block pairs L (paper: 8)
        C_pp:         FullBandLinear bottleneck dimension (paper: 8)
        n_fft:        STFT FFT size
        hop_length:   STFT hop size
        win_length:   STFT window size
        d_state:      Mamba2 SSM state dimension
        d_conv:       Mamba2 local conv width
        expand:       Mamba2 expand ratio
        headdim:      Mamba2 SSD head dimension
        sample_rate:  audio sample rate
    """

    MOUTH_BEAM_IDX = 4   # BEAM_NAMES = ['front', 'left', 'right', 'back', 'mouth']

    @staticmethod
    def _load_or_compute_beam_weights(
        n_fft: int,
        sample_rate: int,
        cache_path: str,
        method: str,
        socp_n_scan: int,
        socp_sector_width: float,
        socp_sector_alpha: float,
        socp_sector_beta: float,
    ) -> torch.Tensor:
        """Load beamforming weights from cache or compute. Returns (5, F, M) complex64."""
        if cache_path and os.path.isfile(cache_path):
            cached = torch.load(cache_path, map_location='cpu', weights_only=True)
            meta = cached.get('meta', {})
            if (meta.get('n_fft') == n_fft and
                meta.get('sample_rate') == sample_rate and
                meta.get('method') == method):
                print(f"[BeamformerMambaSE] Loaded beam weights from {cache_path}")
                return cached['weights']
            print(f"[BeamformerMambaSE] Cache mismatch, recomputing...")

        print(f"[BeamformerMambaSE] Computing beam weights (method={method})...")
        weights_dict, _ = compute_lcmv_weights(
            n_fft=n_fft, sample_rate=sample_rate,
            method=method,
            socp_n_scan=socp_n_scan,
            socp_sector_width=socp_sector_width,
            socp_sector_alpha=socp_sector_alpha,
            socp_sector_beta=socp_sector_beta,
        )
        W = lcmv_weights_to_tensor(weights_dict)   # (5, F, 4) complex64

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            torch.save({
                'weights': W,
                'meta': {'n_fft': n_fft, 'sample_rate': sample_rate, 'method': method},
            }, cache_path)
            print(f"[BeamformerMambaSE] Saved beam weights to {cache_path}")

        return W

    def __init__(
        self,
        in_channels: int = 5,
        n_air: int = 4,
        hidden_dim: int = 96,
        n_blocks: int = 8,
        C_pp: int = 8,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        sample_rate: int = 16000,
        beam_weights_path: str = "data/beam_weights.pt",
        beam_method: str = "lcmv",
        beam_socp_n_scan: int = 180,
        beam_socp_sector_width: float = None,
        beam_socp_sector_alpha: float = 1.0,
        beam_socp_sector_beta: float = 1.0,
    ):
        super().__init__(
            in_channels=in_channels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        self.n_air = n_air
        self.n_beams = 5
        self.n_freq = n_fft // 2 + 1   # F = 257

        # ── Beamforming weights ───────────────────────────────────────────
        W = self._load_or_compute_beam_weights(
            n_fft=n_fft, sample_rate=sample_rate,
            cache_path=beam_weights_path,
            method=beam_method,
            socp_n_scan=beam_socp_n_scan,
            socp_sector_width=beam_socp_sector_width,
            socp_sector_alpha=beam_socp_sector_alpha,
            socp_sector_beta=beam_socp_sector_beta,
        )
        self.register_buffer('beam_weights', W)

        # ── Input Conv2d: (5 beams + 1 BCM) × 2(real/imag) → hidden_dim ──
        # SpatialNet style: freq_kernel=1, time_kernel=5
        n_input_ch = (self.n_beams + 1) * 2   # 12
        self.input_conv = nn.Conv2d(
            in_channels=n_input_ch,
            out_channels=hidden_dim,
            kernel_size=(1, 5),   # freq=1, time=5
            padding=(0, 2),
        )

        # ── BCM FiLM ─────────────────────────────────────────────────────
        self.bone_film = _BoneFilm(self.n_freq, hidden_dim, n_blocks)

        # ── Interleaved blocks: CrossBand + NarrowBand(Mamba×2) ──────────
        self.cross_band_blocks = nn.ModuleList([
            _CrossBandBlock(hidden_dim, C_pp, self.n_freq)
            for _ in range(n_blocks)
        ])
        self.narrow_band_blocks = nn.ModuleList([
            _NarrowBandBlock(hidden_dim, d_state, d_conv, expand, headdim)
            for _ in range(n_blocks)
        ])

        # ── cRM mask output head ─────────────────────────────────────────
        self.mask_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),   # → (real_mask, imag_mask)
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5, T) — 4 air mics + 1 BCM
        Returns:
            (B, 1, T) — Mouth-direction enhanced speech
        """
        B, _, T = x.shape

        # ── Split air / BCM ──────────────────────────────────────────────
        x_air = x[:, :self.n_air].contiguous()   # (B, 4, T)
        x_bcm = x[:, self.n_air:].contiguous()   # (B, 1, T)

        # ── STFT ─────────────────────────────────────────────────────────
        X_air = self.stft(x_air)    # (B, 4, F, T_f) complex
        X_bcm = self.stft(x_bcm)   # (B, 1, F, T_f) complex
        T_f = X_air.shape[-1]

        # ── LCMV beamforming: 4ch → 5 beams ─────────────────────────────
        X_beams = apply_beamforming(X_air, self.beam_weights)  # (B, 5, F, T_f)

        # ── Feature: 6 complex ch → 12 real ch ──────────────────────────
        X_all = torch.cat([X_beams, X_bcm], dim=1)           # (B, 6, F, T_f)
        feat = torch.cat([X_all.real, X_all.imag], dim=1)    # (B, 12, F, T_f)

        # ── Input Conv2d (SpatialNet style) ──────────────────────────────
        h = self.input_conv(feat)                              # (B, H, F, T_f)
        h = h.permute(0, 2, 3, 1)                             # (B, F, T_f, H)

        # ── BCM FiLM parameters ──────────────────────────────────────────
        bcm_mag = X_bcm[:, 0].abs()                           # (B, F, T_f)
        film_params = self.bone_film(bcm_mag)

        # ── Interleaved CrossBand + NarrowBand blocks ────────────────────
        for i in range(len(self.cross_band_blocks)):
            # CrossBandBlock (time-frame independent)
            h = self.cross_band_blocks[i](h)

            # NarrowBandBlock — Mamba×2 (frequency independent)
            h = self.narrow_band_blocks[i](h)

            # BCM FiLM conditioning
            gamma, beta = film_params[i]   # (B, T_f, H)
            h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # broadcast F dim

        # ── cRM mask → Mouth beam ────────────────────────────────────────
        mask = self.mask_head(h)                               # (B, F, T_f, 2)
        mask = mask.permute(0, 3, 1, 2).float()               # (B, 2, F, T_f)
        mask_c = torch.complex(mask[:, 0:1], mask[:, 1:2])    # (B, 1, F, T_f)

        X_mouth = X_beams[:, self.MOUTH_BEAM_IDX:self.MOUTH_BEAM_IDX + 1]
        X_enh = X_mouth * mask_c                               # (B, 1, F, T_f)

        # ── iSTFT ────────────────────────────────────────────────────────
        return self.istft(X_enh, length=T)                     # (B, 1, T)
