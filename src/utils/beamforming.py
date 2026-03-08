"""
LCMV/MVDR beamformer — AR glasses 4ch fixed array.

Coordinate system:
  x: left(-) / right(+)
  y: back(-) / front(+)
  z: down(-) / up(+)

Azimuth: from front(+y), CCW positive
  0=front, 90=left, -90=right, ±180=back

Elevation: from horizontal, up positive
  0=horizontal, -45=down 45deg (self-voice)

Usage:
    weights, freqs = compute_lcmv_weights(n_fft=512, sample_rate=16000)
    W = lcmv_weights_to_tensor(weights)   # (5, F, M) complex64
    beams = apply_beamforming(x_stft, W)  # (B, 5, F, T) complex
"""

import numpy as np
import torch

# ── 물리 상수 ──────────────────────────────────────────────────────────────────
SPEED_OF_SOUND = 343.0  # m/s

# ── AR 글래스 에어컨덕션 마이크 위치 (m) ─────────────────────────────────────
# Mic 0: 좌전 / Mic 1: 우전 / Mic 2: 좌후 / Mic 3: 우후
MIC_POSITIONS = np.array([
    [-0.07,  0.04, 0.0],
    [ 0.07,  0.04, 0.0],
    [-0.07, -0.04, 0.0],
    [ 0.07, -0.04, 0.0],
]).T  # (3, M=4)

# ── 5 beam directions (az_deg, el_deg) ────────────────────────────────────────
BEAM_DIRECTIONS = {
    'front': ( 0.0,   0.0),   # front
    'left':  (90.0,   0.0),   # left
    'right': (-90.0,  0.0),   # right
    'back':  (180.0,  0.0),   # back
    'mouth': ( 0.0, -45.0),   # self-voice (front, down 45deg)
}
BEAM_NAMES = ['front', 'left', 'right', 'back', 'mouth']

# ── Null directions for LCMV (one null per beam, opposite side) ────────────────
BEAM_NULL_DIRECTIONS = {
    'front': (180.0,  0.0),   # null at back
    'left':  (-90.0,  0.0),   # null at right
    'right': ( 90.0,  0.0),   # null at left
    'back':  (  0.0,  0.0),   # null at front
    # Mouth target (az=0, el=-45): planar array (z=0) cannot distinguish
    # elevation, so null must be in a direction that differs in azimuth.
    # Null at back keeps horizontal noise from behind while preserving
    # the self-voice from below-front.
    'mouth': (180.0,  0.0),   # null at back (horizontal)
}


# ── 핵심 계산 함수 ─────────────────────────────────────────────────────────────

def _unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """방향 → 단위벡터 (3,). 정면=+y, 우=+x, 상=+z."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    return np.array([
        -np.sin(az) * np.cos(el),
         np.cos(az) * np.cos(el),
         np.sin(el),
    ])


def steering_vector(
    freqs: np.ndarray,
    mic_pos: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """
    자유장(free-field) 조향 벡터.

    Args:
        freqs:        (F,) Hz
        mic_pos:      (3, M)
        azimuth_deg:  방위각 (정면 기준, CCW 양수)
        elevation_deg: 고도각
    Returns:
        (F, M) complex steering vector
    """
    u = _unit_vector(azimuth_deg, elevation_deg)
    tau = mic_pos.T @ u / SPEED_OF_SOUND  # (M,) 시간 지연
    return np.exp(-1j * 2 * np.pi * freqs[:, None] * tau[None, :])  # (F, M)


def diffuse_noise_covariance(freqs: np.ndarray, mic_pos: np.ndarray) -> np.ndarray:
    """
    확산 소음장(isotropic diffuse noise field) 공분산 행렬.
    Γ_ij(f) = sinc(2π f d_ij / c)

    Args:
        freqs:   (F,) Hz
        mic_pos: (3, M)
    Returns:
        (F, M, M) complex
    """
    M = mic_pos.shape[1]
    F = len(freqs)
    Gamma = np.zeros((F, M, M), dtype=complex)
    for i in range(M):
        for j in range(M):
            d = np.linalg.norm(mic_pos[:, i] - mic_pos[:, j])
            if d < 1e-12:
                Gamma[:, i, j] = 1.0
            else:
                # np.sinc(x) = sin(πx)/(πx)  →  인자에 2fd/c 를 넣으면
                # sin(2πfd/c) / (2πfd/c) 가 됨
                Gamma[:, i, j] = np.sinc(2.0 * freqs * d / SPEED_OF_SOUND)
    return Gamma


def compute_lcmv_beam_weights(
    freqs: np.ndarray,
    mic_pos: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    null_azimuth_deg: float = None,
    null_elevation_deg: float = None,
    diag_load: float = 1e-4,
) -> np.ndarray:
    """
    LCMV beamformer weights.

    Constraints:
      - W^H a(target) = 1   (distortionless)
      - W^H a(null)   = 0   (null, optional)

    Without null: reduces to MVDR  W = Γ⁻¹a / (aᴴΓ⁻¹a)
    With null:    W = Γ⁻¹C (CᴴΓ⁻¹C)⁻¹ f
                  where C = [a_target, a_null], f = [1, 0]

    Note: at low frequencies (d << λ) a_target ≈ a_null for opposite directions,
    making the null constraint near-degenerate. diag_load limits weight growth.

    Returns:
        (F, M) complex
    """
    Gamma = diffuse_noise_covariance(freqs, mic_pos)   # (F, M, M)
    M = mic_pos.shape[1]
    Gamma = Gamma + diag_load * np.eye(M)[None]

    a_t = steering_vector(freqs, mic_pos, azimuth_deg, elevation_deg)  # (F, M)

    if null_azimuth_deg is None:
        # MVDR (single constraint)
        Gi_a = np.linalg.solve(Gamma, a_t[:, :, None]).squeeze(-1)      # (F, M)
        denom = np.einsum('fi,fi->f', a_t.conj(), Gi_a)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        return Gi_a / denom[:, None]

    # LCMV (distortionless + null)
    a_n = steering_vector(freqs, mic_pos, null_azimuth_deg, null_elevation_deg)  # (F, M)
    f = np.array([1.0, 0.0], dtype=complex)   # constraint vector

    _reg = 1e-6 * np.eye(2, dtype=complex)   # regularization for (2,2) solve
    weights = np.zeros((len(freqs), M), dtype=complex)
    for fi in range(len(freqs)):
        C = np.column_stack([a_t[fi], a_n[fi]])             # (M, 2)
        Gi_C = np.linalg.solve(Gamma[fi], C)                # (M, 2) = Γ⁻¹C
        CH_Gi_C = C.conj().T @ Gi_C + _reg                  # (2, 2) regularized
        weights[fi] = Gi_C @ np.linalg.solve(CH_Gi_C, f)   # (M,)
    return weights


# ── [DISABLED] Diagonal-loading binary search (heuristic, kept for reference) ──
# def compute_gain_bounded_weights_bisect(...):
#     """
#     Heuristic: binary search on δ in W(δ) = (Γ+δI)^{-1}a / (a^H(Γ+δI)^{-1}a).
#     Indirect — does not directly enforce |W^H a_i| ≤ 1 as a constraint;
#     relies on the monotonicity of max_gain(δ) as δ increases toward DAS.
#     Replaced by compute_gain_bounded_weights() which solves the exact SOCP.
#     """
#     pass


def compute_gain_bounded_weights(
    freqs: np.ndarray,
    mic_pos: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    null_azimuth_deg: float = None,
    null_elevation_deg: float = 0.0,
    n_scan: int = 180,
    gain_ceil = 1.0,
    diag_load: float = 1e-4,
    sector_half_width: float = None,
    sector_alpha: float = 1.0,
    sector_beta: float = 1.0,
) -> np.ndarray:
    """
    Gain-bounded beamformer via exact SOCP (CLARABEL direct solve).

    Solves per-frequency bin:
        minimize    w^T Q w                 (Q = real repr. of diffuse Γ)
        subject to  W^H a(target) = 1       (distortionless)
                    W^H a(null)   = 0       (hard null, optional)
                    |W^H a(θ_i)| ≤ gain_ceil  for i = 0..n_scan-1

    The null equality constraint forces complete suppression at null_direction.
    Combined with the gain bound, this produces cardioid-like patterns where:
      - target direction = 1.0 (0 dB, guaranteed max)
      - null direction   = 0.0 (−∞ dB, hard null)
      - all others       ≤ gain_ceil

    Real-valued reformulation  (w = [w_r; w_i] ∈ R^{2M}):
        Q = [[Γ_r, -Γ_i], [Γ_i, Γ_r]]      (2M×2M real PSD)
        Equality (ZeroCone):
            distortionless (2 rows): Re(W^H a_t)=1, Im(W^H a_t)=0
            null if given  (2 rows): Re(W^H a_n)=0, Im(W^H a_n)=0
        SOC per scan dir (SecondOrderConeT, dim=3):
            s = b - Ax ∈ SOC(3)
            s[0] = gain_ceil, s[1] = -Re(W^H a_i), s[2] = -Im(W^H a_i)
            → gain_ceil ≥ |W^H a_i|  ✓

    Feasibility note: DAS satisfies |W^H a_i| ≤ 1, but may not satisfy null=0.
    If SOCP with null is infeasible (rare), falls back to SOCP without null,
    then to MVDR if that also fails.

    Args:
        freqs:              (F,) Hz
        mic_pos:            (3, M)
        azimuth_deg:        target azimuth (deg)
        elevation_deg:      target elevation — also the azimuth scan plane
        null_azimuth_deg:   null direction azimuth (deg), or None to skip null
        null_elevation_deg: null direction elevation (deg)
        n_scan:             number of azimuth scan points (default 180 = 2° steps)
        gain_ceil:          scalar or (n_scan,) array of per-direction gain upper bounds.
        diag_load:          diagonal loading on Γ for numerical stability
        sector_half_width:  if set, replaces the diffuse-Γ objective with a
                            sector-aware mixed objective (degrees):
                              inside  |Δaz| ≤ sector_half_width: β·|W^H a_i - 1|²
                              outside |Δaz| > sector_half_width: α·|W^H a_i|²
                            This pulls the inside-sector gain toward 1.0 while
                            suppressing the outside sector.
        sector_alpha:       outside-sector suppression weight (default 1.0)
        sector_beta:        inside-sector tracking-to-1 weight (default 1.0)
    Returns:
        (F, M) complex
    """
    try:
        import clarabel
        import scipy.sparse as sp
    except ImportError:
        raise ImportError(
            "clarabel and scipy required. Install with: uv add clarabel scipy"
        )

    M   = mic_pos.shape[1]
    nw  = 2 * M     # real variables [w_r; w_i]
    F   = len(freqs)

    # ── Precompute steering vectors ───────────────────────────────────────────
    a_target = steering_vector(freqs, mic_pos, azimuth_deg, elevation_deg)   # (F, M)
    scan_az  = np.linspace(0, 360, n_scan, endpoint=False)
    scan_vecs = np.stack([
        steering_vector(freqs, mic_pos, az, elevation_deg) for az in scan_az
    ], axis=0)   # (n_scan, F, M)
    Gamma0 = diffuse_noise_covariance(freqs, mic_pos) + diag_load * np.eye(M)[None]

    # ── Sector mask (for sector-aware objective) ──────────────────────────────
    if sector_half_width is not None:
        delta_az = np.abs(((scan_az - azimuth_deg + 180) % 360) - 180)  # (n_scan,)
        inside_mask  = delta_az <= sector_half_width    # (n_scan,) bool
        outside_mask = ~inside_mask

    has_null = null_azimuth_deg is not None
    if has_null:
        a_null = steering_vector(freqs, mic_pos,
                                 null_azimuth_deg, null_elevation_deg)  # (F, M)
    n_eq = 4 if has_null else 2   # equality constraint rows (2 for target + 2 for null)

    # ── gain_ceil: broadcast to (n_scan,) ────────────────────────────────────
    gain_ceil_arr = np.broadcast_to(
        np.asarray(gain_ceil, dtype=float), (n_scan,)
    ).copy()

    # ── Constant b vector ─────────────────────────────────────────────────────
    b_full = np.zeros(n_eq + 3 * n_scan)
    b_full[0]       = 1.0           # Re(W^H a_t) = 1
    b_full[n_eq::3] = gain_ceil_arr # SOC cone heights (per direction)

    # ── Cone list ─────────────────────────────────────────────────────────────
    cones = [clarabel.ZeroConeT(n_eq)] + [clarabel.SecondOrderConeT(3)] * n_scan

    # ── CLARABEL settings ─────────────────────────────────────────────────────
    settings = clarabel.DefaultSettings()
    settings.verbose     = False
    settings.max_iter    = 200
    settings.tol_gap_abs = 1e-6
    settings.tol_gap_rel = 1e-6
    settings.tol_feas    = 1e-6

    # Preallocate A_dense: (n_eq + 3*n_scan, 2M)
    A_dense  = np.zeros((n_eq + 3 * n_scan, nw))
    weights  = np.zeros((F, M), dtype=complex)

    for fi in range(F):
        a_t   = a_target[fi]         # (M,)  complex
        Gf    = Gamma0[fi]           # (M,M) complex
        svecs = scan_vecs[:, fi, :]  # (n_scan, M) complex

        def _mvdr_fallback():
            Gi_a  = np.linalg.solve(Gf, a_t)
            denom = a_t.conj() @ Gi_a
            return Gi_a / (denom if abs(denom) > 1e-14 else 1e-14)

        if freqs[fi] < 1e-3:
            weights[fi] = _mvdr_fallback()
            continue

        # ── Objective ─────────────────────────────────────────────────────
        if sector_half_width is None:
            # Default: minimize diffuse noise  W^H Γ W
            Gr, Gi_m = Gf.real, Gf.imag
            P_mat = 2.0 * np.block([[Gr, -Gi_m], [Gi_m, Gr]])
            q = np.zeros(nw)
        else:
            # Sector-aware: α·Σ_out|W^H a_i|² + β·Σ_in|W^H a_i - 1|²
            # Real covariance sum helper: C(S) = sum_{i in S} C_i
            #   C_i = [[sr sr^T + si si^T, sr si^T - si sr^T],
            #           [si sr^T - sr si^T, si si^T + sr sr^T]]
            def _cov_sum(sr_s, si_s):
                TL = sr_s.T @ sr_s + si_s.T @ si_s   # (M, M)
                TR = sr_s.T @ si_s - si_s.T @ sr_s   # (M, M)
                return np.block([[TL, TR], [-TR, TL]])

            sr_all, si_all = svecs.real, svecs.imag  # (n_scan, M)
            C_out = _cov_sum(sr_all[outside_mask], si_all[outside_mask])  if outside_mask.any()  else np.zeros((nw, nw))
            C_in  = _cov_sum(sr_all[inside_mask],  si_all[inside_mask])   if inside_mask.any()   else np.zeros((nw, nw))
            P_mat = 2.0 * (sector_alpha * C_out + sector_beta * C_in)
            # Linear term from β·Σ_in(- 2 Re(W^H a_i)):  q = -2β·Σ_in [a_r; a_i]
            q = -2.0 * sector_beta * np.concatenate([
                sr_all[inside_mask].sum(axis=0),
                si_all[inside_mask].sum(axis=0),
            ])
        P = sp.csc_matrix(P_mat)

        # ── Equality rows (ZeroConeT) ─────────────────────────────────────
        a_tr, a_ti = a_t.real, a_t.imag
        A_dense[0, :M] =  a_tr;  A_dense[0, M:] =  a_ti   # Re(W^H a_t) = 1
        A_dense[1, :M] =  a_ti;  A_dense[1, M:] = -a_tr   # Im(W^H a_t) = 0
        if has_null:
            a_nr, a_ni = a_null[fi].real, a_null[fi].imag
            A_dense[2, :M] =  a_nr;  A_dense[2, M:] =  a_ni   # Re(W^H a_n) = 0
            A_dense[3, :M] =  a_ni;  A_dense[3, M:] = -a_nr   # Im(W^H a_n) = 0

        # ── SOC rows (vectorized) ─────────────────────────────────────────
        sr, si = svecs.real, svecs.imag
        A_dense[n_eq::3,   :]  = 0.0                          # cone height: 0
        A_dense[n_eq+1::3, :M] = sr;  A_dense[n_eq+1::3, M:] = si    # Re
        A_dense[n_eq+2::3, :M] = si;  A_dense[n_eq+2::3, M:] = -sr   # Im

        A = sp.csc_matrix(A_dense)

        # ── Solve (with null), fallback to no-null SOCP, then MVDR ────────
        def _solve(A_mat, b_vec, cones_list):
            sol = clarabel.DefaultSolver(P, q, A_mat, b_vec, cones_list, settings).solve()
            return sol if str(sol.status) in ('Solved', 'AlmostSolved') else None

        try:
            sol = _solve(A, b_full, cones)
            if sol is None and has_null:
                # Null constraint infeasible at this freq — try without null
                A_no_null = sp.csc_matrix(A_dense[np.r_[0:2, n_eq:n_eq+3*n_scan]])
                b_no_null = np.concatenate([b_full[:2], b_full[n_eq:]])
                c_no_null = [clarabel.ZeroConeT(2)] + [clarabel.SecondOrderConeT(3)] * n_scan
                sol = _solve(A_no_null, b_no_null, c_no_null)
            if sol is None and np.any(gain_ceil_arr < 1.0 - 1e-6):
                # Tight outside-sector bound infeasible — relax to uniform 1.0
                b_relaxed    = b_full.copy()
                b_relaxed[n_eq::3] = 1.0
                b_relaxed[0] = 1.0
                c_relax = [clarabel.ZeroConeT(2)] + [clarabel.SecondOrderConeT(3)] * n_scan
                A_relax = sp.csc_matrix(A_dense[np.r_[0:2, n_eq:n_eq+3*n_scan]])
                b_relax2 = np.concatenate([b_relaxed[:2], b_relaxed[n_eq:]])
                sol = _solve(A_relax, b_relax2, c_relax)
            if sol is not None:
                w = np.asarray(sol.x)
                weights[fi] = w[:M] + 1j * w[M:]
            else:
                weights[fi] = _mvdr_fallback()
        except Exception:
            weights[fi] = _mvdr_fallback()

    return weights


def compute_lcmv_weights(
    n_fft: int = 512,
    sample_rate: int = 16000,
    mic_pos: np.ndarray = None,
    beam_directions: dict = None,
    null_directions: dict = None,
    method: str = 'lcmv',
    socp_n_scan: int = 180,
    socp_sector_width: float = None,
    socp_sector_alpha: float = 1.0,
    socp_sector_beta: float = 1.0,
) -> tuple:
    """
    Compute beamformer weights for all 5 beams.

    Args:
        method: 'lcmv'  — LCMV with one distortionless + one null constraint.
                          Closed-form, fast. May produce grating lobes > 1.0 at
                          high frequencies.
                'socp'  — Gain-bounded SOCP. Guarantees all non-target directions
                          have gain ≤ 1.0 → target is always the global maximum.
                          Slower (CVXPY per bin) but visually clean patterns.
        socp_n_scan: azimuth scan resolution for SOCP (only used when method='socp').
        null_directions: optional override for LCMV null directions.
    Returns:
        weights_dict: {name: (F, M) complex}
        freqs:        (F,) Hz
    """
    if mic_pos is None:
        mic_pos = MIC_POSITIONS
    if beam_directions is None:
        beam_directions = BEAM_DIRECTIONS
    if null_directions is None:
        null_directions = BEAM_NULL_DIRECTIONS

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)   # (F,)

    weights = {}
    for name, (az, el) in beam_directions.items():
        null_az, null_el = null_directions[name]
        if method == 'socp':
            weights[name] = compute_gain_bounded_weights(
                freqs, mic_pos, az, el,
                null_azimuth_deg=null_az,
                null_elevation_deg=null_el,
                n_scan=socp_n_scan,
                sector_half_width=socp_sector_width,
                sector_alpha=socp_sector_alpha,
                sector_beta=socp_sector_beta,
            )
        else:
            weights[name] = compute_lcmv_beam_weights(
                freqs, mic_pos, az, el,
                null_azimuth_deg=null_az, null_elevation_deg=null_el,
            )
    return weights, freqs


def lcmv_weights_to_tensor(
    weights_dict: dict,
    beam_names: list = None,
) -> torch.Tensor:
    """
    가중치 dict → (K, F, M) complex64 텐서. beam_names 순서 유지.
    """
    if beam_names is None:
        beam_names = BEAM_NAMES
    W = np.stack([weights_dict[name] for name in beam_names], axis=0)  # (K, F, M)
    return torch.from_numpy(W).to(torch.complex64)


# ── GPU 적용 ───────────────────────────────────────────────────────────────────

def apply_beamforming(
    x_stft: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    GPU 빔포밍 적용. 단순 행렬곱 (고정 가중치).

    Args:
        x_stft:  (B, M, F, T) complex — 입력 STFT
        weights: (K, F, M) complex    — 빔 가중치 (register_buffer로 올라온 것)
    Returns:
        (B, K, F, T) complex — 빔포밍된 STFT
    """
    return torch.einsum('kfm,bmft->bkft', weights.conj(), x_stft)


# ── 빔패턴 계산 (시각화용) ────────────────────────────────────────────────────

def beam_pattern_horizontal(
    W_f: np.ndarray,
    freq_hz: float,
    scan_azimuths: np.ndarray,
    mic_pos: np.ndarray = None,
    elevation_deg: float = 0.0,
) -> np.ndarray:
    """
    Azimuth scan at fixed elevation.

    Args:
        W_f:           (M,) complex weights at single freq
        freq_hz:       scan frequency (Hz)
        scan_azimuths: (N,) deg — azimuth scan angles
        mic_pos:       (3, M)
        elevation_deg: fixed elevation angle (deg). Default 0 = horizontal plane.
    Returns:
        (N,) magnitude
    """
    if mic_pos is None:
        mic_pos = MIC_POSITIONS
    pattern = np.array([
        abs(W_f.conj() @ np.exp(
            -1j * 2 * np.pi * freq_hz *
            (mic_pos.T @ _unit_vector(az, elevation_deg)) / SPEED_OF_SOUND
        ))
        for az in scan_azimuths
    ])
    return pattern


def beam_pattern_vertical(
    W_f: np.ndarray,
    freq_hz: float,
    scan_elevations: np.ndarray,
    azimuth_deg: float = 0.0,
    mic_pos: np.ndarray = None,
) -> np.ndarray:
    """
    수직면(고정 방위각) 빔패턴 계산.

    Args:
        W_f:             (M,) complex weights at single freq
        freq_hz:         스캔 주파수 (Hz)
        scan_elevations: (N,) deg
        azimuth_deg:     고정 방위각
        mic_pos:         (3, M)
    Returns:
        (N,) magnitude
    """
    if mic_pos is None:
        mic_pos = MIC_POSITIONS
    pattern = np.array([
        abs(W_f.conj() @ np.exp(
            -1j * 2 * np.pi * freq_hz *
            (mic_pos.T @ _unit_vector(azimuth_deg, el)) / SPEED_OF_SOUND
        ))
        for el in scan_elevations
    ])
    return pattern
