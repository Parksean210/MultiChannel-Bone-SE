"""
MVDR Beam Pattern Visualization.

Polar plots per frequency x beam direction.
Mouth beam: horizontal + vertical planes.

Run:
    uv run python scripts/visualize_beampattern.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils.beamforming import (
    compute_lcmv_weights,
    beam_pattern_horizontal,
    beam_pattern_vertical,
    BEAM_NAMES,
    BEAM_DIRECTIONS,
    MIC_POSITIONS,
)

# ── Config ─────────────────────────────────────────────────────────────────────
N_FFT       = 512
SAMPLE_RATE = 16000
FREQ_LIST   = [500, 1000, 2000, 4000]          # Hz to visualize

# Scan angles — endpoint=False avoids duplicating 0/360 boundary
SCAN_AZ_DEG = np.linspace(0, 360, 720, endpoint=False)   # horizontal scan
SCAN_EL_DEG = np.linspace(-90, 90,  360)                  # vertical scan

# Dynamic range for polar plots.
# Use a tight range so small directional differences are visible.
# At 4 kHz the left beam has ~3.5 dB front-to-back suppression, so -6 dB shows it well.
FLOOR_DB = -6.0
RLIM     = abs(FLOOR_DB)   # radial axis limit (= 6)

BEAM_LABELS = {
    'front': 'Front\n(az=0, el=0)',
    'left':  'Left\n(az=90, el=0)',
    'right': 'Right\n(az=-90, el=0)',
    'back':  'Back\n(az=180, el=0)',
    'mouth': 'Mouth\n(az=0, el=-45)\n[scan @ el=-45]',
}

# Each beam is scanned in the plane that contains its target direction.
# Mouth target is at el=-45 so we scan azimuth at el=-45 (not the horizontal plane).
BEAM_SCAN_ELEVATION = {
    'front': 0.0,
    'left':  0.0,
    'right': 0.0,
    'back':  0.0,
    'mouth': -45.0,
}
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']


def _to_db(pattern: np.ndarray, floor_db: float = FLOOR_DB) -> np.ndarray:
    """magnitude -> dB, normalized to 1.0 reference (LCMV distortionless constraint).
    LCMV guarantees W^H a(target) = 1.0, so target is always at 0 dB.
    Grating lobes with gain > 1.0 are clipped to 0 dB (outer ring).
    Range after shift: [0, abs(floor_db)].  Center = floor_db, outer = 0 dB."""
    db = 20 * np.log10(np.maximum(pattern, 1e-12))   # reference = 1.0 (not max)
    return np.clip(db, floor_db, 0.0) - floor_db


def _polar_setup(ax, title: str = '', rlim: float = RLIM):
    """Common polar axes settings. 0 deg = front (top), CW positive."""
    ax.set_theta_zero_location('N')   # 0 deg at top
    ax.set_theta_direction(-1)         # clockwise positive
    ax.set_rlim(0, rlim)
    half = rlim / 2
    ax.set_rticks([half, rlim])
    ax.set_yticklabels([f'{-half:.0f} dB', '0 dB'], fontsize=6)
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        ['Front\n0', '45', 'Right\n90', '135',
         'Back\n180', '225', 'Left\n270', '315'],
        fontsize=6,
    )
    if title:
        ax.set_title(title, fontsize=9, pad=8)


def plot_main_grid(weights, freqs):
    """
    Main grid: (n_freqs x n_beams) polar — horizontal beam patterns.
    """
    n_rows = len(FREQ_LIST)
    n_cols = len(BEAM_NAMES)

    fig = plt.figure(figsize=(4.5 * n_cols, 4.0 * n_rows))
    fig.suptitle(
        'SOCP Beam Patterns — AR Glasses 4ch Array (Horizontal Scan)\n'
        'Sector-aware: track≈1 within ±30°, suppress outside  |  Red dot: target',
        fontsize=13, y=1.01,
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.55, wspace=0.4)

    for ri, freq_hz in enumerate(FREQ_LIST):
        freq_idx = int(np.argmin(np.abs(freqs - freq_hz)))
        actual_hz = freqs[freq_idx]

        for ci, name in enumerate(BEAM_NAMES):
            ax = fig.add_subplot(gs[ri, ci], projection='polar')

            W_f = weights[name][freq_idx]   # (M,)
            scan_el = BEAM_SCAN_ELEVATION[name]
            pattern = beam_pattern_horizontal(
                W_f, actual_hz, SCAN_AZ_DEG, MIC_POSITIONS,
                elevation_deg=scan_el,
            )
            pattern_db = _to_db(pattern)

            # Close the curve: append the first point to avoid gap at 0/360 boundary
            theta = np.radians(SCAN_AZ_DEG)
            theta_c = np.append(theta, theta[0])
            pat_c   = np.append(pattern_db, pattern_db[0])

            ax.plot(theta_c, pat_c, color=COLORS[ci], linewidth=1.5)
            ax.fill(theta_c, pat_c, alpha=0.15, color=COLORS[ci])

            title = BEAM_LABELS[name] if ri == 0 else ''
            _polar_setup(ax, title=title, rlim=RLIM)

            # Target direction marker — az=0 for all beams on their respective planes
            target_az = BEAM_DIRECTIONS[name][0]
            ax.scatter([np.radians(target_az)], [RLIM * 0.93],
                       color='red', s=25, zorder=5)

            # Frequency label (first column only)
            if ci == 0:
                ax.set_ylabel(f'{actual_hz:.0f} Hz', fontsize=9,
                              labelpad=25, rotation=0, va='center')

    return fig


def plot_mouth_vertical(weights, freqs):
    """
    Mouth beam vertical pattern (azimuth=0 fixed, elevation scan).
    Pattern is mirrored to fill full 360 deg polar plot.
    """
    fig, axes = plt.subplots(1, len(FREQ_LIST),
                              figsize=(4 * len(FREQ_LIST), 4),
                              subplot_kw={'projection': 'polar'})
    fig.suptitle(
        'Mouth Beam — Vertical Pattern (Azimuth=0, Elevation Scan)\n'
        'Right half: +el (down), Left half: mirror',
        fontsize=12,
    )
    for ci, freq_hz in enumerate(FREQ_LIST):
        freq_idx = int(np.argmin(np.abs(freqs - freq_hz)))
        actual_hz = freqs[freq_idx]

        ax = axes[ci]
        W_f = weights['mouth'][freq_idx]
        pattern = beam_pattern_vertical(
            W_f, actual_hz, SCAN_EL_DEG,
            azimuth_deg=0.0,
            mic_pos=MIC_POSITIONS,
        )
        pattern_db = _to_db(pattern)

        # theta: el=+90 -> 0 (top), el=0 -> pi/2 (right), el=-90 -> pi (bottom)
        theta_half = np.radians(90.0 - SCAN_EL_DEG)   # 0 .. pi

        # Mirror: left half is the reverse (bottom -> top on the left)
        theta_mirror = 2 * np.pi - theta_half[::-1]
        pat_mirror   = pattern_db[::-1]

        theta_full = np.concatenate([theta_half, theta_mirror])
        pat_full   = np.concatenate([pattern_db, pat_mirror])

        # Close
        theta_full = np.append(theta_full, theta_full[0])
        pat_full   = np.append(pat_full,   pat_full[0])

        ax.plot(theta_full, pat_full, color=COLORS[4], linewidth=1.5)
        ax.fill(theta_full, pat_full, alpha=0.15, color=COLORS[4])

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlim(0, RLIM)
        ax.set_rticks([RLIM / 2, RLIM])
        ax.set_yticklabels([f'{-RLIM/2:.0f} dB', '0 dB'], fontsize=6)
        ax.set_thetagrids(
            [0, 90, 180, 270],
            ['Up\n+90', 'Horiz\n0', 'Down\n-90', 'Horiz\n0'],
            fontsize=7,
        )
        ax.set_title(f'{actual_hz:.0f} Hz', fontsize=9, pad=8)

        # Target elevation: -45 deg -> theta = 90-(-45) = 135 deg
        target_el = BEAM_DIRECTIONS['mouth'][1]   # -45
        target_theta = np.radians(90.0 - target_el)
        ax.scatter([target_theta], [RLIM * 0.93],
                   color='red', s=25, zorder=5)

    return fig


def main():
    os.makedirs('results', exist_ok=True)

    print("Computing SOCP sector-aware weights...")
    print("  Objective: minimize |W^H a|^2 outside ±30°, track 1 inside ±30°.")
    print("  This solves a SOCP per frequency bin — may take ~30 sec.")
    weights, freqs = compute_lcmv_weights(
        N_FFT, SAMPLE_RATE, method='socp', socp_n_scan=180,
        socp_sector_width=30.0, socp_sector_alpha=1.0, socp_sector_beta=1.0,
    )
    print(f"  Freq bins: {len(freqs)} (0 ~ {freqs[-1]:.0f} Hz)")

    # ── Horizontal grid ────────────────────────────────────────────────────────
    print("Plotting horizontal beam patterns...")
    fig1 = plot_main_grid(weights, freqs)
    out1 = 'results/beam_patterns_horizontal.png'
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out1}")
    plt.close(fig1)

    # ── Mouth vertical ─────────────────────────────────────────────────────────
    print("Plotting Mouth vertical beam patterns...")
    fig2 = plot_mouth_vertical(weights, freqs)
    out2 = 'results/beam_patterns_mouth_vertical.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out2}")
    plt.close(fig2)

    # ── Gain summary ───────────────────────────────────────────────────────────
    print("\n=== Gain at target directions ===")
    for freq_hz in [500, 1000, 2000, 4000]:
        freq_idx = int(np.argmin(np.abs(freqs - freq_hz)))
        print(f"\n  {freqs[freq_idx]:.0f} Hz:")
        for name in BEAM_NAMES:
            az, el = BEAM_DIRECTIONS[name]
            u = np.array([
                -np.sin(np.radians(az)) * np.cos(np.radians(el)),
                 np.cos(np.radians(az)) * np.cos(np.radians(el)),
                 np.sin(np.radians(el)),
            ])
            a = np.exp(-1j * 2 * np.pi * freqs[freq_idx] *
                       (MIC_POSITIONS.T @ u) / 343.0)
            gain = abs(weights[name][freq_idx].conj() @ a)
            print(f"    {name:6s}: gain = {gain:.4f} ({20*np.log10(gain+1e-12):.1f} dB)")

    print("\nDone.")


if __name__ == '__main__':
    main()
