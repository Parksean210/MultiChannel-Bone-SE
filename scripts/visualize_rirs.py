#!/usr/bin/env python
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==========================================
# 🎨 Core Plotting Logic (from visualize_rir)
# ==========================================

def plot_room_2d(ax, room_config, sources, mic_positions):
    """2D Top View Floor Plan"""
    ax.set_title("Room Layout (Top View)", fontsize=12, fontweight='bold')
    
    if room_config['room_type'] == 'shoebox':
        dims = room_config['dimensions']
        rect = plt.Rectangle((0, 0), dims[0], dims[1], fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.set_xlim(-0.5, dims[0] + 0.5)
        ax.set_ylim(-0.5, dims[1] + 0.5)
    else:
        corners = room_config['corners']
        polygon = plt.Polygon(corners.T, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(polygon)
        ax.set_xlim(corners[0].min() - 0.5, corners[0].max() + 0.5)
        ax.set_ylim(corners[1].min() - 0.5, corners[1].max() + 0.5)
    
    for i, s in enumerate(sources):
        color = 'green' if s['type'] == 'target' else 'red'
        marker = '*' if s['type'] == 'target' else 'x'
        size = 200 if s['type'] == 'target' else 100
        label = 'Target' if s['type'] == 'target' and i == 0 else ('Noise' if i == 1 else None)
        ax.scatter(s['pos'][0], s['pos'][1], c=color, marker=marker, s=size, label=label, zorder=5)
    
    ax.scatter(mic_positions[0], mic_positions[1], c='blue', marker='o', s=80, label='Mics', zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_room_3d(ax, room_config, sources, mic_positions):
    """3D Room Perspective"""
    ax.set_title("Room 3D View", fontsize=12, fontweight='bold')
    
    if room_config['room_type'] == 'shoebox':
        dims = room_config['dimensions']
        floor = np.array([[0, 0, 0], [dims[0], 0, 0], [dims[0], dims[1], 0], [0, dims[1], 0], [0, 0, 0]])
        ax.plot(floor[:, 0], floor[:, 1], floor[:, 2], 'k-', alpha=0.5)
        ceiling = floor.copy(); ceiling[:, 2] = dims[2]
        ax.plot(ceiling[:, 0], ceiling[:, 1], ceiling[:, 2], 'k-', alpha=0.5)
        for corner in [[0, 0], [dims[0], 0], [dims[0], dims[1]], [0, dims[1]]]:
            ax.plot([corner[0], corner[0]], [corner[1], corner[1]], [0, dims[2]], 'k-', alpha=0.3)
    else:
        corners = room_config['corners']
        height = room_config['height']
        n_corners = corners.shape[1]
        floor_verts = [np.column_stack((corners.T, np.zeros(n_corners)))]
        ceil_verts = [np.column_stack((corners.T, np.full(n_corners, height)))]
        ax.add_collection3d(Poly3DCollection(floor_verts, facecolors='gray', alpha=0.1))
        ax.add_collection3d(Poly3DCollection(ceil_verts, facecolors='gray', alpha=0.1))
        for i in range(n_corners):
            c1, c2 = corners[:, i], corners[:, (i + 1) % n_corners]
            wall = [np.array([[c1[0], c1[1], 0], [c2[0], c2[1], 0], [c2[0], c2[1], height], [c1[0], c1[1], height]])]
            ax.add_collection3d(Poly3DCollection(wall, facecolors='gray', alpha=0.05))
        ax.set_xlim(corners[0].min(), corners[0].max())
        ax.set_ylim(corners[1].min(), corners[1].max())
        ax.set_zlim(0, height)

    for s in sources:
        color = 'green' if s['type'] == 'target' else 'red'
        marker = '*' if s['type'] == 'target' else 'x'
        ax.scatter(s['pos'][0], s['pos'][1], s['pos'][2], c=color, marker=marker, s=100)
    ax.scatter(mic_positions[0], mic_positions[1], mic_positions[2], c='blue', marker='o', s=60)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

def plot_rir_waveforms_grid(fig, rirs, sources, fs):
    """Grid of RIR Waveforms for each source"""
    num_channels = len(rirs)
    num_sources = len(rirs[0])
    ncols = 2
    nrows = (num_sources + 1) // 2
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_channels))
    
    for s_idx in range(num_sources):
        ax = fig.add_subplot(nrows + 2, ncols, s_idx + 5)
        s_type = sources[s_idx]['type']
        ax.set_title(f"Source {s_idx}: {s_type}", fontsize=10, fontweight='bold')
        for ch in range(num_channels):
            rir = rirs[ch][s_idx]
            t = np.arange(len(rir)) / fs * 1000
            ax.plot(t, rir + ch * 0.4, color=colors[ch], linewidth=0.5, alpha=0.8)
        ax.set_xlabel('Time (ms)', fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.2)

# ==========================================
# 🚀 Business Logic
# ==========================================

def process_file(pkl_path, output_path):
    """Processes a single RIR file and saves visualization"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    meta = data['meta']
    room_config = meta['room_config']
    mic_config = meta['mic_config']
    sources = data['source_info']
    rirs = data['rirs']
    fs = meta['fs']
    
    if 'mic_pos' in meta and meta['mic_pos'] is not None:
        mic_positions = np.array(meta['mic_pos'])
    else:
        target_pos = np.array(sources[0]['pos'])
        mic_rel = np.array(mic_config['relative_positions'])
        mic_positions = mic_rel + target_pos[:, None]
        if mic_config.get('use_bcm'):
            bcm_abs = np.array(mic_config['bcm_rel_pos']).reshape(3, 1) + target_pos[:, None]
            mic_positions = np.hstack([mic_positions, bcm_abs])

    num_sources = len(rirs[0])
    nrows_rir = (num_sources + 1) // 2
    
    fig = plt.figure(figsize=(14, 18))
    fig.suptitle(f"RIR Diagnostic View: {os.path.basename(pkl_path)}", fontsize=16, fontweight='bold')
    
    ax1 = plt.subplot2grid((nrows_rir + 2, 2), (0, 0))
    plot_room_2d(ax1, room_config, sources, mic_positions)
    
    ax2 = plt.subplot2grid((nrows_rir + 2, 2), (0, 1), projection='3d')
    plot_room_3d(ax2, room_config, sources, mic_positions)
    
    plot_rir_waveforms_grid(fig, rirs, sources, fs)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Unified RIR Visualization Tool")
    parser.add_argument("input", help="Path to a .pkl file OR a directory containing .pkl files")
    parser.add_argument("--out", "-o", help="Output path (for single file) or directory (for batch)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        out_dir = args.out if args.out else os.path.join(args.input, "viz")
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(args.input, "*.pkl")))
        print(f"Batch processing {len(files)} files into {out_dir}...")
        for f in tqdm(files):
            out_name = os.path.basename(f).replace('.pkl', '.png')
            process_file(f, os.path.join(out_dir, out_name))
    else:
        out_file = args.out if args.out else args.input.replace('.pkl', '.png')
        print(f"Visualizing {args.input} -> {out_file}")
        process_file(args.input, out_file)

if __name__ == "__main__":
    main()
