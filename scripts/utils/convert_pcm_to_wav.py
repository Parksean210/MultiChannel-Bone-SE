import os
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse
import concurrent.futures
import sys

def convert_single_file(pcm_path, wav_path, sample_rate, channels, dtype, overwrite=False):
    """
    Converts a single raw PCM file to WAV format.
    """
    try:
        if not overwrite and wav_path.exists():
            return True, pcm_path
            
        # Load raw data
        pcm_data = np.fromfile(pcm_path, dtype=dtype)
        
        # Reshape if multi-channel (PCM is interleaved)
        if channels > 1:
            pcm_data = pcm_data.reshape(-1, channels)
            
        sf.write(wav_path, pcm_data, sample_rate)
        return True, pcm_path
    except Exception as e:
        return False, f"Error {pcm_path}: {e}"

def normalize_path(path_str):
    """
    Normalizes paths, handling Windows-style paths on both native Windows and WSL.
    """
    if not path_str:
        return None
    
    p = Path(path_str)
    
    # 1. If the path already exists as-is, use it (standard for native Windows or already-correct Linux paths)
    if p.exists():
        return p
        
    # 2. If we are on Linux (WSL) and get a Windows-style path (e.g., D:\...), try converting to /mnt/d/...
    if os.name == 'posix' and len(path_str) > 1 and path_str[1] == ":" and path_str[2] in ["\\", "/"]:
        drive = path_str[0].lower()
        rest = path_str[3:].replace("\\", "/")
        wsl_path = Path(f"/mnt/{drive}/{rest}")
        if wsl_path.exists():
            return wsl_path
            
    return p

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded PCM to WAV Converter")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing PCM files")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (default: 16000)")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels (default: 1)")
    parser.add_argument("--dtype", type=str, default="int16", help="Numpy dtype of PCM (e.g., int16, float32)")
    parser.add_argument("--delete_pcm", action="store_true", help="Delete original PCM files after conversion")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WAV files")
    parser.add_argument("--dry_run", action="store_true", help="Only list files without conversion")
    
    args = parser.parse_args()

    root_path = normalize_path(args.root)
    if not root_path or not root_path.exists():
        print(f"Error: Path does not exist: {args.root} (Resolved to: {root_path})")
        sys.exit(1)

    print(f"Searching for PCM files in: {root_path}")
    pcm_files = []
    # os.walk is generally faster for massive directory trees
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(".pcm"):
                pcm_files.append(Path(root) / file)
        
        # Give feedback every 100 directories to show it's alive
        if len(pcm_files) % 1000 == 0 and len(pcm_files) > 0:
            print(f"  ... Scanning: Found {len(pcm_files)} files so far...", end='\r')

    print(f"\nFound {len(pcm_files)} PCM files.")

    if not pcm_files:
        print("No PCM files found. Exiting.")
        return

    if args.dry_run:
        print("Dry run mode. Top 5 files to be converted:")
        for f in pcm_files[:5]:
            print(f"  {f} -> {f.with_suffix('.wav')}")
        return

    # Use ThreadPoolExecutor for I/O bound tasks and to avoid process spawn overhead
    dtype = getattr(np, args.dtype)
    success_count = 0
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Define a helper to package arguments
        def task_wrapper(pcm_path):
            wav_path = pcm_path.with_suffix(".wav")
            return convert_single_file(pcm_path, wav_path, args.sr, args.channels, dtype, args.overwrite)

        # map returns an iterator, avoiding the massive overhead of creating 600k Future objects at once
        results = list(tqdm(executor.map(task_wrapper, pcm_files), total=len(pcm_files), desc="Converting"))

    for success, result in results:
        if success:
            success_count += 1
            if args.delete_pcm:
                result.unlink()
        else:
            errors.append(result)

    print(f"\nConversion finished.")
    print(f"Successfully converted: {success_count}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print("  ...")

if __name__ == "__main__":
    main()
