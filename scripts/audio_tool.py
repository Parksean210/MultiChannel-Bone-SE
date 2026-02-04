import os
import glob
import numpy as np
import soundfile as sf
import argparse
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def get_audio_files(root_dir, extension):
    return glob.glob(os.path.join(root_dir, f"**/*.{extension}"), recursive=True)

def pcm_to_wav_worker(task):
    path, sr, delete = task
    try:
        # Assuming int16 PCM
        with open(path, 'rb') as f:
            pcm_data = np.frombuffer(f.read(), dtype=np.int16)
        
        # soundfile scale (normalize to [-1.0, 1.0])
        waveform = pcm_data.astype(np.float32) / 32768.0
        output_path = os.path.splitext(path)[0] + ".wav"
        
        sf.write(output_path, waveform, sr)
        
        if delete:
            os.remove(path)
        return True
    except Exception as e:
        print(f"Error converting PCM {path}: {e}")
        return False

def wav_to_npy_worker(task):
    path, delete = task
    try:
        # soundfile.read returns (Samples, Channels)
        waveform, sr = sf.read(path)
        
        # Ensure mono and flatten
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
            
        waveform_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
        output_path = os.path.splitext(path)[0] + ".npy"
        np.save(output_path, waveform_int16)
        
        if delete:
            os.remove(path)
        return True
    except Exception as e:
        print(f"Error converting to NPY {path}: {e}")
        return False

def npy_to_wav_worker(task):
    path, sr, delete = task
    try:
        data = np.load(path)
        # Convert int16 back to float32
        waveform = data.astype(np.float32) / 32767.0
        output_path = os.path.splitext(path)[0] + ".wav"
        
        sf.write(output_path, waveform, sr)
        
        if delete:
            os.remove(path)
        return True
    except Exception as e:
        print(f"Error restoring WAV from NPY {path}: {e}")
        return False

def resample_worker(task):
    path, target_sr, delete = task
    try:
        # "Fury Fast Mode": Using system ffmpeg directly
        # -ac 1: Ensure mono
        # -ar {target_sr}: Set sample rate
        # -v error: Be quiet
        temp_path = str(Path(path).with_suffix('.tmp.wav'))
        
        cmd = [
            "ffmpeg", "-v", "error", "-y", 
            "-i", str(path), 
            "-ar", str(target_sr), 
            "-ac", "1", 
            temp_path
        ]
        
        subprocess.run(cmd, check=True)
        
        if delete:
            os.remove(path)
            os.rename(temp_path, path)
        else:
            # If not deleting, we might want to keep both, 
            # but usually for 'resample' we want to update the file.
            # Here we follow the previous logic.
            os.replace(temp_path, path)
            
        return True
    except Exception as e:
        print(f"Error turbo-resampling {path}: {e}")
        return False

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes (default: all cores)")

    parser = argparse.ArgumentParser(description="Unified Audio Tool")
    subparsers = parser.add_subparsers(dest="command")
    
    # PCM to WAV
    p2w = subparsers.add_parser("pcm2wav", parents=[parent_parser])
    p2w.add_argument("dir", help="Root directory")
    p2w.add_argument("--sr", type=int, default=16000, help="Sample rate of the PCM data (default: 16000)")
    p2w.add_argument("--delete", action="store_true", help="Delete original pcm files")
    
    # WAV to NPY
    w2n = subparsers.add_parser("wav2npy", parents=[parent_parser])
    w2n.add_argument("dir", help="Root directory")
    w2n.add_argument("--delete", action="store_true", help="Delete original wav files after conversion")
    
    # NPY to WAV (Restore)
    n2w = subparsers.add_parser("npy2wav", parents=[parent_parser])
    n2w.add_argument("dir", help="Root directory")
    n2w.add_argument("--sr", type=int, default=16000, help="Target sample rate for restoration")
    n2w.add_argument("--delete", action="store_true", help="Delete original npy files after restoration")
    
    # Resample
    res = subparsers.add_parser("resample", parents=[parent_parser])
    res.add_argument("dir", help="Root directory")
    res.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    res.add_argument("--delete", action="store_true", help="Explicitly delete old file while saving (safer for space)")
    
    args = parser.parse_args()
    
    if args.command == "pcm2wav":
        files = get_audio_files(args.dir, "pcm")
        tasks = [(f, args.sr, args.delete) for f in files]
        print(f"Converting {len(files)} PCM files to WAV ({args.sr}Hz) using {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(pcm_to_wav_worker, t) for t in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
            
    elif args.command == "wav2npy":
        files = get_audio_files(args.dir, "wav")
        tasks = [(f, args.delete) for f in files]
        print(f"Converting {len(files)} WAV files to NPY (delete={args.delete}) using {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(wav_to_npy_worker, t) for t in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
            
    elif args.command == "npy2wav":
        files = get_audio_files(args.dir, "npy")
        tasks = [(f, args.sr, args.delete) for f in files]
        print(f"Restoring {len(files)} NPY files to WAV ({args.sr}Hz) using {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(npy_to_wav_worker, t) for t in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
            
    elif args.command == "resample":
        files = get_audio_files(args.dir, "wav")
        tasks = [(f, args.sr, args.delete) for f in files]
        print(f"Resampling {len(files)} WAV files to {args.sr}Hz using {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(resample_worker, t) for t in tasks]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
            

if __name__ == "__main__":
    main()
