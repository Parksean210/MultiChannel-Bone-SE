import os
import random
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

# 프로젝트 루트를 경로에 추가
sys.path.append(os.getcwd())
from src.simulation.generator import RandomRoomGenerator, RIRGenerator
from src.simulation.config import MicArrayConfig

def worker(worker_id, num_rirs, output_dir, fs, rir_len, start_id):
    room_gen = RandomRoomGenerator()
    rir_gen = RIRGenerator(fs=fs, rir_len_sec=rir_len)
    mic_config = MicArrayConfig() # 기본 설정 사용
    
    success_count = 0
    while success_count < num_rirs:
        try:
            # 1. 랜덤 방 생성 (shoebox, l_shape, polygon 중 하나)
            room_type = random.choice(["shoebox", "l_shape", "polygon"])
            if room_type == "shoebox":
                room_config = room_gen.generate_random_shoebox()
            elif room_type == "l_shape":
                room_config = room_gen.generate_random_l_shape()
            else:
                room_config = room_gen.generate_random_polygon()
            
            # 2. 룸 생성
            rir_gen.create_room(room_config)
            
            # 3. 안경 마이크 배치
            glasses_center, R = rir_gen.add_ar_glasses_randomly(mic_config)
            
            # 4. 타겟 보이스(입) 추가
            rir_gen.add_target_source(glasses_center, R)
            
            # 5. 랜덤 노이즈 소스 추가 (1~8개)
            num_noise = rir_gen.add_noise_sources_randomly(num_sources_range=(1, 8))
            config_n = f"n{num_noise}"
            
            # 6. RIR 계산 및 저장
            # Filename format: rir_{unique_id}.pkl
            unique_id = start_id + success_count 
            filename = f"{output_dir}/rir_{unique_id:05d}.pkl"
            
            rir_gen.generate_and_save(filename)
            success_count += 1
            
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description='Generate a bank of RIRs.')
    parser.add_argument('--count', type=int, default=1000, help='Total number of RIRs to generate')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(), help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='data/rirs', help='Output directory')
    parser.add_argument('--fs', type=int, default=16000, help='Sampling rate')
    parser.add_argument('--rir_len', type=float, default=1.0, help='RIR truncation length in seconds')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate samples per worker and their start IDs
    base_num_per_worker = args.count // args.workers
    remainder = args.count % args.workers
    
    print(f"Starting RIR bank generation (Total: {args.count}, Workers: {args.workers}, RIR Len: {args.rir_len}s)...")
    processes = []
    current_start_id = 0
    for i in range(args.workers):
        num_for_this_worker = base_num_per_worker + (1 if i < remainder else 0)
        p = mp.Process(target=worker, args=(i, num_for_this_worker, args.output_dir, args.fs, args.rir_len, current_start_id))
        p.start()
        processes.append(p)
        current_start_id += num_for_this_worker
    
    # Progress monitoring of output directory
    pbar = tqdm(total=args.count, desc="Generating RIRs")
    last_count = 0
    while any(p.is_alive() for p in processes):
        current_count = len(list(Path(args.output_dir).glob("*.pkl")))
        if current_count > last_count:
            pbar.update(current_count - last_count)
            last_count = current_count
        import time
        time.sleep(1)
    
    pbar.n = len(list(Path(args.output_dir).glob("*.pkl")))
    pbar.refresh()
    pbar.close()
    
    for p in processes:
        p.join()

    print(f"RIR generation complete. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
