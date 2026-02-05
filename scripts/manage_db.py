import argparse
import sys
import os
import json

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from src.db.engine import create_db_engine
from src.db.manager import DatabaseManager

def main():
    parser = argparse.ArgumentParser(description="Unified Database Management Tool")
    parser.add_argument("--db_path", type=str, default="data/metadata.db", help="Path to SQLite database")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stats Command
    subparsers.add_parser("stats", help="Show database statistics")

    # Sync Command
    subparsers.add_parser("sync", help="Synchronize .wav paths to .npy if missing")

    # Speech Command
    speech_parser = subparsers.add_parser("speech", help="Index speech data")
    speech_parser.add_argument("--path", type=str, required=True, help="Root directory")
    speech_parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    speech_parser.add_argument("--eval", action="store_true", help="Index as eval data")
    speech_parser.add_argument("--sr", type=int, default=16000, help="Sample rate for .npy files")

    # Noise Command
    noise_parser = subparsers.add_parser("noise", help="Index noise data")
    noise_parser.add_argument("--path", type=str, required=True, help="Root directory")
    noise_parser.add_argument("--category", type=str, required=True, help="Noise category")
    noise_parser.add_argument("--sub", type=str, default=None, help="Manual sub-category override")
    noise_parser.add_argument("--sub_depth", type=int, default=1, help="Directory depth for auto sub-category")
    noise_parser.add_argument("--sr", type=int, default=16000, help="Sample rate for .npy files")

    # RIR Command
    rir_parser = subparsers.add_parser("rir", help="Index RIR data (.pkl or .wav)")
    rir_parser.add_argument("--path", type=str, required=True, help="Root directory")

    args = parser.parse_args()

    # Initialize Engine and Manager
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    engine = create_db_engine(args.db_path)
    manager = DatabaseManager(engine)

    if args.command == "stats":
        stats = manager.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.command == "sync":
        manager.sync_paths()

    elif args.command == "speech":
        manager.index_speech(args.path, args.dataset, args.eval, sample_rate=args.sr)
        
    elif args.command == "noise":
        manager.index_noise(args.path, args.category, args.sub, sub_depth=args.sub_depth, sample_rate=args.sr)
        
    elif args.command == "rir":
        manager.index_rirs(args.path)

if __name__ == "__main__":
    main()
