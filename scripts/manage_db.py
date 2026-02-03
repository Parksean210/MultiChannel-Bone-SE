import argparse
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.db.engine import create_db_engine
from src.db.manager import DatabaseManager

def main():
    parser = argparse.ArgumentParser(description="Detailed Database Management CLI")
    parser.add_argument("--db_path", type=str, default="data/metadata.db", help="Path to SQLite database")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # Speech Command
    speech_parser = subparsers.add_parser("speech", help="Index speech data")
    speech_parser.add_argument("--path", type=str, required=True, help="Root directory of speech data")
    speech_parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., KsponSpeech)")
    speech_parser.add_argument("--eval", action="store_true", help="Flag if this is evaluation data")

    # Noise Command
    noise_parser = subparsers.add_parser("noise", help="Index noise data")
    noise_parser.add_argument("--path", type=str, required=True, help="Root directory of noise data")
    noise_parser.add_argument("--category", type=str, required=True, help="Category of noise (e.g., urban, factory)")
    noise_parser.add_argument("--sub", "--sub_category", type=str, default=None, help="Manual sub-category override")

    # RIR Command
    rir_parser = subparsers.add_parser("rir", help="Index RIR data")
    rir_parser.add_argument("--path", type=str, required=True, help="Root directory of RIR data")

    args = parser.parse_args()

    # Initialize Engine and Manager
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    engine = create_db_engine(args.db_path)
    manager = DatabaseManager(engine)

    if args.command == "speech":
        manager.index_speech(args.path, args.dataset, args.eval)
    elif args.command == "noise":
        manager.index_noise(args.path, args.category, args.sub)
    elif args.command == "rir":
        manager.index_rirs(args.path)

if __name__ == "__main__":
    main()
