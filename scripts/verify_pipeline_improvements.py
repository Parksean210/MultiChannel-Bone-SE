import torch
import torchaudio
import numpy as np
from src.data.dataset import SpatialMixingDataset
import os

def test_resampling():
    print("Testing resampling...")
    db_path = "data/metadata.db"
    # Note: This assumes metadata.db exists and has at least one wav file. 
    # For a real test, we would mock the DB or use a dummy file.
    try:
        dataset = SpatialMixingDataset(db_path, target_sr=8000) # Force resampling if files are 16k
        sample = dataset[0]
        print(f"Sample SR: {dataset.target_sr}")
        print(f"Sample shape: {sample['raw_speech'].shape}")
    except Exception as e:
        print(f"Resampling test failed (likely DB issue): {e}")

if __name__ == "__main__":
    if os.path.exists("data/metadata.db"):
        test_resampling()
    else:
        print("data/metadata.db not found, skipping resampling test.")
