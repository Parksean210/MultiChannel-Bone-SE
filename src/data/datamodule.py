import torch
import os
import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import numpy as np
import random
from .dataset import SpatialMixingDataset

def worker_init_fn(worker_id):
    """
    DataLoader 워커별로 고유한 시드를 설정하여 
    멀티프로세싱 환경에서 난수 생성이 중복되는 것을 방지합니다.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_gpu_synthesis(batch):
    """
    Custom collator to handle:
    1. Variable length RIRs (pad to batch max)
    2. Dictionary of scalars (mic_config)
    3. Normal tensors
    """
    # Find max RIR length in this batch
    max_rir_len = max(item['rir_tensor'].shape[-1] for item in batch)
    
    for item in batch:
        # Pad RIR tensor to batch max
        curr_len = item['rir_tensor'].shape[-1]
        if curr_len < max_rir_len:
            item['rir_tensor'] = F.pad(item['rir_tensor'], (0, max_rir_len - curr_len))
            
    return default_collate(batch)

class SEDataModule(L.LightningDataModule):
    """
    LightningDataModule wrapping SpatialMixingDataset.
    Handles Train/Val/Test Dataloaders.
    """
    def __init__(self, 
                 db_path: str = "data/metadata.db",
                 batch_size: int = 4,
                 num_workers: int = 4,
                 target_sr: int = 16000,
                 snr_range: tuple = (-5, 20)):
        super().__init__()
        self.save_hyperparameters()
        self.db_path = db_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_sr = target_sr
        self.snr_range = snr_range

    def setup(self, stage: str = None):
        # Assign Train/Val/Test datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = SpatialMixingDataset(
                db_path=self.db_path,
                target_sr=self.target_sr,
                split="train",
                snr_range=self.snr_range
            )
            
            self.val_dataset = SpatialMixingDataset(
                db_path=self.db_path,
                target_sr=self.target_sr,
                split="val",
                snr_range=self.snr_range
            )

        if stage == "test" or stage is None:
            self.test_dataset = SpatialMixingDataset(
                db_path=self.db_path,
                target_sr=self.target_sr,
                split="test",
                snr_range=self.snr_range
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_gpu_synthesis,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_gpu_synthesis,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_gpu_synthesis,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
