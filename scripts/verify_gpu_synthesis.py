import torch
from src.data.datamodule import SEDataModule
from src.modules.se_module import SEModule
from src.models.baseline import DummyModel
from src.modules.losses import CompositeLoss

def main():
    db_path = "data/metadata.db"
    dm = SEDataModule(db_path=db_path, batch_size=4, num_workers=0)
    dm.setup()
    
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    print("Batch Keys:", batch.keys())
    print("Raw Speech Shape:", batch['raw_speech'].shape) # (B, T)
    print("RIR Tensor Shape:", batch['rir_tensor'].shape) # (B, M, S, L)
    print("Raw Noises Shape:", batch['raw_noises'].shape) # (B, S-1, T)
    
    # Test SEModule GPU Synthesis
    model = DummyModel()
    loss = CompositeLoss()
    module = SEModule(model, loss)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    
    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Test synthesis
    processed_batch = module._apply_gpu_synthesis(batch_gpu)
    
    print("Processed Noisy Shape:", processed_batch['noisy'].shape)
    print("Processed Clean Shape:", processed_batch['clean'].shape)
    
    assert processed_batch['noisy'].shape == (4, 5, 48000)
    print("GPU Synthesis Verification PASSED!")

if __name__ == "__main__":
    main()
