import torch
import sys
from pathlib import Path

# Add src to path to allow relative imports inside models
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.models.ic_conv_tasnet import ICConvTasNet
    
    def test_forward():
        print("Testing ICConvTasNet forward pass...")
        model = ICConvTasNet(in_channels=5)
        
        # (Batch, Mics, Time)
        # 16kHz, 3s = 48000 samples
        dummy_input = torch.randn(2, 5, 48000)
        
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        assert dummy_input.shape == output.shape, "Shape mismatch!"
        print("Forward pass successful! ✅")

    if __name__ == "__main__":
        test_forward()
except Exception as e:
    print(f"Verification failed: {e}")
    import traceback
    traceback.print_exc()
