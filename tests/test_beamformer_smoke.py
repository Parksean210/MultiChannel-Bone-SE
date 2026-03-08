"""Quick smoke test: BeamformerMambaSE forward pass under AMP."""
import torch
from src.models.beamformer_mamba_se import BeamformerMambaSE

def test_forward_amp():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return

    model = BeamformerMambaSE(
        model_dim=64, n_blocks=2,
        beam_weights_path="data/beam_weights_lcmv.pt",
    ).cuda()

    x = torch.randn(2, 5, 16000).cuda()

    # Test under AMP (same as training with precision=16-mixed)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = model(x)

    print(f"OK: input {x.shape} -> output {out.shape}, dtype={out.dtype}")
    assert out.shape == (2, 1, 16000), f"Wrong shape: {out.shape}"
    print("PASS")

if __name__ == "__main__":
    test_forward_amp()
