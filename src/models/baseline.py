import torch
import torch.nn as nn
from .base import BaseSEModel

class DummyModel(BaseSEModel):
    """
    A minimal implementation of BaseSEModel for pipeline verification.
    It performs STFT -> iSTFT (Identity operation) and returns the input.
    Use this to test if the Training Engine (Data, Loss, Logging) is working correctly.
    """
    def __init__(self,
                 in_channels: int = 5,
                 n_fft: int = 512,
                 hop_length: int = 256,
                 window_type: str = "hann"):
        super().__init__(in_channels=in_channels, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window_type=window_type)
        # A dummy learnable parameter to satisfy optimizers
        self.dummy_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Verify dimension logic inside BaseSEModel methods
        # (B, C, T) -> Spec -> (B, C, T)
        spec = self.stft(x)
        wav = self.istft(spec, length=x.shape[-1])
        
        # Add dummy param to computation graph so backward() works
        return wav + self.dummy_param * 0.0
