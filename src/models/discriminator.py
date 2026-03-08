import torch
import torch.nn as nn


class LearnableSigmoid(nn.Module):
    """원본 ABNet과 동일한 LearnableSigmoid (per-scalar slope)."""
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class MetricDiscriminator(nn.Module):
    """
    원본 ABNet/MetricGAN+ Discriminator와 동일한 구조.

    SpectralNorm + InstanceNorm + PReLU Conv2d 4층 +
    AdaptiveMaxPool -> Linear -> LearnableSigmoid

    입력:
        clean_mag:     (B, T, F)  power-compressed magnitude
        processed_mag: (B, T, F)  power-compressed magnitude
    출력:
        list of intermediate features; [-1] = predicted score (B, 1)
    """
    def __init__(self, ndf: int = 16, in_channel: int = 2):
        super().__init__()
        norm_f = nn.utils.spectral_norm
        self.layers = nn.ModuleList([
            nn.Sequential(
                norm_f(nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.InstanceNorm2d(ndf, affine=True),
                nn.PReLU(ndf),
            ),
            nn.Sequential(
                norm_f(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.InstanceNorm2d(ndf * 2, affine=True),
                nn.PReLU(ndf * 2),
            ),
            nn.Sequential(
                norm_f(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.InstanceNorm2d(ndf * 4, affine=True),
                nn.PReLU(ndf * 4),
            ),
            nn.Sequential(
                norm_f(nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.InstanceNorm2d(ndf * 8, affine=True),
                nn.PReLU(ndf * 8),
            ),
            nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
                norm_f(nn.Linear(ndf * 8, ndf * 4)),
                nn.Dropout(0.3),
                nn.PReLU(ndf * 4),
                norm_f(nn.Linear(ndf * 4, 1)),
                LearnableSigmoid(1),
            ),
        ])

    def forward(self, clean_mag: torch.Tensor, processed_mag: torch.Tensor) -> list:
        """
        Args:
            clean_mag:     (B, T, F)
            processed_mag: (B, T, F)
        Returns:
            list of tensors; [-1] is the predicted PESQ score (B, 1)
        """
        assert clean_mag.ndim == 3, f"Expected 3D input, got {clean_mag.ndim}D"
        xy = torch.stack([clean_mag, processed_mag], dim=1)  # (B, 2, T, F)
        outs = []
        for layer in self.layers:
            xy = layer(xy)
            outs.append(xy)
        return outs
