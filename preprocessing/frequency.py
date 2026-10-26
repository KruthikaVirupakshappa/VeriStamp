"""Frequency-domain inputs for M1 (owner: Workstream B).

Both functions take un-normalised images [B, 3, H, W] in 0..1.
"""
import torch
import torchvision.transforms.functional as TF


def high_pass(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Residual = image - blurred image. Keeps fine detail, noise and generator artefacts."""
    return x - TF.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)


def fft_logmag(x: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] -> [B, 1, H, W]: centred log-magnitude spectrum, normalised per image."""
    gray = x.mean(dim=1, keepdim=True)
    spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
    mag = torch.log1p(spec.abs())
    mean = mag.mean(dim=(-2, -1), keepdim=True)
    std = mag.std(dim=(-2, -1), keepdim=True)
    return (mag - mean) / (std + 1e-6)


def m1_input(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """The 4-channel M1 input: residual (3) + FFT log-magnitude (1)."""
    return torch.cat([high_pass(x, kernel_size, sigma), fft_logmag(x)], dim=1)
