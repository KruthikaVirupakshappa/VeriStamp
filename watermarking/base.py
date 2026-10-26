"""The watermark interface and a classical (non-learned) stand-in (owner: Workstream C).

    x_stamped = wm.embed(x01, bits)     # x01 [B,3,H,W] in 0..1, bits LongTensor [B,32]
    logits    = wm.extract(x01)         # [B,32]; bit = logit > 0

M3 (the learned HiDDeN-style model) will implement this same interface.
Until it exists, SpreadSpectrumWatermarker lets the pipeline, tests and demo run.
It is also a sensible non-learned baseline to compare M3 against in RQ2/RQ3.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class Watermarker:
    name = "base"
    n_bits = 32

    def embed(self, x01: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract(self, x01: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SpreadSpectrumWatermarker(Watermarker):
    """Classic spread-spectrum watermark: each bit adds +/- a fixed pseudo-random pattern.

    Patterns are random +/-1 values on a grid of `cell` x `cell` pixel blocks, the
    same for every image (fixed seed). Decoding high-pass filters the image (to
    remove most of the photo's own content) and correlates with each pattern; the
    sign of the correlation is the bit. If the image is smaller than at stamping
    time (a crop), extract() assumes a centre crop and aligns the patterns to it.

    Strength trade-off, measured on 50 natural 224x224 crops (32 bits):
      0.002 -> ~39 dB PSNR, ~0.99 bit accuracy after JPEG-50
      0.003 -> ~36 dB PSNR, ~0.999 bit accuracy after JPEG-50   (default)
    It has no defence against AI edits by design; that is what RQ2 measures.
    """
    name = "spread_spectrum"

    def __init__(self, n_bits: int = 32, strength: float = 0.003, cell: int = 4, seed: int = 1234):
        self.n_bits, self.strength, self.cell, self.seed = n_bits, strength, cell, seed
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    def _patterns(self, h: int, w: int) -> torch.Tensor:
        """[n_bits, h, w] patterns for an image of size h x w, built for a large canvas
        and centre-cropped, so a centre-cropped image still lines up."""
        key = (h, w)
        if key not in self._cache:
            canvas = 1024
            g = torch.Generator().manual_seed(self.seed)
            n = canvas // self.cell
            coarse = torch.randint(0, 2, (self.n_bits, 1, n, n), generator=g).float() * 2 - 1
            full = F.interpolate(coarse, scale_factor=self.cell, mode="nearest")[:, 0]
            top, left = (canvas - h) // 2, (canvas - w) // 2
            self._cache[key] = full[:, top:top + h, left:left + w].contiguous()
        return self._cache[key]


    def embed(self, x01: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x01.shape
        if h > 1024 or w > 1024:
            raise ValueError("SpreadSpectrumWatermarker supports images up to 1024x1024")
        p = self._patterns(h, w).to(x01.device)                    # [K, H, W]
        signs = bits.to(x01.device).float() * 2 - 1                # [B, K]
        residual = torch.einsum("bk,khw->bhw", signs, p)           # [B, H, W]
        return (x01 + self.strength * residual.unsqueeze(1)).clamp(0, 1)

    @staticmethod
    def _high_pass(x: torch.Tensor) -> torch.Tensor:
        return x - TF.gaussian_blur(x, kernel_size=9, sigma=3.0)

    def extract(self, x01: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x01.shape
        p = self._patterns(h, w).to(x01.device)
        r = self._high_pass(x01.mean(dim=1, keepdim=True))[:, 0]   # [B, H, W]
        ph = self._high_pass(p.unsqueeze(1))[:, 0]                 # [K, H, W]
        corr = torch.einsum("bhw,khw->bk", r, ph)
        norm = ph.flatten(1).norm(dim=1).clamp_min(1e-8) * (h * w) ** 0.5
        return corr / norm * 1e3                                   # logits; sign carries the bit
