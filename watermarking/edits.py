"""Image edits for RQ2 / RQ3 / the demo (owner: Workstream C).

All functions take and return [B, 3, H, W] tensors in 0..1. These are the real,
non-differentiable edits used for EVALUATION. Training-time differentiable
approximations belong in noise_layers.py.
"""
from __future__ import annotations

import io

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def _per_image(x01: torch.Tensor, fn) -> torch.Tensor:
    return torch.stack([fn(img) for img in x01])


def jpeg(x01: torch.Tensor, quality: int) -> torch.Tensor:
    def one(img):
        buf = io.BytesIO()
        TF.to_pil_image(img.clamp(0, 1).cpu()).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        with Image.open(buf) as out:
            return TF.to_tensor(out.convert("RGB")).to(x01.device)
    return _per_image(x01, one)


def resize(x01: torch.Tensor, scale: float, back_to_original: bool = True) -> torch.Tensor:
    """Down-scale by `scale`; by default scale back up, as a re-upload pipeline would."""
    _, _, h, w = x01.shape
    small = F.interpolate(x01, scale_factor=scale, mode="bilinear", antialias=True, align_corners=False)
    if not back_to_original:
        return small.clamp(0, 1)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


def center_crop(x01: torch.Tensor, keep: float) -> torch.Tensor:
    """Keep the central `keep` fraction of each side (the image gets smaller)."""
    _, _, h, w = x01.shape
    return TF.center_crop(x01, [int(round(h * keep)), int(round(w * keep))])


def screenshot(x01: torch.Tensor, scale: float = 0.8, gamma: float = 1.1,
               brightness: float = 0.02, quality: int = 90) -> torch.Tensor:
    """Simple screenshot simulation: rescale, colour/gamma shift, re-encode."""
    x = resize(x01, scale)
    x = (x.clamp(0, 1) ** (1 / gamma) + brightness).clamp(0, 1)
    return jpeg(x, quality)


def vae_roundtrip(x01: torch.Tensor) -> torch.Tensor:
    """TODO (Workstream C): diffusers AutoencoderKL ("stabilityai/sd-vae-ft-mse").
    Scale to [-1, 1], encode -> latent_dist.sample() or .mode(), decode, back to [0, 1].
    The VAE expects sizes divisible by 8; upsample 128x128 images (e.g. to 512) first."""
    raise NotImplementedError


def inpaint(x01: torch.Tensor, mask_fraction: float) -> torch.Tensor:
    """TODO (Workstream C): an open-source inpainting pipeline (e.g. a Stable Diffusion
    inpainting checkpoint) with a fixed square mask covering `mask_fraction` of the area."""
    raise NotImplementedError


# name -> function, used by the integration test and the RQ2 suite
NORMAL_EDITS = {
    "none": lambda x: x,
    "jpeg90": lambda x: jpeg(x, 90),
    "jpeg70": lambda x: jpeg(x, 70),
    "jpeg50": lambda x: jpeg(x, 50),
    "resize0.75": lambda x: resize(x, 0.75),
    "resize0.5": lambda x: resize(x, 0.5),
    "crop0.9": lambda x: center_crop(x, 0.9),
    "screenshot": screenshot,
}
