"""THE shared preprocessing function (owner: Workstream A).

Every image the detectors see, real or AI, goes through `preprocess()`.
Using one function for both classes is our bias control: the detector must not
be able to tell real from AI by file format, size or JPEG quality.

Output convention used across the whole repo: float tensor [3, H, W] in 0..1,
NOT normalised. Each model applies its own normalisation internally.
"""
from __future__ import annotations

import io
import random
import zlib

import torch
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def load_rgb(path) -> Image.Image:
    """Open any image file as 3-channel RGB (drops alpha, expands greyscale)."""
    with Image.open(path) as img:
        return img.convert("RGB")


def jpeg_reencode(img: Image.Image, quality: int) -> Image.Image:
    """Save to an in-memory JPEG and read it back."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


def quality_for(image_id: str, quality_range, seed: int) -> int:
    """Deterministic JPEG quality per image, so re-running gives identical tensors."""
    rng = random.Random(zlib.crc32(f"{seed}:{image_id}".encode()))
    lo, hi = quality_range
    return rng.randint(lo, hi)


def preprocess(img: Image.Image, image_id: str, cfg: dict) -> torch.Tensor:
    """resize shorter side -> centre crop -> JPEG re-encode -> tensor in 0..1."""
    p = cfg["preprocess"]
    img = T.Resize(p["resize_short_side"])(img)
    img = T.CenterCrop(p["crop_size"])(img)
    img = jpeg_reencode(img, quality_for(image_id, p["jpeg_quality_range"], cfg["seed"]))
    return T.ToTensor()(img)


def normalize(x01: torch.Tensor, kind: str = "imagenet") -> torch.Tensor:
    """Apply model-specific normalisation. Works on [3,H,W] or [B,3,H,W]."""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if kind == "imagenet" else (CLIP_MEAN, CLIP_STD)
    mean = torch.tensor(mean, device=x01.device).view(-1, 1, 1)
    std = torch.tensor(std, device=x01.device).view(-1, 1, 1)
    return (x01 - mean) / std
