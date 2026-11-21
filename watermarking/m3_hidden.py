"""M3: the trained HiDDeN-style watermark, behind the team's Watermarker interface (owner: Workstream C).

The model, its training and its stamping code live in m3_watermark/ (notebooks 1 to 4 and
m3_watermark.py). This file only wraps m3_watermark.py, unchanged, so pipeline.py, tests/integration.py
and the demo can use M3 in place of the SpreadSpectrumWatermarker stand-in:

    wm = M3Watermarker('path/to/best.pt')
    x_stamped = wm.embed(x01, bits)     # x01 [B,3,H,W] in 0..1, bits from payload.encode
    logits = wm.extract(x01)            # [B,32]; bit = logit > 0

How a large image is stamped by a 128x128 model (docs/decisions.md item 4): the stamp is computed
on a 128 pixel copy, enlarged and added to the full image; reading shrinks the image to 128 pixels.
Images pass through PIL, so they are rounded to 8-bit colours, as in a saved PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

from watermarking import payload
from watermarking.base import Watermarker

M3_DIR = Path(__file__).resolve().parent.parent / 'm3_watermark'


class M3Watermarker(Watermarker):
    name = "m3_hidden"

    def __init__(self, checkpoint: str, device: str | None = None, m3_dir: str | Path = M3_DIR):
        if str(m3_dir) not in sys.path:
            sys.path.insert(0, str(m3_dir))
        import m3_watermark
        self.mw = m3_watermark
        self.m3 = m3_watermark.load_m3(checkpoint, device)

    @torch.no_grad()
    def embed(self, x01: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        out = []
        for img, b in zip(x01, bits):
            fields = payload.decode(b.tolist())
            if not fields["valid"]:
                raise ValueError("M3 stamps payloads made by watermarking.payload.encode only")
            stamped = self.mw.embed(self.m3, TF.to_pil_image(img.clamp(0, 1).cpu()), fields["verdict"],
                                    fields["p_ai_bucket"] / 15, detector_id=fields["detector_id"],
                                    image_tag=fields["image_tag"])
            out.append(TF.to_tensor(stamped))
        return torch.stack(out).to(x01.device)

    @torch.no_grad()
    def extract(self, x01: torch.Tensor) -> torch.Tensor:
        logits = []
        for img in x01:
            small = self.mw._shrink(self.mw._to_tensor(TF.to_pil_image(img.clamp(0, 1).cpu()), self.m3.device))
            logits.append(self.m3.decoder(small * 2 - 1)[0] - 0.5)   # read_bits uses output > 0.5
        return torch.stack(logits).to(x01.device)
