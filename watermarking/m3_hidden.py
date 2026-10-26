"""M3: HiDDeN-style learned watermark (owner: Workstream C).

Plan (proposal): adapt a public HiDDeN implementation rather than writing one from scratch.
TODO:
  1. Vendor or pip-install the public implementation; note its licence and commit in docs.
  2. Train at 128x128 on COCO with noise layers from configs/m3.yaml (normal edits only).
  3. Wrap the trained encoder/decoder in this class so it matches watermarking.base.Watermarker:
       embed(x01 [B,3,H,W], bits [B,32]) -> stamped x01
       extract(x01) -> logits [B,32]
  4. Decide (team) how a 224x224 detector image is stamped by a 128x128 model, e.g.
     compute the residual at 128 and upsample it to 224 before adding it.
"""
from watermarking.base import Watermarker


class M3Watermarker(Watermarker):
    name = "m3_hidden"

    def __init__(self, cfg: dict, checkpoint: str):
        raise NotImplementedError("Workstream C: see the TODO list at the top of this file.")
