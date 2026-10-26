"""VeriStamp's decode-detect-stamp pipeline: the integration layer (owner: Workstream D).

    vs = VeriStamp(detector, watermarker, t_low=0.35, t_high=0.65)
    stamped, records = vs.stamp(x01, image_ids)     # detect, encode verdict, embed
    results = vs.check(edited)                      # decode stamp + fresh detection

Every image is a [3, H, W] tensor in 0..1; batches are [B, 3, H, W].
Works with stubs (models.base.StubDetector, watermarking.base.SpreadSpectrumWatermarker)
until the real M1/M2/Fusion and M3 are ready. Swap them in without changing this file.
"""
from __future__ import annotations

import torch

from models.base import Detector
from models.thresholds import to_verdict
from watermarking import payload
from watermarking.base import Watermarker


class VeriStamp:
    def __init__(self, detector: Detector, watermarker: Watermarker, t_low: float, t_high: float):
        self.detector, self.wm = detector, watermarker
        self.t_low, self.t_high = t_low, t_high

    @torch.no_grad()
    def detect(self, x01: torch.Tensor) -> list[dict]:
        p = self.detector.predict(x01)["p_ai"].detach().cpu().tolist()
        return [{"p_ai": pi, "verdict": to_verdict(pi, self.t_low, self.t_high)} for pi in p]

    @torch.no_grad()
    def read_stamp(self, x01: torch.Tensor) -> list[dict]:
        bits = (self.wm.extract(x01) > 0).long().cpu()
        return [payload.decode(b.tolist()) | {"bits": b} for b in bits]

    @torch.no_grad()
    def stamp(self, x01: torch.Tensor, image_ids: list[str] | None = None):
        """Detect, then embed each image's verdict. Returns (stamped batch, records)."""
        image_ids = image_ids or [""] * len(x01)
        dets = self.detect(x01)
        bits = torch.tensor([payload.encode(d["verdict"], d["p_ai"], self.detector.name,
                                            payload.image_tag_for(i)) for d, i in zip(dets, image_ids)])
        stamped = self.wm.embed(x01, bits.to(x01.device))
        records = [d | {"image_id": i, "bits": b} for d, i, b in zip(dets, image_ids, bits)]
        return stamped, records

    @torch.no_grad()
    def check(self, x01: torch.Tensor) -> list[dict]:
        """The decode-detect step. A valid stamp gives the stored verdict; the detector
        also runs fresh, so evaluation can compare the two (the proposal's pipeline
        reports the stored verdict when the stamp is valid)."""
        stamps, fresh = self.read_stamp(x01), self.detect(x01)
        out = []
        for s, f in zip(stamps, fresh):
            out.append({"stamp_valid": s["valid"], "stored_verdict": s["verdict"],
                        "fresh_verdict": f["verdict"], "p_ai": f["p_ai"],
                        "reported_verdict": s["verdict"] if s["valid"] else f["verdict"],
                        "agree": s["valid"] and s["verdict"] == f["verdict"], "bits": s["bits"]})
        return out
