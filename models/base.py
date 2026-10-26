"""The detector interface every model implements (agreed in docs/interfaces.md).

    out = detector.predict(x01)     # x01: [B, 3, H, W] in 0..1, NOT normalised
    out["p_ai"]       -> FloatTensor [B]   probability the image is AI-generated
    out["features"]   -> FloatTensor [B, D] the vector Fusion consumes (D=0 allowed for stubs)

Each detector normalises internally (ImageNet stats for M1, CLIP stats for M2),
so callers never have to know which normalisation a model needs.
"""
from __future__ import annotations

import torch


class Detector:
    name = "base"
    feature_dim = 0

    def predict(self, x01: torch.Tensor) -> dict:
        raise NotImplementedError


class StubDetector(Detector):
    """Placeholder until real checkpoints exist (owner: Workstream D).

    Deterministic and cheap: P(AI) is computed from the image's high-frequency
    energy, so it reacts (a little) to edits and watermarks, like a real
    frequency detector would. It means NOTHING about real vs AI. Use it only
    to test plumbing.
    """
    name = "stub"
    feature_dim = 8

    def predict(self, x01: torch.Tensor) -> dict:
        x01 = x01.float()
        gray = x01.mean(dim=1)                                    # [B, H, W]
        dx = (gray[:, :, 1:] - gray[:, :, :-1]).abs().mean(dim=(1, 2))
        dy = (gray[:, 1:, :] - gray[:, :-1, :]).abs().mean(dim=(1, 2))
        energy = (dx + dy) / 2
        p_ai = torch.sigmoid((energy - 0.03) * 60)
        feats = torch.stack([energy, dx, dy, gray.mean(dim=(1, 2)), gray.std(dim=(1, 2)),
                             x01[:, 0].mean(dim=(1, 2)), x01[:, 1].mean(dim=(1, 2)),
                             x01[:, 2].mean(dim=(1, 2))], dim=1)
        return {"p_ai": p_ai, "features": feats}
