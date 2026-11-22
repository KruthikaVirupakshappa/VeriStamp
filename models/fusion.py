"""Early and late fusion of the frozen M1 and M2 outputs."""
from __future__ import annotations

import torch
from torch import nn

from models.base import Detector


class FusionDetector(Detector, nn.Module):
    name = "fusion"

    def __init__(self, cfg: dict, m1: Detector | None = None, m2: Detector | None = None,
                 m1_dim: int | None = None, m2_dim: int | None = None):
        nn.Module.__init__(self)
        self.m1, self.m2 = m1, m2
        self.m1_dim = int(m1.feature_dim if m1 is not None else (m1_dim or 0))
        self.m2_dim = int(m2.feature_dim if m2 is not None else (m2_dim or 0))
        if self.m1_dim <= 0 or self.m2_dim <= 0:
            raise ValueError("Both branches must expose a positive feature_dim")
        self.feature_dim = self.m1_dim + self.m2_dim
        self.standardize = bool(cfg["model"].get("standardize", True))
        hidden = int(cfg["model"]["head"]["hidden"])
        dropout = float(cfg["model"]["head"]["dropout"])
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("m1_mean", torch.zeros(self.m1_dim))
        self.register_buffer("m1_std", torch.ones(self.m1_dim))
        self.register_buffer("m2_mean", torch.zeros(self.m2_dim))
        self.register_buffer("m2_std", torch.ones(self.m2_dim))

    @torch.no_grad()
    def fit_standardization(self, m1_features: torch.Tensor, m2_features: torch.Tensor) -> None:
        if not self.standardize:
            return
        self.m1_mean.copy_(m1_features.mean(dim=0))
        self.m1_std.copy_(m1_features.std(dim=0, unbiased=False).clamp_min(1e-6))
        self.m2_mean.copy_(m2_features.mean(dim=0))
        self.m2_std.copy_(m2_features.std(dim=0, unbiased=False).clamp_min(1e-6))

    def fused_features(self, m1_features: torch.Tensor, m2_features: torch.Tensor) -> torch.Tensor:
        if m1_features.ndim != 2 or m2_features.ndim != 2:
            raise ValueError("Fusion inputs must be two-dimensional feature matrices")
        if m1_features.shape[0] != m2_features.shape[0]:
            raise ValueError("M1 and M2 batch sizes do not match")
        if self.standardize:
            m1_features = (m1_features - self.m1_mean) / self.m1_std
            m2_features = (m2_features - self.m2_mean) / self.m2_std
        return torch.cat([m1_features, m2_features], dim=1)

    def logits_from_features(self, m1_features: torch.Tensor, m2_features: torch.Tensor) -> torch.Tensor:
        return self.head(self.fused_features(m1_features, m2_features)).squeeze(1)

    def predict_features(self, m1_features: torch.Tensor, m2_features: torch.Tensor) -> dict:
        features = self.fused_features(m1_features, m2_features)
        logits = self.head(features).squeeze(1)
        return {"logits": logits, "p_ai": torch.sigmoid(logits), "features": features}

    def predict(self, x01: torch.Tensor) -> dict:
        if self.m1 is None or self.m2 is None:
            raise RuntimeError("predict(x01) requires M1 and M2 detector branches")
        m1_out, m2_out = self.m1.predict(x01), self.m2.predict(x01)
        return self.predict_features(m1_out["features"], m2_out["features"])
