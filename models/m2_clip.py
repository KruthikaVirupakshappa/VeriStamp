"""M2: frozen CLIP image encoder plus a trainable binary head."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.base import Detector
from preprocessing.transforms import normalize


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class M2Detector(Detector, nn.Module):
    name = "m2_clip"

    def __init__(self, cfg: dict):
        nn.Module.__init__(self)
        try:
            from transformers import CLIPModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("M2 requires transformers; install requirements.txt first") from exc
        self.cfg = cfg
        model_cfg = cfg["model"]
        self.clip_name = model_cfg.get("clip_name", "openai/clip-vit-base-patch16")
        self.clip = CLIPModel.from_pretrained(self.clip_name)
        self.clip.eval()
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.feature_dim = int(self.clip.config.projection_dim)
        head_cfg = model_cfg.get("head", {})
        self.head = MLPHead(self.feature_dim, int(head_cfg.get("hidden", 256)),
                            float(head_cfg.get("dropout", 0.2)))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.clip.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        return self

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        x = normalize(x01.float(), "clip")
        size = int(self.clip.config.vision_config.image_size)
        if x.shape[-2:] != (size, size):
            x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
        output = self.clip.get_image_features(pixel_values=x)
        # Transformers 5.x returns BaseModelOutputWithPooling, while older
        # versions returned the projected tensor directly.
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, tuple):
            return output[0]
        return output

    def logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings.float())

    def forward(self, x01: torch.Tensor) -> dict:
        features = self.encode(x01)
        logits = self.logits_from_embeddings(features)
        return {"logits": logits, "p_ai": torch.sigmoid(logits), "features": features}

    def predict(self, x01: torch.Tensor) -> dict:
        self.eval()
        with torch.no_grad():
            out = self.forward(x01)
        return {"p_ai": out["p_ai"], "features": out["features"]}

    def save_checkpoint(self, path: str | Path, cfg: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"head": self.head.state_dict(), "config": cfg or self.cfg,
                    "clip_name": self.clip_name, "feature_dim": self.feature_dim}, path)

    def load_checkpoint(self, path: str | Path, map_location="cpu") -> None:
        checkpoint = torch.load(path, map_location=map_location)
        self.head.load_state_dict(checkpoint["head"])
