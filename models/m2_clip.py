"""M2: frozen CLIP image encoder + MLP head (owner: Workstream B).

TODO:
  1. Load CLIPModel + CLIPImageProcessor for cfg["model"]["clip_name"]; freeze every parameter.
  2. cache_embeddings(dataset) -> save {"ids", "emb"} to data/cache/ ONCE per split
     (CLIP never changes, so never recompute it inside a training loop).
  3. Train MLPHead(in_dim=512 for B/16, 768 for L/14) on cached embeddings with BCEWithLogitsLoss.
  4. predict(x01): normalise with CLIP stats (preprocessing.transforms.normalize(x, "clip")),
     resize to 224 if needed, embed, run head -> {"p_ai", "features"}.
"""
from models.base import Detector


class M2Detector(Detector):
    name = "m2_clip"
    feature_dim = 512   # 768 for ViT-L/14

    def __init__(self, cfg: dict):
        raise NotImplementedError("Workstream B: see the TODO list at the top of this file.")
