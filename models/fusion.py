"""Fusion: standardise + concatenate M1 and M2 features, MLP head (owner: Workstream B).

TODO:
  1. Fit per-branch feature mean/std on the TRAIN split only; save them with the checkpoint.
  2. MLP(2048 + 512|768 -> 512 -> 1) with dropout.
  3. Ablation: late fusion = average of M1 and M2 probabilities.
  4. predict(x01): run both branches, standardise, concatenate, head -> {"p_ai", "features"}.
"""
from models.base import Detector


class FusionDetector(Detector):
    name = "fusion"

    def __init__(self, cfg: dict, m1: Detector, m2: Detector):
        raise NotImplementedError("Workstream B: see the TODO list at the top of this file.")
