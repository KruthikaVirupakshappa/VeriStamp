"""M1: ResNet-50 on high-pass residual + FFT inputs (owner: Workstream B).

TODO:
  1. Load torchvision resnet50 with ImageNet weights.
  2. Replace conv1 with a 4-channel conv: copy RGB filters for the residual
     channels and their mean for the FFT channel (guide, coding snippet 5).
  3. Replace fc with nn.Identity() for 2048-d features plus a Linear(2048, 1) head.
  4. predict(x01): build preprocessing.frequency.m1_input(x01), run the network,
     return {"p_ai": sigmoid(logit), "features": 2048-d}.
"""
from models.base import Detector


class M1Detector(Detector):
    name = "m1_resnet"
    feature_dim = 2048

    def __init__(self, cfg: dict):
        raise NotImplementedError("Workstream B: see the TODO list at the top of this file.")
