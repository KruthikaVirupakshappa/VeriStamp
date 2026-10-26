"""One call to make a run repeatable."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # safe to call without a GPU
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(name: str = "auto") -> torch.device:
    """'auto' picks CUDA (Colab/Kaggle/HPC), then MPS (M-series Macs), then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
