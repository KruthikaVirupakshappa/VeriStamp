"""Load YAML configs with simple inheritance.

A config can name a parent with `_base_: base.yaml`. The parent is loaded first
and the child's values are merged on top (nested dicts merge, everything else
is replaced).

    cfg = load_config("configs/m1.yaml")
    cfg["preprocess"]["crop_size"]   # 224, inherited from base.yaml
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("_base_", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Write the fully merged config, e.g. into a run folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
