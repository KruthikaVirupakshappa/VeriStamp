"""Build the split manifest from raw GenImage folders (owner: Workstream A).

Expected layout (GenImage's own convention, one folder per generator):

    data/raw/genimage/<generator>/ai/*.png|jpg       AI images from that generator
    data/raw/genimage/<generator>/nature/*.JPEG      the real images paired with it

Writes data/splits/genimage_<per_class>.csv with columns:
    image_id, path, label (0 real, 1 AI), generator, split (train/val/test)

Leave-one-generator-out folds are NOT separate files: ManifestDataset derives
them from this one manifest, so every fold uses exactly the same split.

    python -m preprocessing.build_splits --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from common.config import load_config

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)


def build(cfg: dict, root: Path) -> pd.DataFrame:
    d = cfg["data"]
    fr = d["split_fractions"]
    rows = []
    for gen in d["generators"]:
        for sub, label in (("nature", 0), ("ai", 1)):
            files = list_images(root / gen / sub)
            if len(files) < d["per_class"]:
                raise ValueError(f"{root / gen / sub}: need {d['per_class']} images, found {len(files)}")
            rng = random.Random(f"{cfg['seed']}:{gen}:{sub}")
            chosen = rng.sample(files, d["per_class"])
            n_train = int(round(fr["train"] * len(chosen)))
            n_val = int(round(fr["val"] * len(chosen)))
            for i, path in enumerate(chosen):
                split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
                rows.append({"image_id": f"{gen}/{sub}/{path.stem}", "path": str(path),
                             "label": label, "generator": gen, "split": split})
    df = pd.DataFrame(rows)
    assert not df["image_id"].duplicated().any(), "duplicate image ids"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--root", default=None, help="defaults to <paths.raw>/genimage")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(args.root or Path(cfg["paths"]["raw"]) / "genimage")
    df = build(cfg, root)
    out = Path(cfg["paths"]["splits"]) / f"genimage_{cfg['data']['per_class']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(df.groupby(["generator", "label", "split"]).size().unstack("split"))


if __name__ == "__main__":
    main()
