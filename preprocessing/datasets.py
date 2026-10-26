"""ManifestDataset: the one data loader every workstream uses (owner: Workstream A).

    ds = ManifestDataset("data/splits/genimage_100.csv", "train", cfg, fold=0)
    x01, label, meta = ds[0]       # x01: [3, 224, 224] in 0..1

`fold=k` gives leave-one-generator-out: generator k is removed from train and val,
and test keeps ONLY generator k (the unseen one). `fold=None` gives the
in-distribution splits with all generators.
"""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.transforms import load_rgb, preprocess


def select(df: pd.DataFrame, split: str, generators: list[str], fold: int | None) -> pd.DataFrame:
    rows = df[df["split"] == split]
    if fold is None:
        return rows
    held_out = generators[fold]
    if split == "test":
        return rows[rows["generator"] == held_out]
    return rows[rows["generator"] != held_out]


class ManifestDataset(Dataset):
    def __init__(self, csv_path: str, split: str, cfg: dict, fold: int | None = None):
        df = pd.read_csv(csv_path)
        self.rows = select(df, split, cfg["data"]["generators"], fold).reset_index(drop=True)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows.iloc[i]
        x01 = preprocess(load_rgb(r["path"]), r["image_id"], self.cfg)
        meta = {"image_id": r["image_id"], "generator": r["generator"], "split": r["split"]}
        return x01, torch.tensor(int(r["label"])), meta
