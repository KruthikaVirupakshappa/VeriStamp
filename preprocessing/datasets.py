"""ManifestDataset: the one data loader every workstream uses (owner: Workstream A).

    ds = ManifestDataset("data/splits/genimage_100.csv", "train", cfg, fold=0)
    x01, label, meta = ds[0]       # x01: [3, 224, 224] in 0..1

`fold=k` gives leave-one-generator-out: generator k is removed from train and val,
and test keeps ONLY generator k (the unseen one). `fold=None` gives the
in-distribution splits with all generators.
"""
from __future__ import annotations

import json
import io
import random
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
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


class NativeCacheDataset(Dataset):
    """Read one of M1's frozen native 256-pixel caches.

    The cache rows are aligned with the rows of the corresponding split CSV.
    This dataset deliberately does not open the original image paths, so M2 can
    run from the shared native caches even when the original Drive paths are not
    mounted on the local machine.
    """

    def __init__(self, split_csv: str | Path, cache_file: str | Path,
                 crop_size: int = 224, align: int = 8):
        self.rows = pd.read_csv(split_csv).reset_index(drop=True)
        self.cache_file = Path(cache_file)
        if not self.cache_file.exists():
            raise FileNotFoundError(f"Native cache not found: {self.cache_file}")
        self.array = np.load(self.cache_file, mmap_mode="r")
        if len(self.array) != len(self.rows):
            raise ValueError(
                f"Cache/CSV row mismatch for {self.cache_file}: "
                f"{len(self.array)} cache rows vs {len(self.rows)} CSV rows"
            )
        self.crop_size = crop_size
        self.align = align
        meta_path = self.cache_file.with_suffix(".json")
        self.valid_h = self.valid_w = None
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            self.valid_h = metadata.get("valid_h")
            self.valid_w = metadata.get("valid_w")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        tile = np.asarray(self.array[i])
        if tile.ndim != 3:
            raise ValueError(f"Expected an image tile, got shape {tile.shape}")
        if tile.shape[-1] == 3:
            h, w = tile.shape[:2]
            tile = tile[:h, :w, :]
        elif tile.shape[0] == 3:
            tile = np.transpose(tile, (1, 2, 0))
            h, w = tile.shape[:2]
        else:
            raise ValueError(f"Expected 3-channel cache tile, got shape {tile.shape}")
        vh = int(self.valid_h[i]) if self.valid_h is not None else h
        vw = int(self.valid_w[i]) if self.valid_w is not None else w
        if min(vh, vw) < self.crop_size:
            raise ValueError(f"Cache row {i} is smaller than {self.crop_size}px")
        top = ((vh - self.crop_size) // 2) // self.align * self.align
        left = ((vw - self.crop_size) // 2) // self.align * self.align
        crop = np.ascontiguousarray(tile[top:top + self.crop_size, left:left + self.crop_size])
        x01 = torch.from_numpy(crop).permute(2, 0, 1).float()
        if x01.max() > 1:
            x01 = x01 / 255.0
        meta = {"image_id": row["image_id"], "generator": row["generator"],
                "split": row.get("split", ""), "cache_index": i,
                "image_path": row.get("image_path", row.get("path", ""))}
        return x01, torch.tensor(int(row["label"])), meta


class NativeImageDataset(Dataset):
    """Read stamped images with the exact native-cache preprocessing."""

    def __init__(self, manifest: str | Path, image_root: str | Path,
                 image_column: str, crop_size: int, tile_size: int, align: int,
                 seed: int, jpeg_quality_range: tuple[int, int]):
        self.rows = pd.read_csv(manifest).reset_index(drop=True)
        required = {"image_id", "label", "generator", image_column}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"{manifest} is missing required columns: {sorted(missing)}")
        self.image_root = Path(image_root)
        self.image_column = image_column
        self.crop_size = int(crop_size)
        self.tile_size = int(tile_size)
        self.align = int(align)
        self.seed = int(seed)
        self.jpeg_low, self.jpeg_high = map(int, jpeg_quality_range)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        path = self.image_root / str(row[self.image_column])
        if not path.exists():
            raise FileNotFoundError(f"Stamped image not found: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            quality_rng = random.Random(zlib.crc32(
                f"{self.seed}:{row['image_id']}".encode()))
            quality = quality_rng.randint(self.jpeg_low, self.jpeg_high)
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=quality, subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as decoded_image:
                decoded = np.asarray(decoded_image.convert("RGB"), dtype=np.uint8).copy()
        valid_h, valid_w = min(self.tile_size, height), min(self.tile_size, width)
        top = ((height - valid_h) // 2) // self.align * self.align
        left = ((width - valid_w) // 2) // self.align * self.align
        tile = decoded[top:top + valid_h, left:left + valid_w]
        crop_top = ((valid_h - self.crop_size) // 2) // self.align * self.align
        crop_left = ((valid_w - self.crop_size) // 2) // self.align * self.align
        crop = np.ascontiguousarray(tile[crop_top:crop_top + self.crop_size,
                                         crop_left:crop_left + self.crop_size])
        if crop.shape[:2] != (self.crop_size, self.crop_size):
            raise ValueError(f"Unexpected native crop for {row['image_id']}: {crop.shape}")
        x01 = torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0)
        return x01, torch.tensor(int(row["label"])), {
            "image_id": str(row["image_id"]),
            "generator": str(row["generator"]),
            "cache_index": -1,
            "image_path": str(path),
            "jpeg_quality": quality,
        }
