"""Evaluate trained Early Fusion on exact whole-image JPEG degradations.

The command reads authoritative fold CSVs, re-encodes each original image at
the requested JPEG quality, applies the same native centre crop used by M1/M2,
extracts M1 and M2 feature vectors, and evaluates the already-trained Early
Fusion head.  No split is created and no artifact path is guessed.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from common.seed import get_device
from models.fusion import FusionDetector
from models.m2_clip import M2Detector
from models.thresholds import to_verdict

SPLITS = ("test_seen", "test_unseen")


class OriginalJpegDataset(Dataset):
    def __init__(self, csv_path: Path, quality: int, crop_size: int, tile_size: int, align: int):
        self.frame = pd.read_csv(csv_path).reset_index(drop=True)
        required = {"image_id", "image_path", "label", "generator"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        self.quality = int(quality)
        self.crop_size = int(crop_size)
        self.tile_size = int(tile_size)
        self.align = int(align)
        for path in self.frame["image_path"].astype(str):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Raw image referenced by split is missing: {path}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = Path(str(row["image_path"]))
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            if min(width, height) < self.crop_size:
                raise ValueError(f"{row['image_id']}: image is smaller than crop size")
            encoded = io.BytesIO()
            rgb.save(encoded, format="JPEG", quality=self.quality, subsampling=2)
            encoded.seek(0)
            with Image.open(encoded) as decoded_image:
                decoded = np.asarray(decoded_image.convert("RGB"), dtype=np.uint8).copy()

        valid_h, valid_w = min(self.tile_size, height), min(self.tile_size, width)
        top = ((height - valid_h) // 2) // self.align * self.align
        left = ((width - valid_w) // 2) // self.align * self.align
        region = decoded[top:top + valid_h, left:left + valid_w]
        crop_top = ((valid_h - self.crop_size) // 2) // self.align * self.align
        crop_left = ((valid_w - self.crop_size) // 2) // self.align * self.align
        crop = region[crop_top:crop_top + self.crop_size,
                      crop_left:crop_left + self.crop_size]
        if crop.shape[:2] != (self.crop_size, self.crop_size):
            raise ValueError(f"{row['image_id']}: native crop has unexpected shape {crop.shape}")
        x01 = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float().div_(255.0)
        return x01, {
            "image_id": str(row["image_id"]),
            "label": int(row["label"]),
            "generator": str(row["generator"]),
            "image_path": str(path),
            "row": int(index),
        }


def collate(batch):
    images, metadata = zip(*batch)
    return torch.stack(images), list(metadata)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_m1(checkpoint: Path, m1_model_dir: Path, m1_feature_dir: Path, device):
    sys.path.insert(0, str(m1_model_dir.resolve()))
    sys.path.insert(0, str(m1_feature_dir.resolve()))
    from vs_models import build_detector

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_cfg = copy.deepcopy(checkpoint_data["config"])
    # The checkpoint contains the complete trained ResNet.  Avoid downloading
    # the initialization weights before the strict checkpoint load.
    model_cfg["model"]["pretrained"] = False
    model = build_detector(checkpoint_data["input_mode"], model_cfg)
    model.load_state_dict(checkpoint_data["model"], strict=True)
    return model.to(device).eval(), checkpoint_data["config"]


def load_m2(checkpoint: Path, device):
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint_data["config"]
    model = M2Detector(cfg).to(device).eval()
    model.load_checkpoint(checkpoint, map_location="cpu")
    return model


@torch.no_grad()
def extract_features(m1, m2, loader, device):
    m1_features, m2_features, labels, ids, generators, paths, rows = [], [], [], [], [], [], []
    for x01, metadata in loader:
        x01 = x01.to(device)
        m1_out = m1(x01)
        m2_features.append(m2.encode(x01).cpu())
        m1_features.append(m1_out["features"].cpu())
        labels.extend(item["label"] for item in metadata)
        ids.extend(item["image_id"] for item in metadata)
        generators.extend(item["generator"] for item in metadata)
        paths.extend(item["image_path"] for item in metadata)
        rows.extend(item["row"] for item in metadata)
    if not ids:
        raise ValueError("No images found in split")
    return {
        "m1": torch.cat(m1_features), "m2": torch.cat(m2_features),
        "labels": np.asarray(labels, dtype=np.int64), "ids": ids,
        "generators": generators, "paths": paths, "rows": rows,
    }


@torch.no_grad()
def evaluate(fusion, features, thresholds, device):
    logits = fusion.logits_from_features(features["m1"].to(device),
                                         features["m2"].to(device)).cpu()
    p_ai = torch.sigmoid(logits).numpy()
    labels = features["labels"]
    verdicts = np.asarray([to_verdict(float(p), thresholds["t_low"], thresholds["t_high"])
                           for p in p_ai])
    expected = np.where(labels == 1, "AI", "Real")
    decided = verdicts != "Uncertain"
    return {
        "n": int(len(labels)),
        "auroc": float(roc_auc_score(labels, p_ai)),
        "accuracy": float(accuracy_score(labels, p_ai >= 0.5)),
        "decided": int(decided.sum()),
        "coverage": float(decided.mean()),
        "accuracy_on_decided": float((verdicts[decided] == expected[decided]).mean())
        if decided.any() else 0.0,
        "uncertain": int((~decided).sum()),
        "p_ai": p_ai,
        "verdicts": verdicts,
    }


def save_features(path: Path, features: dict):
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "m1_features.npy", features["m1"].numpy())
    np.save(path / "m2_features.npy", features["m2"].numpy())
    pd.DataFrame({"row": features["rows"], "image_id": features["ids"],
                  "image_path": features["paths"], "generator": features["generators"],
                  "label": features["labels"]}).to_csv(path / "index.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-spec", required=True,
                        help="Authoritative folds.json containing split_dir entries")
    parser.add_argument("--m1-checkpoints", required=True,
                        help="JSON mapping fold name to the exact M1 hp_fft/best.pt")
    parser.add_argument("--m1-model-dir", required=True,
                        help="Directory containing the authoritative vs_models.py")
    parser.add_argument("--m1-feature-dir", required=True,
                        help="Directory containing the authoritative vs_features.py")
    parser.add_argument("--m2-run-root", required=True,
                        help="Root containing m2_final_n500_fold_<fold>/best.pt")
    parser.add_argument("--fusion-run-root", required=True,
                        help="Root containing fold_<fold>/best.pt and early_thresholds.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quality", type=int, action="append", required=True,
                        help="JPEG quality; repeat for 65 and 30")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    fold_spec = load_json(Path(args.fold_spec))
    m1_checkpoints = load_json(Path(args.m1_checkpoints))
    folds = fold_spec.get("folds")
    if not isinstance(folds, dict) or not folds:
        raise ValueError("fold-spec must contain a non-empty 'folds' object")
    if set(m1_checkpoints) != set(folds):
        raise ValueError("M1 checkpoint map must contain exactly the fold names in fold-spec")

    device = get_device(args.device)
    output_root = Path(args.output_dir)
    m1_model_dir = Path(args.m1_model_dir)
    m1_feature_dir = Path(args.m1_feature_dir)
    m2_root = Path(args.m2_run_root)
    fusion_root = Path(args.fusion_run_root)
    for m1_lib_dir in (m1_model_dir, m1_feature_dir):
        if not m1_lib_dir.is_dir():
            raise FileNotFoundError(f"M1 library directory not found: {m1_lib_dir}")

    summary = {}
    for fold_name, fold in folds.items():
        split_dir = Path(fold["split_dir"])
        m1_checkpoint = Path(m1_checkpoints[fold_name])
        m2_checkpoint = m2_root / f"m2_final_n500_fold_{fold_name}" / "best.pt"
        fusion_dir = fusion_root / f"fold_{fold_name}"
        fusion_checkpoint = fusion_dir / "best.pt"
        threshold_path = fusion_dir / "early_thresholds.json"
        for required in (split_dir, m1_checkpoint, m2_checkpoint, fusion_checkpoint, threshold_path):
            if not required.exists():
                raise FileNotFoundError(f"Required artifact not found: {required}")

        fusion_checkpoint_data = torch.load(fusion_checkpoint, map_location="cpu", weights_only=False)
        fusion_cfg = fusion_checkpoint_data["config"]
        fusion = FusionDetector(fusion_cfg, m1_dim=int(fusion_checkpoint_data["m1_dim"]),
                                m2_dim=int(fusion_checkpoint_data["m2_dim"])).to(device)
        fusion.load_state_dict(fusion_checkpoint_data["model"], strict=True)
        fusion.eval()
        thresholds = load_json(threshold_path)
        m1, m1_cfg = load_m1(m1_checkpoint, m1_model_dir, m1_feature_dir, device)
        m2 = load_m2(m2_checkpoint, device)
        preprocess_cfg = m1_cfg.get("preprocess", {})
        crop_size = int(preprocess_cfg["crop_size"])
        tile_size = int(preprocess_cfg["tile_size"])
        align = int(preprocess_cfg["align"])
        fold_summary = {}

        for quality in args.quality:
            if not 1 <= quality <= 100:
                raise ValueError(f"JPEG quality must be between 1 and 100: {quality}")
            quality_summary = {}
            for split in SPLITS:
                csv_path = split_dir / f"{split}.csv"
                dataset = OriginalJpegDataset(csv_path, quality, crop_size=crop_size,
                                              tile_size=tile_size, align=align)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, collate_fn=collate)
                features = extract_features(m1, m2, loader, device)
                split_output = output_root / f"fold_{fold_name}" / f"jpeg{quality}" / split
                save_features(split_output, features)
                result = evaluate(fusion, features, thresholds, device)
                p_ai = result.pop("p_ai")
                verdicts = result.pop("verdicts")
                pd.DataFrame({"row": features["rows"], "image_id": features["ids"],
                              "generator": features["generators"], "label": features["labels"],
                              "p_ai": p_ai, "verdict": verdicts}).to_csv(
                                  split_output / "predictions.csv", index=False)
                quality_summary[split] = result
            fold_summary[str(quality)] = quality_summary

        (output_root / f"fold_{fold_name}").mkdir(parents=True, exist_ok=True)
        (output_root / f"fold_{fold_name}" / "metrics.json").write_text(
            json.dumps({"method": "whole original image JPEG re-encode, then native crop",
                        "qualities": [int(q) for q in args.quality],
                        "threshold_source": str(threshold_path),
                        "results": fold_summary}, indent=2))
        summary[fold_name] = fold_summary
        print(json.dumps({fold_name: fold_summary}, indent=2))

    (output_root / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved Early-Fusion JPEG evaluation to {output_root}")


if __name__ == "__main__":
    main()
