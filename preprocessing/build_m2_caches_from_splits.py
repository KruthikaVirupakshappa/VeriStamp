"""Build M2 native caches from already-authoritative M1 split CSVs.

This command never creates or reassigns splits.  Each input split directory is
read exactly as supplied, in CSV row order, and one cache is written for each
row.  The generated metadata contains the row IDs and source paths so cache
alignment can be verified independently of the training code.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

SPLITS = ("train", "val", "test_seen", "test_unseen")


def resolve_image_path(raw_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate
    marker = f"{raw_root.name}/"
    text = str(value)
    if marker in text:
        mapped = raw_root / text.split(marker, 1)[1]
        if mapped.exists():
            return mapped
    raise FileNotFoundError(f"Image referenced by manifest was not found: {value}")


def load_split(split_dir: Path, split: str, raw_root: Path) -> pd.DataFrame:
    path = split_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required split CSV: {path}")
    frame = pd.read_csv(path).reset_index(drop=True)
    required = {"image_id", "image_path", "label", "generator"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image_id values in {path}")
    resolved = [resolve_image_path(raw_root, value) for value in frame["image_path"]]
    frame["_resolved_image_path"] = [str(path) for path in resolved]
    frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
    return frame


def jpeg_quality(image_id: str, low: int, high: int) -> int:
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    return low + (int.from_bytes(digest[:8], "big") % (high - low + 1))


def preprocess(path: Path, image_id: str, tile_size: int, min_side: int,
               align: int, jpeg_low: int, jpeg_high: int):
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        if min(width, height) < min_side:
            raise ValueError(f"{image_id}: image is {width}x{height}, below {min_side}px")
        quality = jpeg_quality(image_id, jpeg_low, jpeg_high)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded_source:
            decoded = np.asarray(decoded_source.convert("RGB"), dtype=np.uint8).copy()
    valid_h, valid_w = min(tile_size, height), min(tile_size, width)
    top = ((height - valid_h) // 2) // align * align
    left = ((width - valid_w) // 2) // align * align
    tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    tile[:valid_h, :valid_w] = decoded[top:top + valid_h, left:left + valid_w]
    return tile, {"image_id": image_id, "jpeg_quality": quality,
                  "orig_h": height, "orig_w": width,
                  "valid_h": valid_h, "valid_w": valid_w}


def write_cache(frame: pd.DataFrame, output: Path, args: argparse.Namespace):
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint8,
                                      shape=(len(frame), args.tile_size, args.tile_size, 3))
    metadata = [None] * len(frame)

    def work(index: int):
        row = frame.iloc[index]
        array[index], metadata[index] = preprocess(
            Path(row["_resolved_image_path"]), str(row["image_id"]),
            args.tile_size, args.min_side, args.align, args.jpeg_low, args.jpeg_high)

    with ThreadPoolExecutor(max_workers=args.workers or 1) as pool:
        list(pool.map(work, range(len(frame))))
    array.flush()
    del array

    info = {
        "mode": "native_raw_rebuild_from_m1_split",
        "n": int(len(frame)), "size": args.tile_size,
        "min_side": args.min_side, "align": args.align,
        "jpeg_quality_range": [args.jpeg_low, args.jpeg_high],
        "jpeg_subsampling": "4:2:0",
        "row_order": "identical to the input CSV",
        "image_ids": [str(row["image_id"]) for _, row in frame.iterrows()],
        "image_paths": [str(row["_resolved_image_path"]) for _, row in frame.iterrows()],
        "labels": [int(row["label"]) for _, row in frame.iterrows()],
        "metadata": metadata,
    }
    output.with_suffix(".json").write_text(json.dumps(info, indent=2))


def validate_fold(frames: dict[str, pd.DataFrame], fold_name: str):
    all_ids = []
    for split, frame in frames.items():
        if len(frame) == 0:
            raise ValueError(f"{fold_name}/{split} is empty")
        all_ids.extend(frame["image_id"].astype(str).tolist())
    if len(all_ids) != len(set(all_ids)):
        raise ValueError(f"Image overlap detected across splits in fold {fold_name}")
    unseen_generators = frames["test_unseen"]["generator"].unique().tolist()
    if len(unseen_generators) != 1:
        raise ValueError(f"{fold_name}/test_unseen must contain one generator")
    held_out = unseen_generators[0]
    for split in ("train", "val", "test_seen"):
        if held_out in set(frames[split]["generator"]):
            raise ValueError(f"Held-out generator {held_out} leaked into {fold_name}/{split}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-spec",
                        help="JSON with folds: {name: {split_dir, cache_dir}}")
    parser.add_argument("--fold", action="append", default=[],
                        metavar="NAME::SPLIT_DIR::CACHE_DIR",
                        help="Explicit fold mapping; repeat once per fold")
    parser.add_argument("--output-spec", help="Output JSON consumed by M2 training")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--split-output-root", required=True,
                        help="Directory for local-path copies of the authoritative CSV rows")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--min-side", type=int, default=224)
    parser.add_argument("--align", type=int, default=8)
    parser.add_argument("--jpeg-low", type=int, default=75)
    parser.add_argument("--jpeg-high", type=int, default=95)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.jpeg_low > args.jpeg_high:
        raise ValueError("--jpeg-low must be <= --jpeg-high")
    raw_root = Path(args.raw_root)
    split_output_root = Path(args.split_output_root)
    if bool(args.fold_spec) == bool(args.fold):
        raise ValueError("Provide exactly one of --fold-spec or --fold")
    if args.fold:
        folds = {}
        for value in args.fold:
            parts = value.split("::")
            if len(parts) != 3 or not all(parts):
                raise ValueError("Each --fold must be NAME::SPLIT_DIR::CACHE_DIR")
            name, split_dir, cache_dir = parts
            if name in folds:
                raise ValueError(f"Duplicate fold name: {name}")
            folds[name] = {"split_dir": split_dir, "cache_dir": cache_dir}
        spec = {"folds": folds}
        spec_path = Path(args.output_spec) if args.output_spec else Path("m2_aligned_folds.json")
    else:
        spec_path = Path(args.fold_spec)
        spec = json.loads(spec_path.read_text())
    folds = spec.get("folds")
    if not isinstance(folds, dict) or not folds:
        raise ValueError("fold spec must contain a non-empty 'folds' mapping")

    output_spec = {"raw_root": str(raw_root), "source_spec": str(spec_path), "folds": {}}
    for fold_name, entry in folds.items():
        split_dir, cache_dir = Path(entry["split_dir"]), Path(entry["cache_dir"])
        frames = {split: load_split(split_dir, split, raw_root) for split in SPLITS}
        validate_fold(frames, fold_name)
        local_split_dir = split_output_root / fold_name
        local_split_dir.mkdir(parents=True, exist_ok=True)
        for split, frame in frames.items():
            write_cache(frame, cache_dir / f"{split}_native256.npy", args)
            csv_frame = frame.drop(columns=["_resolved_image_path", "_source_row"]).copy()
            csv_frame["image_path"] = frame["_resolved_image_path"]
            csv_frame.to_csv(local_split_dir / f"{split}.csv", index=False)
        output_spec["folds"][fold_name] = {
            "split_dir": str(local_split_dir), "cache_dir": str(cache_dir),
            "counts": {split: int(len(frame)) for split, frame in frames.items()},
            "held_out_generator": str(frames["test_unseen"]["generator"].iloc[0]),
        }
        print(f"built {fold_name}: {output_spec['folds'][fold_name]}")
    output_path = Path(args.output_spec) if args.output_spec else spec_path.with_name(
        f"{spec_path.stem}_aligned_caches.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_spec, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
