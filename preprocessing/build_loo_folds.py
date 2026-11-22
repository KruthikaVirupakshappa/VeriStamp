"""Build reproducible three-fold LOO manifests and native caches from raw images.

The input is the final M1 selected-image split and the raw dataset folder. The
script does not copy or repurpose an existing cache. It re-encodes each raw
image with the frozen M1 native preprocessing, then writes one cache per fold.
All paths are explicit arguments so the result is portable between machines.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SPLITS = ("train", "val", "test_seen", "test_unseen")
ROLES = ("train", "val", "test_seen")


def quality_for(image_id: str, seed: int, low: int, high: int) -> int:
    rng = random.Random(zlib.crc32(f"{seed}:{image_id}".encode()))
    return rng.randint(low, high)


def local_image_path(raw_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.exists():
        return path
    marker = f"{raw_root.name}/"
    if marker not in str(path):
        raise FileNotFoundError(f"Cannot map manifest image_path into raw root: {image_path}")
    mapped = raw_root / str(path).split(marker, 1)[1]
    if not mapped.exists():
        raise FileNotFoundError(f"Raw image missing: {mapped}")
    return mapped


def preprocess_native(path: Path, image_id: str, seed: int, tile_size: int,
                      min_side: int, align: int, jpeg_low: int, jpeg_high: int):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if min(width, height) < min_side:
            raise ValueError(f"{image_id}: {width}x{height} is below min_side={min_side}")
        quality = quality_for(image_id, seed, jpeg_low, jpeg_high)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded_image:
            decoded = np.asarray(decoded_image.convert("RGB"), dtype=np.uint8).copy()
    valid_h, valid_w = min(tile_size, height), min(tile_size, width)
    top = ((height - valid_h) // 2) // align * align
    left = ((width - valid_w) // 2) // align * align
    tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    tile[:valid_h, :valid_w] = decoded[top:top + valid_h, left:left + valid_w]
    return tile, {"valid_h": valid_h, "valid_w": valid_w, "orig_h": height,
                  "orig_w": width, "jpeg_quality": quality}


def load_selected_rows(split_dir: Path, raw_root: Path):
    frames = []
    for split in SPLITS:
        frame = pd.read_csv(split_dir / f"{split}.csv").reset_index(drop=True)
        frame["_source_split"] = split
        frame["_source_index"] = np.arange(len(frame))
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    if rows.image_id.duplicated().any():
        raise ValueError("Duplicate image_id in selected M1 splits")
    rows["image_path"] = [str(local_image_path(raw_root, p)) for p in rows.image_path]
    return rows


def assign_roles(rows: pd.DataFrame, generator: str, base_held_out: str, seed: int):
    result = {}
    for label in (0, 1):
        pool = rows[(rows.generator == generator) & (rows.label == label)].copy()
        if len(pool) != 500:
            raise ValueError(f"Expected 500 selected {generator} label={label} rows, found {len(pool)}")
        if generator != base_held_out:
            for role, count in (("train", 350), ("val", 75), ("test_seen", 75)):
                part = pool[pool["split"] == role].copy()
                if len(part) != count:
                    raise ValueError(f"Unexpected existing {generator}/{label}/{role} count")
                result.setdefault(role, []).append(part)
        else:
            order = list(range(len(pool)))
            random.Random(f"{seed}:{generator}:{label}").shuffle(order)
            pool = pool.iloc[order].reset_index(drop=True)
            result.setdefault("train", []).append(pool.iloc[:350].copy())
            result.setdefault("val", []).append(pool.iloc[350:425].copy())
            result.setdefault("test_seen", []).append(pool.iloc[425:].copy())
    return {role: pd.concat(parts, ignore_index=True) for role, parts in result.items()}


def write_cache(frame: pd.DataFrame, output: Path, seed: int, tile_size: int,
                min_side: int, align: int, jpeg_low: int, jpeg_high: int, workers: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint8,
                                      shape=(len(frame), tile_size, tile_size, 3))
    infos = [None] * len(frame)

    def work(i):
        array[i], infos[i] = preprocess_native(Path(frame.iloc[i].image_path), frame.iloc[i].image_id,
                                                seed, tile_size, min_side, align, jpeg_low, jpeg_high)

    with ThreadPoolExecutor(max_workers=workers or 1) as pool:
        list(pool.map(work, range(len(frame))))
    array.flush()
    del array
    metadata = {
        "mode": "native_raw_rebuild", "n": len(frame), "size": tile_size,
        "seed": seed, "min_side": min_side, "align": align,
        "jpeg_quality_range": [jpeg_low, jpeg_high], "jpeg_subsampling": "4:2:0",
        "valid_h": [info["valid_h"] for info in infos],
        "valid_w": [info["valid_w"] for info in infos],
        "orig_h": [info["orig_h"] for info in infos],
        "orig_w": [info["orig_w"] for info in infos],
        "jpeg_quality": [info["jpeg_quality"] for info in infos],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def build(args):
    split_dir, raw_root = Path(args.base_split_dir), Path(args.raw_root)
    rows = load_selected_rows(split_dir, raw_root)
    generators = sorted(rows.generator.unique().tolist())
    if len(generators) != 3:
        raise ValueError(f"Expected exactly three generators, found {generators}")
    base_unseen = rows.loc[rows["split"] == "test_unseen", "generator"].unique().tolist()
    if len(base_unseen) != 1:
        raise ValueError(f"Base split must have exactly one test_unseen generator, found {base_unseen}")
    base_held_out = base_unseen[0]
    role_assignments = {generator: assign_roles(rows, generator, base_held_out, args.seed)
                        for generator in generators}
    spec = {"seed": args.seed, "generators": generators, "folds": {}}
    for held_out in generators:
        split_out = Path(args.split_output_root) / f"loo_{held_out}_n{args.per_class}_seed{args.seed}"
        cache_out = Path(args.cache_output_root) / f"loo_{held_out}_n{args.per_class}_seed{args.seed}"
        split_out.mkdir(parents=True, exist_ok=True)
        cache_out.mkdir(parents=True, exist_ok=True)
        fold_frames = {}
        for role in ROLES:
            parts = [role_assignments[g][role] for g in generators if g != held_out]
            fold_frames[role] = pd.concat(parts, ignore_index=True).sort_values(
                ["generator", "label", "image_id"]).reset_index(drop=True)
        fold_frames["test_unseen"] = rows[rows.generator == held_out].copy().sort_values(
            ["label", "image_id"]).reset_index(drop=True)
        for split, frame in fold_frames.items():
            frame["split"] = split
            frame = frame.drop(columns=["_source_split", "_source_index"])
            frame.to_csv(split_out / f"{split}.csv", index=False)
            write_cache(frame, cache_out / f"{split}_native256.npy", args.seed, args.tile_size,
                        args.min_side, args.align, args.jpeg_low, args.jpeg_high, args.workers)
        all_frame = pd.concat([fold_frames[s] for s in SPLITS], ignore_index=True)
        all_frame.to_csv(split_out / "manifest.csv", index=False)
        overlap = []
        for i, left in enumerate(SPLITS):
            for right in SPLITS[i + 1:]:
                overlap.append({"left": left, "right": right,
                                "overlap_count": int(len(set(fold_frames[left].image_id) &
                                                           set(fold_frames[right].image_id)))})
        if any(row["overlap_count"] for row in overlap):
            raise ValueError(f"Split overlap in fold {held_out}: {overlap}")
        info = {"held_out_generator": held_out, "seen_generators": [g for g in generators if g != held_out],
                "seed": args.seed, "per_class": args.per_class, "raw_root": str(raw_root),
                "preprocess": {"mode": "native_raw_rebuild", "tile_size": args.tile_size,
                               "min_side": args.min_side, "align": args.align,
                               "jpeg_quality_range": [args.jpeg_low, args.jpeg_high],
                               "jpeg_subsampling": "4:2:0"},
                "counts": {s: int(len(fold_frames[s])) for s in SPLITS}, "overlap_checks": overlap}
        (split_out / "split_info.json").write_text(json.dumps(info, indent=2))
        pd.DataFrame(overlap).to_csv(split_out / "leakage_checks.csv", index=False)
        spec["folds"][held_out] = {"split_dir": str(split_out), "cache_dir": str(cache_out)}
    spec_path = Path(args.split_output_root) / "folds.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(json.dumps({"folds_spec": str(spec_path), "folds": spec["folds"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-split-dir", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--split-output-root", required=True)
    parser.add_argument("--cache-output-root", required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--min-side", type=int, default=224)
    parser.add_argument("--align", type=int, default=8)
    parser.add_argument("--jpeg-low", type=int, default=75)
    parser.add_argument("--jpeg-high", type=int, default=95)
    parser.add_argument("--workers", type=int, default=4)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
