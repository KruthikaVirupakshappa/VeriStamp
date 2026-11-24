"""Train M2 from the frozen M1 native caches."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from common.config import load_config
from common.seed import get_device, set_seed
from models.m2_clip import M2Detector
from models.thresholds import fit_thresholds, save_thresholds, to_verdict
from preprocessing.datasets import NativeCacheDataset, NativeImageDataset


def collate(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32), list(metas)


def image_loader(dataset, batch_size: int, workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=workers, collate_fn=collate)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    emb, labels, ids, generators, cache_indices = [], [], [], [], []
    for x01, y, metas in loader:
        emb.append(model.encode(x01.to(device)).cpu())
        labels.append(y.cpu())
        ids.extend(m["image_id"] for m in metas)
        generators.extend(m["generator"] for m in metas)
        cache_indices.extend(m.get("cache_index", -1) for m in metas)
    if not emb:
        raise ValueError("No images found in split")
    return {"emb": torch.cat(emb), "labels": torch.cat(labels), "ids": ids,
            "generators": generators, "cache_indices": cache_indices}


def cache_embeddings(model, loader, device, path: Path):
    if path.exists():
        print(f"Loading cached CLIP embeddings: {path}")
        return torch.load(path, map_location="cpu")
    result = extract_embeddings(model, loader, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    print(f"Saved CLIP embeddings: {path}")
    return result


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    logits = model.logits_from_embeddings(data["emb"].to(device)).cpu()
    p_ai = torch.sigmoid(logits).numpy()
    labels = data["labels"].numpy().astype(int)
    return {"n": int(len(labels)), "auroc": float(roc_auc_score(labels, p_ai)),
            "accuracy": float(accuracy_score(labels, p_ai >= 0.5)),
            "labels": labels, "p_ai": p_ai}


def write_predictions(path: Path, data, result, split: str, thresholds=None):
    rows = [{"image_id": image_id, "generator": generator, "label": int(label),
             "p_ai": float(score),
             "verdict": to_verdict(float(score), thresholds["t_low"], thresholds["t_high"])
             if thresholds else None,
             "split": split, "cache_index": int(cache_index)}
            for image_id, generator, label, score, cache_index in zip(
                data["ids"], data["generators"], result["labels"], result["p_ai"],
                data["cache_indices"])]
    pd.DataFrame(rows).to_csv(path, index=False)


def jpeg_original_batch(x01: torch.Tensor, metas, quality: int):
    output = []
    for x, meta in zip(x01, metas):
        image_path = str(meta.get("image_path", ""))
        if not image_path or not Path(image_path).exists():
            raise FileNotFoundError(f"Original image required for exact JPEG evaluation: {image_path}")
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, subsampling=2)
            buf.seek(0)
            with Image.open(buf) as decoded:
                decoded = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        vh, vw = min(256, height), min(256, width)
        top = ((height - vh) // 2) // 8 * 8
        left = ((width - vw) // 2) // 8 * 8
        crop_top = ((vh - 224) // 2) // 8 * 8
        crop_left = ((vw - 224) // 2) // 8 * 8
        crop = decoded[top + crop_top:top + crop_top + 224,
                       left + crop_left:left + crop_left + 224]
        output.append(torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float().div(255))
    return torch.stack(output)


@torch.no_grad()
def degraded_embeddings(model, dataset, batch_size, device, quality):
    result = {"emb": [], "labels": [], "ids": [], "generators": [], "cache_indices": []}
    for x01, y, metas in image_loader(dataset, batch_size):
        degraded = jpeg_original_batch(x01, metas, quality)
        result["emb"].append(model.encode(degraded.to(device)).cpu())
        result["labels"].append(y.cpu())
        result["ids"].extend(m["image_id"] for m in metas)
        result["generators"].extend(m["generator"] for m in metas)
        result["cache_indices"].extend(m["cache_index"] for m in metas)
    result["emb"] = torch.cat(result["emb"])
    result["labels"] = torch.cat(result["labels"])
    return result


def three_way_metrics(result, thresholds):
    verdicts = np.asarray([to_verdict(float(score), thresholds["t_low"], thresholds["t_high"])
                           for score in result["p_ai"]])
    labels = result["labels"]
    decided = verdicts != "Uncertain"
    expected = np.where(labels == 1, "AI", "Real")
    return {"n": int(len(labels)), "decided": int(decided.sum()),
            "coverage": float(decided.mean()),
            "accuracy_on_decided": float((verdicts[decided] == expected[decided]).mean())
            if decided.any() else 0.0,
            "uncertain": int((~decided).sum())}


def train_fold(cfg, split_dir: Path, cache_dir: Path, run_dir: Path, device, args,
               fold_name: str | None = None):
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"Native cache directory not found: {cache_dir}")

    split_names = ("train", "val", "test_seen", "test_unseen")
    for split in split_names:
        csv_path = split_dir / f"{split}.csv"
        cache_path = cache_dir / f"{split}_native256.npy"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing native cache: {cache_path}")
    for row in pd.read_csv(split_dir / "test_unseen.csv").itertuples():
        if not Path(row.image_path).exists():
            raise FileNotFoundError(
                f"Exact JPEG evaluation requires raw test image, but it is missing: {row.image_path}")

    batch_size = int(cfg["train"].get("batch_size", 64))
    crop_size = int(cfg["preprocess"].get("crop_size", 224))
    align = int(cfg["preprocess"].get("align", 8))
    datasets = {split: NativeCacheDataset(split_dir / f"{split}.csv",
                                           cache_dir / f"{split}_native256.npy",
                                           crop_size=crop_size, align=align)
                for split in split_names}

    model = M2Detector(cfg).to(device)
    print(f"device={device} clip={model.clip_name} feature_dim={model.feature_dim}")
    embedding_dir = run_dir / "embeddings"
    data = {split: cache_embeddings(model, image_loader(datasets[split], batch_size, args.workers),
                                    device, embedding_dir / f"{split}.pt")
            for split in split_names}

    if args.augmented_train_csv_root:
        if not fold_name:
            raise ValueError("fold name is required when using augmented training")
        augmented_csv = Path(args.augmented_train_csv_root) / f"{fold_name}.csv"
        if not augmented_csv.exists():
            raise FileNotFoundError(f"Augmented train manifest not found: {augmented_csv}")
        jpeg_range = tuple(cfg["preprocess"]["jpeg_quality_range"])
        augmented_dataset = NativeImageDataset(
            augmented_csv, args.augmented_train_image_root, "stamped_file",
            crop_size=crop_size, tile_size=int(cfg["preprocess"]["tile_size"]),
            align=align, seed=int(cfg["seed"]), jpeg_quality_range=jpeg_range)
        augmented = cache_embeddings(
            model, image_loader(augmented_dataset, batch_size, args.workers), device,
            embedding_dir / "train_stamped.pt")
        train_emb = torch.cat([data["train"]["emb"], augmented["emb"]])
        train_labels = torch.cat([data["train"]["labels"], augmented["labels"]])
        print(f"augmented_train_images={len(augmented_dataset)}")
    else:
        train_emb = data["train"]["emb"]
        train_labels = data["train"]["labels"]

    train_loader = DataLoader(TensorDataset(train_emb, train_labels),
                              batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=float(cfg["train"]["lr"]),
                                  weight_decay=float(cfg["train"]["weight_decay"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_head, stale = -1.0, 0, None, 0
    patience = int(cfg["train"].get("patience", 5))
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        losses = []
        for emb, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.logits_from_embeddings(emb.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, data["val"], device)
        print(f"epoch={epoch:02d} loss={np.mean(losses):.4f} val_auroc={val['auroc']:.4f}")
        if val["auroc"] > best_auc:
            best_auc, best_epoch, stale = val["auroc"], epoch, 0
            best_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_head is None:
        raise RuntimeError("No M2 checkpoint was selected")
    model.head.load_state_dict(best_head)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(run_dir / "best.pt", cfg)
    results = {split: evaluate(model, data[split], device) for split in split_names}
    thresholds = fit_thresholds(results["val"]["p_ai"], results["val"]["labels"])
    save_thresholds(run_dir / "thresholds.json", thresholds["t_low"], thresholds["t_high"],
                    rule="validation-only; AI-called-Real <= 2%; Uncertain <= 20%")
    three_way = {split: three_way_metrics(results[split], thresholds)
                 for split in split_names}
    jpeg_results = {}
    for quality in (65, 30):
        degraded = degraded_embeddings(model, datasets["test_unseen"], batch_size, device, quality)
        scored = evaluate(model, degraded, device)
        jpeg_results[str(quality)] = {
            "method": "whole original image, then native preprocessing",
            "n": scored["n"], "auroc": scored["auroc"], "accuracy": scored["accuracy"],
        }
    metrics = {"best_epoch": best_epoch, "best_val_auroc": best_auc,
               "splits": {split: {k: v for k, v in result.items()
                                   if k not in ("labels", "p_ai")}
                          for split, result in results.items()},
               "uncertain_thresholds": thresholds,
               "three_way": three_way,
               "jpeg_degradation": {"results": jpeg_results}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for split in split_names:
        write_predictions(run_dir / f"predictions_{split}.csv", data[split], results[split], split,
                          thresholds)
    print(f"best_epoch={best_epoch} val_auroc={best_auc:.4f} "
          f"test_unseen_auroc={results['test_unseen']['auroc']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m2.yaml")
    parser.add_argument("--split-dir", default=None,
                        help="One fold's split directory; use --fold-spec for all folds")
    parser.add_argument("--cache-dir", default=None,
                        help="One fold's native cache directory; use --fold-spec for all folds")
    parser.add_argument("--fold-spec", default=None,
                        help="folds.json created by preprocessing.build_loo_folds")
    parser.add_argument("--fold", default="all", help="all or a generator name when using --fold-spec")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--augmented-train-csv-root", default=None,
                        help="Directory with one stamped train CSV per fold")
    parser.add_argument("--augmented-train-image-root", default=None,
                        help="Root directory for stamped training image paths")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if bool(args.augmented_train_csv_root) != bool(args.augmented_train_image_root):
        parser.error("Both augmented training arguments are required together")
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    root_run = Path(args.run_dir)
    if args.fold_spec:
        spec = json.loads(Path(args.fold_spec).read_text())
        folds = list(spec["folds"])
        if args.fold != "all":
            if args.fold not in spec["folds"]:
                raise ValueError(f"Unknown fold {args.fold}; choices: {folds}")
            folds = [args.fold]
        summary = {}
        for held_out in folds:
            entry = spec["folds"][held_out]
            metrics = train_fold(cfg, Path(entry["split_dir"]), Path(entry["cache_dir"]),
                                 root_run / f"m2_final_n500_fold_{held_out}", device, args,
                                 fold_name=held_out)
            summary[held_out] = {"best_val_auroc": metrics["best_val_auroc"],
                                 "test_unseen_auroc": metrics["splits"]["test_unseen"]["auroc"]}
        (root_run / "m2_all_folds_summary.json").parent.mkdir(parents=True, exist_ok=True)
        (root_run / "m2_all_folds_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return
    if not args.split_dir or not args.cache_dir:
        parser.error("--split-dir and --cache-dir are required unless --fold-spec is supplied")
    train_fold(cfg, Path(args.split_dir), Path(args.cache_dir), root_run, device, args)


if __name__ == "__main__":
    main()
