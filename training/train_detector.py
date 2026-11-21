"""Train M2 from the frozen M1 native caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from common.config import load_config
from common.seed import get_device, set_seed
from models.m2_clip import M2Detector
from preprocessing.datasets import NativeCacheDataset


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
        cache_indices.extend(m["cache_index"] for m in metas)
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


def write_predictions(path: Path, data, result, split: str):
    rows = [{"image_id": image_id, "generator": generator, "label": int(label),
             "p_ai": float(score), "split": split, "cache_index": int(cache_index)}
            for image_id, generator, label, score, cache_index in zip(
                data["ids"], data["generators"], result["labels"], result["p_ai"],
                data["cache_indices"])]
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m2.yaml")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    split_dir = Path(args.split_dir or cfg["data"]["split_dir"])
    cache_dir = Path(args.cache_dir or cfg["data"]["native_cache_dir"])
    run_dir = Path(args.run_dir or cfg["paths"]["experiments"]) / "m2_final_n500"
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

    train_loader = DataLoader(TensorDataset(data["train"]["emb"], data["train"]["labels"]),
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
    metrics = {"best_epoch": best_epoch, "best_val_auroc": best_auc,
               "splits": {split: {k: v for k, v in result.items()
                                   if k not in ("labels", "p_ai")}
                          for split, result in results.items()}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for split in split_names:
        write_predictions(run_dir / f"predictions_{split}.csv", data[split], results[split], split)
    print(f"best_epoch={best_epoch} val_auroc={best_auc:.4f} "
          f"test_unseen_auroc={results['test_unseen']['auroc']:.4f}")


if __name__ == "__main__":
    main()
