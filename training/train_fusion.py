"""Train proposal Fusion from saved M1 and M2 features.

All artifact locations are explicit command-line arguments. The script never
guesses a local data folder or silently substitutes another split.
"""
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
from models.fusion import FusionDetector

SPLITS = ("train", "val", "test_seen", "test_unseen")


def load_split(m1_dir: Path, m2_embedding_dir: Path, m2_prediction_dir: Path, split: str):
    m1_features = np.load(m1_dir / f"{split}_features.npy")
    m1_index = pd.read_csv(m1_dir / f"{split}_index.csv")
    m2 = torch.load(m2_embedding_dir / f"{split}.pt", map_location="cpu")
    m2_pred = pd.read_csv(m2_prediction_dir / f"predictions_{split}.csv")
    ids = list(m1_index["image_id"])
    if len(m1_features) != len(m1_index) or len(m2["emb"]) != len(ids) or len(m2_pred) != len(ids):
        raise ValueError(f"row-count mismatch in {split}")
    if ids != list(m2["ids"]) or ids != list(m2_pred["image_id"]):
        raise ValueError(f"image_id row-order mismatch in {split}; refusing positional join")
    labels = m1_index["label"].to_numpy(dtype=np.float32)
    if not np.array_equal(labels.astype(int), m2["labels"].numpy().astype(int)):
        raise ValueError(f"label mismatch between M1 and M2 in {split}")
    if not np.array_equal(labels.astype(int), m2_pred["label"].to_numpy(dtype=int)):
        raise ValueError(f"label mismatch between M1 and M2 predictions in {split}")
    return {"m1": torch.from_numpy(np.asarray(m1_features, dtype=np.float32)),
            "m2": m2["emb"].float(), "labels": torch.from_numpy(labels), "ids": ids,
            "m1_p_ai": m1_index["p_ai"].to_numpy(dtype=np.float32),
            "m2_p_ai": m2_pred["p_ai"].to_numpy(dtype=np.float32)}


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    logits = model.logits_from_features(data["m1"].to(device), data["m2"].to(device)).cpu()
    p_ai = torch.sigmoid(logits).numpy()
    labels = data["labels"].numpy().astype(int)
    return {"n": int(len(labels)), "auroc": float(roc_auc_score(labels, p_ai)),
            "accuracy": float(accuracy_score(labels, p_ai >= 0.5)),
            "labels": labels, "p_ai": p_ai}


def late_fusion_metrics(data):
    p_ai = (data["m1_p_ai"] + data["m2_p_ai"]) / 2.0
    labels = data["labels"].numpy().astype(int)
    return {"n": int(len(labels)), "auroc": float(roc_auc_score(labels, p_ai)),
            "accuracy": float(accuracy_score(labels, p_ai >= 0.5))}


def write_predictions(path: Path, data, result):
    pd.DataFrame({"image_id": data["ids"], "label": data["labels"].numpy().astype(int),
                  "p_ai": result["p_ai"]}).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--m1-feature-dir", required=True)
    parser.add_argument("--m2-embedding-dir", required=True)
    parser.add_argument("--m2-prediction-dir", required=True,
                        help="Directory containing predictions_<split>.csv for late fusion")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    m1_dir, m2_emb_dir, m2_pred_dir, run_dir = map(Path, (
        args.m1_feature_dir, args.m2_embedding_dir, args.m2_prediction_dir, args.run_dir))
    for directory in (m1_dir, m2_emb_dir, m2_pred_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Artifact directory not found: {directory}")
    data = {split: load_split(m1_dir, m2_emb_dir, m2_pred_dir, split) for split in SPLITS}
    m1_dim, m2_dim = data["train"]["m1"].shape[1], data["train"]["m2"].shape[1]
    model = FusionDetector(cfg, m1_dim=m1_dim, m2_dim=m2_dim).to(device)
    model.fit_standardization(data["train"]["m1"], data["train"]["m2"])
    train_set = TensorDataset(data["train"]["m1"], data["train"]["m2"], data["train"]["labels"])
    train_loader = DataLoader(train_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                              num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=float(cfg["train"]["lr"]),
                                  weight_decay=float(cfg["train"]["weight_decay"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_auc, best_epoch, best_head, stale = -1.0, 0, None, 0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        losses = []
        for m1_x, m2_x, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.logits_from_features(m1_x.to(device), m2_x.to(device)), labels.to(device))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        val = evaluate(model, data["val"], device)
        print(f"epoch={epoch:02d} loss={np.mean(losses):.4f} val_auroc={val['auroc']:.4f}")
        if val["auroc"] > best_auc:
            best_auc, best_epoch, stale = val["auroc"], epoch, 0
            best_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        else:
            stale += 1
            if stale >= int(cfg["train"].get("patience", 5)):
                break
    if best_head is None:
        raise RuntimeError("No Fusion checkpoint was selected")
    model.head.load_state_dict(best_head)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "m1_dim": m1_dim, "m2_dim": m2_dim},
               run_dir / "best.pt")
    early = {split: evaluate(model, data[split], device) for split in SPLITS}
    metrics = {"best_epoch": best_epoch, "best_val_auroc": best_auc,
               "feature_dims": {"m1": m1_dim, "m2": m2_dim, "fused": m1_dim + m2_dim},
               "early_fusion": {s: {k: v for k, v in r.items() if k not in ("labels", "p_ai")}
                                for s, r in early.items()},
               "late_fusion_average": {s: late_fusion_metrics(data[s]) for s in SPLITS},
               "standardization_fit_split": "train"}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for split in SPLITS:
        write_predictions(run_dir / f"predictions_{split}.csv", data[split], early[split])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
