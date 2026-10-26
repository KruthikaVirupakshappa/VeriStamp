"""One implementation of every metric (owner: Workstream D; everyone uses it).

Conventions: label 1 = AI (positive class), 0 = Real. Scores are P(AI).
Verdicts are the strings "Real", "AI", "Uncertain".
"""
from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

VERDICTS = ("Real", "Uncertain", "AI")


# ---------------- detector ----------------
def accuracy(labels, preds) -> float:
    labels, preds = np.asarray(labels), np.asarray(preds)
    return float((labels == preds).mean())


def auroc(labels, p_ai) -> float:
    """Threshold-free: P(score of a random AI image > score of a random real image)."""
    return float(roc_auc_score(np.asarray(labels), np.asarray(p_ai)))


def confusion_3way(labels, verdicts) -> dict:
    """{true_label: {verdict: count}} with true labels 'Real'/'AI'."""
    out = {t: {v: 0 for v in VERDICTS} for t in ("Real", "AI")}
    for y, v in zip(labels, verdicts):
        out["AI" if int(y) == 1 else "Real"][v] += 1
    return out


def decided_metrics(labels, verdicts) -> dict:
    """Coverage, accuracy on decided images, precision/recall/F1 for AI.
    Recall counts Uncertain AI images as not caught."""
    cm = confusion_3way(labels, verdicts)
    n = sum(sum(r.values()) for r in cm.values())
    decided = n - cm["Real"]["Uncertain"] - cm["AI"]["Uncertain"]
    tp, fp = cm["AI"]["AI"], cm["Real"]["AI"]
    n_ai = sum(cm["AI"].values())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / n_ai if n_ai else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall and not math.isnan(precision) else float("nan"))
    return {"coverage": decided / n if n else float("nan"),
            "accuracy_decided": (cm["Real"]["Real"] + tp) / decided if decided else float("nan"),
            "precision_ai": precision, "recall_ai": recall, "f1_ai": f1,
            "ai_called_real": cm["AI"]["Real"]}


# ---------------- watermark ----------------
def bit_accuracy(bits_true: torch.Tensor, bits_pred: torch.Tensor) -> float:
    """Share of bits recovered correctly; 0.5 means the decoder is guessing."""
    return float((bits_true.long() == bits_pred.long()).float().mean())


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Images in 0..1. Higher = stamped image closer to the cover."""
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean SSIM over a batch [B,3,H,W] in 0..1 (needs scikit-image)."""
    from skimage.metrics import structural_similarity
    vals = [structural_similarity(a.permute(1, 2, 0).cpu().numpy(), b.permute(1, 2, 0).cpu().numpy(),
                                  channel_axis=2, data_range=1.0) for a, b in zip(x, y)]
    return float(np.mean(vals))


def lpips_distance(x: torch.Tensor, y: torch.Tensor, net: str = "alex") -> float:
    """Mean LPIPS (needs the `lpips` package; downloads weights on first use). Lower = more similar."""
    import lpips
    model = lpips.LPIPS(net=net, verbose=False).to(x.device).eval()
    with torch.no_grad():
        return float(model(x * 2 - 1, y * 2 - 1).mean())


def invalidation_rate(valid_flags) -> float:
    """Share of stamped images whose stamp is INVALID after an edit."""
    v = np.asarray(valid_flags, dtype=bool)
    return float((~v).mean())


# ---------------- system (RQ3) ----------------
def score_change(p_before, p_after) -> dict:
    d = np.asarray(p_after) - np.asarray(p_before)
    return {"mean": float(d.mean()), "std": float(d.std()), "max_abs": float(np.abs(d).max())}


def flip_rate(verdicts_before, verdicts_after) -> float:
    a, b = np.asarray(verdicts_before), np.asarray(verdicts_after)
    return float((a != b).mean())


def ai_stamped_as_real(labels, stamped_verdicts) -> int:
    return int(sum(1 for y, v in zip(labels, stamped_verdicts) if int(y) == 1 and v == "Real"))


def agreement(stored_verdicts, fresh_verdicts) -> float:
    """Share of images whose decoded (stored) verdict equals a fresh detector verdict.
    Images with no valid stamp (stored None) count as disagreement."""
    pairs = list(zip(stored_verdicts, fresh_verdicts))
    return float(np.mean([s is not None and s == f for s, f in pairs])) if pairs else float("nan")
