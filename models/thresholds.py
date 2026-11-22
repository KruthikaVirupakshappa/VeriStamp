"""Turn P(AI) into Real / AI / Uncertain (owners: B fits thresholds, D evaluates them).

Thresholds are chosen on the VALIDATION split, never on test, and saved per model:
    checkpoints/thresholds_<model>.json  ->  {"t_low": 0.31, "t_high": 0.72, "rule": "..."}
"""
from __future__ import annotations

import json
from pathlib import Path

REAL, AI, UNCERTAIN = "Real", "AI", "Uncertain"
VERDICTS = (REAL, AI, UNCERTAIN)


def to_verdict(p_ai: float, t_low: float, t_high: float) -> str:
    if not 0.0 <= t_low <= t_high <= 1.0:
        raise ValueError(f"need 0 <= t_low <= t_high <= 1, got {t_low}, {t_high}")
    if p_ai < t_low:
        return REAL
    if p_ai > t_high:
        return AI
    return UNCERTAIN


def save_thresholds(path: str | Path, t_low: float, t_high: float, rule: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"t_low": t_low, "t_high": t_high, "rule": rule}, indent=2))


def load_thresholds(path: str | Path) -> tuple[float, float]:
    d = json.loads(Path(path).read_text())
    return d["t_low"], d["t_high"]


def fit_thresholds(p_ai, labels, max_ai_as_real: float = 0.02, max_uncertain: float = 0.20):
    """Fit validation-only Real/Uncertain/AI thresholds."""
    import numpy as np

    scores = np.asarray(p_ai, dtype=float)
    y = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or y.ndim != 1 or len(scores) != len(y) or len(scores) == 0:
        raise ValueError("p_ai and labels must be non-empty arrays of equal length")
    candidates = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), scores]))
    best = None
    for low in candidates:
        for high in candidates[candidates >= low]:
            verdict = np.full(len(scores), UNCERTAIN, dtype=object)
            verdict[scores < low] = REAL
            verdict[scores > high] = AI
            uncertain = verdict == UNCERTAIN
            decided = ~uncertain
            ai_as_real = float(((y == 1) & (verdict == REAL)).sum() / max(1, (y == 1).sum()))
            uncertain_rate = float(uncertain.mean())
            if ai_as_real > max_ai_as_real or uncertain_rate > max_uncertain or not decided.any():
                continue
            expected = np.where(y == 1, AI, REAL)
            accuracy = float((verdict[decided] == expected[decided]).mean())
            coverage = float(decided.mean())
            objective = (accuracy, coverage, -float(high - low), float(low), -float(high))
            if best is None or objective > best[0]:
                best = (objective, float(low), float(high), accuracy, coverage,
                        ai_as_real, uncertain_rate)
    if best is None:
        raise ValueError("No threshold pair satisfies the validation constraints")
    _, low, high, accuracy, coverage, ai_as_real, uncertain_rate = best
    return {"t_low": low, "t_high": high, "decided_accuracy": accuracy,
            "coverage": coverage, "ai_called_real_rate": ai_as_real,
            "uncertain_rate": uncertain_rate,
            "max_ai_as_real": float(max_ai_as_real), "max_uncertain": float(max_uncertain)}
