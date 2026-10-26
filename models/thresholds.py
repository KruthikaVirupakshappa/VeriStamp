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
    """TODO (Workstreams B + D): choose t_low / t_high on validation predictions.

    Suggested rule (a team decision, see docs/decisions.md): among threshold pairs
    where the share of AI images called Real is <= max_ai_as_real and the share of
    Uncertain verdicts is <= max_uncertain, pick the pair with the best accuracy
    on decided images. Report the coverage you get.
    """
    raise NotImplementedError("Implement after the team agrees on the threshold rule.")
