"""End-to-end check: stamp -> edit -> check, on CPU, in under a minute.

    python -m tests.integration --n 20

Uses StubDetector + SpreadSpectrumWatermarker until real checkpoints exist; once
they do, swap them in below (or add --detector/--watermark options). Run this
every week on main. If it fails, fixing it comes before new features.

The numbers it prints come from STUB models and say nothing about our research
questions. They only show that the plumbing works.
"""
from __future__ import annotations

import argparse
import sys

import torch

from evaluation import metrics as M
from models.base import StubDetector
from pipeline import VeriStamp
from tests.sample_images import sample_batch
from watermarking.base import SpreadSpectrumWatermarker
from watermarking.edits import NORMAL_EDITS


def run(n: int = 20) -> dict:
    x = sample_batch(n)
    vs = VeriStamp(StubDetector(), SpreadSpectrumWatermarker(), t_low=0.35, t_high=0.65)
    stamped, records = vs.stamp(x, [f"sample_{i}" for i in range(n)])
    results = {}
    print(f"{'edit':<12}{'bit_acc':>9}{'valid':>8}{'agree':>8}{'mean_dP':>9}")
    for name, edit in NORMAL_EDITS.items():
        checked = vs.check(edit(stamped))
        bits_true = [r["bits"] for r in records]
        bits_out = [c["bits"] for c in checked]
        row = {
            "bit_acc": M.bit_accuracy(torch.stack(bits_true), torch.stack(bits_out)),
            "valid": 1 - M.invalidation_rate([c["stamp_valid"] for c in checked]),
            "agree": M.agreement([c["stored_verdict"] for c in checked], [c["fresh_verdict"] for c in checked]),
            "mean_dp": M.score_change([r["p_ai"] for r in records], [c["p_ai"] for c in checked])["mean"],
        }
        results[name] = row
        print(f"{name:<12}{row['bit_acc']:>9.3f}{row['valid']:>8.2f}{row['agree']:>8.2f}{row['mean_dp']:>+9.3f}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    res = run(ap.parse_args().n)
    ok = res["none"]["valid"] >= 0.9 and res["jpeg90"]["valid"] >= 0.8
    print("INTEGRATION", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
