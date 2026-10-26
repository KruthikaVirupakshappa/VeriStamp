"""Create a run folder and append one row to experiments/runs_log.csv.

    run_dir = start_run(cfg, model="m2_clip", owner_ws="B", fold=0)
    ... train / evaluate, write run_dir / "predictions.csv" ...
    finish_run(run_dir, main_metric="auroc_heldout", main_value=0.83, hours=0.4, gpu="T4")
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

from common.config import save_config

LOG = Path("experiments/runs_log.csv")
FIELDS = ["run_id", "date", "owner_ws", "model", "fold", "seed", "config", "git_sha",
          "gpu", "hours", "main_metric", "main_value", "out_dir", "notes"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def start_run(cfg: dict, model: str, owner_ws: str, fold: int | None = None,
              config_name: str = "", notes: str = "") -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{model}" + (f"_fold{fold}" if fold is not None else "")
    run_dir = Path(cfg["paths"]["experiments"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    meta = {"run_id": run_id, "date": dt.date.today().isoformat(), "owner_ws": owner_ws,
            "model": model, "fold": "" if fold is None else fold, "seed": cfg.get("seed"),
            "config": config_name, "git_sha": git_sha(), "out_dir": str(run_dir), "notes": notes}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def finish_run(run_dir: Path, main_metric: str, main_value: float,
               hours: float = 0.0, gpu: str = "", metrics: dict | None = None) -> None:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    row = {**meta, "gpu": gpu, "hours": round(hours, 3),
           "main_metric": main_metric, "main_value": main_value}
    new_file = not LOG.exists() or LOG.stat().st_size == 0
    with open(LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})
