"""
experiments/mini_sweep.py
=========================
Real-data mini-sweep on the 538-sample subset of the merged CHIRP dataset.

Five experiments, sequential, designed to produce Milestone 3 deliverables
on CPU in ~3 hours:

  1. RF baseline on frozen EB3 features (no gradient training)
  2. EB3 fine-tune, T=1 (picture-only baseline)
  3. EB3 fine-tune, T=4
  4. EB3 fine-tune, T=16
  5. Fusion (Swin-T + EB3), T=8, K=4

Outputs:
  - outputs/runs/mini_sweep/<exp>/ — config snapshot + checkpoint
  - outputs/mini_sweep_results.json — structured results for REPORT.md
  - outputs/mini_sweep_results.csv — flat row per experiment
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import csv
import json
import logging
import time
import traceback
from pathlib import Path

from training.config import TrainConfig
from training.train import run_experiment

logger = logging.getLogger(__name__)

BASE_CFG = "configs/mini_sweep.yaml"

EXPERIMENTS = [
    # name, overrides — applied via TrainConfig.apply_overrides
    ("rf_baseline", [
        "model.model_type=baseline_rf",
        "data.num_frames=4",
        "model.n_keyframes=4",
    ]),
    ("eb3_T1_picture", [
        "model.model_type=efficientnet",
        "data.num_frames=1",
        "model.n_keyframes=1",
    ]),
    ("eb3_T4", [
        "model.model_type=efficientnet",
        "data.num_frames=4",
        "model.n_keyframes=4",
    ]),
    ("eb3_T16", [
        "model.model_type=efficientnet",
        "data.num_frames=16",
        "model.n_keyframes=16",
        "data.batch_size=4",                 # halve batch for memory
    ]),
    ("fusion_T8", [
        "model.model_type=fusion",
        "data.num_frames=8",
        "model.n_keyframes=4",
        "data.batch_size=4",
        "num_epochs=5",                       # shorter — Swin is the slow one
    ]),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    base = TrainConfig.from_yaml(BASE_CFG)

    results: list[dict] = []
    out_json = Path("outputs/mini_sweep_results.json")
    out_csv  = Path("outputs/mini_sweep_results.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    overall_start = time.perf_counter()
    for i, (name, overrides) in enumerate(EXPERIMENTS, 1):
        cfg = base.apply_overrides(overrides)
        cfg.output_dir = f"outputs/runs/mini_sweep/{name}"

        logger.info("\n%s\n[%d/%d] %s\n%s", "=" * 78, i, len(EXPERIMENTS), name, "=" * 78)
        t0 = time.perf_counter()
        try:
            metrics = run_experiment(cfg, run_name=name, write_results_csv=False)
            seconds = time.perf_counter() - t0
            row = {
                "name":          name,
                "status":        "ok",
                "seconds":       round(seconds, 1),
                "overrides":     overrides,
                "model_type":    cfg.model.model_type,
                "T":             cfg.data.num_frames,
                "n_keyframes":   cfg.model.n_keyframes,
                "pool":          cfg.model.pool,
                "pretrained":    cfg.model.pretrained,
                "batch_size":    cfg.data.batch_size,
                "lr":            cfg.optim.lr,
                "num_epochs_run": metrics.get("num_epochs_run"),
                "best_val_acc":  metrics.get("best_val_acc"),
                "best_val_f1":   metrics.get("best_val_f1"),
                "best_val_loss": metrics.get("best_val_loss"),
                "test_acc":      metrics.get("test_acc"),
                "test_f1":       metrics.get("test_f1"),
                "test_loss":     metrics.get("test_loss"),
                "checkpoint":    metrics.get("checkpoint_path"),
            }
        except Exception as exc:
            seconds = time.perf_counter() - t0
            logger.error("Experiment %s failed (%.1fs): %s\n%s",
                         name, seconds, exc, traceback.format_exc())
            row = {
                "name":     name,
                "status":   f"error: {type(exc).__name__}",
                "seconds":  round(seconds, 1),
                "overrides": overrides,
                "error_msg": str(exc),
            }
        results.append(row)

        # Persist after each experiment so partial results survive a crash.
        out_json.write_text(json.dumps(results, indent=2, default=str))
        _write_csv(out_csv, results)

        elapsed_total = time.perf_counter() - overall_start
        logger.info(
            "→ %s | val_f1=%s test_f1=%s status=%s (%.1f min for this run, %.1f min total)",
            name, _fmt(row.get("best_val_f1")), _fmt(row.get("test_f1")),
            row["status"], seconds / 60, elapsed_total / 60,
        )

    print("\n" + "=" * 90)
    print(f"{'name':<22} {'model':<14} {'T':>3} {'val_acc':>8} {'val_f1':>8} {'test_acc':>8} {'test_f1':>8} {'sec':>7}")
    print("=" * 90)
    for r in results:
        print(f"{r['name']:<22} "
              f"{r.get('model_type', '?'):<14} "
              f"{str(r.get('T', '?')):>3} "
              f"{_fmt(r.get('best_val_acc')):>8} "
              f"{_fmt(r.get('best_val_f1')):>8} "
              f"{_fmt(r.get('test_acc')):>8} "
              f"{_fmt(r.get('test_f1')):>8} "
              f"{r['seconds']:>7.1f}")
    print("=" * 90)
    print(f"\nResults JSON: {out_json}")
    print(f"Results CSV:  {out_csv}")


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—"


def _write_csv(path: Path, results: list[dict]) -> None:
    fields = [
        "name", "model_type", "T", "n_keyframes", "pool", "pretrained",
        "batch_size", "lr", "num_epochs_run",
        "best_val_acc", "best_val_f1", "best_val_loss",
        "test_acc", "test_f1", "test_loss",
        "seconds", "status",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)


if __name__ == "__main__":
    main()
