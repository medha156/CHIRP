"""
experiments/full_integration_run.py
===================================
End-to-end integration smoke run for CHIRP on the synthetic dataset.

Runs one experiment per model_type + a small ablation subset + an
evaluation pass + a baseline fit, capturing timings and metrics for the
project's REPORT.md.

Outputs:
    outputs/integration_report.json   — structured results for REPORT.md
    outputs/runs/integration/<run>/   — per-run checkpoints + config
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
import time
import traceback
from pathlib import Path

from training.config import TrainConfig
from training.train import run_experiment

logger = logging.getLogger(__name__)


BASE_CFG_PATH = "configs/synthetic_smoke.yaml"


# Each entry: (run_name, override-strings)
EXPERIMENTS = [
    # Picture-vs-video corner: EB3 with single frame
    ("eb3_T1_picture", [
        "model.model_type=efficientnet",
        "data.num_frames=1", "model.n_keyframes=1",
    ]),
    # EB3 with a few frames
    ("eb3_T8_mean", [
        "model.model_type=efficientnet",
        "data.num_frames=8", "model.n_keyframes=8",
        "model.pool=mean",
    ]),
    # EB3 attention pool ablation
    ("eb3_T8_attention", [
        "model.model_type=efficientnet",
        "data.num_frames=8", "model.n_keyframes=8",
        "model.pool=attention",
    ]),
    # Swin alone — temporal modelling path
    ("swin_T8", [
        "model.model_type=swin",
        "data.num_frames=8",
    ]),
    # Fusion ensemble — dual input path
    ("fusion_T8", [
        "model.model_type=fusion",
        "data.num_frames=8", "model.n_keyframes=4",
    ]),
    # Optical-flow + swin — verifies the 5-channel path
    ("swin_T8_flow", [
        "model.model_type=swin",
        "data.num_frames=8",
        "data.use_optical_flow=true",
        "data.optical_flow_backend=farneback",
    ]),
    # Classical baseline: RF on frozen EB3 features
    ("baseline_rf", [
        "model.model_type=baseline_rf",
        "data.num_frames=8", "model.n_keyframes=4",
    ]),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = TrainConfig.from_yaml(BASE_CFG_PATH)

    results: list[dict] = []

    for name, overrides in EXPERIMENTS:
        cfg = base.apply_overrides(overrides)
        cfg.output_dir = f"outputs/runs/integration/{name}"
        logger.info("\n%s\n%s\n%s", "=" * 72, name, "=" * 72)

        t0 = time.perf_counter()
        try:
            metrics = run_experiment(cfg, run_name=name, write_results_csv=False)
            seconds = time.perf_counter() - t0
            row = {
                "name": name, "status": "ok", "seconds": round(seconds, 1),
                "overrides": overrides,
                "model_type": cfg.model.model_type,
                "T": cfg.data.num_frames,
                "n_keyframes": cfg.model.n_keyframes,
                "pool": cfg.model.pool,
                "use_optical_flow": cfg.data.use_optical_flow,
                "best_val_acc": metrics.get("best_val_acc"),
                "best_val_f1":  metrics.get("best_val_f1"),
                "test_acc":     metrics.get("test_acc"),
                "test_f1":      metrics.get("test_f1"),
                "num_epochs_run": metrics.get("num_epochs_run"),
            }
        except Exception as exc:
            seconds = time.perf_counter() - t0
            logger.error("Experiment %s failed (%.1fs): %s\n%s",
                         name, seconds, exc, traceback.format_exc())
            row = {
                "name": name, "status": f"error: {type(exc).__name__}",
                "seconds": round(seconds, 1),
                "overrides": overrides,
                "error_msg": str(exc),
            }
        results.append(row)
        logger.info("→ %s  | val_f1=%s  test_f1=%s  (%.1fs)",
                    row.get("status"),
                    _fmt(row.get("best_val_f1")),
                    _fmt(row.get("test_f1")),
                    seconds)

    out = Path("outputs/integration_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {len(results)} experiment results → {out}")

    # Print a compact summary table
    print("\n" + "=" * 78)
    print(f"{'name':<22} {'model':<14} {'T':>3} {'pool':>10} {'flow':>5} "
          f"{'val_f1':>8} {'test_f1':>8} {'sec':>7}  status")
    print("=" * 78)
    for r in results:
        print(f"{r['name']:<22} "
              f"{r.get('model_type','?'):<14} "
              f"{r.get('T','?'):>3} "
              f"{str(r.get('pool','?')):>10} "
              f"{'on' if r.get('use_optical_flow') else 'off':>5} "
              f"{_fmt(r.get('best_val_f1')):>8} "
              f"{_fmt(r.get('test_f1')):>8} "
              f"{r['seconds']:>7.1f}  {r['status']}")
    print("=" * 78)


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "nan"


if __name__ == "__main__":
    main()
