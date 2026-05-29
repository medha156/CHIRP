"""
Resume the mini-sweep with the now-frozen-backbone config — only run the
experiments that didn't complete in the first pass.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import time
from training.config import TrainConfig
from training.train import run_experiment

logger = logging.getLogger(__name__)

RESULTS = Path("outputs/mini_sweep_results.json")

REMAINING = [
    ("eb3_T4",  ["model.model_type=efficientnet", "data.num_frames=4",  "model.n_keyframes=4"]),
    ("eb3_T16", ["model.model_type=efficientnet", "data.num_frames=16", "model.n_keyframes=16",
                 "data.batch_size=4"]),
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    base = TrainConfig.from_yaml("configs/mini_sweep.yaml")

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done_names = {r["name"] for r in results}
    logger.info("Already done: %s", sorted(done_names))

    for i, (name, overrides) in enumerate(REMAINING, 1):
        if name in done_names:
            logger.info("Skipping %s (already done)", name)
            continue
        cfg = base.apply_overrides(overrides)
        cfg.output_dir = f"outputs/runs/mini_sweep/{name}"
        logger.info("\n%s\n[%d/%d] %s\n%s", "=" * 78, i, len(REMAINING), name, "=" * 78)

        t0 = time.perf_counter()
        try:
            metrics = run_experiment(cfg, run_name=name, write_results_csv=False)
            seconds = time.perf_counter() - t0
            row = {
                "name": name, "status": "ok", "seconds": round(seconds, 1),
                "overrides": overrides, "model_type": cfg.model.model_type,
                "T": cfg.data.num_frames, "n_keyframes": cfg.model.n_keyframes,
                "pool": cfg.model.pool, "pretrained": cfg.model.pretrained,
                "freeze_backbone": cfg.model.freeze_backbone,
                "batch_size": cfg.data.batch_size, "lr": cfg.optim.lr,
                "num_epochs_run":  metrics.get("num_epochs_run"),
                "best_val_acc":    metrics.get("best_val_acc"),
                "best_val_f1":     metrics.get("best_val_f1"),
                "best_val_loss":   metrics.get("best_val_loss"),
                "test_acc":        metrics.get("test_acc"),
                "test_f1":         metrics.get("test_f1"),
                "test_loss":       metrics.get("test_loss"),
                "checkpoint":      metrics.get("checkpoint_path"),
            }
        except Exception as e:
            seconds = time.perf_counter() - t0
            logger.error("Failed: %s", e)
            import traceback; logger.error(traceback.format_exc())
            row = {"name": name, "status": f"error: {type(e).__name__}",
                   "seconds": round(seconds, 1), "error_msg": str(e)}

        results.append(row)
        RESULTS.write_text(json.dumps(results, indent=2, default=str))
        logger.info("→ %s | val_f1=%s test_f1=%s (%.1f min)",
                    name, row.get("best_val_f1"), row.get("test_f1"), seconds / 60)


if __name__ == "__main__":
    main()
