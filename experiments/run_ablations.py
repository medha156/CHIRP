"""
experiments/run_ablations.py
============================
Programmatic ablation sweep for CHIRP.

Runs the four ablation studies called out in the project plan:

  (a) **Swin-T only**            — temporal backbone, no EB3 frame branch.
  (b) **EfficientNet-B3 only**   — frame backbone, no temporal modelling.
  (c) **Classical head swap**    — replace the MLP fusion head with
                                   RandomForest or XGBoost trained on
                                   frozen EB3 embeddings.
  (d) **EfficientNet pooling**   — sweep ``pool ∈ {mean, max, attention}``
                                   with the rest of the EB3 config held
                                   constant.

Each ablation re-uses the project's existing machinery via
:func:`training.train.run_experiment`, so behaviour matches a
normal training run except that:

- ``output_dir`` is isolated per ablation: ``outputs/runs/ablations/<name>/``
- An ablation-only summary CSV is appended at the end of every run to
  ``outputs/ablation_results.csv`` (in addition to the global
  ``outputs/results.csv`` that ``train.py`` already writes).
- ``--dry-run`` builds + validates every config without launching
  training, useful in CI to catch sweep-config bugs cheaply.

Usage
-----
::

    # Full sweep
    python experiments/run_ablations.py --base-config configs/fusion.yaml

    # Validate configs only (no training)
    python experiments/run_ablations.py --base-config configs/fusion.yaml --dry-run

    # Subset of ablations + shorter epochs for a quick sanity check
    python experiments/run_ablations.py --base-config configs/fusion.yaml \\
        --only swin_only effnet_mean baseline_rf \\
        --override num_epochs=3 early_stopping_patience=99
"""

from __future__ import annotations

# Direct-script execution
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import copy
import csv
import datetime as dt
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from training.config import TrainConfig  # noqa: E402
from training.train import run_experiment  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation registry
# ---------------------------------------------------------------------------

@dataclass
class Ablation:
    """A single experiment in the sweep."""
    name:     str                 # short identifier; used as sub-dir name
    group:    str                 # which study (a / b / c / d)
    overrides: list[str] = field(default_factory=list)
    notes:    str = ""

    def apply(self, base: TrainConfig) -> TrainConfig:
        """Return a deep-copied config with this ablation's overrides applied."""
        # apply_overrides re-parses through from_dict → already a fresh object,
        # but we deepcopy first so subsequent mutations don't bleed back.
        cfg = copy.deepcopy(base).apply_overrides(self.overrides)
        # Per-ablation isolated output directory.
        cfg.output_dir = str(Path(base.output_dir).parent / "ablations" / self.name)
        return cfg


ABLATIONS: list[Ablation] = [
    # ── (a) Swin-T only ────────────────────────────────────────────────────
    Ablation(
        name="swin_only",
        group="a_swin_only",
        overrides=["model.model_type=swin"],
        notes="Temporal backbone alone, no EB3 branch.",
    ),

    # ── (b) EfficientNet-B3 only ──────────────────────────────────────────
    Ablation(
        name="effnet_only",
        group="b_effnet_only",
        overrides=["model.model_type=efficientnet"],
        notes="Frame backbone alone with default mean pool.",
    ),

    # ── (c) Replace MLP head with RF / XGBoost ────────────────────────────
    Ablation(
        name="baseline_rf",
        group="c_classical_head",
        overrides=["model.model_type=baseline_rf"],
        notes="Frozen EB3 features → RandomForest head.",
    ),
    Ablation(
        name="baseline_xgb",
        group="c_classical_head",
        overrides=["model.model_type=baseline_xgb"],
        notes="Frozen EB3 features → XGBoost head.",
    ),

    # ── (d) Pooling sweep for EfficientNet ────────────────────────────────
    Ablation(
        name="effnet_pool_mean",
        group="d_pooling",
        overrides=["model.model_type=efficientnet", "model.pool=mean"],
        notes="Arithmetic mean over T frames (default baseline).",
    ),
    Ablation(
        name="effnet_pool_max",
        group="d_pooling",
        overrides=["model.model_type=efficientnet", "model.pool=max"],
        notes="Element-wise max over T frames.",
    ),
    Ablation(
        name="effnet_pool_attention",
        group="d_pooling",
        overrides=["model.model_type=efficientnet", "model.pool=attention"],
        notes="Learned soft-attention pool (F+1 extra params).",
    ),
]


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

ABLATION_FIELDS = [
    "timestamp", "ablation_name", "group", "model_type",
    "pool", "hidden_dim", "dropout", "freeze_backbone",
    "use_optical_flow", "num_frames", "n_keyframes",
    "lr", "batch_size", "num_epochs_run",
    "best_val_acc", "best_val_f1", "best_val_loss",
    "test_acc", "test_f1", "test_loss",
    "seconds", "status", "notes",
]


def append_ablation_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ABLATION_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _row_from_metrics(
    ablation: Ablation,
    cfg: TrainConfig,
    metrics: dict | None,
    seconds: float,
    status: str,
) -> dict:
    """Build a single CSV row from the metrics dict returned by run_experiment."""
    metrics = metrics or {}
    return {
        "timestamp":      dt.datetime.utcnow().isoformat(timespec="seconds"),
        "ablation_name":  ablation.name,
        "group":          ablation.group,
        "model_type":     cfg.model.model_type,
        "pool":           cfg.model.pool,
        "hidden_dim":     cfg.model.hidden_dim,
        "dropout":        cfg.model.dropout,
        "freeze_backbone": cfg.model.freeze_backbone,
        "use_optical_flow": cfg.data.use_optical_flow,
        "num_frames":     cfg.data.num_frames,
        "n_keyframes":    cfg.model.n_keyframes,
        "lr":             cfg.optim.lr,
        "batch_size":     cfg.data.batch_size,
        "num_epochs_run": metrics.get("num_epochs_run", 0),
        "best_val_acc":   metrics.get("best_val_acc", float("nan")),
        "best_val_f1":    metrics.get("best_val_f1",  float("nan")),
        "best_val_loss":  metrics.get("best_val_loss", float("nan")),
        "test_acc":       metrics.get("test_acc", float("nan")),
        "test_f1":        metrics.get("test_f1",  float("nan")),
        "test_loss":      metrics.get("test_loss", float("nan")),
        "seconds":        round(seconds, 1),
        "status":         status,
        "notes":          ablation.notes,
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(
    base_cfg: TrainConfig,
    ablations: list[Ablation],
    out_csv: Path,
    *,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> list[dict]:
    """Execute every ablation in ``ablations`` and append results to ``out_csv``.

    Returns the list of result rows (one per ablation).
    """
    rows: list[dict] = []
    n = len(ablations)
    logger.info("Running %d ablations | base=%s | dry_run=%s | csv=%s",
                n, Path(base_cfg.output_dir), dry_run, out_csv)

    for i, ab in enumerate(ablations, 1):
        cfg = ab.apply(base_cfg)
        logger.info("\n%s\n[%d/%d] ablation=%s  (group=%s)  → %s\n%s",
                    "=" * 72, i, n, ab.name, ab.group, cfg.output_dir, "=" * 72)

        t0 = time.perf_counter()
        if dry_run:
            # Validate config and snapshot it, but don't train.
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            cfg.to_yaml(Path(cfg.output_dir) / "config.yaml")
            row = _row_from_metrics(ab, cfg, None, time.perf_counter() - t0, "dry_run")
        else:
            try:
                metrics = run_experiment(
                    cfg,
                    run_name=ab.name,
                    write_results_csv=True,   # also keep the global outputs/results.csv
                )
                row = _row_from_metrics(ab, cfg, metrics,
                                        time.perf_counter() - t0, "ok")
            except Exception as exc:
                logger.error("Ablation %s failed: %s\n%s",
                             ab.name, exc, traceback.format_exc())
                row = _row_from_metrics(ab, cfg, None,
                                        time.perf_counter() - t0,
                                        f"error: {type(exc).__name__}")
                if not continue_on_error:
                    append_ablation_row(out_csv, row)
                    rows.append(row)
                    raise

        append_ablation_row(out_csv, row)
        rows.append(row)
        logger.info(
            "  → val_f1=%s  test_f1=%s  status=%s  (%.1fs)",
            _fmt(row["best_val_f1"]), _fmt(row["test_f1"]),
            row["status"], row["seconds"],
        )

    return rows


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "nan"


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    """Pretty-print the ablation table grouped by study (a / b / c / d)."""
    if not rows:
        print("(no rows)")
        return

    # Group by ablation.group
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)

    print("\n" + "=" * 78)
    print(f"{'group':<22} {'ablation':<26} {'val_f1':>8} {'test_f1':>8} {'sec':>7}  status")
    print("=" * 78)
    for g in sorted(groups):
        for r in groups[g]:
            print(f"{g:<22} {r['ablation_name']:<26} "
                  f"{_fmt(r['best_val_f1']):>8} {_fmt(r['test_f1']):>8} "
                  f"{r['seconds']:>7.1f}  {r['status']}")
        print("-" * 78)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", required=True,
                        help="Base YAML config (e.g. configs/fusion.yaml).")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Global overrides applied to the base config "
                             "before per-ablation overrides (e.g. num_epochs=5).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these ablation names; default: all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and validate configs without training.")
    parser.add_argument("--out-csv", default="outputs/ablation_results.csv")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Re-raise after the first ablation failure "
                             "(default: log and continue).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    base = TrainConfig.from_yaml(args.base_config).apply_overrides(args.override)
    if args.only:
        unknown = set(args.only) - {a.name for a in ABLATIONS}
        if unknown:
            parser.error(f"Unknown ablation names: {sorted(unknown)}. "
                         f"Choices: {[a.name for a in ABLATIONS]}")
        ablations = [a for a in ABLATIONS if a.name in args.only]
    else:
        ablations = ABLATIONS

    rows = run_sweep(
        base, ablations,
        out_csv=Path(args.out_csv),
        dry_run=args.dry_run,
        continue_on_error=not args.stop_on_error,
    )
    print_summary(rows)
    logger.info("Wrote %d rows → %s", len(rows), args.out_csv)


if __name__ == "__main__":
    main()
