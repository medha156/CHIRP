"""
experiments/run_temporal_sweep.py
=================================
Picture-vs-video comparison: how much temporal context actually helps?

Sweeps the number of frames per clip while holding everything else
constant, and runs the same backbone(s) across the sweep so each row
isolates the *temporal-extent* effect:

  - **EfficientNet-B3** with ``T ∈ {1, 2, 4, 8, 16}``
    - ``T=1`` is the **picture-only** baseline (single keyframe, no
      temporal pool at all → standard image classification).
    - ``T>1`` averages per-frame features over T frames (the default
      EB3 strategy in CHIRP).
  - **Video Swin-T** with ``T ∈ {2, 4, 8, 16}``
    (skip T=1 because Swin3D's temporal patch size = 2 makes T=1 a
    degenerate corner case — would just pad and not measure anything
    meaningful about temporal modelling).
  - **Fusion (Swin + EB3)** with ``T ∈ {8, 16}`` as upper-bound check.

Outputs
-------
- ``outputs/temporal_sweep_results.csv`` — one row per experiment with
  the same schema as ``outputs/ablation_results.csv`` plus a ``T`` column
  for plotting / pivoting.
- ``outputs/figures/temporal_sweep_val_f1.png`` — line chart of val_f1
  vs T, one line per backbone, with the T=1 picture-only point starred.
- ``outputs/figures/temporal_sweep_test_f1.png`` — same for test split.

Usage
-----
::

    # Full sweep (real training)
    python experiments/run_temporal_sweep.py --base-config configs/fusion.yaml

    # Subset + short epochs for quick check
    python experiments/run_temporal_sweep.py --base-config configs/fusion.yaml \\
        --only eb3_T1_picture eb3_T4 swin_T4 --override num_epochs=3

    # Validate configs only
    python experiments/run_temporal_sweep.py --base-config configs/fusion.yaml --dry-run

    # Re-plot from an existing CSV without retraining
    python experiments/run_temporal_sweep.py --plot-only \\
        --csv outputs/temporal_sweep_results.csv
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.config import TrainConfig  # noqa: E402
from training.train import run_experiment  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

@dataclass
class TemporalExperiment:
    """A single (backbone, T) cell of the comparison."""
    name:       str
    backbone:   str       # "efficientnet" | "swin" | "fusion"
    T:          int       # number of frames fed to the model
    n_keyframes: int      # EB3 keyframes (must be ≤ T)
    notes:      str = ""
    extra_overrides: list[str] = field(default_factory=list)

    def overrides(self) -> list[str]:
        ov = [
            f"model.model_type={self.backbone}",
            f"data.num_frames={self.T}",
            f"model.n_keyframes={self.n_keyframes}",
        ]
        return ov + self.extra_overrides

    def apply(self, base: TrainConfig) -> TrainConfig:
        cfg = copy.deepcopy(base).apply_overrides(self.overrides())
        cfg.output_dir = str(Path(base.output_dir).parent / "temporal_sweep" / self.name)
        return cfg


# Picture-only baseline (T=1) → progressively more frames → full video
EXPERIMENTS: list[TemporalExperiment] = [
    # ── EfficientNet-B3: T=1 (picture-only) through T=16 ─────────────────
    TemporalExperiment(
        name="eb3_T1_picture", backbone="efficientnet", T=1, n_keyframes=1,
        notes="Picture-only baseline: single keyframe, pure image classification.",
    ),
    TemporalExperiment(
        name="eb3_T2", backbone="efficientnet", T=2, n_keyframes=2,
        notes="EB3 with 2 frames (mean pool).",
    ),
    TemporalExperiment(
        name="eb3_T4", backbone="efficientnet", T=4, n_keyframes=4,
        notes="EB3 with 4 frames — matches CLAUDE.md keyframe spec.",
    ),
    TemporalExperiment(
        name="eb3_T8", backbone="efficientnet", T=8, n_keyframes=8,
        notes="EB3 with 8 frames.",
    ),
    TemporalExperiment(
        name="eb3_T16", backbone="efficientnet", T=16, n_keyframes=16,
        notes="EB3 sees every frame from the default 16-frame clip.",
    ),

    # ── Video Swin-T: T=2 through T=16 ──────────────────────────────────
    # T=1 omitted: Swin3D's temporal patch size is 2; T=1 would just be
    # padded and wouldn't tell us anything about temporal modelling.
    TemporalExperiment(
        name="swin_T2", backbone="swin", T=2, n_keyframes=2,
        notes="Swin with 2 frames — minimum temporal extent.",
    ),
    TemporalExperiment(
        name="swin_T4", backbone="swin", T=4, n_keyframes=4,
        notes="Swin with 4 frames.",
    ),
    TemporalExperiment(
        name="swin_T8", backbone="swin", T=8, n_keyframes=8,
        notes="Swin with 8 frames.",
    ),
    TemporalExperiment(
        name="swin_T16", backbone="swin", T=16, n_keyframes=16,
        notes="Swin at the CLAUDE.md default of 16 frames.",
    ),

    # ── Fusion: a few points to anchor the upper bound ──────────────────
    TemporalExperiment(
        name="fusion_T8", backbone="fusion", T=8, n_keyframes=4,
        notes="Fusion ensemble with shorter clips.",
    ),
    TemporalExperiment(
        name="fusion_T16", backbone="fusion", T=16, n_keyframes=4,
        notes="Fusion ensemble at CLAUDE.md default.",
    ),
]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

SWEEP_FIELDS = [
    "timestamp", "experiment", "backbone", "T", "n_keyframes",
    "model_type", "pool", "hidden_dim", "dropout",
    "freeze_backbone", "use_optical_flow",
    "lr", "batch_size", "num_epochs_run",
    "best_val_acc", "best_val_f1", "best_val_loss",
    "test_acc", "test_f1", "test_loss",
    "seconds", "status", "notes",
]


def append_sweep_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _row(
    exp: TemporalExperiment,
    cfg: TrainConfig,
    metrics: dict | None,
    seconds: float,
    status: str,
) -> dict:
    m = metrics or {}
    return {
        "timestamp":      dt.datetime.utcnow().isoformat(timespec="seconds"),
        "experiment":     exp.name,
        "backbone":       exp.backbone,
        "T":              exp.T,
        "n_keyframes":    exp.n_keyframes,
        "model_type":     cfg.model.model_type,
        "pool":           cfg.model.pool,
        "hidden_dim":     cfg.model.hidden_dim,
        "dropout":        cfg.model.dropout,
        "freeze_backbone": cfg.model.freeze_backbone,
        "use_optical_flow": cfg.data.use_optical_flow,
        "lr":             cfg.optim.lr,
        "batch_size":     cfg.data.batch_size,
        "num_epochs_run": m.get("num_epochs_run", 0),
        "best_val_acc":   m.get("best_val_acc", float("nan")),
        "best_val_f1":    m.get("best_val_f1",  float("nan")),
        "best_val_loss":  m.get("best_val_loss", float("nan")),
        "test_acc":       m.get("test_acc", float("nan")),
        "test_f1":        m.get("test_f1",  float("nan")),
        "test_loss":      m.get("test_loss", float("nan")),
        "seconds":        round(seconds, 1),
        "status":         status,
        "notes":          exp.notes,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colour-blind-safe palette per backbone for visual consistency.
BACKBONE_COLORS = {
    "efficientnet": "#2E86AB",   # blue
    "swin":         "#A23B72",   # magenta
    "fusion":       "#F18F01",   # orange
}
BACKBONE_LABEL = {
    "efficientnet": "EfficientNet-B3 (per-frame + pool)",
    "swin":         "Video Swin-T (3D conv)",
    "fusion":       "Swin + EB3 fusion",
}


def plot_sweep(
    rows: list[dict],
    out_path: Path,
    *,
    metric: str = "best_val_f1",
    title_suffix: str = "",
) -> None:
    """Line plot of ``metric`` vs T, one line per backbone, T=1 point starred.

    Rows with ``status != "ok"`` and NaN metrics are skipped.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group rows by backbone, sort by T.
    by_backbone: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        try:
            v = float(r[metric])
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue
        by_backbone.setdefault(r["backbone"], []).append((int(r["T"]), v))

    if not by_backbone:
        logger.warning("No usable rows for plotting (metric=%s).", metric)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for bb, pts in by_backbone.items():
        pts.sort()
        xs, ys = zip(*pts)
        color = BACKBONE_COLORS.get(bb, "#666666")
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=8,
                color=color, label=BACKBONE_LABEL.get(bb, bb))
        # Highlight T=1 (picture-only) with a star if present.
        if 1 in xs:
            i = xs.index(1)
            ax.scatter([1], [ys[i]], s=260, marker="*", color=color,
                       edgecolor="black", linewidth=1.2, zorder=5)
            ax.annotate("picture-only",
                        xy=(1, ys[i]), xytext=(1.4, ys[i] - 0.02),
                        fontsize=9, color=color)

    ax.set_xlabel("Frames per clip (T)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Picture vs video: {metric} vs temporal extent{title_suffix}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1\n(picture)", "2", "4", "8", "16"])
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s plot → %s", metric, out_path)


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(
    base_cfg: TrainConfig,
    experiments: list[TemporalExperiment],
    out_csv: Path,
    *,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    n = len(experiments)
    logger.info("Running %d temporal-extent experiments | dry_run=%s | csv=%s",
                n, dry_run, out_csv)

    for i, exp in enumerate(experiments, 1):
        cfg = exp.apply(base_cfg)
        logger.info("\n%s\n[%d/%d] %s  (backbone=%s, T=%d)  → %s\n%s",
                    "=" * 72, i, n, exp.name, exp.backbone, exp.T,
                    cfg.output_dir, "=" * 72)

        t0 = time.perf_counter()
        if dry_run:
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            cfg.to_yaml(Path(cfg.output_dir) / "config.yaml")
            row = _row(exp, cfg, None, time.perf_counter() - t0, "dry_run")
        else:
            try:
                metrics = run_experiment(cfg, run_name=exp.name, write_results_csv=True)
                row = _row(exp, cfg, metrics, time.perf_counter() - t0, "ok")
            except Exception as exc:
                logger.error("Experiment %s failed: %s\n%s",
                             exp.name, exc, traceback.format_exc())
                row = _row(exp, cfg, None,
                           time.perf_counter() - t0,
                           f"error: {type(exc).__name__}")
                if not continue_on_error:
                    append_sweep_row(out_csv, row)
                    rows.append(row)
                    raise

        append_sweep_row(out_csv, row)
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


def load_rows_from_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("(no rows)")
        return

    # Sort: backbone, then T
    rows_sorted = sorted(rows, key=lambda r: (r["backbone"], int(r["T"])))
    print("\n" + "=" * 82)
    print(f"{'experiment':<22} {'backbone':<14} {'T':>3} "
          f"{'val_f1':>8} {'test_f1':>8} {'sec':>7}  status")
    print("=" * 82)
    last_bb = None
    for r in rows_sorted:
        if last_bb is not None and r["backbone"] != last_bb:
            print("-" * 82)
        last_bb = r["backbone"]
        print(f"{r['experiment']:<22} {r['backbone']:<14} {int(r['T']):>3} "
              f"{_fmt(r['best_val_f1']):>8} {_fmt(r['test_f1']):>8} "
              f"{float(r['seconds']):>7.1f}  {r['status']}")
    print("=" * 82)

    # Picture vs video delta per backbone
    print("\nPicture-vs-video delta (val_f1):")
    for bb in sorted({r["backbone"] for r in rows}):
        pts = [(int(r["T"]), float(r["best_val_f1"]))
               for r in rows if r["backbone"] == bb
               and r["status"] == "ok" and not _is_nan(r["best_val_f1"])]
        if not pts:
            continue
        pts.sort()
        lo_T, lo_f1 = pts[0]
        hi_T, hi_f1 = pts[-1]
        delta = hi_f1 - lo_f1
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"  {bb:14s}  T={lo_T:>2d}: {lo_f1:.4f}  →  "
              f"T={hi_T:>2d}: {hi_f1:.4f}   ({arrow} {delta:+.4f})")


def _is_nan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", help="Base YAML config (e.g. configs/fusion.yaml). "
                                              "Required unless --plot-only is set.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Global overrides applied before per-experiment overrides "
                             "(e.g. num_epochs=5).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these experiment names; default: all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build + validate configs without training.")
    parser.add_argument("--out-csv", default="outputs/temporal_sweep_results.csv")
    parser.add_argument("--fig-dir", default="outputs/figures")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; just re-plot from --csv.")
    parser.add_argument("--csv", default=None,
                        help="CSV to re-plot when --plot-only is set "
                             "(defaults to --out-csv).")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Re-raise after first failure (default: continue).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        csv_path = Path(args.csv or args.out_csv)
        if not csv_path.exists():
            parser.error(f"CSV not found: {csv_path}")
        rows = load_rows_from_csv(csv_path)
        plot_sweep(rows, fig_dir / "temporal_sweep_val_f1.png",
                   metric="best_val_f1", title_suffix=" — val")
        plot_sweep(rows, fig_dir / "temporal_sweep_test_f1.png",
                   metric="test_f1", title_suffix=" — test")
        print_summary(rows)
        return

    if not args.base_config:
        parser.error("--base-config is required (unless --plot-only)")

    base = TrainConfig.from_yaml(args.base_config).apply_overrides(args.override)
    if args.only:
        unknown = set(args.only) - {e.name for e in EXPERIMENTS}
        if unknown:
            parser.error(f"Unknown experiments: {sorted(unknown)}. "
                         f"Choices: {[e.name for e in EXPERIMENTS]}")
        experiments = [e for e in EXPERIMENTS if e.name in args.only]
    else:
        experiments = EXPERIMENTS

    rows = run_sweep(base, experiments,
                     out_csv=Path(args.out_csv),
                     dry_run=args.dry_run,
                     continue_on_error=not args.stop_on_error)

    plot_sweep(rows, fig_dir / "temporal_sweep_val_f1.png",
               metric="best_val_f1", title_suffix=" — val")
    plot_sweep(rows, fig_dir / "temporal_sweep_test_f1.png",
               metric="test_f1", title_suffix=" — test")
    print_summary(rows)


if __name__ == "__main__":
    main()
