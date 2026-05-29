"""
experiments/build_milestone3_artifacts.py
=========================================
Consume outputs/mini_sweep_results.json and produce the Milestone 3
analysis artifacts:

- outputs/figures/mini_results_table.png        — clean rendered table
- outputs/figures/mini_per_experiment_f1.png    — bar chart of val/test F1
- outputs/figures/mini_picture_vs_video.png     — EB3 T=1 vs T=4 vs T=16
- outputs/figures/mini_confusion_<exp>.png      — confusion matrix per exp
- outputs/figures/mini_per_class_f1_<exp>.png   — per-class F1 per exp
- outputs/figures/shap_rf_real.png              — SHAP on RF baseline (real data)

Each plot has Stanford species labels where applicable, and the picture-
vs-video chart explicitly marks the T=1 ("picture-only") point.

Usage
-----
::

    python experiments/build_milestone3_artifacts.py
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pipelines.video_dataset import NUM_CLASSES, SPECIES

logger = logging.getLogger(__name__)

RESULTS_JSON = Path("outputs/mini_sweep_results.json")
FIG_DIR      = Path("outputs/figures")
METRICS_DIR  = Path("outputs/metrics")


# ---------------------------------------------------------------------------
# Plot 1 — bar chart of val/test F1 per experiment
# ---------------------------------------------------------------------------

def plot_summary_bars(results: list[dict]) -> Path:
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return None  # type: ignore[return-value]

    labels = [r["name"] for r in ok]
    val_f1  = [_safe_float(r.get("best_val_f1")) for r in ok]
    test_f1 = [_safe_float(r.get("test_f1"))     for r in ok]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.38
    bars_v = ax.bar(x - w/2, val_f1,  w, label="val macro-F1",
                    color="#2E86AB", edgecolor="white")
    bars_t = ax.bar(x + w/2, test_f1, w, label="test macro-F1",
                    color="#F18F01", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Mini-sweep: val vs test macro-F1 per experiment")
    ax.axhline(1.0 / 20, color="gray", linestyle=":", linewidth=1,
               label=f"random chance (1/{NUM_CLASSES})")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for bars in (bars_v, bars_t):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h): continue
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    out = FIG_DIR / "mini_per_experiment_f1.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ---------------------------------------------------------------------------
# Plot 2 — picture-vs-video line (EB3 T=1, 4, 16)
# ---------------------------------------------------------------------------

def plot_picture_vs_video(results: list[dict]) -> Path | None:
    pts = []
    for r in results:
        if r.get("status") != "ok": continue
        if r.get("model_type") != "efficientnet": continue
        t = r.get("T")
        f1 = _safe_float(r.get("best_val_f1"))
        tf1 = _safe_float(r.get("test_f1"))
        if t is None: continue
        pts.append((int(t), f1, tf1))
    if not pts:
        return None
    pts.sort()
    xs = [p[0] for p in pts]
    val = [p[1] for p in pts]
    test = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, val,  marker="o", linewidth=2, markersize=10,
            color="#2E86AB", label="val macro-F1")
    ax.plot(xs, test, marker="s", linewidth=2, markersize=10,
            color="#F18F01", label="test macro-F1")
    if 1 in xs:
        i = xs.index(1)
        ax.scatter([1], [val[i]],  s=280, marker="*", color="#2E86AB",
                   edgecolor="black", linewidth=1.2, zorder=5)
        ax.annotate("picture-only", xy=(1, val[i]), xytext=(1.3, val[i] - 0.02),
                    fontsize=10, color="#2E86AB")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) + ("\n(picture)" if x == 1 else "") for x in xs])
    ax.set_xlabel("Frames per clip (T)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("EfficientNet-B3: picture vs video — does temporal context help?")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axhline(1.0 / 20, color="gray", linestyle=":", linewidth=1,
               label=f"random chance (1/{NUM_CLASSES})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "mini_picture_vs_video.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ---------------------------------------------------------------------------
# Plot 3 — confusion + per-class F1 from saved metrics JSONs
# ---------------------------------------------------------------------------

def plot_per_experiment_eval(results: list[dict]) -> list[Path]:
    """Run eval/evaluate.py-style plots on each successful checkpoint."""
    from eval.evaluate import (
        plot_confusion_matrix, plot_per_class_f1, run_inference,
    )
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
    )

    from pipelines.augment import build_augment
    from pipelines.datamodule import CHIRPDataModule
    from training.config import TrainConfig
    from training.train import build_model_and_step

    saved: list[Path] = []
    for r in results:
        if r.get("status") != "ok": continue
        if r.get("model_type", "").startswith("baseline_"): continue   # no checkpoint
        name = r["name"]
        ckpt = r.get("checkpoint")
        if not ckpt or not Path(ckpt).exists():
            logger.warning("  %s: checkpoint missing (%s) — skipping", name, ckpt)
            continue

        cfg_path = Path(ckpt).parent.parent / "config.yaml"
        if not cfg_path.exists():
            continue
        cfg = TrainConfig.from_yaml(cfg_path)
        dm = CHIRPDataModule(
            data_root=cfg.data.data_root,
            datasets=tuple(cfg.data.datasets),
            splits=tuple(cfg.data.splits),
            seed=cfg.seed,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            n_frames=cfg.data.num_frames,
            height=cfg.data.height,
            width=cfg.data.width,
            backend=cfg.data.backend,
            eval_transform=build_augment("test", {"augment": True}),
        )
        dm.setup()
        device = torch.device("cpu")
        model, step_fn = build_model_and_step(cfg, device)
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])

        preds, labels = run_inference(model, step_fn, dm.test_dataloader(), device)
        if len(labels) == 0:
            continue

        cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
        f1_per_class = f1_score(labels, preds, average=None,
                                labels=list(range(NUM_CLASSES)),
                                zero_division=0)
        macro = float(f1_score(labels, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(labels, preds))

        cm_path = FIG_DIR / f"mini_confusion_{name}.png"
        f1_path = FIG_DIR / f"mini_per_class_f1_{name}.png"
        plot_confusion_matrix(cm, SPECIES, cm_path,
                              title_suffix=f" — {name} (acc={acc:.3f})")
        plot_per_class_f1(f1_per_class, SPECIES, f1_path,
                          title_suffix=f" — {name} (macro-F1={macro:.3f})")
        saved.extend([cm_path, f1_path])

        # Persist a metrics JSON
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        (METRICS_DIR / f"mini_metrics_{name}.json").write_text(json.dumps({
            "run_name":     name,
            "accuracy":     acc,
            "macro_f1":     macro,
            "per_class_f1": {SPECIES[i]: float(f1_per_class[i]) for i in range(NUM_CLASSES)},
            "confusion":    cm.tolist(),
            "report":       classification_report(
                labels, preds, labels=list(range(NUM_CLASSES)),
                target_names=SPECIES, zero_division=0,
            ),
        }, indent=2))

    return saved


# ---------------------------------------------------------------------------
# Plot 4 — SHAP on the RF baseline (real data)
# ---------------------------------------------------------------------------

def plot_shap_real_rf() -> Path | None:
    """Run SHAP on the RF baseline trained during the mini sweep."""
    try:
        import joblib
    except ImportError:
        return None

    rf_path = Path("outputs/runs/mini_sweep/rf_baseline/baselines/rf.joblib")
    feat_train = Path("outputs/runs/mini_sweep/rf_baseline/baselines/features_train.npz")
    if not rf_path.exists() or not feat_train.exists():
        logger.warning("RF baseline artefacts not found at %s / %s", rf_path, feat_train)
        return None

    from eval.shap_analysis import compute_shap_values, plot_shap_summary
    from models.baselines import load_cached_features

    rf = joblib.load(rf_path)
    X_train, _ = load_cached_features(feat_train)

    n = min(60, len(X_train))
    sv, X_explained = compute_shap_values(rf, X_train, name="rf",
                                          n_samples_explain=n)
    out = FIG_DIR / "shap_rf_real.png"
    plot_shap_summary(sv, X_explained, out, top_k=15,
                      title_suffix=" — RF on real EB3 features (mini-sweep)")
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"Run experiments/mini_sweep.py first to produce {RESULTS_JSON}")
    results = json.loads(RESULTS_JSON.read_text())
    logger.info("Loaded %d results", len(results))

    plot_summary_bars(results)
    plot_picture_vs_video(results)
    plot_per_experiment_eval(results)
    plot_shap_real_rf()


if __name__ == "__main__":
    main()
