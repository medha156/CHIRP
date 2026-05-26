"""
eval/evaluate.py
================
Test-set evaluation for CHIRP checkpoints.

Loads a checkpoint produced by :mod:`training.trainer`, runs inference on
the test split of :class:`pipelines.datamodule.CHIRPDataModule`, and
writes three artefacts to ``outputs/figures/`` and ``outputs/metrics/``:

1. ``confusion_matrix_<run>.png`` — 2-panel heatmap (absolute counts
   + row-normalised) with species names on both axes.
2. ``per_class_f1_<run>.png`` — horizontal bar chart of per-class F1.
3. ``metrics_<run>.json`` — top-1 accuracy, macro/weighted F1,
   per-class precision / recall / F1 / support.

Usage
-----
::

    python eval/evaluate.py --config configs/fusion.yaml \\
                            --checkpoint outputs/runs/fusion_baseline/checkpoints/best.pt

The config selects the model architecture; the checkpoint provides the
weights. Use ``--run-name <str>`` to suffix the output files; defaults
to the checkpoint's parent directory name.
"""

from __future__ import annotations

# Direct-script execution
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from pipelines.augment import build_augment  # noqa: E402
from pipelines.datamodule import CHIRPDataModule  # noqa: E402
from pipelines.video_dataset import NUM_CLASSES, SPECIES  # noqa: E402
from training.config import TrainConfig  # noqa: E402
from training.train import build_model_and_step  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model, step_fn, loader, device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predictions, labels)`` over the entire loader."""
    model = model.to(device).eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="test", leave=False):
        logits, y = step_fn(model, batch)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Two-panel heatmap: absolute counts (left) + row-normalised (right)."""
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    sns.heatmap(
        cm, ax=axes[0],
        annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Count"}, square=False, linewidths=0.3,
    )
    axes[0].set_title(f"Confusion matrix (counts){title_suffix}")
    axes[0].set_xlabel("Predicted species")
    axes[0].set_ylabel("True species")

    sns.heatmap(
        cm_norm, ax=axes[1],
        annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Row-normalised"}, square=False, linewidths=0.3,
    )
    axes[1].set_title(f"Confusion matrix (row-normalised){title_suffix}")
    axes[1].set_xlabel("Predicted species")
    axes[1].set_ylabel("True species")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out_path)


def plot_per_class_f1(
    f1_per_class: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Horizontal bar chart of per-class F1, sorted descending."""
    order = np.argsort(-f1_per_class)
    sorted_f1 = f1_per_class[order]
    sorted_names = [class_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2E86AB" if f >= 0.5 else "#A23B72" for f in sorted_f1]
    bars = ax.barh(sorted_names, sorted_f1, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(x=sorted_f1.mean(), color="#F18F01", linestyle="--",
               label=f"macro-F1 = {sorted_f1.mean():.3f}")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 score")
    ax.set_title(f"Per-class F1{title_suffix}")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    for bar, v in zip(bars, sorted_f1):
        ax.text(min(v + 0.01, 0.99), bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-class F1 chart → %s", out_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CHIRP test-set evaluation.")
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pt produced by training.trainer.")
    parser.add_argument("--run-name",   default=None,
                        help="Suffix for output filenames.")
    parser.add_argument("--fig-dir",    default="outputs/figures")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--device",     default=None,
                        help="Override cfg.device (auto/cuda/cpu).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = TrainConfig.from_yaml(args.config)
    if args.device:
        cfg.device = args.device

    run_name = args.run_name or Path(args.checkpoint).parent.parent.name
    fig_dir     = Path(args.fig_dir);     fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir); metrics_dir.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------
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

    # ---- model -------------------------------------------------------
    device = torch.device(
        "cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    model, step_fn = build_model_and_step(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint %s (epoch %d)", args.checkpoint, ckpt.get("epoch", "?"))

    # ---- inference ---------------------------------------------------
    preds, labels = run_inference(model, step_fn, dm.test_dataloader(), device)

    # ---- metrics -----------------------------------------------------
    acc = float(accuracy_score(labels, preds))
    macro_f1    = float(f1_score(labels, preds, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    f1_per_class = f1_score(
        labels, preds, average=None,
        labels=list(range(NUM_CLASSES)), zero_division=0,
    )

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        labels, preds,
        labels=list(range(NUM_CLASSES)),
        target_names=SPECIES,
        zero_division=0,
        output_dict=True,
    )

    print(f"\n{'=' * 70}")
    print(f"TEST  top-1 accuracy = {acc:.4f}")
    print(f"      macro-F1       = {macro_f1:.4f}")
    print(f"      weighted-F1    = {weighted_f1:.4f}")
    print(classification_report(
        labels, preds,
        labels=list(range(NUM_CLASSES)),
        target_names=SPECIES,
        zero_division=0,
    ))

    # ---- artefacts ---------------------------------------------------
    title_suffix = f" — {run_name}"
    plot_confusion_matrix(cm, SPECIES,
                          fig_dir / f"confusion_matrix_{run_name}.png",
                          title_suffix=title_suffix)
    plot_per_class_f1(f1_per_class, SPECIES,
                      fig_dir / f"per_class_f1_{run_name}.png",
                      title_suffix=title_suffix)

    metrics = {
        "run_name":       run_name,
        "checkpoint":     str(args.checkpoint),
        "n_test_samples": int(len(labels)),
        "accuracy":       acc,
        "macro_f1":       macro_f1,
        "weighted_f1":    weighted_f1,
        "per_class_f1":   {SPECIES[i]: float(f1_per_class[i]) for i in range(NUM_CLASSES)},
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = metrics_dir / f"metrics_{run_name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics → %s", metrics_path)


if __name__ == "__main__":
    main()
