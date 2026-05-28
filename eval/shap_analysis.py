"""
eval/shap_analysis.py
=====================
SHAP feature-importance analysis for CHIRP baseline classifiers.

For RF and XGB we use :class:`shap.TreeExplainer` (exact, very fast). For
KNN — which has no tree structure — we fall back to
:class:`shap.KernelExplainer` on a small background sample to keep
runtime bounded.

Produces ``outputs/figures/shap_<model>.png`` containing two panels:

- Left: **mean |SHAP| bar chart** of the top-K EfficientNet embedding
  dimensions (default K=20).
- Right: **SHAP beeswarm summary** showing per-sample direction of
  influence for those same dimensions.

The bird-species view (which features push the model toward each class)
is also printed as a per-class top-5 table.

Usage
-----
::

    # Auto-extract test features and explain RF (or knn / xgb)
    python eval/shap_analysis.py \\
        --config configs/fusion.yaml \\
        --baseline rf

    # Re-use cached features (skip the EfficientNet pass)
    python eval/shap_analysis.py \\
        --baseline xgb \\
        --features outputs/baselines/features_val.npz \\
        --baseline-dir outputs/baselines/
"""

from __future__ import annotations

# Direct-script execution
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import logging
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from models.baselines import (  # noqa: E402
    extract_features,
    load_baseline,
    load_cached_features,
)
from models.efficientnet import EfficientNetB3Encoder  # noqa: E402
from pipelines.video_dataset import SPECIES  # noqa: E402

logger = logging.getLogger(__name__)


BaselineName = Literal["knn", "rf", "xgb"]


# ---------------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X: np.ndarray,
    *,
    name: BaselineName,
    background_size: int = 100,
    n_samples_explain: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(shap_values, X_explained)``.

    Shape of ``shap_values``:

    - RF / XGB / KNN multi-class → ``[C, N, D]`` (one matrix per class).

    For KNN we subsample both the background and explained sets to keep
    the kernel explainer tractable (still O(C·N·D)).
    """
    rng = np.random.default_rng(seed)
    if len(X) > n_samples_explain:
        idx = rng.choice(len(X), size=n_samples_explain, replace=False)
        X_explained = X[idx]
    else:
        X_explained = X

    if name in ("rf", "xgb"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explained)
    else:  # knn — Kernel explainer with a small background
        if len(X) > background_size:
            bg_idx = rng.choice(len(X), size=background_size, replace=False)
            background = X[bg_idx]
        else:
            background = X
        explainer = shap.KernelExplainer(model.predict_proba, background)
        sv = explainer.shap_values(X_explained, nsamples=100)

    # Normalise to a [C, N, D] array regardless of SHAP version. The class
    # count C may be < NUM_CLASSES if the training fold didn't contain
    # every species, so detect orientation by matching against the
    # explained-sample count N (which is always the most reliable axis).
    n_expected = X_explained.shape[0]
    if isinstance(sv, list):                       # older API: list of [N, D]
        sv = np.stack(sv, axis=0)
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        # Possible orderings: [C, N, D] (already right) | [N, D, C] | [N, C, D]
        if sv.shape[0] == n_expected and sv.shape[1] != n_expected:
            # [N, D, C] → [C, N, D]
            sv = sv.transpose(2, 0, 1)
        elif sv.shape[1] == n_expected and sv.shape[0] != n_expected:
            # [C, N, D] — already correct
            pass
        elif sv.shape[1] == n_expected and sv.shape[2] != n_expected:
            # [C, N, D] — still correct, no-op
            pass
    return sv, X_explained


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_shap_summary(
    sv: np.ndarray,                # [C, N, D]
    X_explained: np.ndarray,       # [N, D]
    out_path: Path,
    top_k: int = 20,
    title_suffix: str = "",
) -> None:
    """Two-panel: mean |SHAP| bar chart + beeswarm of top-K dims (class-averaged)."""
    # Aggregate over classes → per-feature global importance.
    mean_abs = np.abs(sv).mean(axis=(0, 1))                          # [D]
    top_idx = np.argsort(-mean_abs)[:top_k]
    top_names = [f"emb_{i:04d}" for i in top_idx]

    # Class-averaged SHAP for beeswarm (per-sample × per-feature).
    sv_avg = sv.mean(axis=0)                                          # [N, D]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: mean |SHAP| bar
    axes[0].barh(top_names[::-1], mean_abs[top_idx][::-1],
                 color="#2E86AB", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("mean(|SHAP value|)")
    axes[0].set_title(f"Top-{top_k} EB3 embedding dims by global SHAP importance{title_suffix}")
    axes[0].grid(axis="x", linestyle=":", alpha=0.4)

    # Right: beeswarm (manual because we want shared axes / no shap.plots.* side-effects)
    plt.sca(axes[1])
    try:
        shap.summary_plot(
            sv_avg[:, top_idx],
            features=X_explained[:, top_idx],
            feature_names=top_names,
            plot_type="dot",
            show=False,
        )
    except Exception as exc:
        logger.warning("Beeswarm plot failed (%s) — falling back to bar.", exc)
        shap.summary_plot(
            sv_avg[:, top_idx],
            features=X_explained[:, top_idx],
            feature_names=top_names,
            plot_type="bar",
            show=False,
        )
    axes[1].set_title(f"Per-sample SHAP distribution (class-averaged){title_suffix}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP summary → %s", out_path)


def print_per_class_top_features(
    sv: np.ndarray,                # [C, N, D]
    top_k: int = 5,
) -> None:
    """Print the top-K embedding dims pushing each class up the most."""
    print("\nTop features by class (positive SHAP push):")
    print("=" * 70)
    per_class_mean = sv.mean(axis=1)                                 # [C, D]
    for c in range(per_class_mean.shape[0]):
        order = np.argsort(-per_class_mean[c])[:top_k]
        feats = ", ".join(f"emb_{i:04d} ({per_class_mean[c, i]:+.4f})" for i in order)
        name  = SPECIES[c] if c < len(SPECIES) else f"class_{c}"
        print(f"  {name:30s}  {feats}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis on CHIRP baselines.")
    parser.add_argument("--baseline", required=True, choices=["knn", "rf", "xgb"])
    parser.add_argument("--baseline-dir", default="outputs/baselines")
    parser.add_argument("--features", default=None,
                        help="Path to a cached .npz (X, y). If absent, "
                             "features are extracted via the config + datamodule.")
    parser.add_argument("--config", default=None,
                        help="Required when --features is not supplied.")
    parser.add_argument("--fig-dir", default="outputs/figures")
    parser.add_argument("--top-k",   type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of test samples to explain.")
    parser.add_argument("--background-size", type=int, default=100,
                        help="Background sample size (KNN only).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ---- features ----------------------------------------------------
    if args.features:
        X, _y = load_cached_features(args.features)
    else:
        if not args.config:
            parser.error("Must provide --features or --config")
        from pipelines.augment import build_augment
        from pipelines.datamodule import CHIRPDataModule
        from training.config import TrainConfig

        cfg = TrainConfig.from_yaml(args.config)
        dm = CHIRPDataModule(
            data_root=cfg.data.data_root,
            datasets=tuple(cfg.data.datasets),
            splits=tuple(cfg.data.splits),
            seed=cfg.seed,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            n_frames=cfg.data.num_frames,
            eval_transform=build_augment("val", {"augment": True}),
        )
        dm.setup()
        encoder = EfficientNetB3Encoder(pretrained=cfg.model.pretrained, freeze=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X, _y = extract_features(encoder, dm.val_dataloader(),
                                 device=device, n_keyframes=cfg.model.n_keyframes)

    # ---- model -------------------------------------------------------
    model = load_baseline(args.baseline, args.baseline_dir)
    logger.info("Loaded %s from %s", args.baseline.upper(), args.baseline_dir)

    # ---- SHAP --------------------------------------------------------
    sv, X_explained = compute_shap_values(
        model, X,
        name=args.baseline,
        n_samples_explain=args.n_samples,
        background_size=args.background_size,
    )
    logger.info("SHAP values shape: %s", sv.shape)

    fig_dir = Path(args.fig_dir)

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_shap_summary(
        sv, X_explained,
        out_path=fig_dir / f"shap_{args.baseline}.png",
        top_k=args.top_k,
        title_suffix=f" — {args.baseline.upper()}",
    )
    print_per_class_top_features(sv)


if __name__ == "__main__":
    main()
