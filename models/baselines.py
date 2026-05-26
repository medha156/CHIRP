"""
models/baselines.py
===================
Classical-ML baselines on frozen EfficientNet-B3 embeddings.

What it does
------------
1. Iterates a CHIRP ``DataLoader`` once with a **frozen**
   :class:`models.efficientnet.EfficientNetB3Encoder`, caching the
   ``[N, 1536]`` feature matrix and ``[N]`` labels to disk.
2. Fits three classifiers on those features:
   - **KNN** (``sklearn.neighbors.KNeighborsClassifier``, k=5, cosine).
   - **Random Forest** (``sklearn.ensemble.RandomForestClassifier``,
     500 trees, balanced class weights).
   - **XGBoost** (``xgboost.XGBClassifier``, 400 trees, depth 6,
     histogram tree method).
3. Reports macro-F1 + per-class accuracy on a held-out validation matrix
   and saves each trained model (``joblib`` for sklearn, native JSON for
   XGBoost) under ``outputs/baselines/``.

Why
---
- **Sanity check.** If a 500-tree RF beats your fancy ensemble, something
  is wrong with the deep model — useful early-warning signal.
- **Inference fallback.** Swap the MLP fusion head for one of these
  trained baselines via :class:`BaselineHeadAdapter`, so you can run the
  full ensemble even without GPU-trained fusion weights.

CLI
---
::

    python models/baselines.py \\
        --train-csv data/splits/train.csv \\
        --val-csv   data/splits/val.csv \\
        --out-dir   outputs/baselines/
"""

from __future__ import annotations

# Allow direct script execution (`python models/baselines.py`).
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.efficientnet import EfficientNetB3Encoder  # noqa: E402
from pipelines.video_dataset import NUM_CLASSES, SPECIES  # noqa: E402

logger = logging.getLogger(__name__)

BaselineName = Literal["knn", "rf", "xgb"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    encoder: EfficientNetB3Encoder,
    loader: DataLoader,
    *,
    device: str | torch.device = "cpu",
    n_keyframes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``encoder`` over ``loader`` and return ``(X, y)`` arrays.

    The encoder is moved to ``device`` and set to ``eval()``. Each batch
    yields ``frames [B, T, 3, H, W]``; uniformly sub-samples ``n_keyframes``
    frames per clip, then takes the temporal mean.
    """
    encoder = encoder.to(device).eval()

    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    pbar = tqdm(loader, desc="extract", leave=False)
    for batch in pbar:
        frames = batch["frames"].to(device, non_blocking=True)   # [B, T, 3, H, W]
        b, t = frames.shape[:2]

        # Uniformly choose K keyframes per clip.
        idx = torch.linspace(0, t - 1, steps=n_keyframes, device=device).round().long()
        keyf = frames[:, idx]                                    # [B, K, 3, H, W]

        emb = encoder(keyf)                                      # [B, 1536]
        feats.append(emb.cpu().numpy())
        labels.append(batch["label"].numpy() if torch.is_tensor(batch["label"])
                      else np.asarray(batch["label"]))

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    logger.info("Extracted features: X=%s, y=%s", X.shape, y.shape)
    return X, y


def cache_features(
    X: np.ndarray, y: np.ndarray, path: str | Path,
) -> Path:
    """Save feature matrix + labels to a single ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    logger.info("Cached features → %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_cached_features(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["y"]


# ---------------------------------------------------------------------------
# Baseline fitting
# ---------------------------------------------------------------------------

def _fit_knn(X: np.ndarray, y: np.ndarray, k: int = 5) -> KNeighborsClassifier:
    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=-1)
    clf.fit(X, y)
    return clf


def _fit_rf(X: np.ndarray, y: np.ndarray, seed: int = 42) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X, y)
    return clf


def _fit_xgb(X: np.ndarray, y: np.ndarray, seed: int = 42):
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        eval_metric="mlogloss",
    )
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_baseline(name: BaselineName, model, out_dir: Path) -> Path:
    """Persist a fitted classifier. We use joblib for all three because
    XGBoost's native ``save_model`` requires extra estimator-type metadata
    that the sklearn wrapper doesn't always set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info("Saved %s → %s", name, path)
    return path


def load_baseline(name: BaselineName, out_dir: str | Path):
    """Inverse of ``_save_baseline``."""
    out_dir = Path(out_dir)
    return joblib.load(out_dir / f"{name}.joblib")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    *,
    out_dir: str | Path = "outputs/baselines",
    which: tuple[BaselineName, ...] = ("knn", "rf", "xgb"),
    seed: int = 42,
) -> dict:
    """Fit, evaluate, and persist the selected baselines.

    Returns a metrics dict::

        {
          "knn": {"accuracy": …, "macro_f1": …, "report": "..."},
          "rf":  {...},
          "xgb": {...},
        }
    """
    out_dir = Path(out_dir)
    fitters = {"knn": _fit_knn, "rf": _fit_rf, "xgb": _fit_xgb}
    results: dict = {}

    for name in which:
        if name not in fitters:
            raise ValueError(f"Unknown baseline {name!r}; choose from {list(fitters)}")
        logger.info("Fitting %s on %d samples…", name.upper(), len(X_train))
        model = fitters[name](X_train, y_train, seed=seed) if name != "knn" \
                else _fit_knn(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1  = f1_score(y_val, preds, average="macro", zero_division=0)

        present_classes = sorted(set(y_val.tolist()) | set(preds.tolist()))
        target_names = [SPECIES[i] for i in present_classes]
        report = classification_report(
            y_val, preds,
            labels=present_classes, target_names=target_names,
            zero_division=0,
        )

        results[name] = {"accuracy": float(acc), "macro_f1": float(f1), "report": report}
        _save_baseline(name, model, out_dir)
        logger.info("%s  acc=%.4f  macro-F1=%.4f", name.upper(), acc, f1)

    # Persist a summary JSON next to the model files.
    summary_path = out_dir / "metrics.json"
    summary_path.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "report"}
         for k, v in results.items()},
        indent=2,
    ))
    logger.info("Saved metrics summary → %s", summary_path)
    return results


# ---------------------------------------------------------------------------
# Inference adapter — lets a sklearn/XGBoost classifier impersonate
# the FusionHead so the rest of the pipeline can stay unchanged.
# ---------------------------------------------------------------------------

class BaselineHeadAdapter(torch.nn.Module):
    """Use a fitted ``rf`` / ``xgb`` / ``knn`` model as the ensemble head.

    Wraps a sklearn-compatible estimator so it behaves like
    :class:`models.fusion.FusionHead`: takes the concatenated
    Swin+EB3 embedding ``[B, 2304]`` (or the EB3 embedding alone if
    ``use_efficientnet_only=True``) and returns class logits ``[B, 20]``.

    Notes
    -----
    - Classical models output probabilities; we convert with
      ``log(p + ε)`` so the result is consumable by ``nn.CrossEntropyLoss``
      or any code that expects logits.
    - The wrapped model is moved to **CPU** for prediction; gradients are
      not flowed through.
    """

    def __init__(
        self,
        sklearn_model,
        use_efficientnet_only: bool = True,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.model = sklearn_model
        self.use_efficientnet_only = use_efficientnet_only
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if len(args) == 2:
            swin, eff = args
            x = eff if self.use_efficientnet_only \
                else torch.cat([swin, eff], dim=-1)
        elif len(args) == 1:
            x = args[0]
        else:
            raise TypeError(f"Expected 1 or 2 tensors; got {len(args)}")

        np_x = x.detach().cpu().numpy()
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(np_x)
        else:
            preds = self.model.predict(np_x)
            probs = np.eye(self.num_classes)[preds]

        # Some classifiers may drop classes that never appeared in training
        # → pad probability matrix to width = num_classes.
        if probs.shape[1] != self.num_classes:
            full = np.zeros((probs.shape[0], self.num_classes), dtype=probs.dtype)
            full[:, self.model.classes_] = probs
            probs = full

        logits = np.log(probs + 1e-9)
        return torch.from_numpy(logits).to(x.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description="Train classical baselines on EB3 features.")
    parser.add_argument("--train-csv", required=False,
                        help="CHIRP split CSV (path,label,species). Required unless --features-only is given.")
    parser.add_argument("--val-csv", required=False,
                        help="Validation split CSV.")
    parser.add_argument("--out-dir", default="outputs/baselines",
                        help="Where to save trained models + metrics.")
    parser.add_argument("--n-keyframes", type=int, default=4)
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cached-train", help="Path to a pre-computed train .npz (X, y).")
    parser.add_argument("--cached-val",   help="Path to a pre-computed val   .npz (X, y).")
    parser.add_argument("--which", nargs="+", default=["knn", "rf", "xgb"],
                        choices=["knn", "rf", "xgb"])
    parser.add_argument("--smoke", action="store_true",
                        help="Run on synthetic features — verifies the fit/save/load loop.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ---- synthetic smoke-test ----
    if args.smoke:
        rng = np.random.default_rng(0)
        X_tr = rng.standard_normal((400, 1536)).astype(np.float32)
        y_tr = rng.integers(0, NUM_CLASSES, size=400)
        X_va = rng.standard_normal((100, 1536)).astype(np.float32)
        y_va = rng.integers(0, NUM_CLASSES, size=100)
    else:
        # ---- real-data path: load cached or extract from videos ----
        if args.cached_train and args.cached_val:
            X_tr, y_tr = load_cached_features(args.cached_train)
            X_va, y_va = load_cached_features(args.cached_val)
        else:
            if not (args.train_csv and args.val_csv):
                parser.error("Must pass --train-csv and --val-csv (or --cached-* / --smoke).")
            from pipelines.video_dataset import build_dataloader  # local import

            encoder = EfficientNetB3Encoder(pretrained=True, freeze=True)
            tr_loader = build_dataloader(
                args.train_csv, split="train",
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            va_loader = build_dataloader(
                args.val_csv, split="val",
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            X_tr, y_tr = extract_features(encoder, tr_loader,
                                          device=args.device,
                                          n_keyframes=args.n_keyframes)
            X_va, y_va = extract_features(encoder, va_loader,
                                          device=args.device,
                                          n_keyframes=args.n_keyframes)
            out = Path(args.out_dir)
            cache_features(X_tr, y_tr, out / "features_train.npz")
            cache_features(X_va, y_va, out / "features_val.npz")

    results = train_baselines(
        X_tr, y_tr, X_va, y_va,
        out_dir=args.out_dir,
        which=tuple(args.which),
    )

    print("\n" + "=" * 70)
    for name, m in results.items():
        print(f"\n{name.upper()}  accuracy={m['accuracy']:.4f}  "
              f"macro-F1={m['macro_f1']:.4f}")
        print(m["report"])


if __name__ == "__main__":
    _main()
