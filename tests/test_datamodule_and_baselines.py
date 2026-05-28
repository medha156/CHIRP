"""Unit tests for datamodule splits / class weights and classical baselines.

These avoid any video decoding by either:
  - using synthetic CSV index files (datamodule split + class-weight logic
    only — no __getitem__ calls), or
  - operating directly on synthetic feature matrices (baselines).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.baselines import (
    BaselineHeadAdapter,
    _save_baseline,
    load_baseline,
    train_baselines,
)
from pipelines.datamodule import CHIRPDataModule
from pipelines.video_dataset import NUM_CLASSES, SPECIES


def _make_synthetic_index(root: Path, name: str, n_clips: int, seed: int = 0) -> None:
    """Write a synthetic <root>/<name>/index.csv with random labels."""
    rng = np.random.default_rng(seed)
    (root / name).mkdir(parents=True)
    labels = rng.integers(0, NUM_CLASSES, size=n_clips)
    df = pd.DataFrame({
        "path":    [f"clip_{i:04d}.mp4" for i in range(n_clips)],
        "label":   labels,
        "species": [SPECIES[i] for i in labels],
    })
    df.to_csv(root / name / "index.csv", index=False)


# ---------------------------------------------------------------------------
# Datamodule
# ---------------------------------------------------------------------------

def test_datamodule_setup_splits_stratified(tmp_path):
    _make_synthetic_index(tmp_path, "fbd_sv_2024", 400, seed=0)
    _make_synthetic_index(tmp_path, "vb100",       250, seed=1)

    dm = CHIRPDataModule(
        data_root=tmp_path,
        splits=(0.70, 0.15, 0.15),
        batch_size=4, num_workers=0,
    )
    dm.setup()

    n_total = 650
    assert len(dm.train_df) + len(dm.val_df) + len(dm.test_df) == n_total

    # Stratification: per-class share in train should be close to full share.
    full = dm.df["label"].value_counts(normalize=True).sort_index()
    train = dm.train_df["label"].value_counts(normalize=True).sort_index()
    assert (full - train).abs().max() < 0.05


def test_datamodule_class_weights_shape_and_range(tmp_path):
    _make_synthetic_index(tmp_path, "fbd_sv_2024", 400)
    _make_synthetic_index(tmp_path, "vb100",       250)
    dm = CHIRPDataModule(data_root=tmp_path, batch_size=4, num_workers=0)
    dm.setup()
    w = dm.class_weights
    assert w.shape == (NUM_CLASSES,)
    assert w.dtype == torch.float32
    assert (w > 0).all()


def test_datamodule_plot_class_distribution_writes_png(tmp_path):
    _make_synthetic_index(tmp_path, "fbd_sv_2024", 200)
    _make_synthetic_index(tmp_path, "vb100",       100)
    dm = CHIRPDataModule(data_root=tmp_path, batch_size=4, num_workers=0)
    dm.setup()
    out = tmp_path / "dist.png"
    dm.plot_class_distribution(out)
    assert out.exists() and out.stat().st_size > 0


def test_datamodule_rejects_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        CHIRPDataModule(data_root=tmp_path, datasets=("not_a_real_dataset",))


def test_datamodule_rejects_bad_splits(tmp_path):
    with pytest.raises(ValueError, match="splits must sum"):
        CHIRPDataModule(data_root=tmp_path, splits=(0.5, 0.3, 0.3))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((200, 1536)).astype(np.float32)
    y_tr = rng.integers(0, NUM_CLASSES, size=200)
    X_va = rng.standard_normal((50, 1536)).astype(np.float32)
    y_va = rng.integers(0, NUM_CLASSES, size=50)
    return X_tr, y_tr, X_va, y_va


def test_train_baselines_fits_all_three(synthetic_features, tmp_path):
    X_tr, y_tr, X_va, y_va = synthetic_features
    results = train_baselines(
        X_tr, y_tr, X_va, y_va,
        out_dir=tmp_path,
        which=("knn", "rf", "xgb"),
    )
    assert set(results) == {"knn", "rf", "xgb"}
    for name, m in results.items():
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0
        assert (tmp_path / f"{name}.joblib").exists()
    assert (tmp_path / "metrics.json").exists()


def test_load_baseline_roundtrip(synthetic_features, tmp_path):
    X_tr, y_tr, _, _ = synthetic_features
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_tr, y_tr)
    _save_baseline("rf", rf, tmp_path)
    loaded = load_baseline("rf", tmp_path)
    assert (loaded.predict(X_tr[:5]) == rf.predict(X_tr[:5])).all()


def test_baseline_adapter_returns_logits(synthetic_features, tmp_path):
    X_tr, y_tr, _, _ = synthetic_features
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_tr, y_tr)
    adapter = BaselineHeadAdapter(rf, use_efficientnet_only=True)
    eff = torch.randn(3, 1536)
    swin = torch.randn(3, 768)
    logits = adapter(swin, eff)
    assert logits.shape == (3, NUM_CLASSES)
    assert logits.dtype == torch.float32


def test_baseline_adapter_handles_missing_classes(tmp_path):
    """Adapter must pad output to NUM_CLASSES even when classifier saw fewer."""
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.RandomState(0).randn(60, 1536).astype(np.float32)
    y = np.random.RandomState(0).randint(0, 5, size=60)        # only 5 classes
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    adapter = BaselineHeadAdapter(rf, use_efficientnet_only=True)
    logits = adapter(torch.randn(3, 768), torch.randn(3, 1536))
    assert logits.shape == (3, NUM_CLASSES)
