"""Unit tests for the CHIRPVideoDataset helpers.

The full ``__getitem__`` path requires real video files, which we don't
ship in CI. Instead we monkey-patch the decode + frame-count helpers to
return synthetic tensors, exercising the dataset's CSV validation,
path resolution, sampling, and transform-application logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pipelines import video_dataset as vd

# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

def test_sample_indices_uniform_no_jitter():
    idx = vd._sample_indices(total_frames=30, n_frames=10, jitter=False)
    assert idx.shape == (10,)
    assert idx.min() >= 0 and idx.max() < 30
    # Without jitter, indices should be strictly increasing.
    assert (np.diff(idx) >= 0).all()


def test_sample_indices_padding_for_short_clips():
    """When total < n_frames, indices wrap (repeat-pad)."""
    idx = vd._sample_indices(total_frames=3, n_frames=8, jitter=False)
    assert idx.shape == (8,)
    assert set(idx.tolist()).issubset({0, 1, 2})


def test_sample_indices_jitter_random_within_segment():
    """With jitter on, two calls should produce different indices."""
    np.random.seed(0)
    a = vd._sample_indices(30, 10, jitter=True)
    b = vd._sample_indices(30, 10, jitter=True)
    assert not np.array_equal(a, b)


def test_sample_indices_rejects_zero_frames():
    with pytest.raises(ValueError):
        vd._sample_indices(0, 8, jitter=False)


# ---------------------------------------------------------------------------
# Dataset with monkey-patched decoder
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_path(tmp_path):
    df = pd.DataFrame({
        "path":    [f"clip_{i}.mp4" for i in range(8)],
        "label":   list(range(8)),
        "species": [vd.SPECIES[i] for i in range(8)],
    })
    p = tmp_path / "split.csv"
    df.to_csv(p, index=False)
    return p


def test_dataset_validates_csv_columns(tmp_path):
    bad = pd.DataFrame({"path": ["a.mp4"], "label_typo": [0]})
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        vd.CHIRPVideoDataset(p)


def test_dataset_rejects_out_of_range_labels(tmp_path):
    bad = pd.DataFrame({"path": ["a.mp4"], "label": [99]})
    p = tmp_path / "bad_label.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="outside"):
        vd.CHIRPVideoDataset(p)


def test_dataset_getitem_returns_expected_dict(csv_path, monkeypatch):
    """Stub decoder so we don't need actual video files."""
    def fake_count(path, backend):
        return 30
    def fake_decode(path, indices, height, width, backend="auto"):
        return torch.rand(len(indices), 3, height, width)
    monkeypatch.setattr(vd, "_count_frames", fake_count)
    monkeypatch.setattr(vd, "decode_clip", fake_decode)

    ds = vd.CHIRPVideoDataset(csv_path, n_frames=8, split="val")
    sample = ds[0]
    assert set(sample) == {"frames", "label", "path"}
    assert sample["frames"].shape == (8, 3, 224, 224)
    assert sample["frames"].dtype == torch.float32
    assert isinstance(sample["label"], int)
    assert isinstance(sample["path"], str)


def test_dataset_class_counts(csv_path, monkeypatch):
    monkeypatch.setattr(vd, "_count_frames", lambda *a, **k: 30)
    monkeypatch.setattr(vd, "decode_clip",
                        lambda *a, **k: torch.zeros(8, 3, 224, 224))
    ds = vd.CHIRPVideoDataset(csv_path, n_frames=8, split="val")
    counts = ds.class_counts()
    # First 8 classes have 1 clip each, rest are 0.
    assert sum(counts.values()) == 8
    assert all(counts[vd.SPECIES[i]] == 1 for i in range(8))
    assert all(counts[vd.SPECIES[i]] == 0 for i in range(8, 20))


def test_dataset_warns_on_corrupted_file(csv_path, monkeypatch, caplog):
    """Decoding error → returns zeros rather than crashing the loader."""
    monkeypatch.setattr(vd, "_count_frames", lambda *a, **k: 30)
    def boom(*a, **k):
        raise RuntimeError("simulated decode failure")
    monkeypatch.setattr(vd, "decode_clip", boom)

    ds = vd.CHIRPVideoDataset(csv_path, n_frames=8, split="val")
    sample = ds[0]
    assert sample["frames"].shape == (8, 3, 224, 224)
    assert torch.equal(sample["frames"], torch.zeros_like(sample["frames"]))
