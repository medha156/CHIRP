"""Unit tests for pipelines.augment — Albumentations video pipeline."""

from __future__ import annotations

import pytest
import torch

from pipelines.augment import VideoAugment, build_augment


@pytest.fixture
def clip():
    """16-frame 3×64×64 RGB clip in [0, 1]."""
    torch.manual_seed(0)
    return torch.rand(16, 3, 64, 64)


def test_train_pipeline_returns_expected_shape(clip):
    aug = build_augment("train", {"augment": True})
    out = aug(clip)
    assert out.shape == (16, 3, 224, 224)


def test_eval_pipeline_returns_expected_shape(clip):
    aug = build_augment("val", {"augment": True})
    out = aug(clip)
    assert out.shape == (16, 3, 224, 224)


def test_eval_pipeline_is_deterministic(clip):
    """val / test apply only resize + centre crop → bit-identical reruns."""
    aug = build_augment("val", {"augment": True})
    assert torch.equal(aug(clip), aug(clip))


def test_train_pipeline_is_stochastic(clip):
    aug = build_augment("train", {"augment": True})
    diff = (aug(clip) - aug(clip)).abs().mean().item()
    assert diff > 0.0


def test_augment_off_collapses_train_to_eval(clip):
    """augment=False makes the train pipeline deterministic (no augmentation)."""
    aug = build_augment("train", {"augment": False})
    assert torch.equal(aug(clip), aug(clip))


def test_output_dtype_and_range(clip):
    aug = build_augment("train", {"augment": True})
    out = aug(clip)
    assert out.dtype == torch.float32
    assert 0.0 <= out.min().item() and out.max().item() <= 1.0


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        VideoAugment(split="bogus")  # type: ignore[arg-type]
