"""Unit tests for pipelines.preprocess — no model weights, CPU-only."""

from __future__ import annotations

import pytest
import torch

from pipelines.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Preprocessor,
    imagenet_normalize,
    resize_preserve_aspect,
    to_efficientnet_layout,
    to_swin_layout,
)


def test_resize_preserves_aspect_and_crops_centre():
    frames = torch.rand(2, 4, 3, 360, 640)            # batched, non-square
    out = resize_preserve_aspect(frames, size=224)
    assert out.shape == (2, 4, 3, 224, 224)


def test_resize_passes_through_when_already_correct():
    frames = torch.rand(4, 3, 224, 224)
    out = resize_preserve_aspect(frames, size=224)
    assert torch.equal(out, frames)                   # no-op fast path


def test_imagenet_normalize_rgb_only():
    x = torch.rand(2, 4, 3, 8, 8)
    out = imagenet_normalize(x)
    assert out.shape == x.shape
    # Mean over a uniformly random tensor after normalisation should drift away from 0.5
    assert out.mean().item() != pytest.approx(x.mean().item(), abs=1e-4)


def test_imagenet_normalize_passes_through_extra_channels():
    """5-channel input (RGB+flow): first 3 normalised, last 2 untouched."""
    x = torch.rand(4, 5, 16, 16)
    out = imagenet_normalize(x)
    assert out.shape == x.shape
    # Last 2 channels (flow) must be byte-equal to input.
    assert torch.equal(out[..., 3:, :, :], x[..., 3:, :, :])
    # First 3 must NOT be equal (they got normalised).
    assert not torch.equal(out[..., :3, :, :], x[..., :3, :, :])


def test_imagenet_normalize_rejects_too_few_channels():
    with pytest.raises(ValueError, match="at least 3 channels"):
        imagenet_normalize(torch.rand(2, 1, 8, 8))


def test_imagenet_normalize_rejects_non_float():
    with pytest.raises(TypeError):
        imagenet_normalize(torch.randint(0, 255, (3, 8, 8), dtype=torch.uint8))


def test_to_swin_layout_shapes():
    x = torch.rand(2, 16, 3, 224, 224)                # [B,T,C,H,W]
    out = to_swin_layout(x)
    assert out.shape == (2, 3, 16, 224, 224)


def test_to_efficientnet_layout_with_keyframes():
    x = torch.rand(2, 16, 3, 224, 224)
    out = to_efficientnet_layout(x, n_keyframes=4)
    assert out.shape == (2 * 4, 3, 224, 224)


def test_to_efficientnet_layout_all_frames():
    x = torch.rand(2, 16, 3, 224, 224)
    out = to_efficientnet_layout(x, n_keyframes=None)
    assert out.shape == (2 * 16, 3, 224, 224)


def test_preprocessor_for_swin_handles_single_clip():
    """[T, C, H, W] auto-batched to [1, ...]."""
    prep = Preprocessor()
    single = torch.rand(16, 3, 360, 640)
    out = prep.for_swin(single)
    assert out.shape == (1, 3, 16, 224, 224)


def test_preprocessor_for_swin_handles_batched():
    prep = Preprocessor()
    batched = torch.rand(2, 16, 3, 360, 640)
    out = prep.for_swin(batched)
    assert out.shape == (2, 3, 16, 224, 224)


def test_preprocessor_dual_returns_both():
    prep = Preprocessor()
    frames = torch.rand(2, 16, 3, 240, 320)
    both = prep(frames, n_keyframes=4)
    assert set(both) == {"swin", "efficientnet"}
    assert both["swin"].shape == (2, 3, 16, 224, 224)
    assert both["efficientnet"].shape == (2 * 4, 3, 224, 224)


def test_imagenet_constants_match_canonical_values():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)
