"""Unit tests for pipelines.optical_flow — Farneback backend only on CI."""

from __future__ import annotations

import pytest
import torch

from pipelines.optical_flow import OpticalFlowChannels, build_optical_flow


@pytest.fixture
def clip():
    torch.manual_seed(0)
    return torch.rand(8, 3, 64, 64)


def test_disabled_is_identity(clip):
    op = build_optical_flow({"enabled": False})
    out = op(clip)
    assert torch.equal(out, clip)
    assert out.shape == clip.shape


def test_farneback_appends_two_channels(clip):
    op = build_optical_flow({"enabled": True, "backend": "farneback"})
    out = op(clip)
    assert out.shape == (8, 5, 64, 64)
    # First 3 channels = original RGB unchanged.
    assert torch.allclose(out[:, :3], clip)


def test_farneback_flow_normalised_to_small_range(clip):
    """flow_scale=20 keeps flow channel magnitudes near ImageNet-normalised RGB."""
    op = build_optical_flow({"enabled": True, "backend": "farneback", "flow_scale": 20})
    out = op(clip)
    flow = out[:, 3:]
    assert flow.abs().max().item() < 5.0    # generous bound; random clip → tiny flow


def test_rejects_non_rgb_input():
    op = build_optical_flow({"enabled": True, "backend": "farneback"})
    bad = torch.rand(8, 1, 64, 64)
    with pytest.raises(ValueError):
        op(bad)


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        OpticalFlowChannels(enabled=True, backend="bogus")  # type: ignore[arg-type]
