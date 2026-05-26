"""
pipelines/optical_flow.py
=========================
Optional optical-flow channel augmentation for CHIRP.

Computes dense optical flow between consecutive frames and concatenates the
2-D ``(u, v)`` displacement field onto each frame, turning the input from
``[T, 3, H, W]`` (RGB) into ``[T, 5, H, W]`` (RGB + flow_x + flow_y).

Two back-ends are supported:

- ``"raft"``      — RAFT-Small from torchvision. Accurate but ~10× slower
                    than Farneback and needs CUDA for real-time work.
- ``"farneback"`` — Classical Gunnar-Farneback dense flow via OpenCV.
                    Pure CPU, no model weights, robust default.

Both are wrapped behind a config flag (``optical_flow.enabled``) so the
whole subsystem can be ablated cleanly::

    cfg:
      optical_flow:
        enabled: true
        backend: farneback       # or "raft"
        raft_model: small        # only used when backend = raft

Padding convention
------------------
Optical flow between *T* frames yields *T - 1* flow fields. To keep the
output time dimension equal to the input, the final flow field is
**duplicated** onto the last frame (so frame *T-1* sees the same flow as
*T-2*). This is the convention used by most Kinetics video pipelines.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

Backend = Literal["raft", "farneback"]


# ---------------------------------------------------------------------------
# Farneback (OpenCV) back-end
# ---------------------------------------------------------------------------

def _farneback_flow(frames_np: np.ndarray) -> np.ndarray:
    """Compute dense flow with Gunnar-Farneback.

    Parameters
    ----------
    frames_np:
        ``[T, H, W, 3]`` uint8 RGB.

    Returns
    -------
    ``[T, H, W, 2]`` float32 — last frame is a duplicate of frame T-2.
    """
    import cv2

    t, h, w, _ = frames_np.shape
    gray = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_np], axis=0
    )

    flows = np.zeros((t, h, w, 2), dtype=np.float32)
    for i in range(t - 1):
        flows[i] = cv2.calcOpticalFlowFarneback(
            gray[i], gray[i + 1],
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
    flows[-1] = flows[-2] if t >= 2 else flows[-1]
    return flows


# ---------------------------------------------------------------------------
# RAFT (torchvision) back-end
# ---------------------------------------------------------------------------

class _RaftFlow:
    """Lazy-loaded RAFT model + per-call inference."""

    def __init__(self, model_size: Literal["small", "large"] = "small",
                 device: str | torch.device = "cpu") -> None:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small,
        )

        if model_size == "small":
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=False)
        else:
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights, progress=False)

        self.transforms = weights.transforms()
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        logger.info("Loaded RAFT-%s on %s", model_size, self.device)

    @torch.no_grad()
    def __call__(self, frames: Tensor) -> Tensor:
        """Return ``[T, 2, H, W]`` flow for a ``[T, 3, H, W]`` clip in [0,1]."""
        t = frames.shape[0]
        if t < 2:
            return torch.zeros(t, 2, *frames.shape[-2:], dtype=frames.dtype)

        # RAFT expects pairs of frames; build (T-1, 3, H, W) batches.
        prev = frames[:-1].to(self.device)
        nxt  = frames[1:].to(self.device)

        # RAFT requires H,W divisible by 8 — pad if needed.
        _, _, h, w = prev.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            prev = torch.nn.functional.pad(prev, (0, pad_w, 0, pad_h))
            nxt  = torch.nn.functional.pad(nxt,  (0, pad_w, 0, pad_h))

        prev_n, nxt_n = self.transforms(prev, nxt)
        flow = self.model(prev_n, nxt_n)[-1]            # final refined flow
        flow = flow[..., :h, :w]                        # un-pad

        # Pad last frame with a copy of the previous flow to keep T frames.
        last = flow[-1:].clone()
        flow = torch.cat([flow, last], dim=0)           # [T, 2, H, W]
        return flow.cpu()


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _tensor_to_np_uint8(frames: Tensor) -> np.ndarray:
    if not frames.is_floating_point():
        raise TypeError(f"Need float frames in [0,1]; got {frames.dtype}")
    arr = (frames.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).contiguous()
    return arr.cpu().numpy()


# ---------------------------------------------------------------------------
# Public callable
# ---------------------------------------------------------------------------

class OpticalFlowChannels:
    """Append optical-flow channels to an RGB clip.

    Parameters
    ----------
    enabled:
        Master toggle. When ``False`` the call is a no-op identity — useful
        for ablation runs that share the same dataset / transform code.
    backend:
        ``"farneback"`` (default, CPU) or ``"raft"`` (more accurate, GPU
        recommended).
    raft_model:
        Either ``"small"`` or ``"large"``. Ignored when backend ≠ raft.
    device:
        Device for RAFT inference. Auto-detected from frame device when
        set to ``"auto"`` (default).
    normalize_flow:
        If ``True`` (default) the flow channels are scaled to roughly
        ``[-1, 1]`` by dividing by ``flow_scale`` (the maximum expected
        per-pixel displacement). Keeps the magnitudes in the same range
        as the ImageNet-normalised RGB channels.
    flow_scale:
        Divisor applied when ``normalize_flow=True``. Default 20 px is a
        reasonable cap for 224×224 bird clips at 25-30 fps.

    Examples
    --------
    >>> flow_op = OpticalFlowChannels(enabled=True, backend="farneback")
    >>> rgb = torch.rand(16, 3, 224, 224)
    >>> rgbf = flow_op(rgb)
    >>> rgbf.shape
    torch.Size([16, 5, 224, 224])

    Off (ablation):

    >>> flow_op = OpticalFlowChannels(enabled=False)
    >>> flow_op(rgb).shape
    torch.Size([16, 3, 224, 224])
    """

    def __init__(
        self,
        enabled: bool = False,
        backend: Backend = "farneback",
        raft_model: Literal["small", "large"] = "small",
        device: str = "auto",
        normalize_flow: bool = True,
        flow_scale: float = 20.0,
    ) -> None:
        if backend not in ("raft", "farneback"):
            raise ValueError(f"Unknown backend {backend!r}")

        self.enabled = enabled
        self.backend = backend
        self.raft_model = raft_model
        self.device = device
        self.normalize_flow = normalize_flow
        self.flow_scale = flow_scale

        self._raft: _RaftFlow | None = None  # lazy init
        if enabled and backend == "raft":
            dev = self._resolve_device(torch.empty(0))
            self._raft = _RaftFlow(model_size=raft_model, device=dev)

        logger.info(
            "OpticalFlowChannels | enabled=%s | backend=%s",
            enabled, backend,
        )

    # ------------------------------------------------------------------

    def _resolve_device(self, ref: Tensor) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if ref.is_cuda:
            return ref.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------

    def __call__(self, frames: Tensor) -> Tensor:
        """Concatenate flow channels: ``[T, 3, H, W]`` → ``[T, 5, H, W]``."""
        if not self.enabled:
            return frames

        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"Expected [T, 3, H, W] tensor, got {tuple(frames.shape)}"
            )

        if self.backend == "farneback":
            np_frames = _tensor_to_np_uint8(frames)        # [T, H, W, 3]
            flow_np = _farneback_flow(np_frames)           # [T, H, W, 2]
            flow = torch.from_numpy(flow_np).permute(0, 3, 1, 2).contiguous()
        else:  # raft
            if self._raft is None:                         # was disabled at init
                self._raft = _RaftFlow(
                    model_size=self.raft_model,
                    device=self._resolve_device(frames),
                )
            flow = self._raft(frames)                      # [T, 2, H, W]

        flow = flow.to(dtype=frames.dtype, device=frames.device)
        if self.normalize_flow:
            flow = flow / self.flow_scale

        return torch.cat([frames, flow], dim=1)            # [T, 5, H, W]

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OpticalFlowChannels(enabled={self.enabled}, "
            f"backend={self.backend!r}, raft_model={self.raft_model!r})"
        )


# ---------------------------------------------------------------------------
# Config-driven factory
# ---------------------------------------------------------------------------

def build_optical_flow(cfg: dict | None) -> OpticalFlowChannels:
    """Construct :class:`OpticalFlowChannels` from a config sub-dict.

    Expected schema::

        optical_flow:
          enabled: true
          backend: farneback        # or "raft"
          raft_model: small
          normalize_flow: true
          flow_scale: 20.0
    """
    cfg = dict(cfg or {})
    return OpticalFlowChannels(
        enabled        = bool(cfg.get("enabled", False)),
        backend        = cfg.get("backend", "farneback"),
        raft_model     = cfg.get("raft_model", "small"),
        device         = cfg.get("device", "auto"),
        normalize_flow = bool(cfg.get("normalize_flow", True)),
        flow_scale     = float(cfg.get("flow_scale", 20.0)),
    )


# ---------------------------------------------------------------------------
# CLI smoke-test (Farneback only — RAFT pulls ~20 MB weights)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    fake = torch.rand(8, 3, 64, 64)
    print(f"Input  : {tuple(fake.shape)}")

    # 1) Disabled — identity
    off = build_optical_flow({"enabled": False})
    out = off(fake)
    assert out.shape == fake.shape, f"Disabled path changed shape: {out.shape}"
    assert torch.equal(out, fake), "Disabled path is not identity"
    print(f"  disabled       → {tuple(out.shape)}  (identity ✓)")

    # 2) Farneback
    fb = build_optical_flow({"enabled": True, "backend": "farneback"})
    out = fb(fake)
    assert out.shape == (8, 5, 64, 64), f"Bad shape: {out.shape}"
    print(f"  farneback (on) → {tuple(out.shape)}  | flow stats: "
          f"mean={out[:, 3:].mean().item():+.3f}, "
          f"std={out[:, 3:].std().item():.3f}")

    # 3) RAFT — only if weights already cached, otherwise skip the download
    import os
    cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    if os.path.isdir(cache) and any("raft" in f.lower() for f in os.listdir(cache)):
        raft = build_optical_flow({"enabled": True, "backend": "raft"})
        out = raft(fake)
        assert out.shape == (8, 5, 64, 64)
        print(f"  raft (cached)  → {tuple(out.shape)}")
    else:
        print("  raft           → skipped (no cached weights; "
              "set backend=raft in config to download on first use)")

    print("\nOK")
