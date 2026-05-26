"""
pipelines/preprocess.py
=======================
Pre-processing pipeline for CHIRP.

Stages
------
1. **Aspect-preserving resize → 224×224**
   Resize the shorter side to 224 with bilinear interpolation, then take a
   centre crop of size 224×224. This matches the standard pre-processing
   used by both Video Swin-T (Kinetics-400 pretrain) and EfficientNet-B3
   (ImageNet pretrain), and avoids the geometric distortion of a plain
   ``F.interpolate`` to 224×224.

2. **ImageNet normalisation**
   ``(x − mean) / std`` with the canonical statistics
   ``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``.

3. **Layout reshape**
   Reorder dimensions for the consuming model:

   - ``for_swin`` → ``[B, C, T, H, W]``  (channel-first, time-after-channel)
   - ``for_efficientnet`` → ``[B·T, C, H, W]``  (frame-flat, optionally
     sub-sampled to ``n_keyframes``)

Input conventions
-----------------
All inputs are float tensors with values in ``[0, 1]`` and shape

- ``[T, C, H, W]`` — single clip (e.g. straight from ``CHIRPVideoDataset``)
- ``[B, T, C, H, W]`` — collated batch from a ``DataLoader``

Channels are RGB. Functions accept both layouts transparently.

Usage
-----
>>> prep = Preprocessor(size=224)
>>> swin_in = prep.for_swin(batch_frames)            # [B, C, T, H, W]
>>> eff_in  = prep.for_efficientnet(batch_frames, n_keyframes=4)
>>> both    = prep(batch_frames)                     # dict for ensemble
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_SIZE: int = 224

# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _ensure_batched(frames: Tensor) -> tuple[Tensor, bool]:
    """Add a batch dim if missing.

    Returns the (possibly-unsqueezed) tensor plus a flag indicating whether
    we added the dim, so callers can squeeze it back at the end.
    """
    if frames.ndim == 4:                       # [T, C, H, W]
        return frames.unsqueeze(0), True       # → [1, T, C, H, W]
    if frames.ndim == 5:                       # [B, T, C, H, W]
        return frames, False
    raise ValueError(
        f"Expected 4D [T,C,H,W] or 5D [B,T,C,H,W] tensor, got shape {tuple(frames.shape)}"
    )


def _validate_dtype(frames: Tensor) -> None:
    if not frames.is_floating_point():
        raise TypeError(
            f"Frames must be a float tensor in [0,1], got dtype={frames.dtype}. "
            "Cast with `.float() / 255.0` first."
        )


# ---------------------------------------------------------------------------
# Stage 1 — aspect-preserving resize + centre crop
# ---------------------------------------------------------------------------

def resize_preserve_aspect(
    frames: Tensor,
    size: int = DEFAULT_SIZE,
    interpolation: str = "bilinear",
) -> Tensor:
    """Resize the shorter side to ``size`` then centre-crop to ``size×size``.

    Works on any of the following layouts (last two dims are H, W):

    - ``[T, C, H, W]``
    - ``[B, T, C, H, W]``
    - ``[N, C, H, W]``

    Bilinear interpolation is used by default; pass ``"bicubic"`` for higher
    fidelity at extra cost.
    """
    if frames.ndim < 3:
        raise ValueError(f"Need at least 3 dims, got shape {tuple(frames.shape)}")

    *lead, h, w = frames.shape
    if h == size and w == size:
        return frames                          # already correct, skip work

    # ---- compute new shape preserving aspect ratio -------------------
    scale = size / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    # ---- flatten leading dims so interpolate sees a 4-D tensor -------
    # Channels must live just before H, W. The supported layouts above all
    # satisfy this; we just need to merge everything before [C, H, W].
    if frames.ndim >= 4:
        # last three dims are [C, H, W]; collapse everything before that
        c = frames.shape[-3]
        flat = frames.reshape(-1, c, h, w)
    else:
        flat = frames.unsqueeze(0)             # [1, C, H, W]

    resized = F.interpolate(
        flat,
        size=(new_h, new_w),
        mode=interpolation,
        align_corners=False if interpolation in ("bilinear", "bicubic") else None,
        antialias=True if interpolation in ("bilinear", "bicubic") else False,
    )

    # ---- centre crop --------------------------------------------------
    top  = (new_h - size) // 2
    left = (new_w - size) // 2
    cropped = resized[..., top:top + size, left:left + size]

    # ---- restore original leading dims -------------------------------
    return cropped.reshape(*lead, size, size)


# ---------------------------------------------------------------------------
# Stage 2 — ImageNet normalisation
# ---------------------------------------------------------------------------

def imagenet_normalize(
    frames: Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Tensor:
    """Apply ``(x − mean) / std`` to the first ``len(mean)`` channels.

    Broadcasts over any layout that ends in ``[..., C, H, W]``.

    When the input has **more** channels than ``mean`` (e.g. RGB + 2
    optical-flow channels → 5 channels), only the leading ``len(mean)``
    are normalised; the trailing channels pass through untouched. This
    lets the optical-flow concatenation happen *before* the colour
    normalisation step without breaking the pipeline.
    """
    _validate_dtype(frames)

    c_total = frames.shape[-3]
    c_norm  = len(mean)
    if c_total < c_norm:
        raise ValueError(
            f"Need at least {c_norm} channels (RGB), got {c_total} "
            f"from shape {tuple(frames.shape)}"
        )

    m = torch.as_tensor(mean, dtype=frames.dtype, device=frames.device)
    s = torch.as_tensor(std,  dtype=frames.dtype, device=frames.device)
    view = (1,) * (frames.ndim - 3) + (c_norm, 1, 1)

    if c_total == c_norm:
        # Fast path — exact match, no slicing.
        return (frames - m.view(*view)) / s.view(*view)

    rgb   = frames[..., :c_norm, :, :]
    extra = frames[..., c_norm:, :, :]
    rgb   = (rgb - m.view(*view)) / s.view(*view)
    return torch.cat([rgb, extra], dim=-3)


# ---------------------------------------------------------------------------
# Stage 3 — layout reshape
# ---------------------------------------------------------------------------

def to_swin_layout(frames: Tensor) -> Tensor:
    """Convert ``[B, T, C, H, W]`` → ``[B, C, T, H, W]`` (Video Swin-T input)."""
    if frames.ndim != 5:
        raise ValueError(
            f"to_swin_layout expects 5-D [B,T,C,H,W], got shape {tuple(frames.shape)}"
        )
    return frames.permute(0, 2, 1, 3, 4).contiguous()


def to_efficientnet_layout(
    frames: Tensor,
    n_keyframes: int | None = None,
) -> Tensor:
    """Convert ``[B, T, C, H, W]`` → ``[B·K, C, H, W]`` for EfficientNet-B3.

    Parameters
    ----------
    frames:
        Batched clip tensor.
    n_keyframes:
        If given, uniformly sub-samples ``K = n_keyframes`` frames per clip
        (this matches the CLAUDE.md spec: *EB3 applied to 4 uniformly
        sampled keyframes, pooled*). When ``None`` all ``T`` frames are
        forwarded — the caller is responsible for any temporal pooling.
    """
    if frames.ndim != 5:
        raise ValueError(
            f"to_efficientnet_layout expects 5-D [B,T,C,H,W], "
            f"got shape {tuple(frames.shape)}"
        )

    b, t, c, h, w = frames.shape

    if n_keyframes is not None:
        if not 1 <= n_keyframes <= t:
            raise ValueError(
                f"n_keyframes must be in [1, T={t}], got {n_keyframes}"
            )
        # Uniform keyframe indices: midpoints of K equal segments.
        idx = torch.linspace(0, t - 1, steps=n_keyframes, device=frames.device)
        idx = idx.round().long()
        frames = frames[:, idx, :, :, :]       # [B, K, C, H, W]

    return frames.reshape(-1, c, h, w).contiguous()


# ---------------------------------------------------------------------------
# Preprocessor — single object that bundles all three stages
# ---------------------------------------------------------------------------

class Preprocessor:
    """Bundled resize → normalise → reshape pipeline.

    Parameters
    ----------
    size:
        Spatial side length after centre-crop. Default 224.
    mean, std:
        ImageNet normalisation statistics. Override if you fine-tune on
        a domain with different statistics.
    interpolation:
        Resize mode passed to ``torch.nn.functional.interpolate``.
        ``"bilinear"`` (default) is what both Video Swin-T and EB3 expect.

    Examples
    --------
    >>> prep = Preprocessor()
    >>> frames = torch.rand(16, 3, 360, 640)         # [T, C, H, W]
    >>> swin_in = prep.for_swin(frames)
    >>> swin_in.shape
    torch.Size([1, 3, 16, 224, 224])
    >>> eff_in = prep.for_efficientnet(frames, n_keyframes=4)
    >>> eff_in.shape
    torch.Size([4, 3, 224, 224])
    """

    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        mean: Sequence[float] = IMAGENET_MEAN,
        std:  Sequence[float] = IMAGENET_STD,
        interpolation: str = "bilinear",
    ) -> None:
        self.size = size
        self.mean = tuple(mean)
        self.std  = tuple(std)
        self.interpolation = interpolation

    # ------------------------------------------------------------------
    # Internal: stages 1 + 2
    # ------------------------------------------------------------------

    def _resize_and_normalize(self, frames: Tensor) -> Tensor:
        """Apply stages 1 + 2; returns a tensor still in ``[B, T, C, H, W]``."""
        frames, _ = _ensure_batched(frames)
        frames = resize_preserve_aspect(
            frames, size=self.size, interpolation=self.interpolation
        )
        frames = imagenet_normalize(frames, mean=self.mean, std=self.std)
        return frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def for_swin(self, frames: Tensor) -> Tensor:
        """Return ``[B, C, T, H, W]`` ready for Video Swin-T."""
        x = self._resize_and_normalize(frames)
        return to_swin_layout(x)

    def for_efficientnet(
        self,
        frames: Tensor,
        n_keyframes: int | None = 4,
    ) -> Tensor:
        """Return ``[B·K, C, H, W]`` ready for EfficientNet-B3.

        Default ``n_keyframes=4`` matches the CLAUDE.md spec.
        """
        x = self._resize_and_normalize(frames)
        return to_efficientnet_layout(x, n_keyframes=n_keyframes)

    def __call__(
        self,
        frames: Tensor,
        n_keyframes: int | None = 4,
    ) -> dict[str, Tensor]:
        """Return both formats in one shot — handy for ensemble training.

        Stages 1 + 2 are computed once, then reused for both layouts.
        """
        x = self._resize_and_normalize(frames)
        return {
            "swin":         to_swin_layout(x),
            "efficientnet": to_efficientnet_layout(x, n_keyframes=n_keyframes),
        }

    def __repr__(self) -> str:
        return (
            f"Preprocessor(size={self.size}, "
            f"interpolation={self.interpolation!r}, "
            f"mean={self.mean}, std={self.std})"
        )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    prep = Preprocessor()
    print(prep)

    # Fake batch: 2 clips × 16 frames × 3×360×640
    fake = torch.rand(2, 16, 3, 360, 640)
    print(f"\nInput  : {tuple(fake.shape)}  dtype={fake.dtype}")

    swin = prep.for_swin(fake)
    print(f"Swin   : {tuple(swin.shape)}  (expected [2, 3, 16, 224, 224])")
    assert swin.shape == (2, 3, 16, 224, 224)

    eff = prep.for_efficientnet(fake, n_keyframes=4)
    print(f"EffNet : {tuple(eff.shape)}  (expected [8, 3, 224, 224])")
    assert eff.shape == (2 * 4, 3, 224, 224)

    both = prep(fake)
    print(f"\nDual call returns keys: {list(both.keys())}")
    print(f"  swin         : {tuple(both['swin'].shape)}")
    print(f"  efficientnet : {tuple(both['efficientnet'].shape)}")

    # Single-clip input (no batch dim)
    single = torch.rand(16, 3, 360, 640)
    swin1 = prep.for_swin(single)
    print(f"\nSingle clip [T,C,H,W] → Swin: {tuple(swin1.shape)}  (expected [1, 3, 16, 224, 224])")
    assert swin1.shape == (1, 3, 16, 224, 224)

    # Verify normalisation roughly worked (mean ≈ 0, std ≈ 1 across many samples).
    sample = both["swin"]
    print(
        f"\nNormalised tensor stats — "
        f"mean={sample.mean().item():+.3f}  std={sample.std().item():.3f}"
    )

    print("\nOK")
