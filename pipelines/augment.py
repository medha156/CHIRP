"""
pipelines/augment.py
====================
Video-clip data augmentation built on Albumentations.

Design notes
------------
* **Temporal consistency** — the *same* random crop / flip / colour-jitter
  parameters are applied to every frame of a single clip. We achieve this
  with ``albumentations.ReplayCompose``: sample params once on the first
  frame, then ``A.ReplayCompose.replay`` on the rest.
* **Split-specific pipelines**
    - ``train`` : ``Resize(short=256) → RandomCrop(224) →
                    HorizontalFlip(p=0.5) → ColorJitter``
    - ``val``   : ``Resize(short=256) → CenterCrop(224)``
    - ``test``  : same as ``val``
* **Toggleable** — passing ``enabled=False`` (or ``cfg["augment"]=False``)
  collapses the train pipeline to the deterministic val pipeline. This is
  the same flag the rest of CHIRP reads from config.
* **Input / output contract** — accepts and returns a
  ``[T, C, H, W]`` float tensor in ``[0, 1]``. This matches
  ``CHIRPVideoDataset``'s ``transform`` slot, so you can plug it in
  directly::

      ds = CHIRPVideoDataset(..., transform=build_augment("train"))
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal

import albumentations as A
import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]

# ---------------------------------------------------------------------------
# Defaults — overridable through build_augment(cfg=...)
# ---------------------------------------------------------------------------

DEFAULTS: dict = {
    "size":          224,    # final crop size
    "resize_short":  256,    # shorter-side resize before cropping
    "hflip_p":       0.5,    # horizontal flip probability
    # ColorJitter params — gentle, plumage colours matter for ID
    "jitter_brightness": 0.2,
    "jitter_contrast":   0.2,
    "jitter_saturation": 0.2,
    "jitter_hue":        0.05,
    "jitter_p":          0.8,
}


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _train_pipeline(cfg: Mapping) -> A.ReplayCompose:
    return A.ReplayCompose([
        A.SmallestMaxSize(max_size=cfg["resize_short"], interpolation=1),  # bilinear
        A.RandomCrop(height=cfg["size"], width=cfg["size"]),
        A.HorizontalFlip(p=cfg["hflip_p"]),
        A.ColorJitter(
            brightness=cfg["jitter_brightness"],
            contrast=cfg["jitter_contrast"],
            saturation=cfg["jitter_saturation"],
            hue=cfg["jitter_hue"],
            p=cfg["jitter_p"],
        ),
    ])


def _eval_pipeline(cfg: Mapping) -> A.ReplayCompose:
    """Deterministic pipeline: resize-shortest-side then centre crop."""
    return A.ReplayCompose([
        A.SmallestMaxSize(max_size=cfg["resize_short"], interpolation=1),
        A.CenterCrop(height=cfg["size"], width=cfg["size"]),
    ])


# ---------------------------------------------------------------------------
# Tensor <-> numpy adapters
# ---------------------------------------------------------------------------

def _tensor_to_np_uint8(frames: Tensor) -> np.ndarray:
    """Convert ``[T, C, H, W]`` float ∈ [0,1] → ``[T, H, W, C]`` uint8 numpy."""
    if frames.ndim != 4:
        raise ValueError(
            f"Expected [T, C, H, W] tensor, got shape {tuple(frames.shape)}"
        )
    if not frames.is_floating_point():
        raise TypeError(
            f"Frames must be float in [0,1], got dtype={frames.dtype}"
        )
    arr = (frames.clamp(0, 1) * 255.0).byte().permute(0, 2, 3, 1).contiguous()
    return arr.cpu().numpy()


def _np_uint8_to_tensor(frames: np.ndarray) -> Tensor:
    """Convert ``[T, H, W, C]`` uint8 numpy → ``[T, C, H, W]`` float ∈ [0,1]."""
    t = torch.from_numpy(frames).float() / 255.0       # [T, H, W, C]
    return t.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Main callable
# ---------------------------------------------------------------------------

class VideoAugment:
    """Stateful, clip-consistent video augmenter.

    Parameters
    ----------
    split:
        ``"train"`` enables stochastic augmentation; ``"val"`` / ``"test"``
        use only the deterministic resize + centre crop.
    enabled:
        Master toggle. When ``False`` the ``train`` split silently falls
        back to the eval pipeline — useful for ablations or quick
        debugging without changing the dataset wiring.
    cfg:
        Optional dict of overrides for any key in ``DEFAULTS``.

    Examples
    --------
    >>> aug = VideoAugment(split="train", enabled=True)
    >>> frames = torch.rand(16, 3, 360, 640)
    >>> aug(frames).shape
    torch.Size([16, 3, 224, 224])

    Toggle off → deterministic resize + centre crop:

    >>> aug = VideoAugment(split="train", enabled=False)
    >>> out = aug(frames)
    >>> out.shape
    torch.Size([16, 3, 224, 224])
    """

    def __init__(
        self,
        split: Split = "train",
        enabled: bool = True,
        cfg: Mapping | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}")

        self.split = split
        self.enabled = enabled
        self.cfg = {**DEFAULTS, **(cfg or {})}

        use_train = (split == "train") and enabled
        self._pipeline = (
            _train_pipeline(self.cfg) if use_train else _eval_pipeline(self.cfg)
        )

        logger.info(
            "VideoAugment | split=%s | enabled=%s | "
            "pipeline=%s | size=%d",
            split,
            enabled,
            "train" if use_train else "eval",
            self.cfg["size"],
        )

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def __call__(self, frames: Tensor) -> Tensor:
        """Apply augmentation to a single clip ``[T, C, H, W]`` → ``[T, C, H, W]``.

        The *same* random parameters (crop coords, flip yes/no, jitter
        magnitudes) are reused for every frame in the clip — essential
        for preserving motion cues that the temporal model relies on.
        """
        np_frames = _tensor_to_np_uint8(frames)        # [T, H, W, C] uint8

        # 1) Apply pipeline to the first frame and capture replay params.
        first = self._pipeline(image=np_frames[0])
        replay = first["replay"]
        out = [first["image"]]

        # 2) Replay the *exact same* augmentation on every subsequent frame.
        for f in np_frames[1:]:
            out.append(A.ReplayCompose.replay(replay, image=f)["image"])

        out_np = np.stack(out, axis=0)                 # [T, H', W', C]
        return _np_uint8_to_tensor(out_np)             # [T, C, H', W']

    def __repr__(self) -> str:
        return (
            f"VideoAugment(split={self.split!r}, enabled={self.enabled}, "
            f"size={self.cfg['size']}, resize_short={self.cfg['resize_short']})"
        )


# ---------------------------------------------------------------------------
# Config-driven factory
# ---------------------------------------------------------------------------

def build_augment(
    split: Split,
    cfg: Mapping | None = None,
) -> VideoAugment:
    """Construct a ``VideoAugment`` from a config-style mapping.

    The factory honours an ``augment`` boolean inside ``cfg`` (default
    ``True``). All other keys override the corresponding entry in
    :data:`DEFAULTS`.

    Example YAML config::

        augment: true            # master toggle
        size: 224
        hflip_p: 0.5
        jitter_brightness: 0.3
        jitter_hue: 0.0          # disable hue jitter

    Loaded as::

        aug_train = build_augment("train", cfg)
        aug_val   = build_augment("val",   cfg)   # always deterministic
    """
    cfg = dict(cfg or {})
    enabled = bool(cfg.pop("augment", True))
    return VideoAugment(split=split, enabled=enabled, cfg=cfg)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    fake = torch.rand(16, 3, 360, 640)
    print(f"Input frames: {tuple(fake.shape)}\n")

    for split in ("train", "val", "test"):
        aug = build_augment(split, cfg={"augment": True})
        out = aug(fake)
        print(f"  {split:5s} (augment=on)  → {tuple(out.shape)}  | {aug}")
        assert out.shape == (16, 3, 224, 224)

    print()
    aug_off = build_augment("train", cfg={"augment": False})
    out = aug_off(fake)
    print(f"  train (augment=off) → {tuple(out.shape)}  | {aug_off}")
    assert out.shape == (16, 3, 224, 224)

    # Temporal-consistency check: when augment is off, two runs on the
    # same input must match exactly (deterministic pipeline).
    out_a = aug_off(fake)
    out_b = aug_off(fake)
    assert torch.allclose(out_a, out_b), "Eval pipeline is not deterministic!"
    print("\n  ✓ Eval pipeline deterministic")

    # When augment is on, repeated runs should differ (stochastic).
    aug_on = build_augment("train", cfg={"augment": True})
    diff = (aug_on(fake) - aug_on(fake)).abs().mean().item()
    print(f"  ✓ Train pipeline stochastic (mean |Δ| = {diff:.4f})")

    print("\nOK")
