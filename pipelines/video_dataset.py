"""
pipelines/video_dataset.py
==========================
PyTorch Dataset for loading short bird-species video clips (3-5 s).

Decoding back-end priority
--------------------------
1. ``decord``  – preferred; fast, zero-copy frame-level seek.
2. ``torchvision.io.read_video`` – fallback when decord is absent or
   raises an error on a particular file.

Returned sample dict
--------------------
{
    "frames" : FloatTensor[T, C, H, W],   # T uniformly-sampled frames
    "label"  : int,                        # class index 0-19
    "path"   : str,                        # absolute path to the clip
}

CSV format expected by ``CHIRPVideoDataset``
--------------------------------------------
Required columns: ``path``, ``label``
Optional columns: ``species`` (human-readable name, ignored at runtime)

Example::

    path,label,species
    data/raw/american_robin/clip_001.mp4,0,American Robin
    data/raw/blue_jay/clip_042.mp4,3,Blue Jay
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class registry (matches CLAUDE.md)
# ---------------------------------------------------------------------------

SPECIES: list[str] = [
    "American Robin",           # 0
    "Baltimore Oriole",         # 1
    "Black-capped Chickadee",   # 2
    "Blue Jay",                 # 3
    "Canada Goose",             # 4
    "Cedar Waxwing",            # 5
    "Common Grackle",           # 6
    "Dark-eyed Junco",          # 7
    "Downy Woodpecker",         # 8
    "House Finch",              # 9
    "Northern Cardinal",        # 10
    "Pileated Woodpecker",      # 11
    "Red-tailed Hawk",          # 12
    "Ruby-throated Hummingbird",# 13
    "Song Sparrow",             # 14
    "Tufted Titmouse",          # 15
    "White-breasted Nuthatch",  # 16
    "Wild Turkey",              # 17
    "Wood Duck",                # 18
    "Yellow Warbler",           # 19
]
NUM_CLASSES: int = len(SPECIES)

# ---------------------------------------------------------------------------
# Decoder helpers
# ---------------------------------------------------------------------------

try:
    import decord  # noqa: F401
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False
    logger.warning(
        "decord not found — falling back to torchvision for video decoding. "
        "Install decord for faster loading: pip install decord"
    )

try:
    import torchvision  # noqa: F401
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


def _sample_indices(total_frames: int, n_frames: int, jitter: bool) -> np.ndarray:
    """Return ``n_frames`` frame indices sampled uniformly from ``[0, total_frames)``.

    Parameters
    ----------
    total_frames:
        Number of frames in the decoded clip.
    n_frames:
        Desired number of output frames.
    jitter:
        When ``True`` each index is randomly offset within its segment
        (training augmentation). When ``False`` the segment mid-point is used.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be > 0, got {total_frames}")

    if total_frames <= n_frames:
        # Clip is too short — repeat-pad to reach n_frames.
        indices = np.arange(total_frames)
        indices = np.resize(indices, n_frames)   # wraps around
        return np.sort(indices)

    # Divide [0, total_frames) into n_frames equal-width segments.
    seg_size = total_frames / n_frames
    if jitter:
        offsets = np.random.uniform(0, seg_size, size=n_frames)
    else:
        offsets = np.full(n_frames, seg_size / 2)

    indices = (np.arange(n_frames) * seg_size + offsets).astype(int)
    return np.clip(indices, 0, total_frames - 1)


def _decode_with_decord(
    path: str,
    indices: np.ndarray,
    height: int,
    width: int,
) -> Tensor:
    """Decode specific frame indices with decord.

    Returns
    -------
    Tensor of shape ``[T, C, H, W]`` in float32 range ``[0, 1]``.
    """
    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(path, width=width, height=height, num_threads=1)
    frames = vr.get_batch(indices.tolist())   # [T, H, W, C] uint8 Tensor
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
    return frames


def _decode_with_torchvision(
    path: str,
    indices: np.ndarray,
    height: int,
    width: int,
) -> Tensor:
    """Decode specific frame indices with torchvision.

    torchvision loads the whole clip then subsamples, so it is slower than
    decord for long files, but perfectly correct for 3-5 s clips.

    Returns
    -------
    Tensor of shape ``[T, C, H, W]`` in float32 range ``[0, 1]``.
    """
    import torchvision.io as tvio
    import torchvision.transforms.functional as TF

    # read_video returns (frames [T, H, W, C] uint8, audio, metadata)
    frames_raw, _, _ = tvio.read_video(path, output_format="TCHW", pts_unit="sec")
    # frames_raw: [T, C, H, W] uint8

    # Subselect requested indices (guard against out-of-bounds).
    total = frames_raw.shape[0]
    safe_idx = np.clip(indices, 0, total - 1)
    frames = frames_raw[safe_idx]                          # [T, C, H, W] uint8

    # Resize each frame.
    frames = torch.stack([
        TF.resize(f, [height, width], antialias=True)
        for f in frames
    ])                                                     # [T, C, H, W] uint8

    return frames.float() / 255.0


def decode_clip(
    path: str,
    indices: np.ndarray,
    height: int,
    width: int,
    backend: Literal["auto", "decord", "torchvision"] = "auto",
) -> Tensor:
    """Decode a set of frame indices from a video file.

    Parameters
    ----------
    path:
        Absolute or relative path to the video clip.
    indices:
        1-D integer array of frame indices to extract.
    height, width:
        Spatial resolution to resize frames to.
    backend:
        ``"auto"``        – try decord, fall back to torchvision.
        ``"decord"``      – force decord (raises if not installed).
        ``"torchvision"`` – force torchvision (raises if not installed).

    Returns
    -------
    FloatTensor ``[T, C, H, W]`` in ``[0, 1]``.
    """
    if backend == "decord" or (backend == "auto" and _DECORD_AVAILABLE):
        try:
            return _decode_with_decord(path, indices, height, width)
        except Exception as exc:
            if backend == "decord":
                raise
            logger.debug("decord failed (%s) — retrying with torchvision", exc)

    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError(
            "Neither decord nor torchvision is available. "
            "Install at least one: pip install decord  OR  pip install torchvision"
        )
    return _decode_with_torchvision(path, indices, height, width)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CHIRPVideoDataset(Dataset):
    """Dataset of short bird-species video clips.

    Parameters
    ----------
    csv_path:
        Path to a CSV file with at least two columns: ``path`` (video file
        path) and ``label`` (integer class index 0-19).
    n_frames:
        Number of frames to uniformly sample from each clip. Recommended
        range 8-16; default 16 (matches CLAUDE.md hyper-params).
    height, width:
        Spatial resolution. Default 224×224 (matches Video Swin-T / EB3).
    split:
        ``"train"`` enables temporal jitter augmentation.
        ``"val"`` / ``"test"`` use deterministic mid-segment sampling.
    transform:
        Optional callable applied to the ``[T, C, H, W]`` float tensor
        **before** it is returned. Use for spatial augmentations
        (normalisation, random crops, etc.).
    backend:
        Video decoding back-end. ``"auto"`` prefers decord with torchvision
        fallback.
    root_dir:
        Optional root directory prepended to relative ``path`` entries in
        the CSV.

    Examples
    --------
    >>> ds = CHIRPVideoDataset("data/splits/train.csv", n_frames=16, split="train")
    >>> sample = ds[0]
    >>> sample["frames"].shape
    torch.Size([16, 3, 224, 224])
    >>> sample["label"]
    3
    >>> sample["path"]
    'data/raw/blue_jay/clip_042.mp4'
    """

    def __init__(
        self,
        csv_path: str | Path,
        n_frames: int = 16,
        height: int = 224,
        width: int = 224,
        split: Literal["train", "val", "test"] = "train",
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        backend: Literal["auto", "decord", "torchvision"] = "auto",
        root_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__()

        if not 8 <= n_frames <= 16:
            logger.warning(
                "n_frames=%d is outside the recommended range [8, 16].", n_frames
            )

        self.n_frames = n_frames
        self.height = height
        self.width = width
        self.split = split
        self.transform = transform
        self.backend = backend
        self.root_dir = Path(root_dir) if root_dir else None
        self._jitter = split == "train"

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self._df = pd.read_csv(csv_path)
        self._validate_csv()

        logger.info(
            "CHIRPVideoDataset | split=%s | clips=%d | n_frames=%d | backend=%s",
            split, len(self._df), n_frames, backend,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_csv(self) -> None:
        required = {"path", "label"}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found: {list(self._df.columns)}"
            )

        bad_labels = self._df["label"][~self._df["label"].between(0, NUM_CLASSES - 1)]
        if not bad_labels.empty:
            raise ValueError(
                f"Found labels outside [0, {NUM_CLASSES - 1}]: "
                f"{bad_labels.unique().tolist()}"
            )

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self, raw: str) -> str:
        p = Path(raw)
        if self.root_dir and not p.is_absolute():
            p = self.root_dir / p
        return str(p)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:
        row = self._df.iloc[idx]
        path = self._resolve_path(str(row["path"]))
        label = int(row["label"])

        # ---- count total frames without decoding all pixels -------------
        total_frames = _count_frames(path, self.backend)

        # ---- sample frame indices ---------------------------------------
        indices = _sample_indices(total_frames, self.n_frames, jitter=self._jitter)

        # ---- decode selected frames -------------------------------------
        try:
            frames = decode_clip(
                path, indices,
                height=self.height,
                width=self.width,
                backend=self.backend,
            )
        except Exception as exc:
            logger.error("Failed to decode %s: %s — returning zeros.", path, exc)
            frames = torch.zeros(
                self.n_frames, 3, self.height, self.width, dtype=torch.float32
            )

        # ---- optional spatial transform ---------------------------------
        if self.transform is not None:
            frames = self.transform(frames)

        return {
            "frames": frames,   # [T, C, H, W] float32 in [0, 1] (or transformed)
            "label": label,
            "path": path,
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def labels(self) -> list[int]:
        """All integer labels in dataset order (for sampler construction)."""
        return self._df["label"].tolist()

    @property
    def class_names(self) -> list[str]:
        return SPECIES

    def class_counts(self) -> dict[str, int]:
        """Return a ``{species_name: count}`` dict."""
        counts = self._df["label"].value_counts().sort_index()
        return {SPECIES[i]: int(counts.get(i, 0)) for i in range(NUM_CLASSES)}

    def __repr__(self) -> str:
        return (
            f"CHIRPVideoDataset(split={self.split!r}, clips={len(self)}, "
            f"n_frames={self.n_frames}, size={self.height}×{self.width}, "
            f"backend={self.backend!r})"
        )


# ---------------------------------------------------------------------------
# Frame-count helper (lightweight — avoids full decode)
# ---------------------------------------------------------------------------

def _count_frames(path: str, backend: Literal["auto", "decord", "torchvision"]) -> int:
    """Return the total number of frames in *path* as cheaply as possible."""
    if backend in ("auto", "decord") and _DECORD_AVAILABLE:
        try:
            import decord
            vr = decord.VideoReader(path, num_threads=1)
            return len(vr)
        except Exception:
            pass  # fall through to torchvision

    if _TORCHVISION_AVAILABLE:
        try:
            import torchvision.io as tvio
            frames, _, _ = tvio.read_video(path, output_format="TCHW", pts_unit="sec")
            return frames.shape[0]
        except Exception:
            pass

    # Last resort: return a reasonable default so sampling doesn't crash.
    logger.warning("Could not count frames for %s — assuming 30.", path)
    return 30


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    csv_path: str | Path,
    split: Literal["train", "val", "test"],
    *,
    n_frames: int = 16,
    height: int = 224,
    width: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    backend: Literal["auto", "decord", "torchvision"] = "auto",
    root_dir: Optional[str | Path] = None,
    balance_classes: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """Construct a ``DataLoader`` for a given split.

    Parameters
    ----------
    csv_path:
        Path to the split CSV.
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    n_frames:
        Frames to sample per clip (8-16).
    height, width:
        Spatial resolution (default 224×224).
    batch_size:
        Mini-batch size (default 16).
    num_workers:
        DataLoader worker processes. Set 0 for debugging.
    transform:
        Optional frame-level transform applied after decoding.
    backend:
        Decoding back-end (``"auto"`` | ``"decord"`` | ``"torchvision"``).
    root_dir:
        Prepended to relative paths found in the CSV.
    balance_classes:
        When ``True`` (training only), use ``WeightedRandomSampler`` so that
        each species is sampled with equal expected frequency.
    pin_memory:
        Pin tensors to page-locked memory for faster GPU transfer.

    Returns
    -------
    ``torch.utils.data.DataLoader`` ready for iteration.
    """
    dataset = CHIRPVideoDataset(
        csv_path=csv_path,
        n_frames=n_frames,
        height=height,
        width=width,
        split=split,
        transform=transform,
        backend=backend,
        root_dir=root_dir,
    )

    sampler = None
    shuffle = split == "train"

    if balance_classes and split == "train":
        labels = dataset.labels
        counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
        counts = np.where(counts == 0, 1, counts)   # avoid div-by-zero
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False   # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=split == "train",   # avoid tiny last batch during training
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Normalisation constants (ImageNet, used by both Video Swin-T and EB3)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])  # [C]
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])  # [C]


def normalize_frames(frames: Tensor) -> Tensor:
    """Apply ImageNet mean/std normalisation to a ``[T, C, H, W]`` tensor.

    Operates in-place on a float tensor already in ``[0, 1]``.
    """
    mean = IMAGENET_MEAN.to(frames.device).view(1, 3, 1, 1)
    std  = IMAGENET_STD.to(frames.device).view(1, 3, 1, 1)
    return (frames - mean) / std


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Smoke-test CHIRPVideoDataset")
    parser.add_argument("csv", help="Path to a split CSV")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "decord", "torchvision"])
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--balance", action="store_true")
    args = parser.parse_args()

    loader = build_dataloader(
        csv_path=args.csv,
        split="train",
        n_frames=args.n_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        backend=args.backend,
        root_dir=args.root_dir,
        balance_classes=args.balance,
    )

    print(loader.dataset)
    print(f"\nIterating first 3 batches …")
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        frames = batch["frames"]
        labels = batch["label"]
        paths  = batch["path"]
        print(
            f"  batch {i}: frames={tuple(frames.shape)}, "
            f"dtype={frames.dtype}, "
            f"labels={labels.tolist()}, "
            f"paths={[Path(p).name for p in paths]}"
        )
        assert frames.shape == (
            args.batch_size, args.n_frames, 3, 224, 224
        ), f"Unexpected shape: {frames.shape}"
        if i >= 2:
            break
    elapsed = time.perf_counter() - t0
    print(f"\nOK — 3 batches in {elapsed:.2f}s")
    sys.exit(0)
