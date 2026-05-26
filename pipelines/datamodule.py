"""
pipelines/datamodule.py
=======================
End-to-end data orchestration for CHIRP.

Responsibilities
----------------
1. **Load** index CSVs for the two bird-video corpora used by the project:

   - **FBD-SV-2024** — *Flying Bird Detection in Surveillance Video 2024*.
     Expected file: ``<data_root>/fbd_sv_2024/index.csv``
   - **VB100**       — *Video-Bird 100*.
     Expected file: ``<data_root>/vb100/index.csv``

   Each CSV must contain columns ``path, label, species`` (label = integer
   in [0, 19] matching CLAUDE.md, species = human-readable name). A
   ``dataset`` column is added automatically.

2. **Merge** both corpora into a unified table.

3. **Stratified split** (default 70 / 15 / 15) preserving per-class
   proportions. Splits are reproducible (``seed``).

4. **Class balancing** — computes ``sklearn``-style balanced class weights
   exposed as ``module.class_weights`` (Tensor of length ``NUM_CLASSES``)
   for use with ``nn.CrossEntropyLoss(weight=...)``.

5. **DataLoaders** — wraps each split in a ``CHIRPVideoDataset`` and
   returns ready-to-iterate ``DataLoader``s.

6. **Sanity check** — ``plot_class_distribution()`` writes a horizontal
   bar chart of class counts to ``outputs/class_distribution.png`` so
   anyone reviewing a run can confirm the splits look right.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

# Allow running this file directly (python pipelines/datamodule.py) by
# making sure the project root is on sys.path before the local import.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipelines.video_dataset import (  # noqa: E402
    NUM_CLASSES,
    SPECIES,
    CHIRPVideoDataset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CHIRPDataModule
# ---------------------------------------------------------------------------

class CHIRPDataModule:
    """Loads FBD-SV-2024 + VB100, splits, balances, and serves DataLoaders.

    Parameters
    ----------
    data_root:
        Directory containing per-dataset sub-folders, each with an
        ``index.csv``. Example layout::

            data_root/
              ├── fbd_sv_2024/index.csv
              └── vb100/index.csv

    datasets:
        Subset of ``("fbd_sv_2024", "vb100")``. Default loads both.
    splits:
        3-tuple of train/val/test fractions (must sum to 1.0).
    seed:
        RNG seed for the stratified split.
    batch_size, num_workers, n_frames, height, width:
        Forwarded to :class:`CHIRPVideoDataset` / DataLoader.
    train_transform, eval_transform:
        Optional per-frame transforms (e.g. from ``pipelines.augment``).
    backend:
        Video decoding backend (``"auto"`` | ``"decord"`` | ``"torchvision"``).
    balance_sampler:
        When ``True``, the train DataLoader uses ``WeightedRandomSampler``
        in addition to ``class_weights``. Belt-and-braces for very
        imbalanced long-tail splits.

    Usage
    -----
    >>> dm = CHIRPDataModule(data_root="data/")
    >>> dm.setup()
    >>> train_dl = dm.train_dataloader()
    >>> weights  = dm.class_weights              # [20] tensor
    >>> dm.plot_class_distribution("outputs/class_distribution.png")
    """

    KNOWN_DATASETS = ("fbd_sv_2024", "vb100")

    def __init__(
        self,
        data_root: str | Path,
        *,
        datasets: tuple[str, ...] = KNOWN_DATASETS,
        splits: tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
        batch_size: int = 16,
        num_workers: int = 4,
        n_frames: int = 16,
        height: int = 224,
        width: int = 224,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
        backend: str = "auto",
        balance_sampler: bool = False,
    ) -> None:
        if abs(sum(splits) - 1.0) > 1e-6:
            raise ValueError(f"splits must sum to 1.0; got {splits} (sum={sum(splits)})")
        for d in datasets:
            if d not in self.KNOWN_DATASETS:
                raise ValueError(
                    f"Unknown dataset {d!r}; known: {self.KNOWN_DATASETS}"
                )

        self.data_root      = Path(data_root)
        self.datasets       = tuple(datasets)
        self.splits         = splits
        self.seed           = seed
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.n_frames       = n_frames
        self.height         = height
        self.width          = width
        self.train_transform = train_transform
        self.eval_transform  = eval_transform
        self.backend         = backend
        self.balance_sampler = balance_sampler

        # populated by setup()
        self.df:         pd.DataFrame | None = None
        self.train_df:   pd.DataFrame | None = None
        self.val_df:     pd.DataFrame | None = None
        self.test_df:    pd.DataFrame | None = None
        self.class_weights: torch.Tensor | None = None
        self._is_set_up = False

    # ------------------------------------------------------------------
    # Setup pipeline
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load CSVs, merge, split, compute class weights. Idempotent."""
        if self._is_set_up:
            return

        self.df = self._load_and_merge()
        self.train_df, self.val_df, self.test_df = self._stratified_split(self.df)
        self.class_weights = self._compute_class_weights(self.train_df)
        self._is_set_up = True

        logger.info(
            "CHIRPDataModule | train=%d | val=%d | test=%d | classes=%d",
            len(self.train_df), len(self.val_df), len(self.test_df), NUM_CLASSES,
        )

    # ------------------------------------------------------------------

    def _load_and_merge(self) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        for name in self.datasets:
            csv = self.data_root / name / "index.csv"
            if not csv.exists():
                raise FileNotFoundError(
                    f"Expected index CSV for {name!r} at {csv}. "
                    "Each dataset must have an index.csv with columns "
                    "path, label, species."
                )
            df = pd.read_csv(csv)
            missing = {"path", "label"} - set(df.columns)
            if missing:
                raise ValueError(f"{csv} is missing columns {missing}")
            df["dataset"] = name
            # Resolve relative paths against <data_root>/<name>/
            df["path"] = df["path"].apply(
                lambda p: str((self.data_root / name / p).resolve())
                if not Path(p).is_absolute() else p
            )
            dfs.append(df)
            logger.info("Loaded %s: %d clips", name, len(df))

        merged = pd.concat(dfs, ignore_index=True)
        bad = merged["label"][~merged["label"].between(0, NUM_CLASSES - 1)]
        if not bad.empty:
            raise ValueError(
                f"Labels outside [0, {NUM_CLASSES - 1}] after merge: "
                f"{sorted(bad.unique().tolist())}"
            )
        return merged

    # ------------------------------------------------------------------

    def _stratified_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_frac, val_frac, test_frac = self.splits

        # First peel off the test set, then split the remainder into train/val.
        train_val, test = train_test_split(
            df,
            test_size=test_frac,
            stratify=df["label"],
            random_state=self.seed,
        )
        relative_val = val_frac / (train_frac + val_frac)
        train, val = train_test_split(
            train_val,
            test_size=relative_val,
            stratify=train_val["label"],
            random_state=self.seed,
        )
        return (
            train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True),
        )

    # ------------------------------------------------------------------

    def _compute_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        present = sorted(train_df["label"].unique())
        weights_present = compute_class_weight(
            class_weight="balanced",
            classes=np.array(present),
            y=train_df["label"].to_numpy(),
        )

        # Fill missing classes with weight 1.0 so indexing by label is safe.
        weights = np.ones(NUM_CLASSES, dtype=np.float32)
        for cls, w in zip(present, weights_present, strict=True):
            weights[cls] = w

        missing = sorted(set(range(NUM_CLASSES)) - set(present))
        if missing:
            logger.warning(
                "Classes absent from training split (weight set to 1.0): %s",
                [SPECIES[i] for i in missing],
            )
        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------

    def _make_loader(
        self,
        df: pd.DataFrame,
        split: str,
        transform: Optional[Callable],
        shuffle: bool,
    ) -> DataLoader:
        # CHIRPVideoDataset expects a CSV path; round-trip through a temp CSV
        # to avoid re-implementing its CSV-validation logic.
        tmp_dir = self.data_root / "_splits_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_csv = tmp_dir / f"{split}_seed{self.seed}.csv"
        df.to_csv(tmp_csv, index=False)

        dataset = CHIRPVideoDataset(
            csv_path=tmp_csv,
            n_frames=self.n_frames,
            height=self.height,
            width=self.width,
            split=split,  # type: ignore[arg-type]
            transform=transform,
            backend=self.backend,  # type: ignore[arg-type]
        )

        sampler = None
        if shuffle and self.balance_sampler:
            from torch.utils.data import WeightedRandomSampler
            sample_w = self.class_weights[df["label"].to_numpy()].double()
            sampler = WeightedRandomSampler(
                weights=sample_w,
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return self._make_loader(
            self.train_df, "train", self.train_transform, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return self._make_loader(
            self.val_df, "val", self.eval_transform, shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        self._ensure_setup()
        return self._make_loader(
            self.test_df, "test", self.eval_transform, shuffle=False
        )

    def _ensure_setup(self) -> None:
        if not self._is_set_up:
            self.setup()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def class_distribution(self) -> pd.DataFrame:
        """Return a tidy DataFrame: rows = species, cols = train/val/test."""
        self._ensure_setup()
        counts = {
            "train": self.train_df["label"].value_counts(),
            "val":   self.val_df["label"].value_counts(),
            "test":  self.test_df["label"].value_counts(),
        }
        out = pd.DataFrame(counts).reindex(range(NUM_CLASSES)).fillna(0).astype(int)
        out.index = pd.Index(SPECIES, name="species")
        return out

    def plot_class_distribution(
        self,
        out_path: str | Path = "outputs/class_distribution.png",
        figsize: tuple[float, float] = (10, 8),
    ) -> Path:
        """Save a stacked horizontal bar chart of per-class counts."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dist = self.class_distribution()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=figsize)
        dist.plot.barh(
            stacked=True,
            ax=ax,
            color=["#2E86AB", "#A23B72", "#F18F01"],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel("Number of clips")
        ax.set_title(
            f"CHIRP class distribution (n={len(self.df)} clips, "
            f"seed={self.seed})"
        )
        ax.invert_yaxis()
        ax.legend(title="Split", loc="lower right")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        logger.info("Saved class distribution → %s", out_path)
        print(f"\nClass distribution saved to {out_path}")
        print(dist.to_string())
        return out_path


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import shutil
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None,
                        help="Path to real data root. If omitted, runs on synthetic CSVs.")
    parser.add_argument("--out", default="outputs/class_distribution.png")
    args = parser.parse_args()

    # ---------------- Synthetic-data path ----------------
    if args.data_root is None:
        tmp = Path(tempfile.mkdtemp(prefix="chirp_dm_"))
        rng = np.random.default_rng(0)
        for ds_name, n_clips in (("fbd_sv_2024", 400), ("vb100", 250)):
            (tmp / ds_name).mkdir(parents=True)
            labels = rng.integers(0, NUM_CLASSES, size=n_clips)
            df = pd.DataFrame({
                "path":    [f"clip_{i:04d}.mp4" for i in range(n_clips)],
                "label":   labels,
                "species": [SPECIES[i] for i in labels],
            })
            df.to_csv(tmp / ds_name / "index.csv", index=False)
        data_root = tmp
        print(f"(using synthetic data root at {data_root})")
    else:
        data_root = Path(args.data_root)

    dm = CHIRPDataModule(data_root=data_root, num_workers=0, batch_size=4)
    dm.setup()

    print(f"\nTrain / Val / Test sizes: "
          f"{len(dm.train_df)} / {len(dm.val_df)} / {len(dm.test_df)}")
    print(f"Class weights (first 5): {dm.class_weights[:5].tolist()}")
    assert dm.class_weights.shape == (NUM_CLASSES,)

    # Distribution + plot
    out_path = dm.plot_class_distribution(args.out)
    assert out_path.exists() and out_path.stat().st_size > 0

    # Stratification sanity check: per-class proportion within ±5 pp.
    full = dm.df["label"].value_counts(normalize=True).sort_index()
    train = dm.train_df["label"].value_counts(normalize=True).sort_index()
    max_drift = (full - train).abs().max()
    print(f"\nStratification drift (max |full − train| share): {max_drift:.4f}")
    assert max_drift < 0.05, f"Stratification drifted by {max_drift:.4f}"

    # Cleanup synthetic data
    if args.data_root is None:
        shutil.rmtree(tmp)

    print("\nOK")
