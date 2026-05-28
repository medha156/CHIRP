"""
training/train.py
=================
Main entry point for CHIRP training runs.

Flow
----
1. Parse ``--config <yaml>`` (+ optional ``--override key=value`` flags).
2. Set seed, build :class:`pipelines.datamodule.CHIRPDataModule`, run
   ``setup()`` to get train/val/test DataLoaders and class weights.
3. Instantiate the right model based on ``cfg.model.model_type``:
   - ``swin``         → :class:`models.swin_t.VideoSwinT`
   - ``efficientnet`` → :class:`models.efficientnet.EfficientNetB3Encoder`
                        (head appended for classification)
   - ``fusion``       → both backbones + :class:`models.fusion.FusionHead`
   - ``baseline_{rf,xgb,knn}`` → no gradient training; extracts EB3
                        features and dispatches to
                        :func:`models.baselines.train_baselines`.
4. Run :class:`training.trainer.Trainer.fit`.
5. Restore the best checkpoint and evaluate on val + test.
6. Append a one-row summary to ``cfg.log.results_csv``.

Usage
-----
::

    python training/train.py --config configs/fusion.yaml
    python training/train.py --config configs/fusion.yaml \\
        --override optim.lr=5e-4 model.dropout=0.4
"""

from __future__ import annotations

# Allow direct script execution
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import datetime as dt
import logging
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from models.baselines import (  # noqa: E402
    extract_features, train_baselines,
)
from models.efficientnet import EfficientNetB3Encoder  # noqa: E402
from models.fusion import FusionHead, SwinEffNetFusion  # noqa: E402
from models.swin_t import VideoSwinT  # noqa: E402
from pipelines.augment import build_augment  # noqa: E402
from pipelines.datamodule import CHIRPDataModule  # noqa: E402
from pipelines.optical_flow import build_optical_flow  # noqa: E402
from pipelines.preprocess import (  # noqa: E402
    Preprocessor, imagenet_normalize, resize_preserve_aspect, to_swin_layout,
)
from pipelines.video_dataset import NUM_CLASSES  # noqa: E402
from training.config import TrainConfig  # noqa: E402
from training.trainer import Trainer  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model + step_fn builders
# ---------------------------------------------------------------------------

def build_model_and_step(
    cfg: TrainConfig, device: torch.device,
) -> tuple[nn.Module, Callable]:
    """Construct the model and a matching ``step_fn`` for the trainer."""
    preprocessor = Preprocessor(size=cfg.data.height)
    flow_op      = build_optical_flow({
        "enabled":  cfg.data.use_optical_flow,
        "backend":  cfg.data.optical_flow_backend,
        "raft_model": cfg.data.raft_model,
    })
    in_channels = 5 if cfg.data.use_optical_flow else 3

    mt = cfg.model.model_type

    # ------------------------------------------------------------------
    if mt == "swin":
        model = VideoSwinT(
            num_classes=cfg.model.num_classes,
            in_channels=in_channels,
            pretrained=cfg.model.pretrained,
        )
        if cfg.model.freeze_backbone:
            model.freeze_backbone(True)

        def step_fn(m: nn.Module, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
            # Resize → flow → normalize → swin layout. We unroll the
            # Preprocessor steps so flow can run on the resized [0,1]
            # RGB frames (Farneback expects uint8 grayscale) *before*
            # ImageNet normalisation runs over the first 3 channels.
            frames = batch["frames"].to(device, non_blocking=True)        # [B,T,3,H,W]
            frames = resize_preserve_aspect(frames, size=cfg.data.height) # [B,T,3,224,224]
            if flow_op.enabled:
                frames = torch.stack([flow_op(c) for c in frames])        # [B,T,5,224,224]
            frames  = imagenet_normalize(frames)                          # first 3 ch only
            swin_in = to_swin_layout(frames)                              # [B,C,T,224,224]
            return m(swin_in), batch["label"].to(device, non_blocking=True)

        return model, step_fn

    # ------------------------------------------------------------------
    if mt == "efficientnet":
        model = EfficientNetB3Encoder(
            pretrained=cfg.model.pretrained,
            freeze=cfg.model.freeze_backbone,
            num_classes=cfg.model.num_classes,
            pool=cfg.model.pool,
        )

        def step_fn(m: nn.Module, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
            frames = batch["frames"].to(device, non_blocking=True)
            eff_in = preprocessor.for_efficientnet(frames, n_keyframes=cfg.model.n_keyframes)
            return m(eff_in, t=cfg.model.n_keyframes), batch["label"].to(device)

        return model, step_fn

    # ------------------------------------------------------------------
    if mt == "fusion":
        swin = VideoSwinT(
            num_classes=cfg.model.num_classes,
            in_channels=in_channels,
            pretrained=cfg.model.pretrained,
        )
        eff = EfficientNetB3Encoder(
            pretrained=cfg.model.pretrained,
            freeze=cfg.model.freeze_backbone,
            num_classes=0,                # raw features for fusion
        )
        if cfg.model.freeze_backbone:
            swin.freeze_backbone(True)

        head = FusionHead(
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.model.dropout,
        )
        model = SwinEffNetFusion(swin, eff, head, keyframes_per_clip=cfg.model.n_keyframes)

        def step_fn(m: nn.Module, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
            frames = batch["frames"].to(device, non_blocking=True)        # [B,T,3,H,W]
            frames = resize_preserve_aspect(frames, size=cfg.data.height)
            if flow_op.enabled:
                # Swin branch gets 5 ch (flow); EB3 branch always sees RGB only.
                frames_swin = torch.stack([flow_op(c) for c in frames])
                frames_swin = imagenet_normalize(frames_swin)
            else:
                frames_swin = imagenet_normalize(frames)
            frames_rgb_norm = imagenet_normalize(frames)                  # [B,T,3,...]

            swin_in = to_swin_layout(frames_swin)
            from pipelines.preprocess import to_efficientnet_layout
            eff_in  = to_efficientnet_layout(frames_rgb_norm, n_keyframes=cfg.model.n_keyframes)

            return m({"swin": swin_in, "efficientnet": eff_in}), \
                   batch["label"].to(device)

        return model, step_fn

    # ------------------------------------------------------------------
    raise ValueError(f"Unsupported model_type for gradient training: {mt!r}")


# ---------------------------------------------------------------------------
# Baseline (classical-ML) dispatch
# ---------------------------------------------------------------------------

def run_baselines(cfg: TrainConfig, dm: CHIRPDataModule) -> dict:
    """Train RF / XGB / KNN on frozen EB3 features.

    ``cfg.model.model_type`` selects which one to fit (other two are
    skipped). Returns a dict with at least ``val_acc`` and ``val_f1``
    keys for the chosen baseline.
    """
    which: dict[str, str] = {
        "baseline_rf":  "rf",
        "baseline_xgb": "xgb",
        "baseline_knn": "knn",
    }
    name = which[cfg.model.model_type]

    encoder = EfficientNetB3Encoder(pretrained=cfg.model.pretrained, freeze=True)
    device  = "cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu"

    logger.info("Extracting EB3 features for baseline=%s …", name)
    X_tr, y_tr = extract_features(
        encoder, dm.train_dataloader(),
        device=device, n_keyframes=cfg.model.n_keyframes,
    )
    X_va, y_va = extract_features(
        encoder, dm.val_dataloader(),
        device=device, n_keyframes=cfg.model.n_keyframes,
    )
    results = train_baselines(
        X_tr, y_tr, X_va, y_va,
        out_dir=Path(cfg.output_dir) / "baselines",
        which=(name,),
    )
    return {"val_acc": results[name]["accuracy"], "val_f1": results[name]["macro_f1"]}


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

RESULTS_FIELDS = [
    "timestamp", "run_name", "model_type", "use_optical_flow",
    "freeze_backbone", "num_frames", "n_keyframes", "hidden_dim", "dropout",
    "lr", "batch_size", "num_epochs_run",
    "best_val_acc", "best_val_f1", "best_val_loss",
    "test_acc", "test_f1", "test_loss",
    "checkpoint_path",
]


def append_results(cfg: TrainConfig, row: dict) -> Path:
    """Append a single-row summary to ``cfg.log.results_csv``."""
    path = Path(cfg.log.results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
    logger.info("Appended results row → %s", path)
    return path


# ---------------------------------------------------------------------------
# run_experiment — programmatic entry point used by main() AND ablations
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: TrainConfig,
    run_name: str | None = None,
    *,
    skip_test: bool = False,
    write_results_csv: bool = True,
) -> dict:
    """Run a single CHIRP experiment from an already-built config.

    Returns a metrics dict with the same keys written to the results CSV
    (best_val_acc, best_val_f1, test_acc, test_f1, num_epochs_run, etc.).
    Suitable for programmatic sweeps — see ``experiments/run_ablations.py``.
    """
    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(Path(cfg.output_dir) / "config.yaml")        # snapshot

    if run_name is None:
        run_name = f"{cfg.model.model_type}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    logger.info("Run %s | model=%s | output=%s",
                run_name, cfg.model.model_type, cfg.output_dir)

    # ---- data --------------------------------------------------------
    dm = CHIRPDataModule(
        data_root=cfg.data.data_root,
        datasets=tuple(cfg.data.datasets),
        splits=tuple(cfg.data.splits),
        seed=cfg.seed,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        n_frames=cfg.data.num_frames,
        height=cfg.data.height,
        width=cfg.data.width,
        backend=cfg.data.backend,
        balance_sampler=cfg.data.balance_sampler,
        train_transform=build_augment("train", {"augment": True}),
        eval_transform =build_augment("val",   {"augment": True}),
    )
    dm.setup()
    dm.plot_class_distribution(Path(cfg.output_dir) / "class_distribution.png")

    # ---- baseline branch --------------------------------------------
    if cfg.model.model_type.startswith("baseline_"):
        bmetrics = run_baselines(cfg, dm)
        row = {
            "timestamp":       dt.datetime.utcnow().isoformat(timespec="seconds"),
            "run_name":        run_name,
            "model_type":      cfg.model.model_type,
            "use_optical_flow": cfg.data.use_optical_flow,
            "freeze_backbone": True,
            "num_frames":      cfg.data.num_frames,
            "n_keyframes":     cfg.model.n_keyframes,
            "hidden_dim":      0,
            "dropout":         0.0,
            "lr":              0.0,
            "batch_size":      cfg.data.batch_size,
            "num_epochs_run":  0,
            "best_val_acc":    bmetrics["val_acc"],
            "best_val_f1":     bmetrics["val_f1"],
            "best_val_loss":   float("nan"),
            "test_acc":        float("nan"),
            "test_f1":         float("nan"),
            "test_loss":       float("nan"),
            "checkpoint_path": str(Path(cfg.output_dir) / "baselines"),
        }
        if write_results_csv:
            append_results(cfg, row)
        return row

    # ---- gradient-training branch -----------------------------------
    device = torch.device(
        "cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    model, step_fn = build_model_and_step(cfg, device)
    trainer = Trainer(
        model, cfg, step_fn=step_fn,
        class_weights=dm.class_weights, device=device,
    )
    history = trainer.fit(dm.train_dataloader(), dm.val_dataloader())

    # ---- final eval --------------------------------------------------
    trainer.load_best()
    best_idx = max(range(len(history)), key=lambda i: history[i].val_f1)
    best = history[best_idx]

    test_loss = test_acc = test_f1 = float("nan")
    if not skip_test:
        test_loss, test_acc, test_f1 = trainer.evaluate(
            dm.test_dataloader(), desc="test",
        )
        logger.info("TEST  loss=%.4f  acc=%.4f  f1=%.4f", test_loss, test_acc, test_f1)

    row = {
        "timestamp":       dt.datetime.utcnow().isoformat(timespec="seconds"),
        "run_name":        run_name,
        "model_type":      cfg.model.model_type,
        "use_optical_flow": cfg.data.use_optical_flow,
        "freeze_backbone": cfg.model.freeze_backbone,
        "num_frames":      cfg.data.num_frames,
        "n_keyframes":     cfg.model.n_keyframes,
        "hidden_dim":      cfg.model.hidden_dim,
        "dropout":         cfg.model.dropout,
        "lr":              cfg.optim.lr,
        "batch_size":      cfg.data.batch_size,
        "num_epochs_run":  len(history),
        "best_val_acc":    best.val_acc,
        "best_val_f1":     best.val_f1,
        "best_val_loss":   best.val_loss,
        "test_acc":        test_acc,
        "test_f1":         test_f1,
        "test_loss":       test_loss,
        "checkpoint_path": str(trainer.ckpt_dir / "best.pt"),
    }
    if write_results_csv:
        append_results(cfg, row)
    return row


# ---------------------------------------------------------------------------
# CLI main — just config parsing + dispatch to run_experiment
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CHIRP training entry point.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Override config fields, e.g. optim.lr=1e-3 model.dropout=0.4")
    parser.add_argument("--run-name", default=None,
                        help="Override run name (defaults to a timestamp).")
    parser.add_argument("--skip-test", action="store_true",
                        help="Don't evaluate on the test set at the end.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = TrainConfig.from_yaml(args.config).apply_overrides(args.override)
    run_experiment(cfg, run_name=args.run_name, skip_test=args.skip_test)


if __name__ == "__main__":
    main()
