"""
training/config.py
==================
Dataclass-based configuration for CHIRP training runs.

A single :class:`TrainConfig` exposes every knob the rest of the project
reads (data paths, model hyper-params, optimiser, scheduling, logging).
Configs are persisted as YAML for reproducibility and can be round-tripped
losslessly via ``TrainConfig.from_yaml`` / ``TrainConfig.to_yaml``.

Why dataclasses instead of Hydra
--------------------------------
- Zero extra dependency (PyYAML is already required for I/O).
- Static type checking via ``mypy`` works out of the box.
- Trivial to introspect and serialise (``asdict``).

The trade-off is no command-line override syntax. ``train.py`` supports
``--override key=value`` for the handful of fields you typically tweak per
run.

Schema
------
The config is grouped into four sub-sections::

    TrainConfig
    ├── data:   DataConfig
    ├── model:  ModelConfig
    ├── optim:  OptimConfig
    └── log:    LogConfig

Top-level fields cover run-wide settings (output dir, seed, num_epochs).

Example YAML
------------
::

    seed: 42
    num_epochs: 50
    output_dir: outputs/runs/baseline
    early_stopping_patience: 10

    data:
      data_root: data/
      datasets: [fbd_sv_2024, vb100]
      batch_size: 16
      num_workers: 4
      num_frames: 16
      use_optical_flow: false
      optical_flow_backend: farneback

    model:
      model_type: fusion          # swin | efficientnet | fusion | baseline_rf | baseline_xgb
      pretrained: true
      freeze_backbone: false
      hidden_dim: 512
      dropout: 0.3
      n_keyframes: 4

    optim:
      lr: 1.0e-4
      weight_decay: 5.0e-2
      warmup_epochs: 3
      grad_clip: 1.0
      label_smoothing: 0.1

    log:
      wandb: false
      wandb_project: chirp
      wandb_run_name: null
      results_csv: outputs/results.csv
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)

ModelType = Literal["swin", "efficientnet", "fusion", "baseline_rf", "baseline_xgb", "baseline_knn"]


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    data_root:            str  = "data/"
    datasets:             list[str] = field(default_factory=lambda: ["fbd_sv_2024", "vb100"])
    splits:               tuple[float, float, float] = (0.70, 0.15, 0.15)
    batch_size:           int  = 16
    num_workers:          int  = 4
    num_frames:           int  = 16
    height:               int  = 224
    width:                int  = 224
    backend:              str  = "auto"       # decord | torchvision | auto
    balance_sampler:      bool = False
    use_optical_flow:     bool = False
    optical_flow_backend: str  = "farneback"  # raft | farneback
    raft_model:           str  = "small"      # small | large


@dataclass
class ModelConfig:
    model_type:      ModelType = "fusion"
    pretrained:      bool      = True
    freeze_backbone: bool      = False
    hidden_dim:      int       = 512          # fusion only
    dropout:         float     = 0.3          # fusion only
    n_keyframes:     int       = 4            # frames fed to EfficientNet
    num_classes:     int       = 20
    pool:            str       = "mean"       # mean | max | attention (efficientnet)


@dataclass
class OptimConfig:
    lr:              float = 1.0e-4
    weight_decay:    float = 5.0e-2
    warmup_epochs:   int   = 3
    min_lr:          float = 1.0e-6
    grad_clip:       float = 1.0              # 0 disables
    label_smoothing: float = 0.1


@dataclass
class LogConfig:
    wandb:          bool          = False
    wandb_project:  str           = "chirp"
    wandb_run_name: str | None    = None
    wandb_entity:   str | None    = None
    results_csv:    str           = "outputs/results.csv"
    log_every:      int           = 20        # steps between train-loss prints


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # run-wide
    seed:                    int  = 42
    num_epochs:              int  = 50
    output_dir:              str  = "outputs/runs/default"
    early_stopping_patience: int  = 10        # epochs of no val-F1 improvement
    early_stopping_metric:   str  = "val_f1"  # val_f1 | val_loss
    device:                  str  = "auto"    # cuda | cpu | auto

    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    log:   LogConfig   = field(default_factory=LogConfig)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.early_stopping_metric not in ("val_f1", "val_loss"):
            raise ValueError(
                f"early_stopping_metric must be 'val_f1' or 'val_loss'; "
                f"got {self.early_stopping_metric!r}"
            )
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.model.dropout}")
        if self.model.model_type.startswith("baseline_") and self.log.wandb:
            logger.warning(
                "WandB logging is ignored for baseline_* model types "
                "(they don't train via gradient descent)."
            )
        if self.data.use_optical_flow and self.model.model_type == "efficientnet":
            logger.warning(
                "use_optical_flow=True has no effect when model_type=efficientnet "
                "(EB3 only sees RGB keyframes)."
            )

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        """Build a config from a nested dict, falling back to defaults."""
        sub = {"data": DataConfig, "model": ModelConfig,
               "optim": OptimConfig, "log": LogConfig}

        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if k in sub:
                kwargs[k] = sub[k](**v) if v else sub[k]()
            else:
                kwargs[k] = v
        instance = cls(**kwargs)
        # YAML can't distinguish tuple from list — re-coerce annotated tuples.
        if isinstance(instance.data.splits, list):
            instance.data.splits = tuple(instance.data.splits)  # type: ignore[assignment]
        return instance

    def to_yaml(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)
        return path

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    # ------------------------------------------------------------------
    # CLI overrides
    # ------------------------------------------------------------------

    def apply_overrides(self, overrides: list[str]) -> TrainConfig:
        """Apply ``key=value`` strings to nested fields.

        Supports dotted paths into sub-configs, e.g.::

            cfg.apply_overrides(["optim.lr=1e-3", "model.dropout=0.5"])

        Values are parsed as YAML so booleans, floats, and lists work
        without quoting.
        """
        d = self.to_dict()
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"Override {ov!r} must be of form key=value")
            key, raw_val = ov.split("=", 1)
            value = yaml.safe_load(raw_val)
            target = d
            *parents, leaf = key.split(".")
            for p in parents:
                if p not in target:
                    raise KeyError(f"Unknown config section {p!r} (in override {ov!r})")
                target = target[p]
            if leaf not in target:
                raise KeyError(f"Unknown config key {key!r}")
            target[leaf] = value
        return type(self).from_dict(d)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = TrainConfig()
    print(f"Default: {cfg.model.model_type=}, {cfg.optim.lr=}, "
          f"{cfg.data.batch_size=}, {cfg.data.use_optical_flow=}")

    # Round-trip via YAML
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        cfg.to_yaml(f.name)
        roundtrip = TrainConfig.from_yaml(f.name)
    assert roundtrip == cfg, "YAML round-trip changed the config"
    print("  ✓ YAML round-trip")

    # CLI overrides
    cfg2 = cfg.apply_overrides([
        "optim.lr=0.001",
        "model.dropout=0.4",
        "data.use_optical_flow=true",
        "num_epochs=5",
    ])
    assert cfg2.optim.lr == 0.001
    assert cfg2.model.dropout == 0.4
    assert cfg2.data.use_optical_flow is True
    assert cfg2.num_epochs == 5
    assert cfg.optim.lr == 1e-4, "apply_overrides mutated original config"
    print("  ✓ CLI overrides applied without mutating original")

    # Unknown key fails loudly
    try:
        cfg.apply_overrides(["optim.bogus=1"])
    except KeyError as ex:
        print(f"  ✓ Unknown key rejected: {ex}")

    # Validation catches bad values
    try:
        TrainConfig(early_stopping_metric="nonsense")
    except ValueError as ex:
        print(f"  ✓ Validation rejects bad value: {ex}")

    print("\nOK")
