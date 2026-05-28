"""Unit tests for TrainConfig YAML I/O and the Trainer loop."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.config import TrainConfig
from training.trainer import (
    EpochMetrics,
    Trainer,
    build_optimizer,
    build_scheduler,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_yaml_roundtrip(tmp_path):
    cfg = TrainConfig()
    out = tmp_path / "cfg.yaml"
    cfg.to_yaml(out)
    loaded = TrainConfig.from_yaml(out)
    assert loaded == cfg


def test_config_apply_overrides_does_not_mutate_original():
    cfg = TrainConfig()
    cfg2 = cfg.apply_overrides(["optim.lr=0.001", "model.dropout=0.4"])
    assert cfg2.optim.lr == 0.001
    assert cfg2.model.dropout == 0.4
    assert cfg.optim.lr == 1e-4
    assert cfg.model.dropout == 0.3


def test_config_apply_overrides_yaml_parsed_values():
    cfg = TrainConfig()
    cfg2 = cfg.apply_overrides([
        "data.use_optical_flow=true",
        "data.datasets=[fbd_sv_2024]",
    ])
    assert cfg2.data.use_optical_flow is True
    assert cfg2.data.datasets == ["fbd_sv_2024"]


def test_config_apply_overrides_rejects_unknown_key():
    cfg = TrainConfig()
    with pytest.raises(KeyError):
        cfg.apply_overrides(["optim.does_not_exist=1"])


def test_config_validation_rejects_bad_metric():
    with pytest.raises(ValueError):
        TrainConfig(early_stopping_metric="nonsense")


def test_config_pool_field_exists():
    """Sanity check: ModelConfig.pool added in the ablations work landed."""
    assert TrainConfig().model.pool == "mean"


# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------

def test_optimizer_splits_decay_groups():
    model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10), nn.Linear(10, 5))
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.05)
    # Two groups: decay + no_decay (bias / LayerNorm / 1-D params).
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.05
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_scheduler_warmup_then_cosine():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = build_scheduler(opt, total_steps=100, warmup_steps=10, min_lr_ratio=0.01)
    # Step 0 → lr * 1/10
    lr_step_0 = opt.param_groups[0]["lr"]
    sched.step()
    # After warmup (step 10), lr should be back near base.
    for _ in range(9):
        sched.step()
    lr_after_warmup = opt.param_groups[0]["lr"]
    # End of schedule → near min
    for _ in range(89):
        sched.step()
    lr_end = opt.param_groups[0]["lr"]
    assert lr_step_0 < lr_after_warmup
    assert lr_end < lr_after_warmup
    assert lr_end == pytest.approx(1e-3 * 0.01, rel=0.1)


# ---------------------------------------------------------------------------
# Trainer (synthetic 2-epoch run)
# ---------------------------------------------------------------------------

def test_trainer_full_run_saves_best_checkpoint(tmp_path):
    torch.manual_seed(0)
    N, D, C = 128, 32, 20
    Xtr, ytr = torch.randn(N, D), torch.randint(0, C, (N,))
    Xva, yva = torch.randn(32, D), torch.randint(0, C, (32,))
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=8, shuffle=True)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=8)

    model = nn.Sequential(nn.Linear(D, 16), nn.GELU(), nn.Linear(16, C))

    cfg = TrainConfig(
        num_epochs=2,
        output_dir=str(tmp_path),
        early_stopping_patience=99,
    )
    cfg.optim.lr = 1e-3
    cfg.optim.warmup_epochs = 1
    cfg.log.wandb = False

    def step_fn(m, batch):
        x, y = batch
        return m(x), y

    trainer = Trainer(model, cfg, step_fn=step_fn, device="cpu")
    history = trainer.fit(tr, va)

    assert len(history) == 2
    assert all(isinstance(m, EpochMetrics) for m in history)
    assert (tmp_path / "checkpoints" / "best.pt").exists()


def test_trainer_load_best_restores_metadata(tmp_path):
    torch.manual_seed(1)
    Xtr, ytr = torch.randn(32, 8), torch.randint(0, 4, (32,))
    Xva, yva = torch.randn(8, 8), torch.randint(0, 4, (8,))
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=4)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=4)
    model = nn.Linear(8, 4)

    cfg = TrainConfig(num_epochs=1, output_dir=str(tmp_path),
                      early_stopping_patience=99)
    cfg.optim.lr, cfg.optim.warmup_epochs = 1e-3, 0
    cfg.log.wandb = False

    trainer = Trainer(model, cfg,
                      step_fn=lambda m, b: (m(b[0]), b[1]), device="cpu")
    trainer.fit(tr, va)
    meta = trainer.load_best()
    assert meta is not None
    assert meta["epoch"] == 1
    assert "metrics" in meta
