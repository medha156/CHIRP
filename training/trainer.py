"""
training/trainer.py
===================
Training loop for CHIRP.

Responsibilities
----------------
- AdamW optimiser with weight-decay groups (no decay on bias / LayerNorm).
- Cosine LR schedule with linear warm-up for the first ``warmup_epochs``.
- Gradient clipping (``clip_grad_norm_``) — 0 disables.
- Early stopping on ``val_f1`` (or ``val_loss``) with configurable patience.
- WandB logging of train_loss, val_loss, val_acc, val_f1, lr per epoch.
- Best-checkpoint persistence to ``<output_dir>/checkpoints/best.pt`` with
  the full state needed to resume:
  ``{"model": ..., "optimizer": ..., "scheduler": ..., "epoch": ..., "metrics": ...}``.

The trainer is **model-agnostic** — it doesn't know whether you're
training Swin-T, EfficientNet, or the fusion ensemble. You give it a
``step_fn(model, batch) → (logits, labels)`` callable; default adapters
for the three model types live in :mod:`training.train`.
"""

from __future__ import annotations

# Direct-script execution support
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.config import TrainConfig  # noqa: E402

logger = logging.getLogger(__name__)

StepFn = Callable[[nn.Module, dict], tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# Optimiser + scheduler helpers
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> AdamW:
    """AdamW with weight-decay disabled on bias and normalisation layers."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    return AdamW(
        [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """Linear warm-up (0 → 1) followed by half-cosine decay (1 → min_lr_ratio)."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Bookkeeping containers
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    epoch:       int
    train_loss:  float
    val_loss:    float
    val_acc:     float
    val_f1:      float
    lr:          float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """End-to-end training loop with early stopping + WandB."""

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        step_fn: StepFn,
        *,
        class_weights: torch.Tensor | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg     = cfg
        self.model   = model
        self.step_fn = step_fn

        # ---- device -------------------------------------------------------
        if device is None:
            device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # ---- loss ---------------------------------------------------------
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg.optim.label_smoothing,
        )

        # ---- paths --------------------------------------------------------
        self.output_dir = Path(cfg.output_dir)
        self.ckpt_dir   = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ---- state --------------------------------------------------------
        self.history:        list[EpochMetrics] = []
        self.best_metric:    float | None = None
        self._best_is_lower_better = (cfg.early_stopping_metric == "val_loss")
        self._stale_epochs:  int = 0

        # ---- wandb (lazy) -------------------------------------------------
        self._wandb = None
        if cfg.log.wandb and not cfg.model.model_type.startswith("baseline_"):
            try:
                import wandb
                self._wandb = wandb.init(
                    project = cfg.log.wandb_project,
                    name    = cfg.log.wandb_run_name,
                    entity  = cfg.log.wandb_entity,
                    config  = cfg.to_dict(),
                    dir     = str(self.output_dir),
                    reinit  = True,
                )
                logger.info("WandB run initialised: %s", self._wandb.name)
            except ImportError:
                logger.warning("wandb not installed — logging disabled.")
            except Exception as exc:                       # offline / no API key
                logger.warning("wandb.init failed (%s) — logging disabled.", exc)

        logger.info("Trainer ready | device=%s | output=%s", self.device, self.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> list[EpochMetrics]:
        """Run the full training loop. Returns per-epoch metrics history."""
        steps_per_epoch = max(1, len(train_loader))
        total_steps     = steps_per_epoch * self.cfg.num_epochs
        warmup_steps    = steps_per_epoch * self.cfg.optim.warmup_epochs
        min_lr_ratio    = self.cfg.optim.min_lr / max(self.cfg.optim.lr, 1e-12)

        self.optimizer = build_optimizer(
            self.model, self.cfg.optim.lr, self.cfg.optim.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

        for epoch in range(1, self.cfg.num_epochs + 1):
            t0 = time.perf_counter()
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_loss, val_acc, val_f1 = self.evaluate(val_loader, desc=f"val   {epoch:03d}")
            lr_now = self.optimizer.param_groups[0]["lr"]

            metrics = EpochMetrics(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                val_acc=val_acc, val_f1=val_f1, lr=lr_now,
            )
            self.history.append(metrics)
            dt = time.perf_counter() - t0
            logger.info(
                "epoch %03d | train_loss=%.4f  val_loss=%.4f  "
                "val_acc=%.4f  val_f1=%.4f  lr=%.2e  (%.1fs)",
                epoch, train_loss, val_loss, val_acc, val_f1, lr_now, dt,
            )
            if self._wandb is not None:
                self._wandb.log({**metrics.as_dict(), "epoch_seconds": dt})

            improved = self._is_improvement(metrics)
            if improved:
                self._save_checkpoint(epoch, metrics)
                self._stale_epochs = 0
            else:
                self._stale_epochs += 1
                if self._stale_epochs >= self.cfg.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (%d stale epochs on %s).",
                        epoch, self._stale_epochs, self.cfg.early_stopping_metric,
                    )
                    break

        if self._wandb is not None:
            self._wandb.finish()

        return self.history

    # ------------------------------------------------------------------
    # Train / eval loops
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss, n_samples = 0.0, 0
        pbar = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
        for step, batch in enumerate(pbar):
            logits, labels = self.step_fn(self.model, batch)
            logits = logits.to(self.device)
            labels = labels.to(self.device)
            loss   = self.criterion(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.optim.grad_clip,
                )
            self.optimizer.step()
            self.scheduler.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n_samples  += bs

            if step % self.cfg.log.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        *,
        desc: str = "eval",
    ) -> tuple[float, float, float]:
        """Return ``(loss, accuracy, macro_f1)`` on ``loader``."""
        self.model.eval()
        total_loss, n_samples = 0.0, 0
        all_preds, all_labels = [], []
        for batch in tqdm(loader, desc=desc, leave=False):
            logits, labels = self.step_fn(self.model, batch)
            logits = logits.to(self.device)
            labels = labels.to(self.device)
            loss   = self.criterion(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n_samples  += bs

            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        import numpy as np
        preds = np.concatenate(all_preds) if all_preds else np.array([])
        gts   = np.concatenate(all_labels) if all_labels else np.array([])
        acc = accuracy_score(gts, preds) if len(gts) else 0.0
        f1  = f1_score(gts, preds, average="macro", zero_division=0) if len(gts) else 0.0
        return total_loss / max(n_samples, 1), float(acc), float(f1)

    # ------------------------------------------------------------------
    # Early-stopping bookkeeping
    # ------------------------------------------------------------------

    def _is_improvement(self, m: EpochMetrics) -> bool:
        current = m.val_loss if self._best_is_lower_better else m.val_f1
        if self.best_metric is None:
            self.best_metric = current
            return True
        better = (current < self.best_metric) if self._best_is_lower_better \
                 else (current > self.best_metric)
        if better:
            self.best_metric = current
        return better

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, metrics: EpochMetrics) -> Path:
        path = self.ckpt_dir / "best.pt"
        torch.save({
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "epoch":      epoch,
            "metrics":    metrics.as_dict(),
            "config":     self.cfg.to_dict(),
        }, path)
        logger.info(
            "Saved best checkpoint → %s (epoch %d, %s=%.4f)",
            path, epoch, self.cfg.early_stopping_metric,
            metrics.val_loss if self._best_is_lower_better else metrics.val_f1,
        )
        return path

    def load_best(self) -> Optional[dict]:
        """Load the best checkpoint into the model and return its metadata."""
        path = self.ckpt_dir / "best.pt"
        if not path.exists():
            logger.warning("No checkpoint found at %s", path)
            return None
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        logger.info("Restored model from %s (epoch %d)", path, ckpt["epoch"])
        return ckpt


# ---------------------------------------------------------------------------
# CLI smoke-test — tiny MLP on synthetic features
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from training.config import TrainConfig

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 1) Synthetic 20-class data
    torch.manual_seed(0)
    N, D, C = 256, 64, 20
    Xtr, ytr = torch.randn(N, D), torch.randint(0, C, (N,))
    Xva, yva = torch.randn(64, D), torch.randint(0, C, (64,))
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True)
    va = DataLoader(TensorDataset(Xva, yva), batch_size=16)

    # 2) Tiny model
    model = nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, C))

    # 3) Config
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(
            num_epochs=4,
            output_dir=tmp,
            early_stopping_patience=99,        # don't stop in smoke test
        )
        cfg.optim.lr = 1e-3
        cfg.optim.warmup_epochs = 1
        cfg.log.wandb = False

        # 4) step_fn for tuple-style TensorDataset
        def step_fn(m: nn.Module, batch) -> tuple[torch.Tensor, torch.Tensor]:
            x, y = batch
            return m(x), y

        trainer = Trainer(model, cfg, step_fn=step_fn, device="cpu")
        history = trainer.fit(tr, va)

        assert len(history) == 4
        assert (Path(tmp) / "checkpoints" / "best.pt").exists()
        print("\n  ✓ trained 4 epochs, best checkpoint written")

        meta = trainer.load_best()
        assert meta is not None and "epoch" in meta
        print(f"  ✓ load_best returned epoch={meta['epoch']}, "
              f"val_f1={meta['metrics']['val_f1']:.4f}")

    print("\nOK")
