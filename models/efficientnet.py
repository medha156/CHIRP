"""
models/efficientnet.py
======================
EfficientNet-B3 frame encoder for CHIRP.

- Loads ``timm/efficientnet_b3`` (ImageNet pretrained by default) with the
  classifier removed (``num_classes=0``) so the backbone returns a 1536-D
  feature vector per frame.
- Encodes each frame independently, then averages embeddings over the
  ``T`` time dimension to produce a single ``[B, 1536]`` clip embedding.
- Optional ``freeze`` flag for fast baseline experiments where the
  backbone is treated as a fixed feature extractor.

Input shape
-----------
``[B, T, C, H, W]``  *or*  ``[B·T, C, H, W]`` (the flat layout produced by
``pipelines.preprocess.to_efficientnet_layout``). When given the flat
layout you must pass ``T`` explicitly so the module can un-flatten before
the temporal mean pool.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class EfficientNetB3Encoder(nn.Module):
    """Per-frame EfficientNet-B3 encoder with temporal mean pooling.

    Parameters
    ----------
    pretrained:
        Load ImageNet weights via timm. Disable for fast smoke tests.
    pool:
        Temporal pooling op applied across T frames. ``"mean"`` (default)
        matches CLAUDE.md; ``"max"`` is provided for ablation.
    freeze:
        Freeze all backbone parameters at construction time. Useful for
        the classical-ML baselines in ``models.baselines``.
    num_classes:
        If ``> 0``, a final ``Linear(1536, num_classes)`` head is appended
        and ``forward()`` returns logits. If ``0`` (default), ``forward()``
        returns the pooled 1536-D embedding directly — this is the mode
        consumed by the fusion head.

    Forward
    -------
    Input  ``[B, T, C, H, W]``  (or ``[B·T, C, H, W]`` with ``t=T``)
    Output ``[B, 1536]``        (or ``[B, num_classes]`` if a head is set)
    """

    FEATURE_DIM: int = 1536      # canonical EfficientNet-B3 output width

    def __init__(
        self,
        pretrained: bool = True,
        pool: Literal["mean", "max"] = "mean",
        freeze: bool = False,
        num_classes: int = 0,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for EfficientNetB3Encoder. "
                "Install with: pip install timm"
            ) from e

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,                    # global-pool features
            global_pool="avg",
        )
        actual_dim = self.backbone.num_features
        if actual_dim != self.FEATURE_DIM:
            logger.warning(
                "EfficientNet-B3 reported %d features (expected %d). "
                "Updating FEATURE_DIM for this instance.",
                actual_dim, self.FEATURE_DIM,
            )
            self.FEATURE_DIM = actual_dim

        if pool not in ("mean", "max"):
            raise ValueError(f"pool must be 'mean' or 'max'; got {pool!r}")
        self.pool = pool

        if freeze:
            self.freeze_backbone(True)

        self.head: Optional[nn.Linear] = None
        if num_classes > 0:
            self.head = nn.Linear(self.FEATURE_DIM, num_classes)

        self.num_classes = num_classes
        logger.info(
            "EfficientNetB3Encoder | pretrained=%s | pool=%s | "
            "freeze=%s | num_classes=%d",
            pretrained, pool, freeze, num_classes,
        )

    # ------------------------------------------------------------------

    def forward(self, x: Tensor, t: Optional[int] = None) -> Tensor:
        """Encode every frame, then pool across time.

        Parameters
        ----------
        x:
            Either ``[B, T, C, H, W]`` or the flat ``[B·T, C, H, W]``
            produced by ``pipelines.preprocess.to_efficientnet_layout``.
        t:
            Required when ``x`` is 4-D — the original number of frames per
            clip. Ignored when ``x`` is 5-D.

        Returns
        -------
        ``[B, 1536]`` if ``num_classes == 0`` else ``[B, num_classes]``.
        """
        if x.ndim == 5:
            b, T, c, h, w = x.shape
            flat = x.reshape(b * T, c, h, w)
        elif x.ndim == 4:
            if t is None:
                raise ValueError(
                    "When passing a flat [B·T, C, H, W] tensor you must "
                    "specify `t` so the module can un-flatten before pooling."
                )
            T = t
            bt = x.shape[0]
            if bt % T != 0:
                raise ValueError(
                    f"Batch dim {bt} is not divisible by t={T}; check layout."
                )
            b = bt // T
            flat = x
        else:
            raise ValueError(
                f"Expected 5-D [B,T,C,H,W] or 4-D [B*T,C,H,W] input, "
                f"got shape {tuple(x.shape)}"
            )

        # ---- per-frame encoding ------------------------------------------
        feats = self.backbone(flat)                       # [B·T, F]
        feats = feats.view(b, T, -1)                      # [B, T, F]

        # ---- temporal pool -----------------------------------------------
        if self.pool == "mean":
            pooled = feats.mean(dim=1)                    # [B, F]
        else:  # "max"
            pooled, _ = feats.max(dim=1)

        if self.head is not None:
            return self.head(pooled)                      # [B, num_classes]
        return pooled                                     # [B, F]

    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_embeddings(self, x: Tensor, t: Optional[int] = None) -> Tensor:
        """Always return the pooled ``[B, 1536]`` features, regardless of head."""
        head_backup = self.head
        self.head = None
        try:
            return self.forward(x, t=t)
        finally:
            self.head = head_backup

    # ------------------------------------------------------------------

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze (or unfreeze) every backbone parameter."""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        # head (if any) remains trainable
        logger.info("EfficientNetB3Encoder backbone frozen=%s", freeze)


# ---------------------------------------------------------------------------
# CLI smoke-test (architecture only — no weight download)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    enc = EfficientNetB3Encoder(pretrained=False, freeze=True).eval()
    print(f"FEATURE_DIM = {enc.FEATURE_DIM}")

    # 5-D input
    x5 = torch.randn(2, 4, 3, 224, 224)
    emb = enc(x5)
    print(f"  5-D input  {tuple(x5.shape)}  → embedding {tuple(emb.shape)}")
    assert emb.shape == (2, enc.FEATURE_DIM)

    # 4-D flat input
    x4 = torch.randn(2 * 4, 3, 224, 224)
    emb = enc(x4, t=4)
    print(f"  4-D input  {tuple(x4.shape)}  (t=4) → embedding {tuple(emb.shape)}")
    assert emb.shape == (2, enc.FEATURE_DIM)

    # With head
    enc_h = EfficientNetB3Encoder(pretrained=False, num_classes=20).eval()
    logits = enc_h(x5)
    print(f"  with head → logits {tuple(logits.shape)}")
    assert logits.shape == (2, 20)

    # extract_embeddings always returns pooled features
    raw = enc_h.extract_embeddings(x5)
    assert raw.shape == (2, enc_h.FEATURE_DIM)
    print(f"  extract_embeddings (head present) → {tuple(raw.shape)}")

    # Freeze check
    enc.freeze_backbone(True)
    n_trainable = sum(p.requires_grad for p in enc.parameters())
    assert n_trainable == 0, f"Expected 0 trainable params after freeze, got {n_trainable}"
    print(f"  freeze_backbone → trainable params = {n_trainable}")

    print("\nOK")
