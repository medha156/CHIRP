"""
models/fusion.py
================
Late-fusion head for CHIRP.

Concatenates the 768-D Video Swin-T embedding with the 1536-D
EfficientNet-B3 embedding (2304-D total) and passes the result through a
two-layer MLP:

    [B, 2304]
        │
        ▼  Linear(2304, hidden_dim)
        │  GELU  +  Dropout(p)
        │
        ▼  Linear(hidden_dim, num_classes)
        │
    [B, 20]  (logits — apply ``softmax`` outside for probabilities)

Both ``hidden_dim`` (default 512) and ``dropout`` (default 0.3) are
exposed as constructor arguments and reflected in the module's repr so
they show up in experiment manifests.

For maximum flexibility ``FusionHead.forward`` accepts the two embeddings
either as separate positional arguments or as a pre-concatenated tensor —
useful if your training loop already does the concat.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard embedding sizes (kept in sync with the backbones)
# ---------------------------------------------------------------------------

SWIN_EMBED_DIM:    int = 768
EFFNET_EMBED_DIM:  int = 1536
FUSED_EMBED_DIM:   int = SWIN_EMBED_DIM + EFFNET_EMBED_DIM   # 2304


# ---------------------------------------------------------------------------
# FusionHead
# ---------------------------------------------------------------------------

class FusionHead(nn.Module):
    """2-layer MLP that fuses Swin-T and EfficientNet-B3 embeddings.

    Parameters
    ----------
    num_classes:
        Output classes. Default 20.
    hidden_dim:
        Width of the hidden layer. Default 512.
    dropout:
        Dropout probability applied after the hidden activation.
        Default 0.3.
    swin_dim, effnet_dim:
        Embedding widths. Override only if you swap in a different
        backbone variant.
    activation:
        Non-linearity between the two ``Linear`` layers. Accepts
        ``"gelu"`` (default), ``"relu"``, or ``"silu"``.
    """

    def __init__(
        self,
        num_classes:  int = 20,
        hidden_dim:   int = 512,
        dropout:      float = 0.3,
        swin_dim:     int = SWIN_EMBED_DIM,
        effnet_dim:   int = EFFNET_EMBED_DIM,
        activation:   str = "gelu",
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0; got {hidden_dim}")

        act_map = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
        if activation not in act_map:
            raise ValueError(
                f"activation must be one of {list(act_map)}; got {activation!r}"
            )

        self.num_classes = num_classes
        self.hidden_dim  = hidden_dim
        self.dropout_p   = dropout
        self.swin_dim    = swin_dim
        self.effnet_dim  = effnet_dim
        self.in_features = swin_dim + effnet_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, hidden_dim),
            act_map[activation](),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        logger.info(
            "FusionHead | in=%d (swin=%d + effnet=%d) | hidden=%d | "
            "dropout=%.2f | activation=%s | classes=%d",
            self.in_features, swin_dim, effnet_dim, hidden_dim,
            dropout, activation, num_classes,
        )

    # ------------------------------------------------------------------

    def forward(self, *args: Tensor) -> Tensor:
        """Accept either ``(swin_emb, effnet_emb)`` or a single fused tensor.

        Examples
        --------
        >>> head = FusionHead()
        >>> s = torch.randn(4, 768)
        >>> e = torch.randn(4, 1536)
        >>> head(s, e).shape
        torch.Size([4, 20])
        >>> head(torch.cat([s, e], dim=-1)).shape
        torch.Size([4, 20])
        """
        if len(args) == 1:
            fused = args[0]
            expected = self.in_features
        elif len(args) == 2:
            swin, eff = args
            self._check_emb(swin, self.swin_dim,   "swin")
            self._check_emb(eff,  self.effnet_dim, "effnet")
            fused = torch.cat([swin, eff], dim=-1)
            expected = self.in_features
        else:
            raise TypeError(
                f"FusionHead.forward takes 1 or 2 tensors; got {len(args)}."
            )

        if fused.ndim != 2 or fused.shape[-1] != expected:
            raise ValueError(
                f"Fused tensor must have shape [B, {expected}]; "
                f"got {tuple(fused.shape)}"
            )
        return self.mlp(fused)

    # ------------------------------------------------------------------

    @staticmethod
    def _check_emb(t: Tensor, dim: int, name: str) -> None:
        if t.ndim != 2 or t.shape[-1] != dim:
            raise ValueError(
                f"{name} embedding must be shape [B, {dim}]; "
                f"got {tuple(t.shape)}"
            )

    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden_dim={self.hidden_dim}, "
            f"dropout={self.dropout_p}, num_classes={self.num_classes}"
        )


# ---------------------------------------------------------------------------
# Convenience: a full Swin + EB3 + Fusion wrapper
# ---------------------------------------------------------------------------

class SwinEffNetFusion(nn.Module):
    """End-to-end model: Swin-T + EB3 + FusionHead.

    Useful when you want a single ``nn.Module`` to pass to a training
    loop. The two sub-models still expose ``extract_embeddings``, so
    feature caching for the classical-ML baselines remains easy.

    Forward
    -------
    Takes a dict with two keys (matching ``Preprocessor.__call__`` output):

        {
            "swin":         [B, C, T, H, W],
            "efficientnet": [B·K, 3, H, W],   # K keyframes per clip
        }
    """

    def __init__(
        self,
        swin: nn.Module,
        effnet: nn.Module,
        fusion: FusionHead,
        keyframes_per_clip: int,
    ) -> None:
        super().__init__()
        self.swin   = swin
        self.effnet = effnet
        self.fusion = fusion
        self.k      = keyframes_per_clip

    def forward(self, batch: dict) -> Tensor:
        swin_emb   = self.swin.extract_embeddings(batch["swin"])         # [B, 768]
        effnet_emb = self.effnet(batch["efficientnet"], t=self.k)        # [B, 1536]
        return self.fusion(swin_emb, effnet_emb)                         # [B, 20]


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Default config — eval() so dropout doesn't randomise the two paths
    head = FusionHead().eval()
    s = torch.randn(4, 768)
    e = torch.randn(4, 1536)

    out_two = head(s, e)
    out_one = head(torch.cat([s, e], dim=-1))
    print(f"  two-arg forward → {tuple(out_two.shape)}")
    print(f"  one-arg forward → {tuple(out_one.shape)}")
    assert out_two.shape == (4, 20) == out_one.shape
    assert torch.allclose(out_two, out_one)
    print("  ✓ both call signatures equivalent")

    # Override hyper-params
    head2 = FusionHead(num_classes=20, hidden_dim=1024, dropout=0.5, activation="relu")
    print(f"\n  custom head: {head2}")
    assert head2(s, e).shape == (4, 20)

    # Validation errors fire correctly
    try:
        FusionHead(dropout=1.5)
    except ValueError as ex:
        print(f"  ✓ dropout validation: {ex}")

    try:
        head(torch.randn(4, 800), e)
    except ValueError as ex:
        print(f"  ✓ shape validation: {ex}")

    print("\nOK")
