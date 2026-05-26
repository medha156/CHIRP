"""
models/swin_t.py
================
Video Swin-T backbone for CHIRP.

Source priority
---------------
1. ``torchvision.models.video.swin3d_t`` with Kinetics-400 weights
   (``Swin3D_T_Weights.KINETICS400_V1``). This is the canonical pure-PyTorch
   implementation and ships with torchvision >= 0.16.
2. ``timm`` fallback — searched at construction time. (timm does not yet
   carry Video Swin-T as of v1.0; the hook is here for future versions.)

Key features
------------
- 20-class classification head (configurable).
- Optional **optical-flow input**: when ``in_channels=5`` the first patch
  embedding conv is replaced with a new ``Conv3d(5, 96, (2,4,4))`` whose
  RGB weights are copied from the pretrained model and whose extra 2 flow
  channels are initialised by averaging the RGB kernels (channel inflation
  trick from *Carreira & Zisserman 2017*).
- ``extract_embeddings(x)`` returns the pre-head feature vector
  ``[B, 768]`` for fusion or downstream classical-ML baselines.

Input shape
-----------
``[B, C, T, H, W]`` with ``C ∈ {3, 5}``, ``T=16``, ``H=W=224``.
This matches the layout produced by ``pipelines.preprocess.to_swin_layout``.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backbone loader
# ---------------------------------------------------------------------------

def _load_torchvision_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    """Return (backbone, feature_dim).

    The backbone keeps its built-in classification head; the caller is
    responsible for replacing ``backbone.head``.
    """
    from torchvision.models.video import Swin3D_T_Weights, swin3d_t

    weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
    backbone = swin3d_t(weights=weights, progress=False)
    feat_dim = backbone.head.in_features          # 768 for swin3d_t
    logger.info(
        "Loaded torchvision swin3d_t (pretrained=%s, feat_dim=%d)",
        pretrained, feat_dim,
    )
    return backbone, feat_dim


def _load_timm_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    """Future-proof hook for timm-hosted Video Swin-T. Raises today."""
    import timm  # noqa: F401  (presence check)
    raise RuntimeError(
        "timm does not currently host Video Swin-T weights. "
        "Use source='torchvision' (the default)."
    )


# ---------------------------------------------------------------------------
# Channel inflation for optical-flow input
# ---------------------------------------------------------------------------

def _inflate_patch_embed(
    proj: nn.Conv3d,
    new_in_channels: int,
) -> nn.Conv3d:
    """Replace a 3-channel patch-embed Conv3d with one accepting more channels.

    The original RGB weights are preserved verbatim; each extra channel is
    initialised with the **mean of the RGB kernels**, scaled by
    ``3 / new_in_channels`` so the activation magnitude is unchanged.
    """
    if proj.in_channels >= new_in_channels:
        return proj                                # nothing to do

    new_proj = nn.Conv3d(
        in_channels=new_in_channels,
        out_channels=proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=proj.bias is not None,
    )
    with torch.no_grad():
        # Copy original RGB kernels.
        new_proj.weight[:, :proj.in_channels] = proj.weight
        # Channel inflation: mean of RGB kernels for the extra channels.
        if new_in_channels > proj.in_channels:
            extra = new_in_channels - proj.in_channels
            mean_kernel = proj.weight.mean(dim=1, keepdim=True)          # [O,1,t,h,w]
            new_proj.weight[:, proj.in_channels:] = mean_kernel.expand(
                -1, extra, -1, -1, -1
            )
        # Rescale so total input variance is preserved.
        new_proj.weight.mul_(proj.in_channels / new_in_channels)

        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias)

    logger.info(
        "Inflated patch_embed: in_channels %d → %d (mean-init + rescale)",
        proj.in_channels, new_in_channels,
    )
    return new_proj


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class VideoSwinT(nn.Module):
    """Video Swin-T wrapper with a CHIRP 20-class head.

    Parameters
    ----------
    num_classes:
        Output classes. Default 20 (matches ``video_dataset.NUM_CLASSES``).
    in_channels:
        ``3`` (RGB only) or ``5`` (RGB + optical-flow u, v).
    pretrained:
        Load Kinetics-400 weights. Disable for fast tests / fresh training.
    source:
        Backbone provider — currently only ``"torchvision"`` is functional.

    Forward
    -------
    Input  ``[B, C, T, H, W]``  →  Output ``[B, num_classes]`` logits.
    Apply ``softmax`` externally if you need probabilities; loss functions
    such as ``nn.CrossEntropyLoss`` expect raw logits.
    """

    FEATURE_DIM: int = 768

    def __init__(
        self,
        num_classes: int = 20,
        in_channels: int = 3,
        pretrained: bool = True,
        source: Literal["torchvision", "timm"] = "torchvision",
    ) -> None:
        super().__init__()

        if in_channels not in (3, 5):
            raise ValueError(
                f"in_channels must be 3 (RGB) or 5 (RGB+flow); got {in_channels}"
            )

        if source == "torchvision":
            self.backbone, feat_dim = _load_torchvision_backbone(pretrained)
        elif source == "timm":
            self.backbone, feat_dim = _load_timm_backbone(pretrained)
        else:
            raise ValueError(f"Unknown source {source!r}")

        # ---- inflate input channels if needed ----------------------------
        if in_channels != 3:
            self.backbone.patch_embed.proj = _inflate_patch_embed(
                self.backbone.patch_embed.proj, in_channels
            )

        # ---- replace classification head ---------------------------------
        self.backbone.head = nn.Linear(feat_dim, num_classes)

        self.num_classes = num_classes
        self.in_channels = in_channels
        logger.info(
            "VideoSwinT | classes=%d | in_channels=%d | source=%s | pretrained=%s",
            num_classes, in_channels, source, pretrained,
        )

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """``[B, C, T, H, W]`` → logits ``[B, num_classes]``."""
        self._check_input(x)
        return self.backbone(x)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_embeddings(self, x: Tensor) -> Tensor:
        """Return the 768-D pre-head feature vector for fusion / baselines.

        We swap the classification ``Linear`` head with ``nn.Identity``
        for the duration of the call so we get back the pooled features
        the head would normally consume. This avoids hard-coding the
        backbone's internal layer names (which differ between
        torchvision and timm).
        """
        self._check_input(x)
        original_head = self.backbone.head
        self.backbone.head = nn.Identity()
        try:
            return self.backbone(x)                  # [B, 768]
        finally:
            self.backbone.head = original_head

    # ------------------------------------------------------------------

    def _check_input(self, x: Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B,C,T,H,W] tensor, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input has {x.shape[1]} channels but model expects "
                f"{self.in_channels}. Did you toggle optical flow correctly?"
            )

    # ------------------------------------------------------------------

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze (or unfreeze) every parameter except the classification head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("head."):
                param.requires_grad = not freeze
        logger.info("VideoSwinT backbone frozen=%s", freeze)


# ---------------------------------------------------------------------------
# CLI smoke-test (architecture only — no weight download)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for c in (3, 5):
        model = VideoSwinT(num_classes=20, in_channels=c, pretrained=False).eval()
        x = torch.randn(2, c, 16, 224, 224)
        logits = model(x)
        emb    = model.extract_embeddings(x)
        print(
            f"  in_channels={c}: logits={tuple(logits.shape)}  "
            f"embeddings={tuple(emb.shape)}"
        )
        assert logits.shape == (2, 20)
        assert emb.shape == (2, 768)

    print("\nOK")
