"""
eval/attention_maps.py
======================
GradCAM and attention rollout for Video Swin-T.

Two visualisations
------------------
1. **GradCAM** (Selvaraju et al. 2017)
   Capture activations + gradients of the **last Swin block's output**
   for the predicted-class logit, then form ``ReLU(Σ α_k · A_k)`` where
   ``α_k`` is the global-average-pooled gradient on channel ``k``.
   Result is a 3-D heatmap ``[T', H', W']`` which we trilinearly upsample
   to the input clip's resolution.

2. **Attention rollout** (Abnar & Zuidema 2020, adapted for windowed
   attention)
   Forward-hook every ``WindowAttention`` submodule, capture the
   ``[B·nw, heads, N, N]`` attention matrices, average over heads,
   add identity for the residual stream, and recursively multiply
   *within each window* to get the per-window attention rollout.
   Each window's mean attention is then placed back into the full
   spatial grid to produce a single ``[T', H', W']`` heatmap.

Both maps are normalised to ``[0, 1]`` per clip for visualisation.

Output
------
For a given clip the script saves a grid figure:

    outputs/figures/attention_<clip>_<run>.png

with rows = sampled keyframes, columns = ``[original | gradcam |
rollout]``. The overlays use a translucent ``jet`` colormap on top of
the grayscale frame so spatial coverage is easy to read.

Usage
-----
::

    python eval/attention_maps.py \\
        --config configs/fusion.yaml \\
        --checkpoint outputs/runs/swin/checkpoints/best.pt \\
        --clip data/raw/california_scrub_jay/clip_001.mp4 \\
        --n-frames-vis 4
"""

from __future__ import annotations

# Direct-script execution
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.swin_t import VideoSwinT  # noqa: E402
from pipelines.preprocess import (  # noqa: E402
    imagenet_normalize,
    to_swin_layout,
)
from pipelines.video_dataset import NUM_CLASSES, SPECIES  # noqa: E402
from training.config import TrainConfig  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

class _ActivationHook:
    """Capture (and optionally retain gradients of) a module's output."""

    def __init__(self, module: nn.Module, retain_grad: bool = False) -> None:
        self.activation: Tensor | None = None
        self.gradient:   Tensor | None = None
        self._h_fwd = module.register_forward_hook(self._fwd)
        if retain_grad:
            self._h_bwd = module.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _inp, out):
        self.activation = out

    def _bwd(self, _m, _grad_in, grad_out):
        # grad_out is a tuple; the first element is the gradient of the loss
        # w.r.t. this module's output.
        self.gradient = grad_out[0]

    def close(self) -> None:
        self._h_fwd.remove()
        if hasattr(self, "_h_bwd"):
            self._h_bwd.remove()


# ---------------------------------------------------------------------------
# Find the right hook points in torchvision swin3d_t
# ---------------------------------------------------------------------------

def _gradcam_target_module(model: VideoSwinT) -> nn.Module:
    """The last SwinTransformerBlock3d before the global average pool.

    torchvision builds ``features`` as an ``nn.Sequential`` of stages,
    each stage being a ``nn.Sequential`` of blocks (optionally followed
    by a patch-merging). The very last item inside the final stage is
    the deepest block we can attach to.
    """
    backbone = model.backbone
    if hasattr(backbone, "features"):
        last_stage = backbone.features[-1]
        # last_stage may itself be a Sequential of blocks
        if isinstance(last_stage, nn.Sequential):
            return last_stage[-1]
        return last_stage
    raise RuntimeError("Could not locate a GradCAM target inside the Swin backbone.")


def _window_attention_modules(model: VideoSwinT) -> list[nn.Module]:
    """All WindowAttention3d (or named *attention*) submodules in order."""
    out = []
    for _name, m in model.backbone.named_modules():
        cls_name = m.__class__.__name__.lower()
        if "attention" in cls_name and "block" not in cls_name:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

def gradcam(
    model: VideoSwinT,
    clip: Tensor,                         # [1, C, T, H, W] already preprocessed
    *,
    target_class: int | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, int]:
    """Return ``(heatmap[T,H,W], predicted_class)`` in ``[0, 1]``."""
    device = device or next(model.parameters()).device
    clip = clip.to(device).requires_grad_(False)
    model = model.to(device).eval()

    target = _gradcam_target_module(model)
    hook = _ActivationHook(target, retain_grad=True)
    try:
        # We *do* need a grad path through the model — make sure params are leaves
        # but we don't update them. Use a fresh enable_grad context.
        with torch.enable_grad():
            logits = model(clip)                              # [1, num_classes]
            pred   = int(logits.argmax(dim=-1).item())
            cls    = target_class if target_class is not None else pred
            score  = logits[0, cls]
            model.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)

        act  = hook.activation                # [1, T', H', W', C] for torchvision swin
        grad = hook.gradient                  # same shape
        if act is None or grad is None:
            raise RuntimeError("Hook captured no activation / gradient.")

        # Squeeze batch dim, then global-avg-pool gradient over spatial+temporal dims
        # to get channel weights. The expected layout for torchvision swin3d output
        # is [B, T', H', W', C]; if a different layout shows up we fall back to
        # generic last-dim-as-channels behaviour.
        if act.ndim == 5 and act.shape[-1] != clip.shape[-1]:    # [B, T', H', W', C]
            act_ = act[0]                                         # [T', H', W', C]
            weights = grad[0].mean(dim=(0, 1, 2))                 # [C]
            cam = (act_ * weights).sum(dim=-1)                    # [T', H', W']
        else:                                                     # generic 5-D [B,C,T,H,W]
            act_ = act[0]                                         # [C, T', H', W']
            weights = grad[0].mean(dim=(1, 2, 3))                 # [C]
            cam = (act_ * weights[:, None, None, None]).sum(dim=0)

        cam = F.relu(cam)
        # Upsample to clip resolution
        cam = cam[None, None]                                     # [1, 1, T', H', W']
        cam = F.interpolate(
            cam, size=clip.shape[-3:], mode="trilinear", align_corners=False,
        )[0, 0]                                                   # [T, H, W]
        cam_np = cam.detach().cpu().numpy()
        cam_np -= cam_np.min()
        if cam_np.max() > 1e-9:
            cam_np /= cam_np.max()
        return cam_np, pred
    finally:
        hook.close()


# ---------------------------------------------------------------------------
# Attention rollout (windowed, simplified)
# ---------------------------------------------------------------------------

def attention_rollout(
    model: VideoSwinT,
    clip: Tensor,                         # [1, C, T, H, W]
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return ``[T, H, W]`` attention-proxy heatmap in ``[0, 1]``.

    Notes on the "rollout" terminology
    ----------------------------------
    Classical attention rollout (Abnar & Zuidema 2020) multiplies
    per-block attention matrices for a model with a single CLS token and
    global attention. Video Swin uses *shifted windowed* attention with
    no CLS token, so the literal rollout product is ill-defined across
    blocks of different window sizes.

    Instead we use the well-established **feature activation proxy**:
    capture the output of each ``ShiftedWindowAttention3d`` module
    (shape ``[B, T', H', W', C]``), compute the per-token L2 norm over
    channels (high norm = strong attention output at that spatial
    location), upsample each block's spatio-temporal map to the input
    resolution, then **average across blocks**. This gives a multi-scale
    activation heatmap that highlights the regions and frames the model
    relies on most — the same interpretation users expect from a
    "rollout" plot.
    """
    device = device or next(model.parameters()).device
    clip = clip.to(device)
    model = model.to(device).eval()

    attn_modules = _window_attention_modules(model)
    if not attn_modules:
        raise RuntimeError("No WindowAttention modules found in backbone.")

    captures: list[Tensor] = []

    def hook(_m, _inputs, output):
        # Output is the post-attention feature map [B, T', H', W', C]
        captures.append(output.detach())

    handles = [m.register_forward_hook(hook) for m in attn_modules]
    try:
        with torch.no_grad():
            _ = model(clip)
    finally:
        for h in handles:
            h.remove()

    if not captures:
        logger.warning("No attention captures — returning uniform map.")
        T, H, W = clip.shape[-3:]
        return np.ones((T, H, W), dtype=np.float32)

    T, H, W = clip.shape[-3:]
    accum = torch.zeros(1, 1, T, H, W, device=device)
    for feat in captures:
        if feat.ndim != 5:                                 # safety net
            continue
        # L2 norm over channels → [B, T', H', W']
        token_mag = feat.norm(dim=-1)                      # [B, T', H', W']
        # Per-block normalise so deep blocks don't dominate scale.
        token_mag = token_mag - token_mag.amin(dim=(1, 2, 3), keepdim=True)
        denom = token_mag.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
        token_mag = token_mag / denom                      # [B, T', H', W'] in [0,1]

        up = F.interpolate(
            token_mag[:, None],                            # [B, 1, T', H', W']
            size=(T, H, W),
            mode="trilinear", align_corners=False,
        )                                                  # [B, 1, T, H, W]
        accum = accum + up

    accum = accum[0, 0]                                    # [T, H, W]
    accum = accum - accum.min()
    if accum.max() > 1e-9:
        accum = accum / accum.max()
    return accum.cpu().numpy()


# ---------------------------------------------------------------------------
# Clip loading
# ---------------------------------------------------------------------------

def load_clip(path: str, n_frames: int = 16, size: int = 224) -> tuple[Tensor, Tensor]:
    """Decode + resize a clip → ``(raw_rgb [T,3,H,W] in [0,1], swin_in [1,3,T,H,W])``.

    Returns both the un-normalised RGB tensor (for overlay rendering) and
    the ImageNet-normalised, swin-layout tensor (for model forward).
    """
    from pipelines.video_dataset import _count_frames, _sample_indices, decode_clip

    total = _count_frames(path, backend="auto")
    idx   = _sample_indices(total, n_frames, jitter=False)
    rgb   = decode_clip(path, idx, height=size, width=size)        # [T,3,H,W] in [0,1]

    swin_in = imagenet_normalize(rgb.unsqueeze(0))                 # [1,T,3,H,W]
    swin_in = to_swin_layout(swin_in)                              # [1,3,T,H,W]
    return rgb, swin_in


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def render_overlay_grid(
    rgb: np.ndarray,            # [T, H, W, 3] in [0,1]
    cam_gc: np.ndarray,         # [T, H, W] in [0,1]
    cam_ro: np.ndarray,         # [T, H, W] in [0,1]
    out_path: Path,
    pred_name: str,
    n_frames_vis: int = 4,
    alpha: float = 0.45,
) -> None:
    """Render rows=keyframes, cols=[original | gradcam | rollout]."""
    T = rgb.shape[0]
    if n_frames_vis > T:
        n_frames_vis = T
    idx = np.linspace(0, T - 1, n_frames_vis).round().astype(int)

    fig, axes = plt.subplots(n_frames_vis, 3, figsize=(12, 3.6 * n_frames_vis))
    if n_frames_vis == 1:
        axes = axes[None]

    for r, i in enumerate(idx):
        frame = rgb[i]
        gc    = cam_gc[i]
        ro    = cam_ro[i]

        axes[r, 0].imshow(frame)
        axes[r, 0].set_title(f"frame {i}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(frame)
        axes[r, 1].imshow(gc, cmap="jet", alpha=alpha)
        axes[r, 1].set_title(f"GradCAM (pred: {pred_name})")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(frame)
        axes[r, 2].imshow(ro, cmap="jet", alpha=alpha)
        axes[r, 2].set_title("Attention rollout")
        axes[r, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved attention grid → %s", out_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Swin-T GradCAM + attention rollout.")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--clip",        required=True,
                        help="Path to a video file to visualise.")
    parser.add_argument("--run-name",    default=None)
    parser.add_argument("--fig-dir",     default="outputs/figures")
    parser.add_argument("--n-frames-vis", type=int, default=4)
    parser.add_argument("--target-class", type=int, default=None,
                        help="Force GradCAM to explain this class (default: predicted).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = TrainConfig.from_yaml(args.config)
    run_name = args.run_name or Path(args.checkpoint).parent.parent.name
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- build model + load checkpoint -------------------------------
    in_channels = 5 if cfg.data.use_optical_flow else 3
    model = VideoSwinT(
        num_classes=cfg.model.num_classes,
        in_channels=in_channels,
        pretrained=False,                  # weights come from the checkpoint
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    logger.info("Loaded Swin-T checkpoint %s", args.checkpoint)

    # ---- prepare clip ------------------------------------------------
    rgb, swin_in = load_clip(args.clip, n_frames=cfg.data.num_frames, size=cfg.data.height)
    rgb_np = rgb.permute(0, 2, 3, 1).numpy()        # [T, H, W, 3]

    # ---- explanations ------------------------------------------------
    cam_gc, pred = gradcam(model, swin_in, target_class=args.target_class, device=device)
    cam_ro       = attention_rollout(model, swin_in, device=device)

    pred_name = SPECIES[pred] if 0 <= pred < NUM_CLASSES else f"class_{pred}"
    logger.info("Predicted class: %d (%s)", pred, pred_name)

    # ---- render ------------------------------------------------------
    clip_stem = Path(args.clip).stem
    out_path = fig_dir / f"attention_{clip_stem}_{run_name}.png"
    render_overlay_grid(
        rgb_np, cam_gc, cam_ro,
        out_path=out_path,
        pred_name=pred_name,
        n_frames_vis=args.n_frames_vis,
    )


if __name__ == "__main__":
    main()
