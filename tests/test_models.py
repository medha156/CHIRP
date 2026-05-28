"""Unit tests for the model classes — no pretrained weights, CPU only."""

from __future__ import annotations

import pytest
import torch

from models.efficientnet import EfficientNetB3Encoder
from models.fusion import FusionHead, SwinEffNetFusion
from models.swin_t import VideoSwinT

# ---------------------------------------------------------------------------
# VideoSwinT
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def swin_rgb() -> VideoSwinT:
    return VideoSwinT(num_classes=20, in_channels=3, pretrained=False).eval()


@pytest.fixture(scope="module")
def swin_flow() -> VideoSwinT:
    return VideoSwinT(num_classes=20, in_channels=5, pretrained=False).eval()


def test_swin_forward_rgb(swin_rgb):
    x = torch.randn(2, 3, 8, 224, 224)
    with torch.no_grad():
        logits = swin_rgb(x)
    assert logits.shape == (2, 20)


def test_swin_forward_optical_flow(swin_flow):
    x = torch.randn(1, 5, 8, 224, 224)
    with torch.no_grad():
        logits = swin_flow(x)
    assert logits.shape == (1, 20)


def test_swin_extract_embeddings(swin_rgb):
    x = torch.randn(2, 3, 8, 224, 224)
    emb = swin_rgb.extract_embeddings(x)
    assert emb.shape == (2, swin_rgb.FEATURE_DIM)


def test_swin_rejects_wrong_channel_count(swin_rgb):
    with pytest.raises(ValueError, match="channels"):
        swin_rgb(torch.randn(1, 5, 8, 224, 224))      # rgb model fed 5 ch


def test_swin_rejects_wrong_ndim(swin_rgb):
    with pytest.raises(ValueError, match=r"\[B,C,T,H,W\]"):
        swin_rgb(torch.randn(2, 3, 224, 224))         # missing T axis


def test_swin_freeze_backbone_disables_grads(swin_rgb):
    swin_rgb.freeze_backbone(True)
    trainable = sum(p.requires_grad for n, p in swin_rgb.named_parameters()
                    if not n.startswith("backbone.head"))
    assert trainable == 0
    swin_rgb.freeze_backbone(False)                   # restore for other tests


# ---------------------------------------------------------------------------
# EfficientNetB3Encoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pool", ["mean", "max", "attention"])
def test_efficientnet_all_pools_shape(pool):
    enc = EfficientNetB3Encoder(pretrained=False, pool=pool).eval()
    x = torch.randn(2, 4, 3, 224, 224)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, enc.FEATURE_DIM)


def test_efficientnet_accepts_flat_layout():
    enc = EfficientNetB3Encoder(pretrained=False).eval()
    x = torch.randn(2 * 4, 3, 224, 224)
    with torch.no_grad():
        out = enc(x, t=4)
    assert out.shape == (2, enc.FEATURE_DIM)


def test_efficientnet_with_head_returns_logits():
    enc = EfficientNetB3Encoder(pretrained=False, num_classes=20).eval()
    out = enc(torch.randn(2, 4, 3, 224, 224))
    assert out.shape == (2, 20)


def test_efficientnet_extract_embeddings_with_head():
    enc = EfficientNetB3Encoder(pretrained=False, num_classes=20).eval()
    emb = enc.extract_embeddings(torch.randn(2, 4, 3, 224, 224))
    assert emb.shape == (2, enc.FEATURE_DIM)


def test_efficientnet_freeze_disables_grads():
    enc = EfficientNetB3Encoder(pretrained=False, freeze=True)
    assert sum(p.requires_grad for p in enc.backbone.parameters()) == 0


def test_efficientnet_attention_pool_head_trainable_when_frozen():
    """Attention-pool params should NOT be frozen even with freeze=True."""
    enc = EfficientNetB3Encoder(pretrained=False, pool="attention", freeze=True)
    assert enc.attn_pool is not None
    assert all(p.requires_grad for p in enc.attn_pool.parameters())


def test_efficientnet_rejects_bad_pool():
    with pytest.raises(ValueError):
        EfficientNetB3Encoder(pretrained=False, pool="median")  # type: ignore[arg-type]


def test_efficientnet_flat_layout_requires_t():
    enc = EfficientNetB3Encoder(pretrained=False).eval()
    with pytest.raises(ValueError, match="specify `t`"):
        enc(torch.randn(8, 3, 224, 224))   # no t given


# ---------------------------------------------------------------------------
# FusionHead
# ---------------------------------------------------------------------------

def test_fusion_head_two_arg_forward():
    head = FusionHead().eval()
    s = torch.randn(4, 768)
    e = torch.randn(4, 1536)
    out = head(s, e)
    assert out.shape == (4, 20)


def test_fusion_head_one_arg_forward_equals_two_arg():
    head = FusionHead().eval()
    s = torch.randn(4, 768)
    e = torch.randn(4, 1536)
    assert torch.allclose(head(s, e), head(torch.cat([s, e], dim=-1)))


def test_fusion_head_configurable_hidden_dim_and_dropout():
    head = FusionHead(hidden_dim=1024, dropout=0.5)
    assert head.hidden_dim == 1024
    assert head.dropout_p == 0.5


def test_fusion_head_rejects_bad_dropout():
    with pytest.raises(ValueError):
        FusionHead(dropout=1.5)


def test_fusion_head_rejects_bad_shape():
    head = FusionHead().eval()
    with pytest.raises(ValueError):
        head(torch.randn(4, 800), torch.randn(4, 1536))


def test_swin_effnet_fusion_end_to_end():
    swin = VideoSwinT(num_classes=20, in_channels=3, pretrained=False).eval()
    eff  = EfficientNetB3Encoder(pretrained=False, num_classes=0).eval()
    head = FusionHead().eval()
    model = SwinEffNetFusion(swin, eff, head, keyframes_per_clip=4)
    batch = {
        "swin": torch.randn(2, 3, 8, 224, 224),
        "efficientnet": torch.randn(2 * 4, 3, 224, 224),
    }
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2, 20)
