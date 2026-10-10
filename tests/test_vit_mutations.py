from __future__ import annotations

from typing import Any

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

import gnn_archs.mutations as mutations


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "changes",
    [
        {"depth": 1},
        {"num_heads": 8},
        {"mlp_hidden_dim": 96},
        {"embed_dim": 24, "num_heads": 3, "mlp_hidden_dim": 48},
    ],
)
def test_vit_rebuild_supports_legacy_constructor_and_preserves_runtime(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    changes: dict[str, int],
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA-compatible accelerator is unavailable")

    device = torch.device(device_name)
    dtype = torch.float64 if device_name == "cpu" else torch.float32
    model = VisionTransformer(
        img_size=32,
        patch_size=16,
        num_classes=4,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
    ).to(device=device, dtype=dtype)
    expected_device = next(model.parameters()).device

    def legacy_constructor(**kwargs: Any) -> VisionTransformer:
        # Exercise the timm 1.0.20 API even when tests use a newer timm.
        for argument in ("device", "dtype"):
            if argument in kwargs:
                raise TypeError(f"unexpected keyword argument '{argument}'")
        return VisionTransformer(**kwargs)

    monkeypatch.setattr(mutations, "VisionTransformer", legacy_constructor)
    rebuilt = mutations.rebuild_vit_model(model, **changes)

    assert len(rebuilt.blocks) == changes.get("depth", 2)
    assert rebuilt.embed_dim == changes.get("embed_dim", 32)
    assert rebuilt.blocks[0].attn.num_heads == changes.get("num_heads", 4)
    assert rebuilt.blocks[0].mlp.fc1.out_features == changes.get("mlp_hidden_dim", 64)
    assert all(parameter.device == expected_device for parameter in rebuilt.parameters())
    assert all(parameter.dtype == dtype for parameter in rebuilt.parameters())

    logits = rebuilt(torch.randn(1, 3, 32, 32, device=device, dtype=dtype))
    assert logits.shape == (1, 4)
    logits.sum().backward()
    assert rebuilt.head.weight.grad is not None
