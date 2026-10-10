from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import gnn_archs.graph_export as graph_export
from common.graph_artifact import load_graph_artifact, save_graph_artifact


class AttentionModel(nn.Module):
    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(query, key, value)


def attention_flags() -> tuple[bool, bool, bool]:
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_corex_attention_graph_round_trips_and_restores_runtime(
    tmp_path: Path, device_name: str
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA-compatible accelerator is unavailable")
    model = AttentionModel().eval()
    args = tuple(torch.randn(1, 2, 8, 16, device=device_name) for _ in range(3))
    original_flags = attention_flags()

    exported = graph_export._capture_inference_graph_for_backend(
        model, args, device_backend="corex"
    )
    assert attention_flags() == original_flags
    assert not any(
        "flash_attention" in str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function"
    )

    path = tmp_path / "attention.pt2"
    save_graph_artifact(exported, path, runtime_input_names=["query", "key", "value"])
    loaded, _ = load_graph_artifact(path)
    with torch.no_grad():
        torch.testing.assert_close(loaded.module()(*args), model(*args))


@pytest.mark.parametrize("device_backend", ["corex", "nvidia", "cpu"])
def test_attention_settings_are_restored_when_graph_capture_fails(
    monkeypatch: pytest.MonkeyPatch, device_backend: str
) -> None:
    original_flags = attention_flags()

    def fail_capture(model: nn.Module, args: tuple[object, ...]) -> None:
        expected = (False, False, True) if device_backend == "corex" else original_flags
        assert attention_flags() == expected
        raise ValueError("capture failed")

    monkeypatch.setattr(graph_export, "capture_inference_graph", fail_capture)
    with pytest.raises(ValueError, match="capture failed"):
        graph_export._capture_inference_graph_for_backend(
            AttentionModel(), (), device_backend=device_backend
        )
    assert attention_flags() == original_flags
