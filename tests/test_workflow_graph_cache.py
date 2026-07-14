from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from workflow.cache_config import (
    WorkflowPhase,
    expand_graph_cache_specs,
    load_graph_cache_config,
)
from workflow.model import build_model_graph_feature

ROOT = Path(__file__).resolve().parents[1]


def test_workflow_cache_yaml_keeps_all_model_workload_combinations() -> None:
    config = load_graph_cache_config(ROOT / "config" / "workflow" / "cache.yaml")

    specs = expand_graph_cache_specs(config)

    assert len(config.cache_groups) == 6
    assert len(specs) == 1980
    assert {spec.key.model_name for spec in specs} == {
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
    }


def build_tiny_qwen() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=None,
            use_cache=True,
        )
    ).eval()


@pytest.mark.parametrize(
    ("phase", "decode_output_length", "expected_phase_id"),
    [("prefill", 0, 2.0), ("decode", 3, 3.0)],
)
def test_workflow_builds_graph_features_without_intermediate_files(
    phase: WorkflowPhase,
    decode_output_length: int,
    expected_phase_id: float,
) -> None:
    model = build_tiny_qwen()
    input_map = {
        "input_ids": torch.zeros((2, 8), dtype=torch.long),
        "attention_mask": torch.ones((2, 8), dtype=torch.long),
    }

    data = build_model_graph_feature(
        "Qwen3-test",
        model,
        input_map,
        ["input_ids", "attention_mask"],
        phase=phase,
        batch_size=2,
        decode_output_length=decode_output_length,
    )
    x = data.x
    edge_attr = data.edge_attr
    assert isinstance(x, torch.Tensor)
    assert isinstance(edge_attr, torch.Tensor)

    assert x.shape[1] == 22
    assert edge_attr.shape[1] == 15
    assert data.graph_features.shape == (1, 30)
    assert data.graph_features[0, 0].item() == expected_phase_id
    assert data.graph_path == ""
