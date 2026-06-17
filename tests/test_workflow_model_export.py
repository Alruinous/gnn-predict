from __future__ import annotations

from workflow.model_export import build_resolved_variant_spec
from workflow.schema import Workflow


def build_workflow(node: dict[str, object]) -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                node,
                {"name": "output", "type": "output"},
            ],
            "edges": [
                {"source": "input", "target": str(node["name"]), "attributes": {}},
                {"source": str(node["name"]), "target": "output", "attributes": {}},
            ],
        }
    )


def test_build_resolved_variant_spec_uses_workflow_detection_parameters() -> None:
    workflow = build_workflow(
        {
            "name": "detector",
            "type": "tool",
            "task": "object_detection",
            "model": {
                "name": "yolov8n",
                "parameters": {
                    "input_channels": 3,
                    "output_classes": 80,
                },
            },
            "runtime": {
                "batch_size": 2,
                "input_shape": [2, 3, 224, 224],
                "phase": "inference",
            },
        }
    )

    spec = build_resolved_variant_spec(workflow.node_map()["detector"])

    assert spec.base_model.name == "yolov8n"
    assert spec.base_model.pretrained is False
    assert spec.mutations == []
    assert spec.variant_config.target_input_channels == 3
    assert spec.variant_config.target_output_classes == 80
    assert spec.variant_config.example_input_shape == [2, 3, 224, 224]


def test_build_resolved_variant_spec_uses_qwen_parameters() -> None:
    workflow = build_workflow(
        {
            "name": "summarizer",
            "type": "tool",
            "task": "text_generation",
            "model": {
                "name": "qwen3",
                "parameters": {
                    "vocab_size": 151936,
                    "hidden_size": 768,
                    "intermediate_size": 2304,
                    "num_hidden_layers": 20,
                    "num_attention_heads": 12,
                    "num_key_value_heads": 4,
                    "max_position_embeddings": 40960,
                },
            },
            "runtime": {
                "batch_size": 1,
                "sequence_length": 128,
                "phase": "prefill",
            },
        }
    )

    spec = build_resolved_variant_spec(workflow.node_map()["summarizer"])

    assert spec.variant_config.qwen3_config is not None
    assert spec.variant_config.qwen3_config.hidden_size == 768
    assert spec.variant_config.example_input_shape == [1, 128]
