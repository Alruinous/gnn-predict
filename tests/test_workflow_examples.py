from __future__ import annotations

from pathlib import Path

import yaml

from workflow.artifacts import GpuKind, load_prediction_cache
from workflow.schema import AgentNodeConfig, FunctionNodeConfig, Workflow

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "example" / "workflow"
GPU_KINDS: tuple[GpuKind, ...] = ("v100", "a100")
OUTPUT_BUCKETS = (128, 512, 1024)


def load_workflow(path: Path) -> Workflow:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Workflow.model_validate(payload)


def test_runtime_examples_validate() -> None:
    workflow = load_workflow(EXAMPLE_DIR / "runtime.yaml")
    predictions = load_prediction_cache(EXAMPLE_DIR / "predictions.yaml")

    assert workflow.graph.entry_node == "prepare"
    assert workflow.graph.terminal_node == "summarize"
    assert predictions.version == 2
    assert predictions.environment == {
        "storage_kind": "shared_model_dir",
        "model_root": "/data/Models",
        "prediction_scope": "gpu_kind",
    }


def test_runtime_prediction_cache_covers_agent_resource_keys() -> None:
    workflow = load_workflow(EXAMPLE_DIR / "runtime.yaml")
    predictions = load_prediction_cache(EXAMPLE_DIR / "predictions.yaml")
    nodes = workflow.node_map()
    prepare = nodes["prepare"]
    summarize = nodes["summarize"]

    assert isinstance(prepare, FunctionNodeConfig)
    assert prepare.routing == "broadcast"
    assert isinstance(summarize, AgentNodeConfig)

    for gpu_kind in GPU_KINDS:
        for output_length in OUTPUT_BUCKETS:
            entry = predictions.lookup_decode(
                model_name=summarize.model.name,
                gpu_kind=gpu_kind,
                sequence_length=2048,
                decode_output_length=output_length,
            )
            assert entry.key.model_dump(mode="python") == {
                "model_name": summarize.model.name,
                "phase": "decode",
                "gpu_name": gpu_kind,
                "batch_size": 1,
                "sequence_length": 2048,
                "decode_output_length": output_length,
            }
            assert entry.predicted_load_sec > 0
            assert entry.predicted_run_sec > 0
            assert entry.predicted_peak_vram_mb > 0
