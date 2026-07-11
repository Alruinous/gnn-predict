from __future__ import annotations

from pathlib import Path

import yaml

from workflow.artifacts import GpuKind, load_deployment_profile, load_prediction_cache
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
    profile = load_deployment_profile(EXAMPLE_DIR / "profile.yaml")

    assert workflow.graph.entry_node == "prepare"
    assert workflow.graph.terminal_node == "summarize"
    assert predictions.version == 1
    assert profile.version == 1


def test_runtime_artifacts_cover_agent_prediction_and_profile_keys() -> None:
    workflow = load_workflow(EXAMPLE_DIR / "runtime.yaml")
    predictions = load_prediction_cache(EXAMPLE_DIR / "predictions.yaml")
    profile = load_deployment_profile(EXAMPLE_DIR / "profile.yaml")
    nodes = workflow.node_map()
    prepare = nodes["prepare"]
    summarize = nodes["summarize"]

    assert isinstance(prepare, FunctionNodeConfig)
    assert prepare.routing == "broadcast"
    assert isinstance(summarize, AgentNodeConfig)

    matching_profiles = {
        entry.gpu_kind
        for entry in profile.entries
        if entry.model_name == summarize.model.name
        and entry.model_path == summarize.execution.model_path
        and entry.dtype == summarize.execution.dtype
    }
    assert matching_profiles == set(GPU_KINDS)

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
