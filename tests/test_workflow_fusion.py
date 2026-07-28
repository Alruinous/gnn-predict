from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from workflow.fusion import find_fusable_chains, fuse_workflow
from workflow.replica import ModelDeploymentConfig
from workflow.schema import FusedAgentNodeConfig, Workflow
from workflow.worker import estimate_chain_input_tokens, execute_fused_agent

from test_workflow_worker import (
    FakeGrant,
    FakeReplica,
    FakeScheduler,
    FakeTokenizer,
    inference_result,
)

ROOT = Path(__file__).resolve().parents[1]


def agent_node(
    name: str,
    model_name: str = "Qwen3-8B",
    *,
    model_path: str = "/models/Qwen3-8B",
    max_new_tokens: int = 384,
    max_model_len: int = 4096,
    prompt_template: str = "{content}",
) -> dict[str, Any]:
    return {
        "name": name,
        "type": "agent",
        "model": {"name": model_name},
        "execution": {
            "model_path": model_path,
            "max_new_tokens": max_new_tokens,
            "dtype": "float16",
            "serving": {
                "max_model_len": max_model_len,
                "max_num_seqs": 3,
                "max_num_batched_tokens": max_model_len * 3,
            },
        },
        "prompt_template": prompt_template,
    }


def chain_workflow(**overrides: Any) -> Workflow:
    tail = overrides.pop("tail", agent_node("finalize"))
    return Workflow.model_validate(
        {
            "workflow_name": "chain",
            "nodes": [
                {"name": "split", "type": "function", "function": "split"},
                agent_node("merge"),
                agent_node("expand"),
                tail,
            ],
            "edges": [
                {"source": "split", "target": "merge"},
                {"source": "merge", "target": "expand"},
                {"source": "expand", "target": tail["name"]},
            ],
        }
    )


def test_committed_serve_workflows_have_no_fusable_chains() -> None:
    """The frozen paper workloads are all heterogeneous or parallel, never chained."""
    for name in ("qmsum1", "mbpp1", "gsm8k1"):
        path = ROOT / "config" / "workflow" / "serve" / f"{name}.yaml"
        workflow = Workflow.model_validate(yaml.safe_load(path.read_text()))
        assert find_fusable_chains(workflow) == ()
        assert fuse_workflow(workflow) is workflow


def test_same_model_chain_collapses_into_one_node() -> None:
    workflow = chain_workflow()

    fused = fuse_workflow(workflow)

    assert [node.name for node in fused.nodes] == [
        "split",
        "merge+expand+finalize",
    ]
    assert [(edge.source, edge.target) for edge in fused.edges] == [
        ("split", "merge+expand+finalize")
    ]
    node = fused.node_map()["merge+expand+finalize"]
    assert isinstance(node, FusedAgentNodeConfig)
    assert [stage.name for stage in node.stages] == ["merge", "expand", "finalize"]
    assert fused.graph.entry_node == "split"
    assert fused.graph.terminal_node == "merge+expand+finalize"


def test_fused_node_keeps_the_chain_model_and_widest_output() -> None:
    workflow = chain_workflow(tail=agent_node("finalize", max_new_tokens=512))

    node = fuse_workflow(workflow).node_map()["merge+expand+finalize"]

    assert isinstance(node, FusedAgentNodeConfig)
    assert node.execution.max_new_tokens == 512
    assert [stage.max_new_tokens for stage in node.stages] == [384, 384, 512]
    original = workflow.node_map()["merge"]
    assert (
        ModelDeploymentConfig.from_node(node).model_key
        == ModelDeploymentConfig.from_node(original).model_key
    )


def test_a_different_model_breaks_the_chain() -> None:
    workflow = chain_workflow(
        tail=agent_node("finalize", "Qwen3-14B", model_path="/models/Qwen3-14B")
    )

    fused = fuse_workflow(workflow)

    assert [node.name for node in fused.nodes] == ["split", "merge+expand", "finalize"]
    assert [(edge.source, edge.target) for edge in fused.edges] == [
        ("split", "merge+expand"),
        ("merge+expand", "finalize"),
    ]


def test_a_differing_serving_config_breaks_the_chain() -> None:
    # max_model_len is part of model_key, so these two cannot share an engine.
    workflow = chain_workflow(tail=agent_node("finalize", max_model_len=8192))

    fused = fuse_workflow(workflow)

    assert [node.name for node in fused.nodes] == ["split", "merge+expand", "finalize"]


def test_fan_in_head_and_fan_out_tail_still_fuse() -> None:
    workflow = Workflow.model_validate(
        {
            "workflow_name": "wide",
            "nodes": [
                {
                    "name": "split",
                    "type": "function",
                    "function": "split",
                    "routing": "targeted",
                },
                agent_node("chunk_0", "Qwen3-4B", model_path="/models/Qwen3-4B"),
                agent_node("chunk_1", "Qwen3-4B", model_path="/models/Qwen3-4B"),
                agent_node("merge"),
                agent_node("expand"),
                {"name": "end", "type": "function", "function": "end"},
            ],
            "edges": [
                {"source": "split", "target": "chunk_0"},
                {"source": "split", "target": "chunk_1"},
                {"source": "chunk_0", "target": "merge"},
                {"source": "chunk_1", "target": "merge"},
                {"source": "merge", "target": "expand"},
                {"source": "expand", "target": "end"},
            ],
        }
    )

    fused = fuse_workflow(workflow)

    node_names = [node.name for node in fused.nodes]
    assert "merge+expand" in node_names
    # Parallel same-model siblings are a scale-out workload, not a fusion target.
    assert "chunk_0" in node_names and "chunk_1" in node_names
    assert set(fused.graph.dependencies["merge+expand"]) == {"chunk_0", "chunk_1"}
    assert fused.graph.adjacency["merge+expand"] == ("end",)


def test_fusion_is_idempotent() -> None:
    once = fuse_workflow(chain_workflow())

    assert fuse_workflow(once) is once


def test_fused_node_round_trips_through_yaml() -> None:
    fused = fuse_workflow(chain_workflow())

    restored = Workflow.model_validate(yaml.safe_load(yaml.safe_dump(fused.model_dump(mode="json"))))

    node = restored.node_map()["merge+expand+finalize"]
    assert isinstance(node, FusedAgentNodeConfig)
    assert [stage.name for stage in node.stages] == ["merge", "expand", "finalize"]


def test_fused_execution_must_reserve_the_widest_stage() -> None:
    fused = fuse_workflow(
        chain_workflow(tail=agent_node("finalize", max_new_tokens=512))
    ).node_map()["merge+expand+finalize"]
    assert isinstance(fused, FusedAgentNodeConfig)
    data = fused.model_dump(mode="json")
    data["execution"]["max_new_tokens"] = 384

    with pytest.raises(ValueError, match="widest stage output"):
        FusedAgentNodeConfig.model_validate(data)


def test_fused_node_needs_at_least_two_stages() -> None:
    fused = fuse_workflow(chain_workflow()).node_map()["merge+expand+finalize"]
    assert isinstance(fused, FusedAgentNodeConfig)
    data = fused.model_dump(mode="json")
    data["stages"] = data["stages"][:1]

    with pytest.raises(ValueError, match="at least two stages"):
        FusedAgentNodeConfig.model_validate(data)


def test_admission_envelope_covers_the_widest_stage_prompt() -> None:
    node = fuse_workflow(
        chain_workflow(tail=agent_node("finalize", max_new_tokens=512))
    ).node_map()["merge+expand+finalize"]
    assert isinstance(node, FusedAgentNodeConfig)

    # Stages 2 and 3 replace the upstream output, so the bound is the first prompt
    # plus the largest generation that feeds a later stage (384, not the tail's 512).
    assert estimate_chain_input_tokens(node, 1000) == 1384


def fused_node(stage_outputs: tuple[int, ...] = (16, 16, 16)) -> FusedAgentNodeConfig:
    return FusedAgentNodeConfig.model_validate(
        {
            "name": "a+b+c",
            "type": "fused_agent",
            "model": {"name": "test-model"},
            "execution": {
                "model_path": "/models/test-model",
                "max_new_tokens": max(stage_outputs),
                "use_chat_template": False,
                "serving": {
                    "max_model_len": 1024,
                    "max_num_seqs": 3,
                    "max_num_batched_tokens": 1536,
                },
            },
            "prompt_template": "Summarize {content}",
            "stages": [
                {
                    "name": name,
                    "prompt_template": template,
                    "max_new_tokens": limit,
                }
                for name, template, limit in zip(
                    ("a", "b", "c"),
                    ("Summarize {content}", "Critique {content}", "Polish {content}"),
                    stage_outputs,
                    strict=True,
                )
            ],
        }
    )


def test_fused_chain_runs_every_stage_under_one_grant() -> None:
    node = fused_node()
    replica = FakeReplica([inference_result() for _ in range(3)])
    grant = FakeGrant(
        acquire_id="acquire-1",
        task_id="task-s1-chain",
        replica_id="replica-1",
        backend_handle=replica,
        accelerator_ids=("host/v100:0",),
        model_key="model-key",
        gpu_kind="v100",
        input_tokens=7 + 16,
        max_new_tokens=16,
    )
    scheduler = FakeScheduler(grants=[grant])
    tokenizer = FakeTokenizer()

    execution = asyncio.run(
        execute_fused_agent(
            node=node,
            task_id="task-s1-chain",
            session_id="s1",
            input_item_ids=["item-1"],
            prompt_context={"content": "text"},
            session_inputs={},
            scheduler=scheduler,
            tokenizer=tokenizer,
            acquire_timeout_sec=0.01,
            grant_poll_interval_sec=0.001,
        )
    )

    # One handshake for the whole chain: that is the cost fusion removes.
    acquires = [call for call in scheduler.calls if call[0] == "request_acquire"]
    assert len(acquires) == 1
    assert acquires[0][1][1] == 7 + 16

    # Three serial generations on the same engine handle, distinct request ids.
    assert [call[0] for call in replica.calls] == [
        "acquire-1#0",
        "acquire-1#1",
        "acquire-1#2",
    ]
    # Each stage reads the previous stage's output.
    assert [call[0] for call in tokenizer.calls] == [
        "Summarize text",
        "Critique generated",
        "Polish generated",
    ]

    report = execution.report
    assert [stage.node_id for stage in report.stage_reports] == ["a", "b", "c"]
    assert report.output_tokens == 12
    assert report.input_tokens == grant.input_tokens
    assert report.duration_sec == 3.0
    assert execution.output is not None


def test_fused_chain_stops_at_the_first_failing_stage() -> None:
    node = fused_node()
    replica = FakeReplica([inference_result(), inference_result("failed")])
    grant = FakeGrant(
        acquire_id="acquire-1",
        task_id="task-s1-chain",
        replica_id="replica-1",
        backend_handle=replica,
        accelerator_ids=("host/v100:0",),
        model_key="model-key",
        gpu_kind="v100",
        input_tokens=7 + 16,
        max_new_tokens=16,
    )
    scheduler = FakeScheduler(grants=[grant])

    execution = asyncio.run(
        execute_fused_agent(
            node=node,
            task_id="task-s1-chain",
            session_id="s1",
            input_item_ids=["item-1"],
            prompt_context={"content": "text"},
            session_inputs={},
            scheduler=scheduler,
            tokenizer=FakeTokenizer(),
            acquire_timeout_sec=0.01,
            grant_poll_interval_sec=0.001,
        )
    )

    assert len(replica.calls) == 2
    assert execution.report.status == "failed"
    assert execution.output is None
    assert [stage.status for stage in execution.report.stage_reports] == [
        "success",
        "failed",
    ]


def test_scheduler_registers_a_session_for_a_fused_node() -> None:
    """A fused chain is an agent task; NodeTaskRecord must accept it as one."""
    from test_workflow_scheduler_elastic import _agent_node, _predictions, elastic_core

    from workflow.scheduler import NodeTaskState

    chain = Workflow.model_validate(
        {
            "workflow_name": "agent-workflow",
            "nodes": [_agent_node("agent", "test-model", 1), _agent_node("tail", "test-model", 1)],
            "edges": [{"source": "agent", "target": "tail"}],
        }
    )
    fused = fuse_workflow(chain)
    assert [node.name for node in fused.nodes] == ["agent+tail"]

    core = elastic_core(workflow=fused, predictions=_predictions())
    core.register_session("s1", "agent-workflow")

    task = core.tasks[core.sessions["s1"].task_ids["agent+tail"]]
    assert task.node_kind == "agent"
    assert task.state is NodeTaskState.PENDING


def test_targeted_routing_follows_a_fused_chain_head() -> None:
    """Fusion renames a chain head; a targeted function still routes to it."""
    from workflow.schema import FunctionNodeConfig
    from workflow.worker import _route_function_output

    node = FunctionNodeConfig.model_validate(
        {"name": "split", "type": "function", "function": "split", "routing": "targeted"}
    )
    state = {"messages": []}
    routed = _route_function_output(
        node,
        {"lane_a_0": state, "lane_b_0": state},
        ("lane_a_0+lane_a_1", "lane_b_0+lane_b_1"),
    )

    assert set(routed) == {"lane_a_0+lane_a_1", "lane_b_0+lane_b_1"}


def test_targeted_routing_rejects_an_ambiguous_key() -> None:
    from workflow.schema import FunctionNodeConfig
    from workflow.worker import _route_function_output

    node = FunctionNodeConfig.model_validate(
        {"name": "split", "type": "function", "function": "split", "routing": "targeted"}
    )
    with pytest.raises(ValueError, match="no unique successor"):
        _route_function_output(node, {"lane": {"messages": []}}, ("other_a", "other_b"))
