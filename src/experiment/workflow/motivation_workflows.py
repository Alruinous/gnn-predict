"""Motivation workflows: identical graphs, only the role-to-model mapping varies.

The motivation experiment's single independent variable is model heterogeneity H —
how many distinct models the three workflows draw on. Graph shape, datasets, prompts,
output limits and the serving profile are held constant across H so that every measured
difference is attributable to heterogeneity alone.

One serving profile is shared by every model (see MOTIVATION_SERVING). ``model_key``
(``replica.py``) hashes the whole serving block, so an identical profile is what lets a
model loaded for one workflow serve another; a per-role profile would silently split the
key and make cross-workflow reuse impossible.

Workflow names carry no ``H`` suffix on purpose: ``poisson_arrival_offsets``
(``plan.py``) derives its RNG seed from the workflow name, so H levels must share a name
to receive byte-identical arrival traces. The H level lives in the exported filename.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from pydantic import JsonValue

from experiment.workflow.gsm8k import (
    GSM8K_REFINE_PROMPT,
    GSM8K_SOLVE_PROMPT,
    GSM8K_SOLVER_COUNT,
)
from experiment.workflow.mbpp import MBPP_CODER_PROMPT
from experiment.workflow.qmsum import (
    QMSUM_CHUNK_PROMPT,
    QMSUM_MERGE_PROMPT,
    _split_evenly,
)
from experiment.workflow.scenario_functions import build_registry as build_base_registry
from workflow.schema import Workflow

Heterogeneity = Literal[1, 5]
HETEROGENEITY_LEVELS: tuple[Heterogeneity, ...] = (1, 5)

MODEL_PATHS: Mapping[str, str] = {
    "Qwen3-0.6B": "/data/Models/Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "/data/Models/Qwen/Qwen3-1.7B",
    "Qwen3-4B": "/data/Models/Qwen/Qwen3-4B",
    "Qwen3-8B": "/data/Models/Qwen/Qwen3-8B",
    "Qwen3-14B": "/data/Models/Qwen/Qwen3-14B",
}

# 8192 covers every role's longest prompt, so bucket selection is driven purely by the
# actual input length and never by a per-role window. Measured worst cases: QMSum chunk
# 5739 tokens at MOTIVATION_CHUNK_COUNT=7, MBPP coder 244, GSM8K solve 160.
MOTIVATION_SERVING: Mapping[str, JsonValue] = {
    "max_model_len": 8192,
    "max_num_seqs": 3,
    "max_num_batched_tokens": 24576,
    "gpu_memory_utilization": 0.98,
}

# Seven chunks keep every QMSum prompt at or below 6144 tokens, the largest decode
# bucket profiled for Qwen3-8B on a V100. At six chunks the longest 1.1% spill into
# 7168, which Qwen3-8B cannot serve on a V100 and which would land them on the A100
# in H=1 only.
MOTIVATION_CHUNK_COUNT = 7

MBPP_DIAGNOSER_PROMPT = (
    "Diagnose the Python solution and its test result.\n"
    "Identify the likely root cause, failing edge cases, and concrete correction "
    "requirements.\n"
    "Return only a concise diagnostic analysis, no code.\n\n"
    "Task:\n{task}\n\nCurrent solution and test result:\n{previous_output}\n\n"
    "Tests:\n{tests}"
)
MBPP_REVIEWER_PROMPT = (
    "Review the Python solution for correctness.\n"
    "Focus on logic errors, edge cases, and API misuse.\n"
    "Return only the review analysis, no code.\n\n"
    "Task:\n{task}\n\nCurrent solution, test result, and diagnostic analysis:\n"
    "{node_outputs_json}\n\nTests:\n{tests}"
)
MBPP_REPAIR_PROMPT = (
    "Repair the Python solution so that it passes the tests.\n"
    "Return only Python code in one fenced code block.\n"
    "The execution context JSON contains the tester's code and initial result, the "
    "diagnostic analysis, and the reviewer's analysis.\n\n"
    "Task:\n{task}\n\nExecution context:\n{node_outputs_json}\n\nTests:\n{tests}"
)

# Role -> model, per heterogeneity level. H=1 is the deployment assumption stated by
# Kairos (every agent shares one LLM); H=5 is role specialisation with a cost-quality
# cascade. Every other node attribute is identical between the two.
ROLE_MODELS: Mapping[Heterogeneity, Mapping[str, str]] = {
    1: {
        "solve_0": "Qwen3-8B",
        "solve_1": "Qwen3-8B",
        "solve_2": "Qwen3-8B",
        "solve_3": "Qwen3-8B",
        "solve_4": "Qwen3-8B",
        "refine": "Qwen3-8B",
        "coder": "Qwen3-8B",
        "diagnoser": "Qwen3-8B",
        "reviewer": "Qwen3-8B",
        "repair": "Qwen3-8B",
        "chunk": "Qwen3-8B",
        "merge": "Qwen3-8B",
    },
    5: {
        "solve_0": "Qwen3-0.6B",
        "solve_1": "Qwen3-1.7B",
        "solve_2": "Qwen3-4B",
        "solve_3": "Qwen3-8B",
        "solve_4": "Qwen3-14B",
        "refine": "Qwen3-8B",
        "coder": "Qwen3-14B",
        "diagnoser": "Qwen3-1.7B",
        "reviewer": "Qwen3-4B",
        "repair": "Qwen3-8B",
        "chunk": "Qwen3-4B",
        "merge": "Qwen3-8B",
    },
}

# Rolling-summary QMSum. Two lanes of three steps each keep a session's chunk work
# 2-way parallel while giving every lane a same-model chain; the tail is a second
# chain on the merge model. Each step reads its own slice_i from the session inputs
# (build_qmsum_session_inputs), so no slice has to travel along the chain.
CHAIN_LANES: tuple[str, ...] = ("lane_a", "lane_b")
CHAIN_DEPTH = 3

QMSUM_LANE_HEAD_PROMPT = (
    "Summarize transcript details relevant to the query.\n"
    "Preserve decisions, reasons, participants, and concrete outcomes.\n"
    "Return only the running summary.\n\n"
    "Query:\n{query}\n\nTranscript chunk:\n{slice_INDEX}"
)
QMSUM_LANE_STEP_PROMPT = (
    "Update the running summary with the next transcript chunk.\n"
    "Keep every query-relevant fact already present and add what the new chunk "
    "contributes.\n"
    "Return only the updated running summary.\n\n"
    "Query:\n{query}\n\nRunning summary:\n{previous_output}\n\n"
    "Next chunk:\n{slice_INDEX}"
)
QMSUM_LANE_MERGE_PROMPT = (
    "Merge the lane summaries into one answer to the query.\n"
    "Remove duplication, preserve query-relevant facts, keep the answer grounded.\n"
    "Return only the answer.\n\n"
    "Query:\n{query}\n\nLane summaries:\n{node_outputs_json}"
)
QMSUM_EXPAND_PROMPT = (
    "Verify the draft answer against the partial answers and expand any point that is "
    "thin or unsupported.\n"
    "Do not introduce facts that the partial answers do not support.\n"
    "Return only the revised answer.\n\n"
    "Query:\n{query}\n\nDraft answer:\n{previous_output}"
)
QMSUM_FINALIZE_PROMPT = (
    "Tighten the revised answer into its final form.\n"
    "Remove hedging and repetition, keep every query-relevant fact.\n"
    "Return only the final answer.\n\n"
    "Query:\n{query}\n\nRevised answer:\n{previous_output}"
)

ROLE_OUTPUT_LIMITS: Mapping[str, int] = {
    "solve_0": 512,
    "solve_1": 512,
    "solve_2": 512,
    "solve_3": 512,
    "solve_4": 512,
    "refine": 512,
    "coder": 512,
    "diagnoser": 384,
    "reviewer": 384,
    "repair": 512,
    "chunk": 384,
    "merge": 384,
}


def split_qmsum_chunks(
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> dict[str, AgentState]:
    """QMSum transcript fan-out with the chunk count carried by the node config."""
    if states:
        raise ValueError("QMSum split expects no input states")
    chunk_count = parameters.get("chunk_count", MOTIVATION_CHUNK_COUNT)
    if not isinstance(chunk_count, int) or isinstance(chunk_count, bool):
        raise TypeError("chunk_count must be an integer")
    if chunk_count < 1:
        raise ValueError("chunk_count must be positive")
    transcript = session_inputs.get("transcript")
    if not isinstance(transcript, str):
        raise TypeError("QMSum transcript must be text")
    chunks = _split_evenly(transcript.split("\n\n"), chunk_count)
    return {
        f"chunk_{index}": AgentState(messages=[AIMessage(content="\n\n".join(chunk))])
        for index, chunk in enumerate(chunks)
    }


def build_registry() -> dict[str, Callable[..., object]]:
    registry = build_base_registry()
    if "split_qmsum_chunks" in registry:
        raise ValueError("split_qmsum_chunks already registered")
    registry["split_qmsum_chunks"] = split_qmsum_chunks
    return registry


def build_moa_gsm8k(
    heterogeneity: Heterogeneity, *, queue_capacity: int = 16
) -> Workflow:
    """Heterogeneous ensemble: independent solvers vote, then one model refines."""
    nodes: list[dict[str, object]] = [
        _function_node("fanout", "fanout_gsm8k", queue_capacity, routing="targeted")
    ]
    nodes.extend(
        _agent_node(f"solve_{index}", heterogeneity, GSM8K_SOLVE_PROMPT, queue_capacity)
        for index in range(GSM8K_SOLVER_COUNT)
    )
    nodes.append(_function_node("aggregate", "aggregate_gsm8k_votes", queue_capacity))
    nodes.append(
        _agent_node("refine", heterogeneity, GSM8K_REFINE_PROMPT, queue_capacity)
    )
    edges = [
        {"source": "fanout", "target": f"solve_{index}"}
        for index in range(GSM8K_SOLVER_COUNT)
    ]
    edges.extend(
        {"source": f"solve_{index}", "target": "aggregate"}
        for index in range(GSM8K_SOLVER_COUNT)
    )
    edges.append({"source": "aggregate", "target": "refine"})
    return _workflow("moa_gsm8k", nodes, edges)


def build_repair_mbpp(
    heterogeneity: Heterogeneity, *, queue_capacity: int = 16
) -> Workflow:
    """Tool-in-the-loop repair: test execution gates a diagnose-review-repair chain."""
    nodes: list[dict[str, object]] = [
        _agent_node("coder", heterogeneity, MBPP_CODER_PROMPT, queue_capacity),
        _function_node("tester", "evaluate_mbpp_initial", queue_capacity),
        _agent_node("diagnoser", heterogeneity, MBPP_DIAGNOSER_PROMPT, queue_capacity),
        _agent_node("reviewer", heterogeneity, MBPP_REVIEWER_PROMPT, queue_capacity),
        _agent_node("repair", heterogeneity, MBPP_REPAIR_PROMPT, queue_capacity),
        _function_node("final_tester", "evaluate_mbpp_final", queue_capacity),
    ]
    edges = [
        {"source": "coder", "target": "tester"},
        {"source": "tester", "target": "diagnoser"},
        {"source": "tester", "target": "reviewer"},
        {"source": "diagnoser", "target": "reviewer"},
        {"source": "tester", "target": "repair"},
        {"source": "diagnoser", "target": "repair"},
        {"source": "reviewer", "target": "repair"},
        {"source": "tester", "target": "final_tester"},
        {"source": "repair", "target": "final_tester"},
    ]
    return _workflow("repair_mbpp", nodes, edges)


def build_mapreduce_qmsum(
    heterogeneity: Heterogeneity,
    *,
    queue_capacity: int = 16,
    chunk_count: int = MOTIVATION_CHUNK_COUNT,
) -> Workflow:
    """Map-reduce summarisation: per-chunk partial answers merged by one model."""
    nodes: list[dict[str, object]] = [
        _function_node(
            "split",
            "split_qmsum_chunks",
            queue_capacity,
            routing="targeted",
            parameters={"chunk_count": chunk_count},
        )
    ]
    nodes.extend(
        _agent_node(
            f"chunk_{index}",
            heterogeneity,
            QMSUM_CHUNK_PROMPT,
            queue_capacity,
            role="chunk",
        )
        for index in range(chunk_count)
    )
    nodes.append(
        _agent_node("merge", heterogeneity, QMSUM_MERGE_PROMPT, queue_capacity)
    )
    edges = [
        {"source": "split", "target": f"chunk_{index}"} for index in range(chunk_count)
    ]
    edges.extend(
        {"source": f"chunk_{index}", "target": "merge"} for index in range(chunk_count)
    )
    return _workflow("mapreduce_qmsum", nodes, edges)


def build_chain_qmsum(
    heterogeneity: Heterogeneity, *, queue_capacity: int = 16
) -> Workflow:
    """Rolling summarisation: parallel lanes of same-model steps, then same-model tail.

    Same models and the same serving profile as the map-reduce variant, so the two
    are interchangeable in a workload without changing which models compete for
    cards. What changes is the graph: the steps within a lane, and the three tail
    steps, are one-to-one edges over one model, which is what a chain-level
    scheduling unit acts on.
    """
    nodes: list[dict[str, object]] = [
        _function_node(
            "split", "fanout_qmsum_lanes", queue_capacity, routing="targeted"
        )
    ]
    edges: list[dict[str, str]] = []
    slice_index = 0
    for lane in CHAIN_LANES:
        for depth in range(CHAIN_DEPTH):
            template = (
                QMSUM_LANE_HEAD_PROMPT if depth == 0 else QMSUM_LANE_STEP_PROMPT
            ).replace("slice_INDEX", f"slice_{slice_index}")
            nodes.append(
                _agent_node(
                    f"{lane}_{depth}",
                    heterogeneity,
                    template,
                    queue_capacity,
                    role="chunk",
                )
            )
            slice_index += 1
            source = "split" if depth == 0 else f"{lane}_{depth - 1}"
            edges.append({"source": source, "target": f"{lane}_{depth}"})
        edges.append(
            {"source": f"{lane}_{CHAIN_DEPTH - 1}", "target": "merge"},
        )

    for role, template in (
        ("merge", QMSUM_LANE_MERGE_PROMPT),
        ("expand", QMSUM_EXPAND_PROMPT),
        ("finalize", QMSUM_FINALIZE_PROMPT),
    ):
        nodes.append(
            _agent_node(role, heterogeneity, template, queue_capacity, role="merge")
        )
    edges.append({"source": "merge", "target": "expand"})
    edges.append({"source": "expand", "target": "finalize"})
    return _workflow("chain_qmsum", nodes, edges)


def build_all(heterogeneity: Heterogeneity) -> tuple[Workflow, ...]:
    return (
        build_moa_gsm8k(heterogeneity),
        build_repair_mbpp(heterogeneity),
        build_mapreduce_qmsum(heterogeneity),
        build_chain_qmsum(heterogeneity),
    )


def _workflow(
    name: str,
    nodes: Sequence[Mapping[str, object]],
    edges: Sequence[Mapping[str, str]],
) -> Workflow:
    return Workflow.model_validate(
        {"workflow_name": name, "nodes": list(nodes), "edges": list(edges)}
    )


def _function_node(
    name: str,
    function: str,
    queue_capacity: int,
    *,
    routing: str = "broadcast",
    parameters: Mapping[str, JsonValue] | None = None,
) -> dict[str, object]:
    node: dict[str, object] = {
        "name": name,
        "type": "function",
        "function": function,
        "routing": routing,
        "queue_capacity": queue_capacity,
        # A targeted fan-out writes one state per successor, so it stays serial.
        "max_concurrency": 1 if routing == "targeted" else 8,
    }
    if parameters is not None:
        node["parameters"] = dict(parameters)
    return node


def _agent_node(
    name: str,
    heterogeneity: Heterogeneity,
    prompt_template: str,
    queue_capacity: int,
    *,
    role: str | None = None,
) -> dict[str, object]:
    role = role or name
    model_name = ROLE_MODELS[heterogeneity][role]
    return {
        "name": name,
        "type": "agent",
        "task": "text_generation",
        "queue_capacity": queue_capacity,
        "model": {"name": model_name},
        "execution": {
            "model_path": MODEL_PATHS[model_name],
            "dtype": "float16",
            "max_new_tokens": ROLE_OUTPUT_LIMITS[role],
            "do_sample": False,
            "use_chat_template": True,
            "enable_thinking": False,
            "serving": dict(MOTIVATION_SERVING),
        },
        "prompt_template": prompt_template,
    }
