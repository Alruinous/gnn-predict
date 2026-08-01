"""cross_workflow_lifecycle: scoping the lifecycle decisions, not the replica pool."""

from __future__ import annotations

from workflow.artifacts import (
    AcceleratorConfig,
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
    SchedulerConfig,
)
from workflow.replica import ReplicaLoadResult
from workflow.scheduler import (
    AgentTaskRuntimeReport,
    FunctionTaskRuntimeReport,
    LoadReplicaAction,
    ModelReplicaRecord,
    OutputReport,
    SchedulerCore,
)
from workflow.schema import Workflow
from workflow.types import ModelReplicaState, WorkflowModelFeatureKey

MODEL_NAME = "shared-model"


def agent_node(name: str) -> dict[str, object]:
    return {
        "name": name,
        "type": "agent",
        "model": {"name": MODEL_NAME},
        "execution": {
            "model_path": "/models/shared-model",
            "max_new_tokens": 16,
            "dtype": "float16",
            "serving": {
                "max_model_len": 2048,
                "max_num_seqs": 1,
                "max_num_batched_tokens": 2048,
            },
        },
        "prompt_template": "{content}",
    }


def flat_workflow(workflow_name: str) -> Workflow:
    """One agent node: enough to contend for a replica and to be ordered."""
    return Workflow.model_validate(
        {"workflow_name": workflow_name, "nodes": [agent_node("agent")], "edges": []}
    )


def chain_workflow(workflow_name: str) -> Workflow:
    """function -> agent, so the agent can become a near-ready candidate."""
    return Workflow.model_validate(
        {
            "workflow_name": workflow_name,
            "nodes": [
                {"name": "upstream", "type": "function", "function": "upstream"},
                agent_node("agent"),
            ],
            "edges": [{"source": "upstream", "target": "agent"}],
        }
    )


def prediction() -> ResourceContract:
    return ResourceContract(
        key=WorkflowModelFeatureKey(
            model_name=MODEL_NAME,
            phase="decode",
            gpu_name="v100",
            batch_size=1,
            sequence_length=2048,
            decode_output_length=512,
        ),
        source=ResourceContractSource.SYNTHETIC_FIXTURE,
        predicted_load_sec=5.0,
        predicted_run_sec=1.0,
        predicted_peak_vram_mb=8_000,
        peak_vram_mb_upper_bound=8_000,
        peak_vram_mb_evidence=ResourceEvidence(
            method="point_estimate_only", sample_count=1
        ),
    )


def scoped_core(workflow: Workflow, *, cross_workflow_lifecycle: bool) -> SchedulerCore:
    return SchedulerCore(
        workflow,
        scheduler_config=SchedulerConfig(
            accelerators=(
                AcceleratorConfig(
                    hostname="node",
                    gpu_kind="v100",
                    local_index=0,
                    total_mem_mb=16_000,
                ),
            ),
            eps_time_sec=0.5,
            cross_workflow_lifecycle=cross_workflow_lifecycle,
        ),
        predictions=ResourceContractCache(version=1, entries=(prediction(),)),
    )


def request_agent(
    core: SchedulerCore,
    workflow_name: str,
    session_id: str,
    *,
    created_at: float,
) -> tuple[str, str]:
    core.register_session(session_id, workflow_name)
    task_id = core.begin_node(session_id, "agent", [f"input-{session_id}"])
    acquire_id = core.request_acquire(task_id, input_tokens=1100, created_at=created_at)
    return task_id, acquire_id


def load_replica(core: SchedulerCore, action: LoadReplicaAction, *, now: float) -> None:
    core.complete_load(
        action.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=1.0,
            idle_vram_mb=4_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle="backend",
        now=now,
    )


def release_lease(core: SchedulerCore, task_id: str, acquire_id: str) -> None:
    grant = core.grants[acquire_id]
    task = core.tasks[task_id]
    core.complete(
        task_id,
        AgentTaskRuntimeReport(
            acquire_id=grant.acquire_id,
            task_id=task.task_id,
            session_id=task.session_id,
            node_id=task.node_id,
            input_item_ids=list(task.input_item_ids),
            model_key=grant.model_key,
            accelerator_id=grant.accelerator_ids[0],
            gpu_kind=grant.gpu_kind,
            input_tokens=grant.input_tokens,
            max_new_tokens=grant.max_new_tokens,
            output_tokens=8,
            hit_token_limit=False,
            started_at=1.0,
            finished_at=2.0,
            duration_sec=1.0,
            status="success",
        ),
        acquire_id=acquire_id,
    )


def finish_upstream(core: SchedulerCore, session_id: str) -> None:
    task_id = core.sessions[session_id].task_ids["upstream"]
    core.complete(
        task_id,
        FunctionTaskRuntimeReport(
            task_id=task_id,
            session_id=session_id,
            node_id="upstream",
            input_item_ids=["input"],
            started_at=0.0,
            finished_at=0.5,
            duration_sec=0.5,
            status="success",
        ),
    )
    core.finish_node(
        task_id,
        OutputReport(
            task_id=task_id,
            output_item_ids=["out"],
            persisted_terminal_result=False,
        ),
    )


def seed_idle_replica(core: SchedulerCore, workflow_name: str) -> ModelReplicaRecord:
    """Serve one request of `workflow_name`, leaving its replica loaded and idle."""
    session_id = f"s-owner-{workflow_name}"
    core.register_session(session_id, workflow_name)
    core.begin_node(session_id, "upstream", ["input"])
    finish_upstream(core, session_id)
    task_id = core.begin_node(session_id, "agent", ["out"])
    acquire_id = core.request_acquire(task_id, input_tokens=1100, created_at=0.75)

    actions = core.tick_once(now=1.0)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, LoadReplicaAction)
    load_replica(core, action, now=2.0)
    assert core.tick_once(now=2.0) == []
    grant = core.poll_grant(acquire_id)
    assert grant is not None
    release_lease(core, task_id, acquire_id)

    replica = core.replicas[grant.replica_id]
    assert replica.state == ModelReplicaState.IDLE
    assert replica.owner_workflow == workflow_name
    return replica


def arm_near_ready(
    core: SchedulerCore,
    workflow_name: str,
    session_id: str,
    *,
    finish_at: float,
) -> None:
    core.register_session(session_id, workflow_name)
    core.begin_node(session_id, "upstream", ["input"])
    core.record_running_upstream(session_id, "upstream", finish_at=finish_at)


def test_scoped_lifecycle_still_shares_the_replica_across_workflows() -> None:
    # The pool cannot afford one replica per workflow, so the ablation leaves
    # model_key alone: a resident model keeps serving whoever asks for it.
    core = scoped_core(flat_workflow("workflow-a"), cross_workflow_lifecycle=False)
    core.register_workflow(flat_workflow("workflow-b"))

    task_a, acquire_a = request_agent(core, "workflow-a", "s-a", created_at=1.0)
    actions = core.tick_once(now=2.0)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, LoadReplicaAction)
    assert action.owner_workflow == "workflow-a"
    load_replica(core, action, now=3.0)
    assert core.tick_once(now=3.0) == []
    grant_a = core.poll_grant(acquire_a)
    assert grant_a is not None
    release_lease(core, task_a, acquire_a)

    _, acquire_b = request_agent(core, "workflow-b", "s-b", created_at=4.0)
    assert core.tick_once(now=5.0) == []
    grant_b = core.poll_grant(acquire_b)

    assert grant_b is not None
    assert grant_b.replica_id == grant_a.replica_id
    assert len(core.replicas) == 1
    assert core.replicas[grant_b.replica_id].owner_workflow == "workflow-a"


def test_joint_lifecycle_lets_another_workflow_protect_a_replica() -> None:
    core = scoped_core(chain_workflow("workflow-a"), cross_workflow_lifecycle=True)
    core.register_workflow(chain_workflow("workflow-b"))
    replica = seed_idle_replica(core, "workflow-a")
    arm_near_ready(core, "workflow-b", "s-near-b", finish_at=40.0)

    near = core._collect_near_ready_tasks()

    assert len(near) == 1
    assert core._future_reuse_distance(replica, 10.0, near) == 30.0


def test_scoped_lifecycle_ignores_another_workflows_demand_for_the_same_model() -> None:
    core = scoped_core(chain_workflow("workflow-a"), cross_workflow_lifecycle=False)
    core.register_workflow(chain_workflow("workflow-b"))
    replica = seed_idle_replica(core, "workflow-a")
    arm_near_ready(core, "workflow-b", "s-near-b", finish_at=40.0)

    near = core._collect_near_ready_tasks()
    assert len(near) == 1  # workflow-b really does want this model

    # It just no longer extends workflow-a's residency, and no pending workflow-a
    # node wants the model either, so nothing is keeping this replica alive.
    assert core._future_reuse_distance(replica, 10.0, near) == float("inf")


def test_scoped_lifecycle_falls_back_to_per_workflow_interleaving() -> None:
    # Global locality-first ordering is itself a cross-workflow lifecycle mechanism:
    # it deliberately breaks fair merging to protect a residency. Scoped, each
    # workflow orders its own queue and the merge across workflows is fair again.
    core = scoped_core(flat_workflow("workflow-a"), cross_workflow_lifecycle=False)
    core.register_workflow(flat_workflow("workflow-b"))
    request_agent(core, "workflow-a", "s-a1", created_at=1.0)
    request_agent(core, "workflow-a", "s-a2", created_at=2.0)
    request_agent(core, "workflow-b", "s-b1", created_at=3.0)

    values = tuple(core.pending_acquires.values())
    ordered = core._order_pending(values, now=4.0)

    assert ordered == core._interleave_by_workflow(values, 4.0)
    assert [core.tasks[pending.task_id].session_id for pending in ordered] == [
        "s-a1",
        "s-b1",
        "s-a2",
    ]


def test_joint_lifecycle_keeps_global_locality_first_ordering() -> None:
    core = scoped_core(flat_workflow("workflow-a"), cross_workflow_lifecycle=True)
    core.register_workflow(flat_workflow("workflow-b"))
    request_agent(core, "workflow-a", "s-a1", created_at=1.0)
    request_agent(core, "workflow-a", "s-a2", created_at=2.0)
    request_agent(core, "workflow-b", "s-b1", created_at=3.0)

    values = tuple(core.pending_acquires.values())
    ordered = core._order_pending(values, now=4.0)

    assert [core.tasks[pending.task_id].session_id for pending in ordered] == [
        "s-a1",
        "s-a2",
        "s-b1",
    ]
