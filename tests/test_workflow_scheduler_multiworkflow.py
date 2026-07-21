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
    LoadReplicaAction,
    SchedulerCore,
)
from workflow.schema import Workflow
from workflow.types import ModelReplicaState, SessionState, WorkflowModelFeatureKey

SHARED_MODEL_PATH = "/models/shared-model"


def agent_workflow(workflow_name: str, node_name: str) -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": workflow_name,
            "nodes": [
                {
                    "name": node_name,
                    "type": "agent",
                    "model": {"name": "shared-model"},
                    "execution": {
                        "model_path": SHARED_MODEL_PATH,
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
            ],
            "edges": [],
        }
    )


def prediction(gpu_kind: str = "v100") -> ResourceContract:
    return ResourceContract(
        key=WorkflowModelFeatureKey(
            model_name="shared-model",
            phase="decode",
            gpu_name=gpu_kind,
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


def multiworkflow_core(*, accelerator_count: int = 1) -> SchedulerCore:
    accelerators = tuple(
        AcceleratorConfig(
            hostname="node",
            gpu_kind="v100",
            local_index=index,
            total_mem_mb=16_000,
        )
        for index in range(accelerator_count)
    )
    return SchedulerCore(
        agent_workflow("workflow-a", "agent_a"),
        scheduler_config=SchedulerConfig(accelerators=accelerators),
        predictions=ResourceContractCache(version=1, entries=(prediction(),)),
    )


def request_agent(
    core: SchedulerCore,
    workflow_name: str,
    node_name: str,
    session_id: str,
    *,
    created_at: float,
) -> tuple[str, str]:
    core.register_session(session_id, workflow_name)
    task_id = core.begin_node(session_id, node_name, [f"input-{session_id}"])
    acquire_id = core.request_acquire(task_id, input_tokens=1100, created_at=created_at)
    return task_id, acquire_id


def agent_report(core: SchedulerCore, grant_task_id: str, acquire_id: str) -> None:
    grant = core.grants[acquire_id]
    task = core.tasks[grant_task_id]
    report = AgentTaskRuntimeReport(
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
    )
    core.complete(task.task_id, report, acquire_id=acquire_id)


def test_register_and_deregister_workflow_lifecycle() -> None:
    core = multiworkflow_core()
    second = agent_workflow("workflow-b", "agent_b")

    core.register_workflow(second)

    assert set(core._workflows) == {"workflow-a", "workflow-b"}
    assert set(core._nodes) == {"workflow-a", "workflow-b"}
    assert set(core._node_order) == {"workflow-a", "workflow-b"}

    core.deregister_workflow("workflow-b")

    assert set(core._workflows) == {"workflow-a"}
    assert "workflow-b" not in core._nodes
    assert "workflow-b" not in core._node_order


def test_registering_duplicate_workflow_name_raises() -> None:
    core = multiworkflow_core()

    try:
        core.register_workflow(agent_workflow("workflow-a", "duplicate-agent"))
    except ValueError as error:
        assert "already registered" in str(error)
    else:
        raise AssertionError("expected a ValueError for a duplicate workflow_name")


def test_deregistering_workflow_with_active_sessions_raises() -> None:
    core = multiworkflow_core()
    core.register_workflow(agent_workflow("workflow-b", "agent_b"))
    request_agent(core, "workflow-b", "agent_b", "s1", created_at=1.0)

    try:
        core.deregister_workflow("workflow-b")
    except ValueError as error:
        assert "active sessions" in str(error)
    else:
        raise AssertionError("expected a ValueError for a workflow with active sessions")


def test_unknown_workflow_name_is_rejected_at_session_registration() -> None:
    core = multiworkflow_core()

    try:
        core.register_session("s1", "not-registered")
    except KeyError:
        pass
    else:
        raise AssertionError("expected a KeyError for an unregistered workflow_name")


def test_cross_workflow_replica_sharing_reuses_one_replica() -> None:
    core = multiworkflow_core()
    core.register_workflow(agent_workflow("workflow-b", "agent_b"))

    task_a, acquire_a = request_agent(core, "workflow-a", "agent_a", "s-a", created_at=1.0)
    actions = core.tick_once(now=2.0)
    assert len(actions) == 1
    load = actions[0]
    assert isinstance(load, LoadReplicaAction)
    core.complete_load(
        load.replica_id,
        ReplicaLoadResult(
            physical_gpu_id=0,
            duration_sec=1.0,
            idle_vram_mb=4_000,
            block_size=16,
            num_gpu_blocks=100,
            gpu_kv_tokens=1_600,
        ),
        backend_handle="backend",
        now=3.0,
    )
    assert core.tick_once(now=3.0) == []
    grant_a = core.poll_grant(acquire_a)
    assert grant_a is not None
    # Release workflow-a's lease so the replica goes IDLE again: cross-workflow
    # sharing still requires an exact batch-1 prediction match, same as any
    # other reuse, since the runtime never extrapolates missing batch buckets.
    agent_report(core, task_a, acquire_a)
    assert core.replicas[grant_a.replica_id].state == ModelReplicaState.IDLE

    _, acquire_b = request_agent(core, "workflow-b", "agent_b", "s-b", created_at=4.0)
    assert core.tick_once(now=5.0) == []
    grant_b = core.poll_grant(acquire_b)

    assert grant_b is not None
    assert grant_b.replica_id == grant_a.replica_id
    assert len(core.replicas) == 1


def test_duration_history_does_not_cross_pollute_between_workflows() -> None:
    core = multiworkflow_core()
    core.register_workflow(agent_workflow("workflow-b", "agent_a"))

    task_a, acquire_a = request_agent(core, "workflow-a", "agent_a", "s-a", created_at=1.0)
    task_b, acquire_b = request_agent(core, "workflow-b", "agent_a", "s-b", created_at=1.0)

    core.record_runtime_report(
        AgentTaskRuntimeReport(
            acquire_id=acquire_a,
            task_id=task_a,
            session_id="s-a",
            node_id="agent_a",
            input_item_ids=["input-s-a"],
            model_key="shared-model-key",
            accelerator_id="node/v100:0",
            gpu_kind="v100",
            input_tokens=64,
            max_new_tokens=16,
            output_tokens=8,
            hit_token_limit=False,
            replica_inflight_at_start=1,
            started_at=1.0,
            finished_at=11.0,
            duration_sec=10.0,
            status="success",
        ),
        "workflow-a",
    )
    core.record_runtime_report(
        AgentTaskRuntimeReport(
            acquire_id=acquire_b,
            task_id=task_b,
            session_id="s-b",
            node_id="agent_a",
            input_item_ids=["input-s-b"],
            model_key="shared-model-key",
            accelerator_id="node/v100:0",
            gpu_kind="v100",
            input_tokens=64,
            max_new_tokens=16,
            output_tokens=8,
            hit_token_limit=False,
            replica_inflight_at_start=1,
            started_at=1.0,
            finished_at=2.0,
            duration_sec=1.0,
            status="success",
        ),
        "workflow-b",
    )

    history_a = core.history[("workflow-a", "agent_a", "shared-model-key", "v100")]
    history_b = core.history[("workflow-b", "agent_a", "shared-model-key", "v100")]
    assert history_a.duration_sec_ema == 10.0
    assert history_b.duration_sec_ema == 1.0


def test_deregistering_one_workflow_does_not_disturb_another() -> None:
    core = multiworkflow_core()
    core.register_workflow(agent_workflow("workflow-b", "agent_b"))

    task_a, acquire_a = request_agent(core, "workflow-a", "agent_a", "s-a", created_at=1.0)
    _, acquire_b = request_agent(core, "workflow-b", "agent_b", "s-b", created_at=1.0)

    with_active_session_error = None
    try:
        core.deregister_workflow("workflow-a")
    except ValueError as error:
        with_active_session_error = error
    assert with_active_session_error is not None

    core.cancel_acquire(acquire_a)
    core.fail_node(task_a, "Stopped", "test cleanup")
    assert core.sessions["s-a"].state == SessionState.FAILED

    core.deregister_workflow("workflow-a")

    assert "workflow-a" not in core._workflows
    assert core.sessions["s-b"].state == SessionState.ACTIVE
    assert acquire_b in core.pending_acquires
