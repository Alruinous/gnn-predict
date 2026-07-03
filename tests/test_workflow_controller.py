from __future__ import annotations

from pathlib import Path

from workflow.controller import WorkflowController, prepare_queues_for_workflow
from workflow.types import (
    FailureRecord,
    WorkerState,
    Workflow,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "config/workflow/runtime_smoke_20260702/fan_in_smoke.yaml"


def make_workflow(queue_capacity: int = 3) -> Workflow:
    execution = {
        "model_name": "stub",
        "model_path": "/models/stub",
        "devices": ["cuda:0"],
    }
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "reader", "type": "input", "queue_capacity": queue_capacity},
                {
                    "name": "worker_a",
                    "type": "agent",
                    "queue_capacity": queue_capacity,
                    "execution": execution,
                },
                {"name": "sink", "type": "output", "queue_capacity": queue_capacity},
            ],
            "edges": [
                {"source": "reader", "target": "worker_a"},
                {"source": "worker_a", "target": "sink"},
            ],
        }
    )


def test_prepare_queues_for_workflow_shares_queue_instance_between_edge_endpoints(
    ray_session,
):
    input_queue_map, output_queues_map, adjacency_list, dependencies = (
        prepare_queues_for_workflow(make_workflow())
    )
    assert set(input_queue_map) == {"reader", "worker_a", "sink"}
    assert output_queues_map["reader"]["worker_a"] is input_queue_map["worker_a"]
    assert output_queues_map["worker_a"]["sink"] is input_queue_map["sink"]
    assert output_queues_map["sink"] == {}
    assert adjacency_list["sink"] == []
    assert dependencies["reader"] == []


def test_prepare_queues_for_workflow_applies_queue_capacity(ray_session):
    input_queue_map, _, _, _ = prepare_queues_for_workflow(make_workflow(queue_capacity=3))
    assert input_queue_map["worker_a"].maxsize == 3


def test_workflow_controller_builds_from_yaml(ray_session):
    controller = WorkflowController.from_yaml(str(SMOKE_CONFIG))
    assert controller.entry_node_name == "reader"
    assert controller.terminal_node_names == ["reducer"]
    assert set(controller.workers) == {"reader", "chunk_a", "chunk_b", "reducer"}
    assert set(controller.input_queue_map) == set(controller.workers)


def test_stop_workflow_sends_stopped_sentinel_through_input_queue(ray_session):
    controller = WorkflowController(make_workflow())
    controller.stop_workflow()
    for node_name in controller.workers:
        sentinel = controller.input_queue_map[node_name].get(timeout=10)
        assert sentinel.worker_state == WorkerState.STOPPED


def test_get_workflow_status_reports_node_states_and_queue_sizes(ray_session):
    controller = WorkflowController(make_workflow())
    status = controller.get_workflow_status()
    assert set(status.node_states) == {"reader", "worker_a", "sink"}
    assert all(state == WorkerState.IDLE for state in status.node_states.values())
    assert status.queue_sizes == {"reader": 0, "worker_a": 0, "sink": 0}
    assert status.failures == []

    controller.failure_queue.put(
        FailureRecord(
            node_name="worker_a",
            session_id="s1",
            item_id="i1",
            attempt=1,
            error_type="RuntimeError",
            error_message="boom",
        )
    )
    status = controller.get_workflow_status()
    assert [record.node_name for record in status.failures] == ["worker_a"]
    # failure 已被收编进 controller.failures，再次查询仍然可见
    status = controller.get_workflow_status()
    assert [record.node_name for record in status.failures] == ["worker_a"]
