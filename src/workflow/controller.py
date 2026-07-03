from __future__ import annotations

from collections.abc import Callable

import ray
import yaml
from langchain.agents import AgentState
from langgraph.graph.state import CompiledStateGraph
from ray.util.queue import Empty, Queue

from common.log import get_logger
from workflow.types import (
    FailureRecord,
    WorkerQueueItem,
    WorkerState,
    Workflow,
    WorkflowDataItem,
    WorkflowStatus,
)
from workflow.worker import NodeWorker

log_name = "workflow_controller"


class WorkflowController:
    def __init__(
        self,
        workflow: Workflow,
        agent_factory: Callable[..., CompiledStateGraph] | None = None,
    ):
        self.workflow = workflow
        input_queue_map, output_queues_map, adjacency_list, dependencies = (
            prepare_queues_for_workflow(workflow)
        )
        entry_nodes = [name for name in workflow.node_names() if not dependencies[name]]
        assert len(entry_nodes) == 1, entry_nodes  # 多入口节点不在当前阶段支持范围
        self.entry_node_name = entry_nodes[0]
        self.terminal_node_names = [
            name for name in workflow.node_names() if not adjacency_list[name]
        ]
        self.input_queue_map = input_queue_map
        self.entry_queue: Queue = input_queue_map[self.entry_node_name]
        self.failure_queue: Queue = Queue()
        self.failures: list[FailureRecord] = []
        self.workers = prepare_node_workers(
            workflow,
            input_queue_map,
            output_queues_map,
            dependencies,
            self.failure_queue,
            agent_factory,
        )

    def start_workflow(self) -> None:
        logger = get_logger(log_name)
        node_map = self.workflow.node_map()
        load_refs = []
        for node_name, worker in self.workers.items():
            if node_map[node_name].execution is None:
                continue
            logger.info(f"Loading model for node: {node_name}")
            load_refs.append(worker.load.remote())
        ray.get(load_refs)
        for node_name, worker in self.workers.items():
            logger.info(f"Starting worker for node: {node_name}")
            worker.loop.remote()  # 限定每个 worker 使用一个设备

    def submit(self, session_id: str, item_id: str, message: AgentState) -> None:
        self.entry_queue.put(
            WorkerQueueItem(
                data=WorkflowDataItem(
                    session_id=session_id,
                    item_id=item_id,
                    source_node=self.entry_node_name,
                    target_node=self.entry_node_name,
                    message=message,
                )
            )
        )

    def is_session_complete(self, session_id: str) -> bool:
        checks = [
            self.workers[name].has_result.remote(session_id)
            for name in self.terminal_node_names
        ]
        return all(ray.get(checks))

    def get_session_results(self, session_id: str) -> dict[str, AgentState]:
        return {
            name: ray.get(self.workers[name].get_result.remote(session_id))
            for name in self.terminal_node_names
        }

    def stop_workflow(self) -> None:
        logger = get_logger(log_name)
        for node_name in self.workers:
            logger.info(f"Stopping worker for node: {node_name}")
            self.input_queue_map[node_name].put(
                WorkerQueueItem(worker_state=WorkerState.STOPPED)
            )

    def get_workflow_status(self) -> WorkflowStatus:
        node_states = {
            name: ray.get(worker.get_state.remote())
            for name, worker in self.workers.items()
        }
        queue_sizes = {
            name: queue.qsize() for name, queue in self.input_queue_map.items()
        }
        while True:
            try:
                self.failures.append(self.failure_queue.get_nowait())
            except Empty:
                break
        return WorkflowStatus(
            node_states=node_states,
            queue_sizes=queue_sizes,
            failures=list(self.failures),
        )

    @classmethod
    def from_yaml(
        cls,
        path: str,
        agent_factory: Callable[..., CompiledStateGraph] | None = None,
    ) -> WorkflowController:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(Workflow.model_validate(raw), agent_factory=agent_factory)


def prepare_node_workers(
    workflow: Workflow,
    input_queue_map: dict[str, Queue],
    output_queues_map: dict[str, dict[str, Queue]],
    dependencies: dict[str, list[str]],
    failure_queue: Queue,
    agent_factory: Callable[..., CompiledStateGraph] | None = None,
) -> dict[str, ray.actor.ActorHandle]:
    node_map = workflow.node_map()
    node_workers = {}
    for node_name in workflow.node_names():
        node_info = node_map[node_name]
        node_workers[node_name] = NodeWorker.remote(
            node_name=node_name,
            execution_config=node_info.execution,
            prompt_template=node_info.prompt_template,
            input_queue=input_queue_map[node_name],
            output_queues=output_queues_map[node_name],
            tools=[],  # 当前场景暂时不需要
            system_prompt=node_info.system_prompt,
            dependencies=dependencies[node_name],
            retry_config=node_info.retry,
            failure_queue=failure_queue,
            agent_factory=agent_factory,
        )
    return node_workers


def prepare_queues_for_workflow(
    workflow: Workflow,
) -> tuple[
    dict[str, Queue],
    dict[str, dict[str, Queue]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    node_names = workflow.node_names()
    node_map = workflow.node_map()
    adjacency_list: dict[str, list[str]] = {node_name: [] for node_name in node_names}
    dependencies: dict[str, list[str]] = {node_name: [] for node_name in node_names}
    for edge in workflow.edges:
        adjacency_list[edge.source].append(edge.target)
        dependencies[edge.target].append(edge.source)
    # 每个节点一个共享 input queue 上游 output queue 必须复用同一对象 否则边不连通
    input_queue_map = {
        node_name: Queue(maxsize=node_map[node_name].queue_capacity)
        for node_name in node_names
    }
    output_queues_map = {
        source_node: {
            target_node: input_queue_map[target_node]
            for target_node in adjacency_list[source_node]
        }
        for source_node in node_names
    }
    return input_queue_map, output_queues_map, adjacency_list, dependencies
