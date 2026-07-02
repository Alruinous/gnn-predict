from __future__ import annotations

from ray.util.queue import Queue

from src.common.log import get_logger
from src.workflow.types import NodeConfig, Workflow
from src.workflow.worker import NodeWorker

log_name = "workflow_controller"


class WorkflowController:
    def __init__(
        self,
        workflow: Workflow,
    ):
        self.workflow = workflow
        workers, entry_queue = prepare_node_workers(workflow)
        self.entry_queue: Queue = entry_queue
        self.workers: dict[str, NodeWorker] = workers

    def start_workflow(self):
        logger = get_logger(log_name)
        for node_name, worker in self.workers.items():
            logger.info(f"Starting worker for node: {node_name}")
            worker.loop.remote()  # 限定每个 worker 使用一个设备

    def stop_workflow(self):
        logger = get_logger(log_name)
        for node_name, worker in self.workers.items():
            logger.info(f"Stopping worker for node: {node_name}")
            worker.stop.remote()

    def get_workflow_status(self):
        # Implement the logic to get the current status of the workflow
        pass

    @classmethod
    def from_yaml(cls, path: str) -> WorkflowController:
        import yaml
        with open(path, "r") as f:
            dc = yaml.safe_load(f)
            


def prepare_node_workers(
    workflow: Workflow,
) -> tuple[dict[str, NodeWorker], Queue]:
    input_queue_map, output_queues_map, _, dependencies = prepare_queues_for_workflow(
        workflow,
    )
    node_names = workflow.node_names()
    node_map = workflow.node_map()
    node_workers = {}
    for node_name in node_names:
        node_info: NodeConfig = node_map[node_name]
        input_queue = input_queue_map[node_name]
        output_queues = output_queues_map[node_name]
        if len(dependencies[node_name]) == 0:
            entry_queue = input_queue

        node_worker = NodeWorker(
            node_name=node_name,
            execution_config=node_info.execution,
            prompt_template=node_info.prompt_template,
            input_queue=input_queue,
            output_queues=output_queues,
            tools=[],  # 当前场景暂时不需要
            system_prompt="",  # TODO NodeConfig 中还没有 system_prompt 字段，后续添加
            denpendencies=dependencies[node_info.name],
        )
        node_workers[node_name] = node_worker

    return node_workers, entry_queue


def prepare_queues_for_workflow(
    workflow: Workflow,
) -> tuple[
    dict[str, Queue],
    dict[str, dict[str, Queue]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    node_names = workflow.node_names()
    edges = workflow.edges
    adjacency_list: dict[str, list[str]] = {node_name: [] for node_name in node_names}
    dependencies: dict[str, list[str]] = {node_name: [] for node_name in node_names}
    for edge in edges:
        adjacency_list[edge.source].append(edge.target)
        dependencies[edge.target].append(edge.source)
    input_queue_map = {
        node_name: Queue()
        for node_name in node_names
        if len(dependencies[node_name]) > 0
    }
    output_queues_map = {
        source_node: {
            target_node: Queue() for target_node in adjacency_list[source_node]
        }
        for source_node in node_names
        if len(adjacency_list[source_node]) > 0
    }
    return input_queue_map, output_queues_map, adjacency_list, dependencies
