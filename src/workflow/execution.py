from __future__ import annotations

from queue import Queue

from workflow.types import (
    Workflow,
    WorkflowQueueItem,
    WorkflowRunMode,
)


def run_workflow(
    workflow: Workflow,
    mode: WorkflowRunMode = "serial",
    parallelism: int | None = None,
):
    if mode == "serial":
        run_workflow_serial(workflow)
    elif mode == "parallel":
        assert parallelism is not None and parallelism > 0, (
            "parallelism must be a positive integer for parallel run mode"
        )
        run_workflow_parallel(workflow, parallelism)
    else:
        raise ValueError(f"unsupported workflow run mode: {mode}")


def run_workflow_serial(workflow: Workflow):
    nodes = workflow.get_topological_node_list()
    results = []
    for node in nodes:
        res = node.run()
        # TODO 汇总给结果
    # TODO 处理和返回结果


def run_workflow_parallel(workflow: Workflow, parallelism: int):
    sender = Queue()
    receiver = Queue()
    results = []
    for node in workflow.nodes:
        while all(
            (
                len(node.previous_nodes) > 0,
                any(prev_node.status == "pending" for prev_node in node.previous_nodes),
            )
        ):
            res = receiver.get()
            results.append(res)
            receiver.task_done()

        node.status = "pending"
        sender.put(WorkflowQueueItem(node=node))

    sender.join()
    for _ in range(receiver.qsize()):
        res = receiver.get()
        results.append(res)
        receiver.task_done()
    for _ in range(parallelism):
        sender.put(WorkflowQueueItem(signal="done"))
    sender.join()
    return results


def worker(sender: Queue, receiver: Queue):
    while True:
        item: WorkflowQueueItem = sender.get()
        if item.signal == "done":
            sender.task_done()
            break
        assert item.node is not None
        res = item.node.run()
        item.node.status = "completed"
        receiver.put(res)
        sender.task_done()
