from __future__ import annotations

import gc
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import ray
import torch
from langchain.agents import AgentState, create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.graph.state import CompiledStateGraph
from ray.util.queue import Queue

from common.log import get_logger
from workflow.types import (
    ExecutionConfig,
    FailureRecord,
    RetryConfig,
    WorkerQueueItem,
    WorkerState,
    WorkflowDataItem,
)

log_name = "workflow_worker"


def resolve_session_content(
    store: dict[str, dict[str, AgentState]],
    dependencies: list[str],
    item: WorkflowDataItem,
) -> tuple[bool, list[AgentState]]:
    if not dependencies:
        return True, [item.message]

    assert item.source_node in dependencies, (item.source_node, dependencies)
    prev_items = store.setdefault(item.session_id, {})
    assert item.source_node not in prev_items, (item.session_id, item.source_node)
    prev_items[item.source_node] = item.message
    if len(prev_items) < len(dependencies):
        return False, []
    content = [prev_items[source] for source in dependencies]
    del store[item.session_id]
    return True, content


def message_text(message: AgentState) -> str:
    content = message["messages"][-1].content
    assert isinstance(content, str), type(content)
    return content


# max_concurrency 让 get_state/has_result 等查询不被常驻的 loop() 阻塞
@ray.remote(max_concurrency=8)
class NodeWorker:
    def __init__(
        self,
        node_name: str,
        execution_config: ExecutionConfig | None,
        input_queue: Queue,
        output_queues: dict[str, Queue],
        prompt_template: str | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        dependencies: list[str] | None = None,
        retry_config: RetryConfig | None = None,
        failure_queue: Queue | None = None,
        agent_factory: Callable[..., CompiledStateGraph] | None = None,
    ) -> None:
        self.status = WorkerState.IDLE
        self.node_name = node_name
        self.execution_config = execution_config
        self.prompt_template = prompt_template
        self.input_queue = input_queue
        self.output_queues = output_queues
        self.system_prompt = system_prompt or ""
        self.tools = tools or []
        self.dependencies = dependencies or []
        self.retry_config = retry_config or RetryConfig()
        self.failure_queue = failure_queue
        self.agent_factory = agent_factory or build_agent
        self.store: dict[str, dict[str, AgentState]] = {}
        self.pending_items: deque[WorkflowDataItem] = deque()
        self.terminal_results: dict[str, AgentState] = {}
        self.agent: CompiledStateGraph | None = None

    def load(self) -> None:
        assert self.execution_config is not None
        self.agent = self.agent_factory(
            config=self.execution_config,
            system_prompt=self.system_prompt,
            tools=self.tools,
            device=int(self.execution_config.devices[0].split(":")[1]),
        )

    def evict(self) -> None:
        self.agent = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _process_item(self, item: WorkflowDataItem) -> None:
        ready, content = resolve_session_content(self.store, self.dependencies, item)
        if not ready:
            return

        text = "\n\n".join(message_text(message) for message in content)
        if self.execution_config is None:
            output = text
        else:
            prompt = text
            if self.prompt_template is not None:
                prompt = self.prompt_template.format(content=text)
            invoked = self._invoke_agent_with_retry(item, prompt)
            if invoked is None:
                return
            output = invoked

        out_message = AgentState(messages=[AIMessage(content=output)])
        if not self.output_queues:
            self.terminal_results[item.session_id] = out_message
            return
        for target_name, output_queue in self.output_queues.items():
            output_queue.put(
                WorkerQueueItem(
                    data=WorkflowDataItem(
                        session_id=item.session_id,
                        item_id=item.item_id,
                        source_node=self.node_name,
                        target_node=target_name,
                        message=out_message,
                    )
                )
            )

    def _invoke_agent_with_retry(
        self, item: WorkflowDataItem, prompt: str
    ) -> str | None:
        assert self.agent is not None
        logger = get_logger(log_name)
        last_error: Exception | None = None
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                state = AgentState(messages=[HumanMessage(content=prompt)])
                return message_text(self.agent.invoke(state))
            except Exception as error:  # 第三方 agent 调用边界 耗尽后按策略上报
                last_error = error
                max_attempts = self.retry_config.max_attempts
                logger.warning(
                    f"node {self.node_name} attempt {attempt}/{max_attempts} "
                    f"failed for item {item.item_id}: {error}"
                )
                if attempt < self.retry_config.max_attempts:
                    time.sleep(self.retry_config.retry_delay_sec)

        assert last_error is not None
        if self.failure_queue is not None:
            self.failure_queue.put(
                FailureRecord(
                    node_name=self.node_name,
                    session_id=item.session_id,
                    item_id=item.item_id,
                    attempt=self.retry_config.max_attempts,
                    error_type=type(last_error).__name__,
                    error_message=str(last_error),
                )
            )
        if self.retry_config.on_exhausted == "fail_workflow":
            raise last_error
        return None

    def loop(self) -> None:
        self.status = WorkerState.RUNNING
        while True:
            if self.agent is not None and self.pending_items:
                data = self.pending_items.popleft()
            else:
                queue_item: WorkerQueueItem = self.input_queue.get()
                if queue_item.worker_state == WorkerState.STOPPED:
                    self.stop()
                    break
                if queue_item.data is None:
                    continue
                data = queue_item.data
                if self.execution_config is not None and self.agent is None:
                    self.pending_items.append(data)
                    continue

            try:
                self._process_item(data)
            except Exception:
                self.stop()
                raise

    def get_state(self) -> WorkerState:
        return self.status

    def has_result(self, session_id: str) -> bool:
        return session_id in self.terminal_results

    def get_result(self, session_id: str) -> AgentState:
        return self.terminal_results[session_id]

    def stop(self) -> None:
        self.status = WorkerState.STOPPED
        self.evict()


def build_agent(
    config: ExecutionConfig,
    system_prompt: str = "",
    tools: list[BaseTool] | None = None,
    device: int | None = None,
) -> CompiledStateGraph:
    llm = HuggingFacePipeline.from_model_id(
        model_id=config.model_path,
        task="text-generation",
        device=device or int(config.devices[0].split(":")[1]),
        pipeline_kwargs={
            "max_new_tokens": config.max_new_tokens,
            "do_sample": False,
            "return_full_text": False,
        },
    )
    return create_agent(
        model=ChatHuggingFace(llm=llm),
        tools=tools or [],
        system_prompt=system_prompt,
    )
