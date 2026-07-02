from __future__ import annotations

import gc
from collections import deque
from typing import Any

import ray
import torch
from langchain.agents import AgentState, create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.graph.state import CompiledStateGraph
from ray.util.queue import Queue

from src.workflow.types import (
    ExecutionConfig,
    WorkerQueueItem,
    WorkerState,
    WorkflowDataItem,
)


@ray.remote
class NodeWorker:
    def __init__(
        self,
        node_name: str,
        execution_config: ExecutionConfig,
        input_queue: Queue,
        output_queues: dict[str, Queue],
        prompt_template: str | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        denpendencies: list[str] | None = None,
    ) -> None:
        self.status = WorkerState.IDLE
        self.node_name = node_name
        self.execution_config = execution_config
        self.prompt_template = prompt_template
        self.input_queue: Queue[WorkerQueueItem] = input_queue
        self.output_queues: dict[str, Queue[WorkerQueueItem]] = output_queues
        self.system_prompt = system_prompt or ""
        self.tools = tools or []
        self.dependencies = denpendencies or []
        self.store: dict[str, dict[str, Any]] = {}
        self.pending_items: deque[WorkflowDataItem] = deque()
        self.agent: CompiledStateGraph | None = None

    def load(self, device_name: str) -> None:
        self.agent = build_agent(
            config=self.execution_config,
            system_prompt=self.system_prompt,
            tools=self.tools,
            device=int(device_name.split(":")[1]),
        )

    def evict(self) -> None:

        self.agent = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _process_item(self, item: WorkflowDataItem) -> None:
        prev_items = self.store.get(item.session_id, {})
        prev_items[item.item_id] = item.message
        self.store[item.session_id] = prev_items

        if len(self.store[item.session_id]) < len(self.dependencies):
            return

        content = [
            self.store[item.session_id][source] for source in self.dependencies
        ]
        prompt = content
        if self.prompt_template is not None:
            prompt = self.prompt_template.format(content=content)

        assert self.agent is not None
        result = self.agent.invoke(prompt)
        output = result["messages"][-1].content
        for output_name, output_queue in self.output_queues.items():
            output_queue.put(
                WorkflowDataItem(
                    session_id=item.session_id,
                    item_id=item.item_id,
                    source_node=self.node_name,
                    target_node=output_name,
                    message=AgentState(
                        messages=[AIMessage(content=output)]
                    ),
                )
            )

    def loop(self) -> None:
        self.status = WorkerState.RUNNING
        while True:
            if self.agent is not None and self.pending_items:
                item = self.pending_items.popleft()
                self._process_item(item)
                continue

            item: WorkerQueueItem = self.input_queue.get()
            state = item.worker_state
            if state is not None and state == WorkerState.STOPPED:
                self.stop()
                break
            data = item.data
            if data is None:
                continue

            if self.agent is None:
                self.pending_items.append(data)
                continue

            self._process_item(data)

    def invoke(self, messages: AgentState) -> dict[str, Any]:
        assert self.agent is not None
        return self.agent.invoke(messages)
    
    def stop(self):
        # TODO 完善停止逻辑
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
