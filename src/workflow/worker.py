from __future__ import annotations

from typing import Any

import ray
from langchain.agents import AgentState, create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from ray.util.queue import Queue

from src.workflow.types import WorkflowDataItem


@ray.remote
class NodeWorker:
    def __init__(
        self,
        node_name: str
        model: BaseChatModel,
        prompt_template: str,
        input_queue: Queue,
        output_queues: dict[str, Queue], # {output_name: Queue}
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        denpendencies: list[str] | None = None,
        buffer_max_size: int = 20,
    ) -> None:
        """
        工具为空时，agent 只负责一次模型推理，一进一出，不会有 ReAct loop
        """
        if tools is None:
            tools = []
        self.agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )
        self.node_name = node_name
        self.input_queue = input_queue
        self.output_queues = output_queues
        self.prompt_template = prompt_template
        if denpendencies is None:
            denpendencies = []
        self.dependencies = denpendencies
        self.store: dict[
            str, dict[str, Any]
        ] = {}  # {session_id: {source_node: message}}

    def invoke(self, messages: AgentState) -> dict[str, Any]:
        return self.agent.invoke(messages)

    def loop(self) -> None:
        while True:
            item: WorkflowDataItem = self.input_queue.get()
            prev_items = self.store.get(item.session_id, {})
            prev_items[item.item_id] = item.message
            self.store[item.source_node] = prev_items

            if len(self.store[item.session_id]) < len(self.dependencies):
                continue

            content = [
                self.store[item.session_id][source] for source in self.dependencies
            ]
            prompt = self.prompt_template.format(content=content)  # TODO: 使用更强的模版应用方式

            result = self.agent.invoke(prompt)
            output = result["messages"][-1].content
            for output_name, output_queue in self.output_queues.items():
                output_item = WorkflowDataItem(
                    session_id=item.session_id,
                    item_id=item.item_id,
                    source_node=self.node_name,
                    target_node=output_name,
                    message=AgentState(messages=[{"role": "assistant", "content": output}]),
                )
                output_queue.put(output_item)
