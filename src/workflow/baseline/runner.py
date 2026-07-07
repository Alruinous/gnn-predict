from __future__ import annotations

from typing import Protocol, cast

from langchain.agents import AgentState, create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.graph.state import CompiledStateGraph

from workflow.baseline.types import BaselineNodeConfig, BaselineWorkflow


class BaselineRunner(Protocol):
    def load_workflow(self, workflow: BaselineWorkflow) -> None: ...

    def run_node(self, node: BaselineNodeConfig, prompt: str) -> str: ...


def message_text(state: AgentState) -> str:
    content = state["messages"][-1].content
    assert isinstance(content, str), type(content)
    return content


class StubBaselineRunner:
    def __init__(self) -> None:
        self.loaded_nodes: list[str] = []
        self.calls: list[tuple[str, str]] = []

    def load_workflow(self, workflow: BaselineWorkflow) -> None:
        self.loaded_nodes = [
            node.name for node in workflow.nodes if node.execution is not None
        ]

    def run_node(self, node: BaselineNodeConfig, prompt: str) -> str:
        self.calls.append((node.name, prompt))
        return f"{node.name}:{prompt}"


class StaticHuggingFaceRunner:
    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self.tools = tools or []
        self.agents: dict[str, CompiledStateGraph] = {}

    def load_workflow(self, workflow: BaselineWorkflow) -> None:
        for node in workflow.nodes:
            if node.execution is None:
                continue
            self.agents[node.name] = self._load_agent(node)

    def run_node(self, node: BaselineNodeConfig, prompt: str) -> str:
        agent = self.agents[node.name]
        state = AgentState(messages=[HumanMessage(content=prompt)])
        invoked = cast(AgentState, agent.invoke(state))
        return message_text(invoked)

    def _load_agent(self, node: BaselineNodeConfig) -> CompiledStateGraph:
        assert node.execution is not None
        device = int(node.execution.devices[0].split(":")[1])
        llm = HuggingFacePipeline.from_model_id(
            model_id=node.execution.model_path,
            task="text-generation",
            device=device,
            pipeline_kwargs={
                "max_new_tokens": node.execution.max_new_tokens,
                "do_sample": node.execution.do_sample,
                "return_full_text": False,
            },
        )
        return create_agent(
            model=ChatHuggingFace(llm=llm),
            tools=self.tools,
            system_prompt=node.system_prompt or "",
        )
