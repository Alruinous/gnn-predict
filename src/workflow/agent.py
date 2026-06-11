from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from workflow.handlers import NodeHandler
from workflow.schema import Workflow
from workflow.tool_nodes import BaseToolNode, build_tool_nodes
from workflow.types import WorkflowContext
from workflow.validation import validate_workflow

DEFAULT_REACT_PROMPT = (
    "You are the main LLM node in a ReAct workflow. "
    "Choose tool nodes when they are useful, pass the task and inputs clearly, "
    "and use the returned tool result to answer the user."
)


def build_workflow_model() -> ChatOpenAI:
    load_dotenv()
    base_url = os.environ["LLM_BASE_URL"]
    api_key = os.environ["LLM_API_KEY"]
    model_name = os.environ["LLM_NAME"]
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        extra_body={"reasoning": {"enabled": True}},
    )


def build_workflow_agent(
    workflow: Workflow,
    handler: NodeHandler | None = None,
    context: WorkflowContext | None = None,
    tool_nodes: list[BaseToolNode] | None = None,
    prompt: str = DEFAULT_REACT_PROMPT,
) -> Any:
    validate_workflow(workflow)
    if tool_nodes is None:
        if handler is None:
            raise ValueError("handler is required when tool_nodes is not provided")
        nodes = build_tool_nodes(
            workflow,
            handler=handler,
            context=context,
        )
    else:
        nodes = tool_nodes
    tools = [node.as_tool() for node in nodes]
    return create_react_agent(
        build_workflow_model(),
        tools,
        prompt=prompt,
        version="v2",
    )


def run_workflow_agent(
    workflow: Workflow,
    content: str | list[dict[str, Any]],
    handler: NodeHandler | None = None,
    context: WorkflowContext | None = None,
    tool_nodes: list[BaseToolNode] | None = None,
) -> str:
    agent = build_workflow_agent(
        workflow,
        handler=handler,
        context=context,
        tool_nodes=tool_nodes,
    )
    result = agent.invoke({"messages": [HumanMessage(content=content)]})
    return str(result["messages"][-1].content)
