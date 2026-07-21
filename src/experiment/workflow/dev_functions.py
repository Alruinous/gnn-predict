"""Trivial echo function node for the GPU-free dev-sim (--functions target)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from langchain.agents import AgentState
from langchain_core.messages import AIMessage


def echo(
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    text = session_inputs.get("text")
    if not isinstance(text, str):
        raise TypeError("echo requires a 'text' string input")
    return AgentState(messages=[AIMessage(content=text)])


def build_registry() -> dict[str, Callable[..., object]]:
    return {"echo": echo}
