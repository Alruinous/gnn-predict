from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict

from workflow.baseline.types import BaselineSessionRequest


def merge_node_outputs(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    overlap = set(left) & set(right)
    assert not overlap, overlap
    merged = dict(left)
    merged.update(right)
    return merged


class BaselineState(TypedDict):
    session_id: str
    inputs: dict[str, str]
    node_outputs: Annotated[dict[str, str], merge_node_outputs]
    trace: Annotated[list[dict[str, Any]], operator.add]
    final_output: str | None


class BaselineStateUpdate(TypedDict):
    node_outputs: dict[str, str]
    trace: list[dict[str, Any]]
    final_output: NotRequired[str]


def initial_baseline_state(request: BaselineSessionRequest) -> BaselineState:
    return {
        "session_id": request.session_id,
        "inputs": dict(request.inputs),
        "node_outputs": {},
        "trace": [],
        "final_output": None,
    }
