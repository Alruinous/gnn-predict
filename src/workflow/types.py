from __future__ import annotations

from typing import Any, Literal

NodeType = Literal[
    "input",
    "output",
    "agent",
    "tool",
]
WorkflowPhase = Literal["inference", "prefill", "decode"]
WorkflowRunMode = Literal["serial", "parallel", "adaptive"]
WorkflowNodeStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]
type WorkflowContext = dict[str, Any]
