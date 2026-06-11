from __future__ import annotations

from typing import Any, Protocol

from workflow.schema import WorkflowNodeConfig
from workflow.types import WorkflowContext


class NodeHandler(Protocol):
    def run(
        self,
        node: WorkflowNodeConfig,
        context: WorkflowContext,
    ) -> dict[str, Any]: ...
