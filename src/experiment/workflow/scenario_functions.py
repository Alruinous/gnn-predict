"""Node-function registry for the QMSum + MBPP master (--functions target)."""

from __future__ import annotations

from collections.abc import Callable

from experiment.workflow.gsm8k import gsm8k_functions
from experiment.workflow.mbpp import mbpp_functions
from experiment.workflow.qmsum import qmsum_functions


def build_registry() -> dict[str, Callable[..., object]]:
    registry: dict[str, Callable[..., object]] = {}
    for functions in (qmsum_functions(), mbpp_functions(), gsm8k_functions()):
        overlap = set(registry) & set(functions)
        if overlap:
            raise ValueError(
                f"function name collision across scenarios: {sorted(overlap)}"
            )
        registry.update(functions)
    return registry
