"""Node-function registry for the QMSum + MBPP master (--functions target)."""

from __future__ import annotations

from collections.abc import Callable

from experiment.workflow.mbpp import mbpp_functions
from experiment.workflow.qmsum import qmsum_functions


def build_registry() -> dict[str, Callable[..., object]]:
    qmsum = qmsum_functions()
    mbpp = mbpp_functions()
    overlap = set(qmsum) & set(mbpp)
    if overlap:
        raise ValueError(f"function name collision across scenarios: {sorted(overlap)}")
    return {**qmsum, **mbpp}
