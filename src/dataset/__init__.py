from __future__ import annotations

from importlib import import_module
from typing import Any

from dataset.schema import (
    Gsm8kEvaluation,
    MbppEvaluation,
    SummaryEvaluation,
    TaskSample,
)

_LAZY_EXPORTS = {
    "evaluate_gsm8k": ("dataset.gsm8k", "evaluate_gsm8k"),
    "evaluate_mbpp": ("dataset.mbpp", "evaluate_mbpp"),
    "evaluate_summary_rouge": ("dataset.summarization", "evaluate_summary_rouge"),
    "evaluate_summary_with_judge": (
        "dataset.summarization",
        "evaluate_summary_with_judge",
    ),
    "load_gsm8k_split": ("dataset.gsm8k", "load_gsm8k_split"),
    "load_mbpp_samples": ("dataset.mbpp", "load_mbpp_samples"),
    "load_multi_news_split": ("dataset.summarization", "load_multi_news_split"),
    "load_qmsum_split": ("dataset.summarization", "load_qmsum_split"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'dataset' has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "Gsm8kEvaluation",
    "MbppEvaluation",
    "SummaryEvaluation",
    "TaskSample",
    "evaluate_gsm8k",
    "evaluate_mbpp",
    "evaluate_summary_rouge",
    "evaluate_summary_with_judge",
    "load_gsm8k_split",
    "load_mbpp_samples",
    "load_multi_news_split",
    "load_qmsum_split",
]
