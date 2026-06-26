from __future__ import annotations

import json
from pathlib import Path

from dataset.mbpp import evaluate_mbpp, load_mbpp_samples


def write_mbpp(path: Path) -> None:
    payload = [
        {
            "task_id": 2,
            "prompt": "Write a function.",
            "code": "def add_one(x):\n    return x + 1",
            "test_imports": [],
            "test_list": ["assert add_one(1) == 2"],
            "source_file": "fixture",
        },
        {
            "task_id": 11,
            "prompt": "Write another function.",
            "code": "def square(x):\n    return x * x",
            "test_imports": [],
            "test_list": ["assert square(3) == 9"],
            "source_file": "fixture",
        },
        {
            "task_id": 511,
            "prompt": "Write validation function.",
            "code": "def double(x):\n    return x * 2",
            "test_imports": [],
            "test_list": ["assert double(3) == 6"],
            "source_file": "fixture",
        },
        {
            "task_id": 601,
            "prompt": "Write train function.",
            "code": "def triple(x):\n    return x * 3",
            "test_imports": [],
            "test_list": ["assert triple(3) == 9"],
            "source_file": "fixture",
        },
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_mbpp_samples_normalizes_records_and_splits(tmp_path: Path) -> None:
    path = tmp_path / "sanitized-mbpp.json"
    write_mbpp(path)

    samples = load_mbpp_samples(path)

    assert [sample.sample_id for sample in samples] == ["2", "11", "511", "601"]
    assert [sample.split for sample in samples] == [
        "few_shot",
        "test",
        "validation",
        "train",
    ]
    assert samples[1].source_dataset == "mbpp_sanitized"
    assert samples[1].task_type == "python_function_generation"
    assert samples[1].quality_metric == "pass_at_1"
    assert samples[1].metadata == {"task_id": 11}


def test_evaluate_mbpp_passes_code_block_solution() -> None:
    result = evaluate_mbpp(
        "```python\ndef add_one(x):\n    return x + 1\n```",
        [],
        ["assert add_one(1) == 2", "assert add_one(5) == 6"],
    )

    assert result.passed
    assert result.passed_tests == 2
    assert result.total_tests == 2
    assert result.error_type is None


def test_evaluate_mbpp_reports_assertion_error() -> None:
    result = evaluate_mbpp(
        "def add_one(x):\n    return x",
        [],
        ["assert add_one(1) == 2", "assert add_one(5) == 6"],
    )

    assert not result.passed
    assert result.passed_tests == 0
    assert result.error_type == "assertion_error"


def test_evaluate_mbpp_reports_syntax_error() -> None:
    result = evaluate_mbpp(
        "def broken(:\n    return 1",
        [],
        ["assert broken() == 1"],
    )

    assert not result.passed
    assert result.error_type == "syntax_error"


def test_evaluate_mbpp_reports_runtime_error() -> None:
    result = evaluate_mbpp(
        "def broken():\n    return 1 / 0",
        [],
        ["assert broken() == 1"],
    )

    assert not result.passed
    assert result.error_type == "runtime_error"


def test_evaluate_mbpp_reports_timeout() -> None:
    result = evaluate_mbpp(
        "def never():\n    while True:\n        pass",
        [],
        ["never()"],
        timeout_sec=0.2,
    )

    assert not result.passed
    assert result.error_type == "timeout"


def test_evaluate_mbpp_keeps_constants_before_function() -> None:
    result = evaluate_mbpp(
        "LIMIT = 3\n\ndef clamp(x):\n    return min(x, LIMIT)",
        [],
        ["assert clamp(5) == 3"],
    )

    assert result.passed


def test_evaluate_mbpp_allows_common_reference_imports() -> None:
    result = evaluate_mbpp(
        "import sys\nimport cmath\n\ndef f(x):\n    return x < sys.maxsize and cmath.phase(1)",
        [],
        ["assert f(1) == 0.0"],
    )

    assert result.passed


def test_evaluate_mbpp_exposes_only_sys_maxsize() -> None:
    result = evaluate_mbpp(
        (
            "import sys\n\n"
            "def f():\n"
            "    try:\n"
            "        sys.modules\n"
            "    except AttributeError:\n"
            "        return sys.maxsize > 0 and sys.getsizeof(1) > 0"
        ),
        [],
        ["assert f()"],
    )

    assert result.passed


def test_evaluate_mbpp_allows_common_exception_classes() -> None:
    result = evaluate_mbpp(
        "def f():\n    try:\n        {}['x']\n    except KeyError:\n        return 1",
        [],
        ["assert f() == 1"],
    )

    assert result.passed


def test_evaluate_mbpp_reports_no_code() -> None:
    result = evaluate_mbpp("No code here.", [], ["assert f() == 1"])

    assert not result.passed
    assert result.error_type == "no_code"
