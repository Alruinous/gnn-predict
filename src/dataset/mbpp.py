from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
from pathlib import Path
from queue import Empty
from types import MappingProxyType, SimpleNamespace
from typing import Any

from dataset.schema import MbppEvaluation, TaskSample

CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_START_PATTERN = re.compile(
    r"(?m)^\s*(?:from\s+\S+\s+import\s+\S+|import\s+\S+|def\s+\w+\s*\(|[A-Za-z_]\w*\s*=)"
)
SAFE_IMPORT_ROOTS = {
    "array",
    "bisect",
    "cmath",
    "collections",
    "copy",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "sys",
}
PROCESS_START_METHOD = "spawn"


def load_mbpp_samples(path: str | Path) -> list[TaskSample]:
    mbpp_path = Path(path)
    raw_samples = json.loads(mbpp_path.read_text(encoding="utf-8"))
    if not isinstance(raw_samples, list):
        raise ValueError("MBPP sanitized file must contain a JSON array")
    return [normalize_mbpp_record(raw_sample) for raw_sample in raw_samples]


def normalize_mbpp_record(raw_sample: dict[str, Any]) -> TaskSample:
    task_id = raw_sample["task_id"]
    prompt = raw_sample["prompt"]
    test_imports = list(raw_sample["test_imports"])
    test_list = list(raw_sample["test_list"])
    return TaskSample(
        sample_id=str(task_id),
        source_dataset="mbpp_sanitized",
        split=resolve_mbpp_split(task_id),
        task_type="python_function_generation",
        input_text=prompt,
        reference_code=raw_sample["code"],
        test_imports=test_imports,
        test_list=test_list,
        quality_metric="pass_at_1",
        metadata={"task_id": task_id},
    )


def resolve_mbpp_split(task_id: int) -> str:
    if 1 <= task_id <= 10:
        return "few_shot"
    if 11 <= task_id <= 510:
        return "test"
    if 511 <= task_id <= 600:
        return "validation"
    if 601 <= task_id <= 974:
        return "train"
    raise ValueError(f"MBPP task_id is outside official ranges: {task_id}")


def evaluate_mbpp(
    model_output: str,
    test_imports: list[str],
    test_list: list[str],
    timeout_sec: float = 3.0,
) -> MbppEvaluation:
    code = extract_python_code(model_output)
    if not code:
        return MbppEvaluation(
            passed=False,
            passed_tests=0,
            total_tests=len(test_list),
            error_type="no_code",
            error_message="no Python code found in model output",
        )

    context = mp.get_context(PROCESS_START_METHOD)
    queue = context.Queue()
    with tempfile.TemporaryDirectory() as temp_dir:
        process = context.Process(
            target=_run_mbpp_tests,
            args=(code, test_imports, test_list, temp_dir, queue),
        )
        process.start()
        process.join(timeout_sec)
        if process.is_alive():
            process.terminate()
            process.join()
            return MbppEvaluation(
                passed=False,
                passed_tests=0,
                total_tests=len(test_list),
                error_type="timeout",
                error_message=f"execution exceeded {timeout_sec} seconds",
            )
        try:
            result = queue.get_nowait()
        except Empty:
            return MbppEvaluation(
                passed=False,
                passed_tests=0,
                total_tests=len(test_list),
                error_type="runtime_error",
                error_message="worker exited without a result",
            )
    return MbppEvaluation.model_validate(result)


def extract_python_code(model_output: str) -> str:
    blocks = CODE_BLOCK_PATTERN.findall(model_output)
    if blocks:
        return blocks[-1].strip()
    match = CODE_START_PATTERN.search(model_output)
    if not match:
        return ""
    return model_output[match.start() :].strip()


def _run_mbpp_tests(
    code: str,
    test_imports: list[str],
    test_list: list[str],
    temp_dir: str,
    queue: mp.Queue,
) -> None:
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(temp_dir)
    namespace = {"__builtins__": build_safe_builtins()}
    try:
        for import_statement in test_imports:
            exec(import_statement, namespace)
        exec(code, namespace)
    except SyntaxError as exc:
        queue.put(_mbpp_result(False, 0, test_list, "syntax_error", str(exc)))
        return
    except Exception as exc:
        queue.put(_mbpp_result(False, 0, test_list, "runtime_error", str(exc)))
        return

    passed_tests = 0
    for test in test_list:
        try:
            exec(test, namespace)
        except AssertionError as exc:
            queue.put(
                _mbpp_result(
                    False,
                    passed_tests,
                    test_list,
                    "assertion_error",
                    str(exc),
                )
            )
            return
        except Exception as exc:
            queue.put(
                _mbpp_result(False, passed_tests, test_list, "runtime_error", str(exc))
            )
            return
        passed_tests += 1
    queue.put(_mbpp_result(True, passed_tests, test_list, None, None))


def _mbpp_result(
    passed: bool,
    passed_tests: int,
    test_list: list[str],
    error_type: str | None,
    error_message: str | None,
) -> dict[str, object]:
    return {
        "passed": passed,
        "passed_tests": passed_tests,
        "total_tests": len(test_list),
        "error_type": error_type,
        "error_message": error_message,
    }


def build_safe_builtins() -> MappingProxyType:
    builtins = {
        "__import__": safe_import,
        "ArithmeticError": ArithmeticError,
        "AttributeError": AttributeError,
        "Exception": Exception,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "RuntimeError": RuntimeError,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "ZeroDivisionError": ZeroDivisionError,
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "hash": hash,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "ord": ord,
        "pow": pow,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
    }
    return MappingProxyType(builtins)


def safe_import(
    name: str,
    globals: object | None = None,
    locals: object | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> object:
    root_name = name.split(".", maxsplit=1)[0]
    if level != 0 or root_name not in SAFE_IMPORT_ROOTS:
        raise ImportError(f"import is not allowed: {name}")
    if root_name == "sys":
        return SimpleNamespace(getsizeof=sys.getsizeof, maxsize=sys.maxsize)
    return __import__(name, globals, locals, fromlist, level)
