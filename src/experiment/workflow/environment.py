from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Self

import pynvml
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt

from common.validate import NonEmptyStr
from experiment.workflow.config import ExperimentConfig

EXPECTED_VLLM_VERSION = "0.10.2"
GPU_ID_ENV = "WORKFLOW_EXPERIMENT_GPU_IDS"


class EnvironmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GpuSnapshot(EnvironmentModel):
    physical_index: NonNegativeInt
    visible_index: NonNegativeInt
    uuid: NonEmptyStr
    name: NonEmptyStr
    total_memory_mb: NonNegativeFloat
    used_memory_mb: NonNegativeFloat
    compute_process_pids: tuple[NonNegativeInt, ...]
    graphics_process_pids: tuple[NonNegativeInt, ...]


class ServingEnvironment(EnvironmentModel):
    vllm_version: NonEmptyStr
    package_versions: dict[str, NonEmptyStr]
    engine_mode: str = "V0"
    attention_backend: str = "XFORMERS"
    dtype: str = "float16"
    enforce_eager: bool = True
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    block_size: int = 16
    scheduling_policy: str = "fcfs"
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    quantization: None = None
    cpu_offload_gb: float = 0.0
    swap_space_gb: float = 0.0
    cuda_visible_devices: NonEmptyStr
    driver_version: NonEmptyStr
    gpus: tuple[GpuSnapshot, ...]

    def assert_idle(self, maximum_used_memory_mb: float) -> Self:
        busy = [
            gpu
            for gpu in self.gpus
            if gpu.used_memory_mb > maximum_used_memory_mb
            or gpu.compute_process_pids
            or gpu.graphics_process_pids
        ]
        if busy:
            values = ", ".join(
                f"GPU {gpu.physical_index}: {gpu.used_memory_mb:.1f} MiB, "
                f"compute={gpu.compute_process_pids}, "
                f"graphics={gpu.graphics_process_pids}"
                for gpu in busy
            )
            raise RuntimeError(f"selected GPUs are not empty: {values}")
        return self


def selected_gpu_ids(gpu_count: int) -> tuple[int, ...]:
    raw = os.environ.get(GPU_ID_ENV)
    if raw is None:
        ids = tuple(range(gpu_count))
    else:
        try:
            ids = tuple(int(value) for value in raw.split(","))
        except ValueError as error:
            raise ValueError(f"{GPU_ID_ENV} must contain integer GPU ids") from error
    if (
        len(ids) != gpu_count
        or len(ids) != len(set(ids))
        or any(value < 0 for value in ids)
    ):
        raise ValueError(f"{GPU_ID_ENV} must contain {gpu_count} unique GPU ids")
    return ids


def validate_visible_devices(gpu_ids: tuple[int, ...]) -> None:
    expected = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    if actual != expected:
        raise RuntimeError(
            "experiment worker requires "
            f"CUDA_VISIBLE_DEVICES={expected}, got {actual!r}"
        )


def inspect_serving_environment(
    vllm_python: Path,
    gpu_ids: tuple[int, ...],
) -> ServingEnvironment:
    package_versions = _serving_package_versions(vllm_python)
    version = package_versions["vllm"]
    if version != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"formal experiment requires vLLM {EXPECTED_VLLM_VERSION}, got {version}"
        )
    pynvml.nvmlInit()
    try:
        driver = _text(pynvml.nvmlSystemGetDriverVersion())
        gpus = tuple(
            _gpu_snapshot(physical_index, visible_index)
            for visible_index, physical_index in enumerate(gpu_ids)
        )
    finally:
        pynvml.nvmlShutdown()
    return ServingEnvironment(
        vllm_version=version,
        package_versions=package_versions,
        cuda_visible_devices=",".join(str(index) for index in gpu_ids),
        driver_version=driver,
        gpus=gpus,
    )


def validate_runtime_paths(config: ExperimentConfig) -> None:
    paths = (
        config.vllm_python,
        config.qwen3_4b.path,
        config.qwen3_8b.path,
        config.qwen3_14b.path,
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    if not config.vllm_python.is_file() or not os.access(config.vllm_python, os.X_OK):
        raise PermissionError(config.vllm_python)


def _serving_package_versions(vllm_python: Path) -> dict[str, str]:
    packages = ("vllm", "torch", "transformers", "ray", "xformers")
    expression = (
        "import importlib.metadata,json; "
        f"names={packages!r}; "
        "print(json.dumps({name:importlib.metadata.version(name) for name in names}))"
    )
    result = subprocess.run(
        [
            str(vllm_python.absolute()),
            "-c",
            expression,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or set(value) != set(packages):
        raise TypeError("serving environment returned invalid package versions")
    if any(
        not isinstance(name, str) or not isinstance(version, str)
        for name, version in value.items()
    ):
        raise TypeError("serving package versions must be strings")
    return value


def _gpu_snapshot(physical_index: int, visible_index: int) -> GpuSnapshot:
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return GpuSnapshot(
        physical_index=physical_index,
        visible_index=visible_index,
        uuid=_text(pynvml.nvmlDeviceGetUUID(handle)),
        name=_text(pynvml.nvmlDeviceGetName(handle)),
        total_memory_mb=memory.total / 1024**2,
        used_memory_mb=memory.used / 1024**2,
        compute_process_pids=tuple(
            process.pid
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        ),
        graphics_process_pids=tuple(
            process.pid
            for process in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
        ),
    )


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
