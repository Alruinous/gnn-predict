"""Profile workflow generation buckets on idle GPUs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing
import os
import platform
import socket
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pynvml
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn_model.data.causal_lm_cache_config import (  # noqa: E402
    GraphCacheExportSpec,
    expand_graph_cache_specs,
    load_graph_cache_config,
)
from workflow.artifacts import (  # noqa: E402
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
    load_resource_contract_cache,
)
from workflow.types import WorkflowModelFeatureKey  # noqa: E402

MIB = 1024**2
POWER_SAMPLE_INTERVAL_SEC = 0.05
LOAD_SAMPLE_COUNT = 3
TERMINAL_SPEC_STATUSES = {
    "success",
    "oom",
    "skipped_dominated_oom",
    "skipped_load_oom",
}


class GpuInterferenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    name: str
    uuid: str
    total_memory_mb: int


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    group_name: str
    model_name: str
    model_path: str
    dtype: str
    phase: Literal["decode"]
    batch_size: int
    sequence_length: int
    decode_output_length: int

    @property
    def spec_id(self) -> str:
        payload = "|".join(
            (
                self.model_name,
                self.phase,
                str(self.batch_size),
                str(self.sequence_length),
                str(self.decode_output_length),
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.batch_size, self.sequence_length, self.decode_output_length


@dataclass(frozen=True, slots=True)
class GroupTask:
    name: str
    model_name: str
    model_path: str
    dtype: str
    weight_bytes: int
    specs: tuple[ProfileSpec, ...]


@dataclass(frozen=True, slots=True)
class GpuSample:
    memory_used_mb: float
    power_watts: float
    foreign_pids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GenerationMeasurement:
    duration_sec: float
    actual_output_length: int
    peak_vram_mb: int
    nvml_peak_used_mb: float
    torch_peak_allocated_mb: float
    torch_peak_reserved_mb: float
    power_watts_avg: float
    power_sample_count: int


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_gpu_kind(value: str) -> str:
    gpu_kind = value.strip().casefold()
    if not gpu_kind:
        raise argparse.ArgumentTypeError("GPU kind must not be empty")
    return gpu_kind


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_yaml(path: Path, payload: object) -> None:
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def load_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, Any], raw)


def decode_specs(
    cache_yaml: Path,
    selected_groups: set[str] | None,
    limit: int | None,
) -> tuple[list[GroupTask], list[ProfileSpec]]:
    config = load_graph_cache_config(cache_yaml)
    expanded = expand_graph_cache_specs(
        config,
        selected_group_names=selected_groups,
        selected_phases={"decode"},
    )
    if limit is not None:
        expanded = expanded[:limit]
    specs = [to_profile_spec(spec) for spec in expanded]
    by_group: dict[str, list[ProfileSpec]] = {}
    for spec in specs:
        by_group.setdefault(spec.group_name, []).append(spec)

    tasks: list[GroupTask] = []
    for group in config.cache_groups:
        group_specs = by_group.get(group.name)
        if not group_specs:
            continue
        if Path(group.name).name != group.name:
            raise ValueError(f"cache group name is not path-safe: {group.name!r}")
        model_path = Path(group.model_path)
        weight_bytes = sum(
            path.stat().st_size for path in model_path.glob("*.safetensors")
        )
        tasks.append(
            GroupTask(
                name=group.name,
                model_name=group.model_name,
                model_path=str(model_path),
                dtype=group.dtype,
                weight_bytes=weight_bytes,
                specs=tuple(sorted(group_specs, key=lambda item: item.shape)),
            )
        )
    return tasks, specs


def to_profile_spec(spec: GraphCacheExportSpec) -> ProfileSpec:
    if spec.key.phase != "decode":
        raise ValueError(f"workflow profiling only supports decode: {spec.key}")
    return ProfileSpec(
        group_name=spec.group_name,
        model_name=spec.key.model_name,
        model_path=str(spec.model_path),
        dtype=spec.dtype,
        phase="decode",
        batch_size=spec.key.batch_size,
        sequence_length=spec.key.sequence_length,
        decode_output_length=spec.key.decode_output_length,
    )


def selection_sha256(specs: list[ProfileSpec]) -> str:
    payload = "\n".join(spec.spec_id for spec in specs)
    return hashlib.sha256(payload.encode()).hexdigest()


def running_gpu_pids(handle: Any) -> set[int]:
    pids: set[int] = set()
    for getter in (
        pynvml.nvmlDeviceGetComputeRunningProcesses,
        pynvml.nvmlDeviceGetGraphicsRunningProcesses,
    ):
        try:
            processes = getter(handle)
        except pynvml.NVMLError as error:
            if getattr(error, "value", None) != pynvml.NVML_ERROR_NOT_SUPPORTED:
                raise
            continue
        pids.update(int(process.pid) for process in processes)
    return pids


def discover_idle_gpus(gpu_kind: str) -> list[GpuDevice]:
    devices: list[GpuDevice] = []
    pynvml.nvmlInit()
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = str(pynvml.nvmlDeviceGetName(handle))
            if gpu_kind.casefold() not in name.casefold():
                continue
            if running_gpu_pids(handle):
                continue
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                GpuDevice(
                    index=index,
                    name=name,
                    uuid=str(pynvml.nvmlDeviceGetUUID(handle)),
                    total_memory_mb=round(float(memory.total) / MIB),
                )
            )
    finally:
        pynvml.nvmlShutdown()
    return devices


def select_gpus(
    gpu_kind: str,
    count: int,
    expected: list[dict[str, object]] | None,
) -> list[GpuDevice]:
    idle = discover_idle_gpus(gpu_kind)
    by_uuid = {gpu.uuid: gpu for gpu in idle}
    if expected is not None:
        expected_uuids = [cast(str, raw["uuid"]) for raw in expected]
        unavailable = [uuid for uuid in expected_uuids if uuid not in by_uuid]
        if unavailable:
            raise RuntimeError(f"profile GPUs are not idle or available: {unavailable}")
        selected = [by_uuid[uuid] for uuid in expected_uuids]
        if len(selected) != count:
            raise RuntimeError(
                f"manifest expected {len(selected)} GPUs, requested {count}"
            )
        return selected
    if len(idle) < count:
        raise RuntimeError(f"requires {count} idle {gpu_kind} GPUs, found {len(idle)}")
    return idle[:count]


def assert_gpu_exclusive(handle: Any, owner_pids: set[int]) -> None:
    foreign = running_gpu_pids(handle) - owner_pids
    if foreign:
        raise GpuInterferenceError(
            f"foreign processes appeared on profile GPU: {sorted(foreign)}"
        )


def sample_gpu(
    handle: Any,
    owner_pids: set[int],
    stop: threading.Event,
    samples: list[GpuSample],
) -> None:
    while True:
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        foreign = tuple(sorted(running_gpu_pids(handle) - owner_pids))
        samples.append(
            GpuSample(
                memory_used_mb=float(memory.used) / MIB,
                power_watts=float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000,
                foreign_pids=foreign,
            )
        )
        if stop.wait(POWER_SAMPLE_INTERVAL_SEC):
            return


def cleanup_cuda(torch: Any, device: str) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def load_runtime(
    model_path: str,
    dtype_name: str,
    device: str,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        device_map={"": device},
    )
    model.eval()
    torch.cuda.synchronize(device)
    return tokenizer, model


def measure_loads(
    task: GroupTask,
    gpu: GpuDevice,
    device: str,
    handle: Any,
    owner_pids: set[int],
    load_lock: Any,
) -> tuple[dict[str, Any], Any | None, Any | None]:
    import torch

    durations: list[float] = []
    idle_vram: list[float] = []
    tokenizer: Any | None = None
    model: Any | None = None
    with load_lock:
        for sample_index in range(LOAD_SAMPLE_COUNT):
            assert_gpu_exclusive(handle, owner_pids)
            started = time.perf_counter()
            try:
                tokenizer, model = load_runtime(task.model_path, task.dtype, device)
            except torch.cuda.OutOfMemoryError as error:
                cleanup_cuda(torch, device)
                return (
                    {
                        "record_type": "model_load",
                        "group_name": task.name,
                        "model_name": task.model_name,
                        "model_path": task.model_path,
                        "dtype": task.dtype,
                        "status": "oom",
                        "gpu": asdict(gpu),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "recorded_at": utc_now(),
                    },
                    None,
                    None,
                )
            duration = time.perf_counter() - started
            assert model is not None
            assert tokenizer is not None
            durations.append(duration)
            idle_vram.append(float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / MIB)
            if sample_index + 1 < LOAD_SAMPLE_COUNT:
                del model
                del tokenizer
                model = None
                tokenizer = None
                cleanup_cuda(torch, device)

    assert model is not None
    assert tokenizer is not None
    return (
        {
            "record_type": "model_load",
            "group_name": task.name,
            "model_name": task.model_name,
            "model_path": task.model_path,
            "dtype": task.dtype,
            "status": "success",
            "gpu": asdict(gpu),
            "sample_count": len(durations),
            "duration_samples_sec": durations,
            "predicted_load_sec": max(durations),
            "idle_vram_samples_mb": idle_vram,
            "recorded_at": utc_now(),
        },
        tokenizer,
        model,
    )


def load_for_resume(
    task: GroupTask,
    gpu: GpuDevice,
    device: str,
    handle: Any,
    owner_pids: set[int],
    load_lock: Any,
) -> tuple[Any, Any]:
    import torch

    with load_lock:
        assert_gpu_exclusive(handle, owner_pids)
        try:
            return load_runtime(task.model_path, task.dtype, device)
        except torch.cuda.OutOfMemoryError as error:
            cleanup_cuda(torch, device)
            raise RuntimeError(
                f"previously successful model load now OOMs: {task.name}"
            ) from error


def generate_full_length(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    output_length: int,
    pad_token_id: int,
) -> Any:
    import torch

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=output_length,
            min_new_tokens=output_length,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=None,
            forced_eos_token_id=None,
            stop_strings=None,
            return_dict_in_generate=False,
        )
    if not isinstance(generated, torch.Tensor):
        raise TypeError(f"model.generate returned {type(generated)}")
    return generated


def measure_generation(
    model: Any,
    tokenizer: Any,
    spec: ProfileSpec,
    gpu: GpuDevice,
    device: str,
    handle: Any,
    owner_pids: set[int],
) -> GenerationMeasurement:
    import torch

    cleanup_cuda(torch, device)
    generator = torch.Generator().manual_seed(int(spec.spec_id, 16))
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(
        0,
        vocab_size,
        (spec.batch_size, spec.sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to(device)  # [B,S]
    attention_mask = torch.ones(
        (spec.batch_size, spec.sequence_length),
        dtype=torch.long,
        device=device,
    )  # [B,S]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_nvml_mb = float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / MIB
    baseline_reserved_mb = float(torch.cuda.memory_reserved(device)) / MIB
    non_allocator_mb = max(0.0, baseline_nvml_mb - baseline_reserved_mb)

    samples: list[GpuSample] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=sample_gpu,
        args=(handle, owner_pids, stop, samples),
        daemon=True,
    )
    sampler.start()
    try:
        started = time.perf_counter()
        generated = generate_full_length(
            model,
            input_ids,
            attention_mask,
            spec.decode_output_length,
            int(tokenizer.pad_token_id),
        )  # [B,S+O]
        torch.cuda.synchronize(device)
        duration_sec = time.perf_counter() - started
    finally:
        stop.set()
        sampler.join()

    expected_shape = (
        spec.batch_size,
        spec.sequence_length + spec.decode_output_length,
    )
    if tuple(generated.shape) != expected_shape:
        raise ValueError(
            f"generated shape changed for {spec.spec_id}: "
            f"{tuple(generated.shape)} != {expected_shape}"
        )
    if not samples:
        raise RuntimeError("NVML sampler returned no samples")
    foreign = sorted({pid for sample in samples for pid in sample.foreign_pids})
    if foreign:
        raise GpuInterferenceError(
            f"foreign processes appeared during {spec.spec_id}: {foreign}"
        )

    torch_peak_allocated_mb = float(torch.cuda.max_memory_allocated(device)) / MIB
    torch_peak_reserved_mb = float(torch.cuda.max_memory_reserved(device)) / MIB
    nvml_peak_used_mb = max(sample.memory_used_mb for sample in samples)
    peak_vram_mb = math.ceil(
        max(nvml_peak_used_mb, torch_peak_reserved_mb + non_allocator_mb)
    )
    power_watts = [sample.power_watts for sample in samples]
    return GenerationMeasurement(
        duration_sec=duration_sec,
        actual_output_length=spec.decode_output_length,
        peak_vram_mb=peak_vram_mb,
        nvml_peak_used_mb=nvml_peak_used_mb,
        torch_peak_allocated_mb=torch_peak_allocated_mb,
        torch_peak_reserved_mb=torch_peak_reserved_mb,
        power_watts_avg=sum(power_watts) / len(power_watts),
        power_sample_count=len(power_watts),
    )


def group_state_path(output_dir: Path, group_name: str) -> Path:
    return output_dir / "groups" / f"{group_name}.json"


def load_state_path(output_dir: Path, group_name: str) -> Path:
    return output_dir / "loads" / f"{group_name}.json"


def load_group_records(path: Path, group_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = load_json_object(path)
    if raw.get("version") != 1 or raw.get("group_name") != group_name:
        raise ValueError(f"invalid group state: {path}")
    records = raw.get("records")
    if not isinstance(records, list):
        raise TypeError(f"group records must be a list: {path}")
    typed = [cast(dict[str, Any], record) for record in records]
    ids = [record.get("spec_id") for record in typed]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate spec records: {path}")
    return typed


def save_group_records(
    path: Path,
    group_name: str,
    records: list[dict[str, Any]],
) -> None:
    write_json(
        path,
        {"version": 1, "group_name": group_name, "records": records},
    )


def spec_record(
    spec: ProfileSpec,
    gpu: GpuDevice,
    status: str,
    **fields: object,
) -> dict[str, object]:
    return {
        "record_type": "spec",
        "spec_id": spec.spec_id,
        "group_name": spec.group_name,
        "key": {
            "model_name": spec.model_name,
            "phase": spec.phase,
            "batch_size": spec.batch_size,
            "sequence_length": spec.sequence_length,
            "decode_output_length": spec.decode_output_length,
        },
        "model_path": spec.model_path,
        "dtype": spec.dtype,
        "status": status,
        "gpu": asdict(gpu),
        "recorded_at": utc_now(),
        **fields,
    }


def dominates(oom_shape: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    return all(
        candidate >= failed for candidate, failed in zip(shape, oom_shape, strict=True)
    )


def append_record(
    path: Path,
    group_name: str,
    records: list[dict[str, Any]],
    record: dict[str, object],
) -> None:
    spec_id = record.get("spec_id")
    if any(existing.get("spec_id") == spec_id for existing in records):
        raise ValueError(f"duplicate result for {spec_id}")
    records.append(cast(dict[str, Any], record))
    save_group_records(path, group_name, records)


def profile_group(
    task: GroupTask, gpu: GpuDevice, output_dir: str, load_lock: Any
) -> None:
    import torch

    output = Path(output_dir)
    state_path = group_state_path(output, task.name)
    model_load_path = load_state_path(output, task.name)
    records = load_group_records(state_path, task.name)
    completed = {
        str(record["spec_id"])
        for record in records
        if record.get("status") in TERMINAL_SPEC_STATUSES
    }
    oom_shapes = [
        (
            int(record["key"]["batch_size"]),
            int(record["key"]["sequence_length"]),
            int(record["key"]["decode_output_length"]),
        )
        for record in records
        if record.get("status") == "oom"
    ]
    pending = [spec for spec in task.specs if spec.spec_id not in completed]
    if not pending:
        print(f"{task.name}: already complete", flush=True)
        return

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
    device = f"cuda:{gpu.index}"
    torch.cuda.set_device(gpu.index)
    probe = torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    del probe
    owner_pids = running_gpu_pids(handle)
    if len(owner_pids) != 1:
        raise GpuInterferenceError(
            f"expected one profile process on {gpu.name}:{gpu.index}, got {owner_pids}"
        )
    tokenizer: Any | None = None
    model: Any | None = None
    try:
        assert_gpu_exclusive(handle, owner_pids)
        if model_load_path.exists():
            load_record = load_json_object(model_load_path)
            if load_record.get("status") == "oom":
                new_records = [
                    spec_record(spec, gpu, "skipped_load_oom") for spec in pending
                ]
                records.extend(cast(list[dict[str, Any]], new_records))
                save_group_records(state_path, task.name, records)
                return
            if load_record.get("status") != "success":
                raise ValueError(f"invalid load state: {model_load_path}")
            tokenizer, model = load_for_resume(
                task,
                gpu,
                device,
                handle,
                owner_pids,
                load_lock,
            )
        else:
            load_record, tokenizer, model = measure_loads(
                task,
                gpu,
                device,
                handle,
                owner_pids,
                load_lock,
            )
            write_json(model_load_path, load_record)
            if load_record["status"] == "oom":
                new_records = [
                    spec_record(spec, gpu, "skipped_load_oom") for spec in pending
                ]
                records.extend(cast(list[dict[str, Any]], new_records))
                save_group_records(state_path, task.name, records)
                print(
                    f"{task.name}: model load OOM, skipped {len(pending)} specs",
                    flush=True,
                )
                return

        assert tokenizer is not None
        assert model is not None
        warmup = min(task.specs, key=lambda item: item.shape)
        try:
            measure_generation(
                model,
                tokenizer,
                warmup,
                gpu,
                device,
                handle,
                owner_pids,
            )
        except torch.cuda.OutOfMemoryError as error:
            if warmup.spec_id in completed:
                raise RuntimeError(
                    f"previously successful warmup now OOMs: {task.name}"
                ) from error
            append_record(
                state_path,
                task.name,
                records,
                spec_record(
                    warmup,
                    gpu,
                    "oom",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    during_warmup=True,
                ),
            )
            completed.add(warmup.spec_id)
            oom_shapes.append(warmup.shape)
        cleanup_cuda(torch, device)

        for spec in task.specs:
            if spec.spec_id in completed:
                continue
            dominating = next(
                (shape for shape in oom_shapes if dominates(shape, spec.shape)),
                None,
            )
            if dominating is not None:
                append_record(
                    state_path,
                    task.name,
                    records,
                    spec_record(
                        spec,
                        gpu,
                        "skipped_dominated_oom",
                        dominating_shape=list(dominating),
                    ),
                )
                continue
            assert_gpu_exclusive(handle, owner_pids)
            try:
                measurement = measure_generation(
                    model,
                    tokenizer,
                    spec,
                    gpu,
                    device,
                    handle,
                    owner_pids,
                )
            except torch.cuda.OutOfMemoryError as error:
                append_record(
                    state_path,
                    task.name,
                    records,
                    spec_record(
                        spec,
                        gpu,
                        "oom",
                        error_type=type(error).__name__,
                        error_message=str(error),
                    ),
                )
                oom_shapes.append(spec.shape)
                print(f"{task.name}: OOM at {spec.shape}", flush=True)
            else:
                assert_gpu_exclusive(handle, owner_pids)
                append_record(
                    state_path,
                    task.name,
                    records,
                    spec_record(
                        spec,
                        gpu,
                        "success",
                        **asdict(measurement),
                    ),
                )
            cleanup_cuda(torch, device)
            if len(records) % 10 == 0:
                counts = Counter(str(record["status"]) for record in records)
                print(
                    f"{task.name}: {len(records)}/{len(task.specs)} {dict(counts)}",
                    flush=True,
                )
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        cleanup_cuda(torch, device)
        pynvml.nvmlShutdown()

    counts = Counter(str(record["status"]) for record in records)
    print(f"{task.name}: complete {dict(counts)}", flush=True)


def terminate_processes(active: dict[int, tuple[Any, GroupTask]]) -> None:
    for process, _ in active.values():
        if process.is_alive():
            process.terminate()
    for process, _ in active.values():
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join()


def run_groups(tasks: list[GroupTask], gpus: list[GpuDevice], output_dir: Path) -> None:
    context = multiprocessing.get_context("spawn")
    load_lock = context.Lock()
    pending = deque(sorted(tasks, key=lambda task: task.weight_bytes, reverse=True))
    available = deque(gpus)
    active: dict[int, tuple[Any, GroupTask]] = {}

    def launch() -> None:
        while pending and available:
            gpu = available.popleft()
            task = pending.popleft()
            process = context.Process(
                target=profile_group,
                args=(task, gpu, str(output_dir), load_lock),
                name=f"workflow-profile-{task.name}",
            )
            process.start()
            active[gpu.index] = (process, task)
            print(f"launched {task.name} on {gpu.name}:{gpu.index}", flush=True)

    launch()
    try:
        while active:
            time.sleep(0.5)
            finished: list[int] = []
            for gpu_index, (process, task) in active.items():
                if process.is_alive():
                    continue
                process.join()
                if process.exitcode != 0:
                    message = (
                        f"profile process failed for {task.name}: "
                        f"exit={process.exitcode}"
                    )
                    raise RuntimeError(message)
                gpu = next(gpu for gpu in gpus if gpu.index == gpu_index)
                print(f"finished {task.name} on {gpu.name}:{gpu_index}", flush=True)
                finished.append(gpu_index)
            for gpu_index in finished:
                active.pop(gpu_index)
                available.append(next(gpu for gpu in gpus if gpu.index == gpu_index))
            launch()
    finally:
        terminate_processes(active)


def manifest_payload(
    cache_yaml: Path,
    specs: list[ProfileSpec],
    gpus: list[GpuDevice],
    gpu_kind: str,
) -> dict[str, object]:
    import torch
    import transformers

    pynvml.nvmlInit()
    try:
        driver_version = str(pynvml.nvmlSystemGetDriverVersion())
    finally:
        pynvml.nvmlShutdown()
    return {
        "version": 2,
        "created_at": utc_now(),
        "completed_at": None,
        "complete": False,
        "cache_yaml": str(cache_yaml.resolve()),
        "cache_yaml_sha256": file_sha256(cache_yaml),
        "selection_sha256": selection_sha256(specs),
        "selected_spec_count": len(specs),
        "gpu_kind": gpu_kind,
        "gpu_count": len(gpus),
        "gpus": [asdict(gpu) for gpu in gpus],
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "cuda": str(torch.version.cuda),
            "driver": driver_version,
        },
        "measurement": {
            "phase": "decode",
            "semantics": "full_generate_forced_output_length",
            "input_kind": "deterministic_synthetic_token_ids",
            "run_samples_per_spec": 1,
            "load_samples_per_model": LOAD_SAMPLE_COUNT,
            "load_statistic": "max_as_conservative_p95",
            "power_sample_interval_sec": POWER_SAMPLE_INTERVAL_SEC,
            "oom_policy": "skip_dominated_batch_sequence_output_shapes",
        },
    }


def prepare_manifest(
    path: Path,
    cache_yaml: Path,
    specs: list[ProfileSpec],
    gpu_kind: str,
    gpu_count: int,
    resume: bool,
) -> tuple[dict[str, Any], list[GpuDevice]]:
    if path.exists():
        if not resume:
            raise FileExistsError(f"profile manifest already exists: {path}")
        manifest = cast(
            dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        expected = {
            "cache_yaml_sha256": file_sha256(cache_yaml),
            "selection_sha256": selection_sha256(specs),
            "selected_spec_count": len(specs),
            "gpu_kind": gpu_kind,
            "gpu_count": gpu_count,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                message = (
                    f"resume manifest mismatch for {field}: "
                    f"{manifest.get(field)!r} != {value!r}"
                )
                raise RuntimeError(message)
        raw_gpus = cast(list[dict[str, object]], manifest["gpus"])
        gpus = select_gpus(
            gpu_kind,
            gpu_count,
            raw_gpus,
        )
        return manifest, gpus

    output_dir = path.parent
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"profile output directory is not empty: {output_dir}")
    gpus = select_gpus(gpu_kind, gpu_count, None)
    manifest = manifest_payload(cache_yaml, specs, gpus, gpu_kind)
    write_yaml(path, manifest)
    return cast(dict[str, Any], manifest), gpus


def read_all_results(
    output_dir: Path,
    tasks: list[GroupTask],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    loads: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = load_state_path(output_dir, task.name)
        if path.exists():
            load_record = load_json_object(path)
            loads[task.name] = load_record
            records.append(load_record)
        records.extend(
            load_group_records(group_state_path(output_dir, task.name), task.name)
        )
    return records, loads


def prediction_environment(
    manifest: dict[str, Any],
    complete: bool,
) -> dict[str, Any]:
    return {
        "source": "empirical_gpu_profile",
        "profile_complete": complete,
        "gpu_kind": manifest["gpu_kind"],
        "source_cache_yaml": manifest["cache_yaml"],
        "source_cache_yaml_sha256": manifest["cache_yaml_sha256"],
        "selection_sha256": manifest["selection_sha256"],
        "storage_kind": "shared_model_dir",
        "model_root": "/data/Models",
        "prediction_scope": "gpu_kind",
        "hardware": manifest["gpus"],
        "software": manifest["environment"],
        "measurement": manifest["measurement"],
    }


def build_prediction_cache(
    output_dir: Path,
    tasks: list[GroupTask],
    specs: list[ProfileSpec],
    manifest: dict[str, Any],
    complete: bool,
) -> ResourceContractCache:
    _, loads = read_all_results(output_dir, tasks)
    records_by_id: dict[str, dict[str, Any]] = {}
    for task in tasks:
        for record in load_group_records(
            group_state_path(output_dir, task.name), task.name
        ):
            spec_id = str(record["spec_id"])
            if spec_id in records_by_id:
                raise ValueError(f"duplicate profile record: {spec_id}")
            records_by_id[spec_id] = record

    entries: list[ResourceContract] = []
    for spec in specs:
        record = records_by_id.get(spec.spec_id)
        if record is None or record.get("status") != "success":
            continue
        load = loads.get(spec.group_name)
        if load is None or load.get("status") != "success":
            raise RuntimeError(f"successful spec has no load profile: {spec.spec_id}")
        gpu = cast(dict[str, object], record["gpu"])
        gpu_kind = str(manifest["gpu_kind"])
        peak_vram_mb = float(record["peak_vram_mb"])
        vram_margin_fraction = 0.10
        entries.append(
            ResourceContract(
                key=WorkflowModelFeatureKey(
                    model_name=spec.model_name,
                    phase="decode",
                    gpu_name=gpu_kind,
                    batch_size=spec.batch_size,
                    sequence_length=spec.sequence_length,
                    decode_output_length=spec.decode_output_length,
                ),
                source=ResourceContractSource.EMPIRICAL_PROFILE,
                predicted_load_sec=float(load["predicted_load_sec"]),
                predicted_run_sec=float(record["duration_sec"]),
                predicted_peak_vram_mb=peak_vram_mb,
                peak_vram_mb_upper_bound=peak_vram_mb * (1.0 + vram_margin_fraction),
                peak_vram_mb_evidence=ResourceEvidence(
                    method="fixed_margin_fallback",
                    sample_count=1,
                    margin_fraction=vram_margin_fraction,
                ),
                predicted_power_watts=float(record["power_watts_avg"]),
                predictor_metadata={
                    "source": "empirical_gpu_profile",
                    "source_gpu_kind": gpu_kind,
                    "source_gpu_index": cast(int, gpu["index"]),
                    "source_gpu_uuid": cast(str, gpu["uuid"]),
                    "measurement_semantics": "full_generate_forced_output_length",
                    "input_kind": "deterministic_synthetic_token_ids",
                    "run_sample_count": 1,
                    "load_sample_count": int(load["sample_count"]),
                    "load_statistic": "max_as_conservative_p95",
                    "power_statistic": "sample_mean",
                    "memory_statistic": "peak_total_vram_estimate",
                },
            )
        )
    return ResourceContractCache(
        version=2,
        environment=prediction_environment(manifest, complete),
        entries=tuple(entries),
    )


def consolidate_outputs(
    output_dir: Path,
    tasks: list[GroupTask],
    specs: list[ProfileSpec],
    manifest: dict[str, Any],
) -> bool:
    records, loads = read_all_results(output_dir, tasks)
    raw_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    atomic_write_text(output_dir / "raw_results.jsonl", raw_text)

    spec_records = [record for record in records if record.get("record_type") == "spec"]
    status_counts = Counter(str(record["status"]) for record in spec_records)
    complete = len(spec_records) == len(specs) and all(
        record.get("status") in TERMINAL_SPEC_STATUSES for record in spec_records
    )
    by_model: dict[str, Counter[str]] = {}
    for record in spec_records:
        key = record.get("key")
        if not isinstance(key, dict):
            raise TypeError("spec record key must be an object")
        model_name = str(key["model_name"])
        by_model.setdefault(model_name, Counter())[str(record["status"])] += 1

    cache = build_prediction_cache(
        output_dir,
        tasks,
        specs,
        manifest,
        complete,
    )
    predictions_path = output_dir / "predictions.yaml"
    write_yaml(predictions_path, cache.model_dump(mode="json"))
    load_resource_contract_cache(predictions_path)

    summary = {
        "complete": complete,
        "selected_specs": len(specs),
        "terminal_specs": len(spec_records),
        "pending_specs": len(specs) - len(spec_records),
        "status_counts": dict(sorted(status_counts.items())),
        "by_model": {
            model: dict(sorted(counts.items())) for model, counts in by_model.items()
        },
        "load_statuses": {
            group: str(record.get("status")) for group, record in loads.items()
        },
        "gpu_kind": manifest["gpu_kind"],
        "successful_prediction_entries": len(cache.entries),
        "generated_at": utc_now(),
    }
    write_yaml(output_dir / "summary.yaml", summary)
    return complete


def print_dry_run(tasks: list[GroupTask], specs: list[ProfileSpec]) -> None:
    payload = {
        "group_count": len(tasks),
        "spec_count": len(specs),
        "groups": {task.name: len(task.specs) for task in tasks},
        "batch_sizes": sorted({spec.batch_size for spec in specs}),
        "sequence_lengths": sorted({spec.sequence_length for spec in specs}),
        "decode_output_lengths": sorted({spec.decode_output_length for spec in specs}),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile cache.yaml generation buckets into ResourceContractCache."
    )
    parser.add_argument(
        "--cache-yaml",
        type=Path,
        default=ROOT / "config" / "workflow" / "cache.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--gpu-kind", type=parse_gpu_kind, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--group", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpu_count <= 0:
        raise ValueError("gpu-count must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    cache_yaml = args.cache_yaml.resolve()
    selected_groups = set(args.group) if args.group else None
    tasks, specs = decode_specs(cache_yaml, selected_groups, args.limit)
    if not specs:
        raise ValueError("no decode profile specs selected")
    if args.dry_run:
        print_dry_run(tasks, specs)
        return 0

    output_dir = args.output_dir
    if output_dir is None:
        run_name = f"{args.gpu_kind}-{datetime.now(UTC):%Y%m%d}"
        output_dir = ROOT / "cache" / "profile" / "runs" / run_name
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "run_manifest.yaml"
    manifest, gpus = prepare_manifest(
        manifest_path,
        cache_yaml,
        specs,
        args.gpu_kind,
        args.gpu_count,
        args.resume,
    )
    complete = False
    try:
        run_groups(tasks, gpus, output_dir)
    finally:
        complete = consolidate_outputs(output_dir, tasks, specs, manifest)
        manifest["complete"] = complete
        manifest["completed_at"] = utc_now() if complete else None
        write_yaml(manifest_path, manifest)
    if not complete:
        raise RuntimeError("profile run ended with pending specs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
