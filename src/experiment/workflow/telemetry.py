from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from threading import Event, Thread
from typing import Any, Protocol, Self, TextIO

import pynvml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)

from experiment.workflow.artifacts import canonical_json

TelemetryRow = Mapping[str, object]


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    interval_sec: PositiveFloat
    physical_gpu_ids: tuple[NonNegativeInt, ...] = Field(min_length=1)
    raw_samples_retained: bool = True
    energy_baseline_method: str
    memory_baseline_method: str

    @model_validator(mode="after")
    def validate_gpu_ids(self) -> Self:
        if len(self.physical_gpu_ids) != len(set(self.physical_gpu_ids)):
            raise ValueError("telemetry GPU ids must be unique")
        return self


class Sampler(Protocol):
    def start(self) -> None: ...

    def sample(self) -> Sequence[TelemetryRow]: ...

    def stop(self) -> None: ...


class PeriodicRecorder:
    def __init__(self, path: Path, sampler: Sampler, interval_sec: float) -> None:
        if interval_sec <= 0:
            raise ValueError("telemetry interval must be positive")
        self.path = path
        self.sampler = sampler
        self.interval_sec = interval_sec
        self._stop = Event()
        self._thread: Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry recorder has already started")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as setup:
            self.sampler.start()
            setup.callback(self.sampler.stop)
            file = setup.enter_context(self.path.open("x", encoding="utf-8"))
            _write_sample(file, self.sampler)
            resources = setup.pop_all()

        def record() -> None:
            try:
                while not self._stop.wait(self.interval_sec):
                    _write_sample(file, self.sampler)
            except Exception as error:
                self._error = error
                self._stop.set()
            finally:
                resources.close()

        self._thread = Thread(target=record, name=f"telemetry:{self.path.name}")
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            raise RuntimeError("telemetry recorder has not started")
        self._stop.set()
        thread.join()
        if self._error is not None:
            raise RuntimeError(
                f"telemetry recorder failed: {self.path}"
            ) from self._error


def _write_sample(file: TextIO, sampler: Sampler) -> None:
    for row in sampler.sample():
        file.write(canonical_json(row))
        file.write("\n")
    file.flush()


class NvmlSampler:
    def __init__(self, gpu_indices: Sequence[int]) -> None:
        if not gpu_indices:
            raise ValueError("GPU telemetry requires at least one index")
        if len(gpu_indices) != len(set(gpu_indices)):
            raise ValueError("GPU telemetry indices must be unique")
        self.gpu_indices = tuple(gpu_indices)
        self._handles: tuple[Any, ...] = ()

    def start(self) -> None:
        pynvml.nvmlInit()
        self._handles = tuple(
            pynvml.nvmlDeviceGetHandleByIndex(index) for index in self.gpu_indices
        )

    def sample(self) -> Sequence[TelemetryRow]:
        if not self._handles:
            raise RuntimeError("NVML sampler has not started")
        ts = time.time()
        rows = []
        for index, handle in zip(self.gpu_indices, self._handles, strict=True):
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            rows.append(
                {
                    "ts": ts,
                    "gpu_index": index,
                    "gpu_uuid": uuid,
                    "memory_used_mb": memory.used / 1024**2,
                    "memory_total_mb": memory.total / 1024**2,
                    "gpu_utilization_percent": utilization.gpu,
                    "memory_utilization_percent": utilization.memory,
                    "power_watts": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                }
            )
        return rows

    def stop(self) -> None:
        self._handles = ()
        pynvml.nvmlShutdown()


class QueueSampler:
    def __init__(self, queues: Mapping[str, object]) -> None:
        self.queues = dict(queues)

    def start(self) -> None:
        pass

    def sample(self) -> Sequence[TelemetryRow]:
        ts = time.time()
        rows = []
        for node_id, queue in sorted(self.queues.items()):
            qsize = getattr(queue, "qsize", None)
            if not callable(qsize):
                raise TypeError(f"queue has no qsize method: {node_id}")
            rows.append({"ts": ts, "node_id": node_id, "queue_size": qsize()})
        return rows

    def stop(self) -> None:
        pass


class CallableSampler:
    def __init__(self, sample: Callable[[], Sequence[TelemetryRow]]) -> None:
        self._sample = sample

    def start(self) -> None:
        pass

    def sample(self) -> Sequence[TelemetryRow]:
        return self._sample()

    def stop(self) -> None:
        pass
