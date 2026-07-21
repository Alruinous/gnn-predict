from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import ClassVar, Literal, Protocol

import ray
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)

from common.validate import NonEmptyStr
from workflow.schema import AgentNodeConfig, ExecutionConfig, ServingConfig

PhysicalGpuId = int | str
FinishReason = Literal["stop", "length", "abort"]
ReplicaStatus = Literal["success", "failed", "oom", "cancelled"]

VLLM_VERSION = "0.10.2"
VLLM_ENGINE_MODE = "V0"
VLLM_ATTENTION_BACKEND = "XFORMERS"


class ReplicaContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelDeploymentConfig(ReplicaContract):
    model_name: NonEmptyStr
    model_path: NonEmptyStr
    dtype: Literal["float16"]
    serving: ServingConfig

    @classmethod
    def from_node(cls, node: AgentNodeConfig) -> ModelDeploymentConfig:
        return cls(
            model_name=node.model.name,
            model_path=node.execution.model_path,
            dtype=node.execution.dtype,
            serving=node.execution.serving,
        )

    @property
    def model_key(self) -> str:
        payload = json.dumps(
            {
                "backend": f"vllm-{VLLM_VERSION}-{VLLM_ENGINE_MODE}",
                "model_name": self.model_name,
                "model_path": self.model_path,
                "dtype": self.dtype,
                "serving": self.serving.model_dump(mode="json"),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class PromptEncoding(ReplicaContract):
    text: str
    token_ids: tuple[NonNegativeInt, ...]

    @property
    def input_tokens(self) -> int:
        return len(self.token_ids)


class GenerationRequest(ReplicaContract):
    request_id: NonEmptyStr
    prompt_token_ids: tuple[NonNegativeInt, ...]
    max_new_tokens: PositiveInt
    do_sample: bool
    temperature: float | None

    @property
    def input_tokens(self) -> int:
        return len(self.prompt_token_ids)


class BackendGeneration(ReplicaContract):
    output_token_ids: tuple[NonNegativeInt, ...]
    finish_reason: FinishReason
    queue_time_sec: NonNegativeFloat | None = None
    time_to_first_token_sec: NonNegativeFloat | None = None


class ReplicaLoadResult(ReplicaContract):
    physical_gpu_id: PhysicalGpuId
    duration_sec: NonNegativeFloat
    idle_vram_mb: NonNegativeFloat
    vllm_version: NonEmptyStr = VLLM_VERSION
    engine_mode: Literal["V0"] = VLLM_ENGINE_MODE
    attention_backend: Literal["XFORMERS"] = VLLM_ATTENTION_BACKEND
    max_num_seqs: PositiveInt = 1
    block_size: PositiveInt
    num_gpu_blocks: PositiveInt
    gpu_kv_tokens: PositiveInt


class ReplicaInferenceResult(ReplicaContract):
    request_id: NonEmptyStr
    output_token_ids: tuple[NonNegativeInt, ...]
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    hit_token_limit: bool
    finish_reason: FinishReason | None
    queue_time_sec: NonNegativeFloat | None
    time_to_first_token_sec: NonNegativeFloat | None
    replica_inflight_at_start: PositiveInt
    started_at: float
    finished_at: float
    duration_sec: NonNegativeFloat
    status: ReplicaStatus
    engine_failed: bool = False
    error_type: str | None
    error_message: str | None


class ReplicaStats(ReplicaContract):
    active_requests: NonNegativeInt
    peak_active_requests: NonNegativeInt
    stopping: bool


class GenerationAbortedError(RuntimeError):
    pass


class GenerationEngineFailedError(RuntimeError):
    pass


class GenerationOutOfMemoryError(RuntimeError):
    pass


class GenerationBackend(Protocol):
    idle_vram_mb: float
    vllm_version: str
    engine_mode: Literal["V0"]
    attention_backend: Literal["XFORMERS"]
    block_size: int
    num_gpu_blocks: int
    gpu_kv_tokens: int

    def load(self) -> None: ...

    async def generate(self, request: GenerationRequest) -> BackendGeneration: ...

    async def abort(self, request_id: str) -> None: ...

    def shutdown(self) -> None: ...


BackendFactory = Callable[[ModelDeploymentConfig], GenerationBackend]
GpuIdProvider = Callable[[], Sequence[PhysicalGpuId]]


class GenerationEngine:
    engine_label: ClassVar[str] = "generation engine"
    # Subclasses opt into rejecting requests beyond serving.max_num_seqs. The
    # static baseline engine deliberately leaves this off: it has no scheduler
    # admission control in front of it and relies on vLLM's own batching
    # instead (see workflow/baseline/vllm_engine.py and the tests asserting
    # it accepts more concurrent requests than max_num_seqs).
    enforce_capacity: ClassVar[bool] = False

    def __init__(
        self,
        deployment: ModelDeploymentConfig,
        *,
        backend_factory: BackendFactory | None = None,
        gpu_id_provider: GpuIdProvider | None = None,
    ) -> None:
        self.deployment = deployment
        self._backend_factory = backend_factory or _create_vllm_backend
        self._gpu_id_provider = gpu_id_provider or ray.get_gpu_ids
        self._backend: GenerationBackend | None = None
        self._active_request_ids: set[str] = set()
        self._peak_active_requests = 0
        self._stopping = False

    def load(self) -> ReplicaLoadResult:
        if self._backend is not None:
            raise RuntimeError(f"{self.engine_label} is already loaded")
        if self._stopping:
            raise RuntimeError(f"{self.engine_label} is stopping")
        gpu_ids = tuple(self._gpu_id_provider())
        if len(gpu_ids) != 1:
            raise RuntimeError(
                f"{self.engine_label} requires exactly one Ray GPU id, got {gpu_ids}"
            )

        started = time.perf_counter()
        backend = self._backend_factory(self.deployment)
        backend.load()
        duration_sec = time.perf_counter() - started
        self._backend = backend
        return ReplicaLoadResult(
            physical_gpu_id=gpu_ids[0],
            duration_sec=duration_sec,
            idle_vram_mb=backend.idle_vram_mb,
            vllm_version=backend.vllm_version,
            engine_mode=backend.engine_mode,
            attention_backend=backend.attention_backend,
            max_num_seqs=self.deployment.serving.max_num_seqs,
            block_size=backend.block_size,
            num_gpu_blocks=backend.num_gpu_blocks,
            gpu_kv_tokens=backend.gpu_kv_tokens,
        )

    async def invoke(
        self,
        request_id: str,
        prompt_token_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        generation: ExecutionConfig,
    ) -> ReplicaInferenceResult:
        backend = self._require_backend()
        if self._stopping:
            raise RuntimeError(f"{self.engine_label} is stopping")
        if generation.serving != self.deployment.serving:
            raise ValueError(
                f"generation serving config does not match the {self.engine_label}"
            )
        if max_new_tokens != generation.max_new_tokens:
            raise ValueError("generation output limit does not match the workflow")
        if request_id in self._active_request_ids:
            raise ValueError(f"duplicate vLLM request id: {request_id}")
        inflight = len(self._active_request_ids) + 1
        if self.enforce_capacity and inflight > self.deployment.serving.max_num_seqs:
            raise RuntimeError(f"{self.engine_label} request capacity exceeded")
        request = GenerationRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            do_sample=generation.do_sample,
            temperature=generation.temperature,
        )
        self._active_request_ids.add(request_id)
        self._peak_active_requests = max(self._peak_active_requests, inflight)
        started_at = time.time()
        started = time.perf_counter()
        try:
            result = await backend.generate(request)
        except GenerationOutOfMemoryError as error:
            return _failure_result(
                request=request,
                status="oom",
                error=error,
                inflight=inflight,
                started_at=started_at,
                started=started,
            )
        except GenerationAbortedError as error:
            return _failure_result(
                request=request,
                status="cancelled",
                error=error,
                inflight=inflight,
                started_at=started_at,
                started=started,
                finish_reason="abort",
            )
        except GenerationEngineFailedError as error:
            return _failure_result(
                request=request,
                status="failed",
                error=error,
                inflight=inflight,
                started_at=started_at,
                started=started,
                engine_failed=True,
            )
        except asyncio.CancelledError:
            await backend.abort(request_id)
            raise
        except Exception as error:
            return _failure_result(
                request=request,
                status="failed",
                error=error,
                inflight=inflight,
                started_at=started_at,
                started=started,
            )
        finally:
            self._active_request_ids.discard(request_id)

        status: ReplicaStatus = (
            "cancelled" if result.finish_reason == "abort" else "success"
        )
        return ReplicaInferenceResult(
            request_id=request.request_id,
            output_token_ids=result.output_token_ids,
            input_tokens=request.input_tokens,
            output_tokens=len(result.output_token_ids),
            hit_token_limit=result.finish_reason == "length",
            finish_reason=result.finish_reason,
            queue_time_sec=result.queue_time_sec,
            time_to_first_token_sec=result.time_to_first_token_sec,
            replica_inflight_at_start=inflight,
            started_at=started_at,
            finished_at=time.time(),
            duration_sec=time.perf_counter() - started,
            status=status,
            engine_failed=False,
            error_type=None,
            error_message=None,
        )

    async def abort(self, request_id: str) -> None:
        if request_id not in self._active_request_ids:
            return
        await self._require_backend().abort(request_id)

    async def shutdown(self) -> None:
        if self._backend is None:
            self._stopping = True
            return
        self._stopping = True
        aborts = [
            self._backend.abort(request_id)
            for request_id in tuple(self._active_request_ids)
        ]
        abort_failure: BaseException | None = None
        if aborts:
            results = await asyncio.gather(*aborts, return_exceptions=True)
            failures = [
                result for result in results if isinstance(result, BaseException)
            ]
            if failures:
                abort_failure = failures[0]
        self._backend.shutdown()
        self._backend = None
        if abort_failure is not None:
            raise RuntimeError(
                f"failed to abort active {self.engine_label} requests"
            ) from abort_failure

    def get_stats(self) -> ReplicaStats:
        return ReplicaStats(
            active_requests=len(self._active_request_ids),
            peak_active_requests=self._peak_active_requests,
            stopping=self._stopping,
        )

    def _require_backend(self) -> GenerationBackend:
        if self._backend is None:
            raise RuntimeError(f"{self.engine_label} must load before invoke")
        return self._backend


class ModelReplica(GenerationEngine):
    engine_label = "model replica"
    enforce_capacity = True


def _create_vllm_backend(deployment: ModelDeploymentConfig) -> GenerationBackend:
    # The driver and serving actor intentionally use different Python environments.
    from workflow.vllm_backend import VLLMBackend

    return VLLMBackend(deployment)


def _failure_result(
    *,
    request: GenerationRequest,
    status: Literal["failed", "oom", "cancelled"],
    error: Exception,
    inflight: int,
    started_at: float,
    started: float,
    finish_reason: FinishReason | None = None,
    engine_failed: bool = False,
) -> ReplicaInferenceResult:
    return ReplicaInferenceResult(
        request_id=request.request_id,
        output_token_ids=(),
        input_tokens=request.input_tokens,
        output_tokens=0,
        hit_token_limit=False,
        finish_reason=finish_reason,
        queue_time_sec=None,
        time_to_first_token_sec=None,
        replica_inflight_at_start=inflight,
        started_at=started_at,
        finished_at=time.time(),
        duration_sec=time.perf_counter() - started,
        status=status,
        engine_failed=engine_failed,
        error_type=type(error).__name__,
        error_message=str(error),
    )


ModelReplicaActor = ray.remote(num_gpus=1)(ModelReplica)
