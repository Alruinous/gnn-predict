from __future__ import annotations

import asyncio
import time
from typing import Literal

import ray

from workflow.replica import (
    BackendFactory,
    BackendGeneration,
    FinishReason,
    GenerationAbortedError,
    GenerationBackend,
    GenerationEngineFailedError,
    GenerationOutOfMemoryError,
    GenerationRequest,
    GpuIdProvider,
    ModelDeploymentConfig,
    ReplicaInferenceResult,
    ReplicaLoadResult,
    ReplicaStats,
)
from workflow.schema import ExecutionConfig


class StaticVLLMEngine:
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
            raise RuntimeError("static vLLM engine is already loaded")
        if self._stopping:
            raise RuntimeError("static vLLM engine is stopping")
        gpu_ids = tuple(self._gpu_id_provider())
        if len(gpu_ids) != 1:
            raise RuntimeError(
                f"static vLLM engine requires exactly one Ray GPU id, got {gpu_ids}"
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
            raise RuntimeError("static vLLM engine is stopping")
        if generation.serving != self.deployment.serving:
            raise ValueError("generation serving config does not match the engine")
        if max_new_tokens != generation.max_new_tokens:
            raise ValueError("generation output limit does not match the workflow")
        if request_id in self._active_request_ids:
            raise ValueError(f"duplicate vLLM request id: {request_id}")

        inflight = len(self._active_request_ids) + 1
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
        result: BackendGeneration
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

        status: Literal["success", "cancelled"] = (
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
                "failed to abort active static vLLM requests"
            ) from abort_failure

    def get_stats(self) -> ReplicaStats:
        return ReplicaStats(
            active_requests=len(self._active_request_ids),
            peak_active_requests=self._peak_active_requests,
            stopping=self._stopping,
        )

    def _require_backend(self) -> GenerationBackend:
        if self._backend is None:
            raise RuntimeError("static vLLM engine must load before invoke")
        return self._backend


def _create_vllm_backend(deployment: ModelDeploymentConfig) -> GenerationBackend:
    # vLLM is imported only by the isolated serving actor environment.
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


StaticVLLMEngineActor = ray.remote(num_gpus=1)(StaticVLLMEngine)
