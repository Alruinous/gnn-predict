from __future__ import annotations

import os
from typing import Literal, cast

import torch
import vllm
from vllm import SamplingParams
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncEngineDeadError, AsyncLLMEngine
from vllm.outputs import RequestOutput

from workflow.replica import (
    VLLM_ATTENTION_BACKEND,
    VLLM_ENGINE_MODE,
    VLLM_VERSION,
    BackendGeneration,
    FinishReason,
    GenerationAbortedError,
    GenerationEngineFailedError,
    GenerationOutOfMemoryError,
    GenerationRequest,
    ModelDeploymentConfig,
)


class VLLMBackend:
    def __init__(self, deployment: ModelDeploymentConfig) -> None:
        self.deployment = deployment
        self.idle_vram_mb = 0.0
        self.vllm_version = VLLM_VERSION
        self.engine_mode: Literal["V0"] = VLLM_ENGINE_MODE
        self.attention_backend: Literal["XFORMERS"] = VLLM_ATTENTION_BACKEND
        self._engine: AsyncLLMEngine | None = None

    def load(self) -> None:
        if vllm.__version__ != VLLM_VERSION:
            raise RuntimeError(
                "vLLM version mismatch: "
                f"expected {VLLM_VERSION}, got {vllm.__version__}"
            )
        if os.environ.get("VLLM_USE_V1") != "0":
            raise RuntimeError("VLLM_USE_V1=0 is required")
        if os.environ.get("VLLM_ATTENTION_BACKEND") != VLLM_ATTENTION_BACKEND:
            raise RuntimeError("VLLM_ATTENTION_BACKEND=XFORMERS is required")
        capability = torch.cuda.get_device_capability(0)
        if capability < (7, 0):
            raise RuntimeError(
                f"vLLM serving requires compute capability >= 7.0: {capability}"
            )

        serving = self.deployment.serving
        engine_args = AsyncEngineArgs(
            model=self.deployment.model_path,
            tokenizer=self.deployment.model_path,
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype=self.deployment.dtype,
            kv_cache_dtype="auto",
            seed=0,
            max_model_len=serving.max_model_len,
            distributed_executor_backend="uni",
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            block_size=16,
            enable_prefix_caching=False,
            swap_space=0.0,
            cpu_offload_gb=0.0,
            gpu_memory_utilization=serving.gpu_memory_utilization,
            max_num_batched_tokens=serving.max_num_batched_tokens,
            max_num_seqs=serving.max_num_seqs,
            quantization=None,
            enforce_eager=True,
            num_lookahead_slots=0,
            enable_chunked_prefill=False,
            speculative_config=None,
            scheduling_policy="fcfs",
            enable_log_requests=False,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.idle_vram_mb = float(torch.cuda.memory_allocated(0)) / 1024**2

    async def generate(self, request: GenerationRequest) -> BackendGeneration:
        engine = self._require_engine()
        sampling_params = SamplingParams(
            temperature=(
                request.temperature if request.temperature is not None else 1.0
            )
            if request.do_sample
            else 0.0,
            max_tokens=request.max_new_tokens,
            detokenize=False,
            skip_special_tokens=False,
        )
        final_output: RequestOutput | None = None
        try:
            outputs = engine.generate(
                {"prompt_token_ids": list(request.prompt_token_ids)},
                sampling_params,
                request.request_id,
            )
            async for output in outputs:
                final_output = output
        except torch.cuda.OutOfMemoryError as error:
            raise GenerationOutOfMemoryError(str(error)) from error
        except AsyncEngineDeadError as error:
            raise GenerationEngineFailedError(str(error)) from error
        if final_output is None:
            raise GenerationAbortedError(request.request_id)
        if len(final_output.outputs) != 1:
            raise RuntimeError("vLLM must return exactly one completion")

        completion = final_output.outputs[0]
        raw_finish_reason = completion.finish_reason
        if raw_finish_reason not in ("stop", "length", "abort"):
            raise RuntimeError(f"unsupported vLLM finish reason: {raw_finish_reason!r}")
        finish_reason = cast(FinishReason, raw_finish_reason)
        metrics = final_output.metrics
        queue_time_sec = None
        time_to_first_token_sec = None
        if metrics is not None:
            queue_time_sec = metrics.time_in_queue
            if metrics.first_token_time is not None:
                time_to_first_token_sec = (
                    metrics.first_token_time - metrics.arrival_time
                )
        return BackendGeneration(
            output_token_ids=tuple(completion.token_ids),
            finish_reason=finish_reason,
            queue_time_sec=queue_time_sec,
            time_to_first_token_sec=time_to_first_token_sec,
        )

    async def abort(self, request_id: str) -> None:
        await self._require_engine().abort(request_id)

    def shutdown(self) -> None:
        engine = self._engine
        if engine is None:
            return
        engine.shutdown_background_loop()
        engine.engine.model_executor.shutdown()
        self._engine = None
        del engine
        cleanup_dist_env_and_memory()

    def _require_engine(self) -> AsyncLLMEngine:
        if self._engine is None:
            raise RuntimeError("vLLM backend is not loaded")
        return self._engine
