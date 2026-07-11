from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, cast

import ray
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)

from common.validate import NonEmptyStr
from workflow.schema import AgentNodeConfig, ExecutionConfig

PhysicalGpuId = int | str


class ReplicaContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelDeploymentConfig(ReplicaContract):
    model_name: NonEmptyStr
    model_path: NonEmptyStr
    dtype: NonEmptyStr

    @classmethod
    def from_node(cls, node: AgentNodeConfig) -> ModelDeploymentConfig:
        return cls(
            model_name=node.model.name,
            model_path=node.execution.model_path,
            dtype=node.execution.dtype,
        )

    @property
    def model_key(self) -> str:
        payload = json.dumps(
            (self.model_name, self.model_path, self.dtype),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class GenerationRequest(ReplicaContract):
    prompt: str
    input_tokens: NonNegativeInt
    max_new_tokens: PositiveInt
    do_sample: bool
    temperature: float | None


class BackendGeneration(ReplicaContract):
    output_text: str
    output_tokens: NonNegativeInt


class ReplicaLoadResult(ReplicaContract):
    physical_gpu_id: PhysicalGpuId
    duration_sec: NonNegativeFloat
    idle_vram_mb: NonNegativeFloat


class ReplicaInferenceResult(ReplicaContract):
    output_text: str | None
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    hit_token_limit: bool
    started_at: float
    finished_at: float
    duration_sec: NonNegativeFloat
    status: Literal["success", "failed", "oom"]
    error_type: str | None
    error_message: str | None


class GenerationBackend(Protocol):
    def generate(self, request: GenerationRequest) -> BackendGeneration: ...


class PromptTokenizerBackend(Protocol):
    truncation_side: str

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        /,
        **kwargs: object,
    ) -> object: ...

    def encode(self, prompt: str, /, **kwargs: object) -> list[int]: ...


class TransformersTokenizerBackend(Protocol):
    def __call__(self, prompt: str, /, **kwargs: object) -> Any: ...

    def decode(self, token_ids: object, /, **kwargs: object) -> object: ...


class TransformersModelBackend(Protocol):
    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> Any: ...


BackendFactory = Callable[[ModelDeploymentConfig], GenerationBackend]
GpuIdProvider = Callable[[], Sequence[PhysicalGpuId]]
PromptTokenizerFactory = Callable[[ExecutionConfig], PromptTokenizerBackend]


class PromptTokenizer:
    def __init__(
        self,
        execution: ExecutionConfig,
        *,
        tokenizer_factory: PromptTokenizerFactory | None = None,
    ) -> None:
        factory = tokenizer_factory or _load_prompt_tokenizer
        self._execution = execution
        self._tokenizer = factory(execution)
        if execution.truncation_side is not None:
            self._tokenizer.truncation_side = execution.truncation_side

    def build_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> tuple[str, int]:
        if self._execution.use_chat_template:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._execution.enable_thinking,
            )
            if not isinstance(rendered, str):
                raise TypeError("chat template must render a string")
            prompt = rendered
        elif system_prompt is None:
            prompt = user_prompt
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}"

        token_ids = self._tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )
        return prompt, len(token_ids)


class TransformersBackend:
    def __init__(self, deployment: ModelDeploymentConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        try:
            dtype = dtype_map[deployment.dtype]
        except KeyError as error:
            supported = ", ".join(sorted(dtype_map))
            raise ValueError(
                f"unsupported dtype {deployment.dtype!r}; expected one of {supported}"
            ) from error

        tokenizer = AutoTokenizer.from_pretrained(
            deployment.model_path,
            local_files_only=True,
        )
        if tokenizer is None:
            raise RuntimeError("AutoTokenizer returned no tokenizer")
        self._tokenizer = cast(TransformersTokenizerBackend, tokenizer)
        self._model = cast(
            TransformersModelBackend,
            AutoModelForCausalLM.from_pretrained(
                deployment.model_path,
                local_files_only=True,
                dtype=dtype,
                device_map="cuda:0",
            ),
        )
        self._model.eval()
        self._idle_vram_mb = float(torch.cuda.memory_allocated("cuda:0")) / 1024**2

    @property
    def idle_vram_mb(self) -> float:
        return self._idle_vram_mb

    def generate(self, request: GenerationRequest) -> BackendGeneration:
        import torch

        encoded = self._tokenizer(
            request.prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
        encoded = encoded.to("cuda:0")
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens != request.input_tokens:
            raise ValueError(
                "prompt token count changed: "
                f"expected {request.input_tokens}, got {input_tokens}"
            )
        generation_kwargs: dict[str, object] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "return_dict_in_generate": False,
        }
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **encoded,
                **generation_kwargs,
            )
        new_token_ids = generated_ids[0, input_tokens:]
        output_text = self._tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
        )
        if not isinstance(output_text, str):
            raise TypeError("tokenizer decode must return a string")
        return BackendGeneration(
            output_text=output_text,
            output_tokens=int(new_token_ids.shape[-1]),
        )


class ModelReplica:
    def __init__(
        self,
        deployment: ModelDeploymentConfig,
        *,
        backend_factory: BackendFactory | None = None,
        gpu_id_provider: GpuIdProvider | None = None,
    ) -> None:
        self.deployment = deployment
        self._backend_factory = backend_factory or TransformersBackend
        self._gpu_id_provider = gpu_id_provider or ray.get_gpu_ids
        self._backend: GenerationBackend | None = None

    def load(self) -> ReplicaLoadResult:
        if self._backend is not None:
            raise RuntimeError("model replica is already loaded")
        gpu_ids = tuple(self._gpu_id_provider())
        if len(gpu_ids) != 1:
            raise RuntimeError(
                f"model replica requires exactly one Ray GPU id, got {gpu_ids}"
            )

        started = time.perf_counter()
        backend = self._backend_factory(self.deployment)
        duration_sec = time.perf_counter() - started
        self._backend = backend
        idle_vram_mb = (
            backend.idle_vram_mb if isinstance(backend, TransformersBackend) else 0.0
        )
        return ReplicaLoadResult(
            physical_gpu_id=gpu_ids[0],
            duration_sec=duration_sec,
            idle_vram_mb=idle_vram_mb,
        )

    def invoke(
        self,
        prompt: str,
        *,
        input_tokens: int,
        max_new_tokens: int,
        generation: ExecutionConfig,
    ) -> ReplicaInferenceResult:
        import torch

        backend = self._backend
        if backend is None:
            raise RuntimeError("model replica must load before invoke")
        request = GenerationRequest(
            prompt=prompt,
            input_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=generation.do_sample,
            temperature=generation.temperature,
        )
        started_at = time.time()
        started = time.perf_counter()
        try:
            result = backend.generate(request)
        except torch.cuda.OutOfMemoryError as error:
            return _failure_result(
                request=request,
                status="oom",
                error=error,
                started_at=started_at,
                started=started,
            )
        except Exception as error:
            return _failure_result(
                request=request,
                status="failed",
                error=error,
                started_at=started_at,
                started=started,
            )

        return ReplicaInferenceResult(
            output_text=result.output_text,
            input_tokens=request.input_tokens,
            output_tokens=result.output_tokens,
            hit_token_limit=result.output_tokens >= request.max_new_tokens,
            started_at=started_at,
            finished_at=time.time(),
            duration_sec=time.perf_counter() - started,
            status="success",
            error_type=None,
            error_message=None,
        )


def _load_prompt_tokenizer(execution: ExecutionConfig) -> PromptTokenizerBackend:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        execution.model_path,
        local_files_only=True,
    )
    if tokenizer is None:
        raise RuntimeError("AutoTokenizer returned no tokenizer")
    return cast(PromptTokenizerBackend, tokenizer)


def _failure_result(
    *,
    request: GenerationRequest,
    status: Literal["failed", "oom"],
    error: Exception,
    started_at: float,
    started: float,
) -> ReplicaInferenceResult:
    return ReplicaInferenceResult(
        output_text=None,
        input_tokens=request.input_tokens,
        output_tokens=0,
        hit_token_limit=False,
        started_at=started_at,
        finished_at=time.time(),
        duration_sec=time.perf_counter() - started,
        status=status,
        error_type=type(error).__name__,
        error_message=str(error),
    )


ModelReplicaActor = ray.remote(num_gpus=1)(ModelReplica)
