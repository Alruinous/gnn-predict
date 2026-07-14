from __future__ import annotations

import asyncio
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

from workflow.replica import (
    BackendGeneration,
    GenerationEngineFailedError,
    GenerationOutOfMemoryError,
    GenerationRequest,
    ModelDeploymentConfig,
    ModelReplica,
    PromptEncoding,
)
from workflow.schema import AgentNodeConfig, ExecutionConfig, ServingConfig
from workflow.tokenizer import PromptTokenizer


def serving(**updates: object) -> ServingConfig:
    values: dict[str, object] = {
        "max_model_len": 1024,
        "max_num_seqs": 3,
        "max_num_batched_tokens": 1536,
        "gpu_memory_utilization": 0.98,
    }
    values.update(updates)
    return ServingConfig.model_validate(values)


def agent_node(
    *,
    model_name: str = "test-model",
    model_path: str = "/models/test-model",
    do_sample: bool = False,
    temperature: float | None = None,
    enable_thinking: bool = False,
    serving_config: ServingConfig | None = None,
) -> AgentNodeConfig:
    return AgentNodeConfig.model_validate(
        {
            "name": "agent-a",
            "type": "agent",
            "model": {"name": model_name},
            "execution": {
                "model_path": model_path,
                "dtype": "float16",
                "do_sample": do_sample,
                "temperature": temperature,
                "enable_thinking": enable_thinking,
                "serving": (serving_config or serving()).model_dump(),
            },
            "token_budget": {
                "min_max_new_tokens": 8,
                "default_max_new_tokens": 16,
                "max_max_new_tokens": 32,
            },
            "prompt_template": "Question: {content}",
            "system_prompt": "Be concise.",
        }
    )


DEPLOYMENT = ModelDeploymentConfig.from_node(agent_node())
GENERATION = ExecutionConfig(
    model_path=DEPLOYMENT.model_path,
    dtype=DEPLOYMENT.dtype,
    do_sample=True,
    temperature=0.7,
    serving=DEPLOYMENT.serving,
)


class FakeBackend:
    idle_vram_mb: float = 123.0
    vllm_version: str = "0.10.2"
    engine_mode: Literal["V0"] = "V0"
    attention_backend: Literal["XFORMERS"] = "XFORMERS"

    def __init__(
        self,
        generation: BackendGeneration | None = None,
        error: Exception | None = None,
    ) -> None:
        self.generation = generation or BackendGeneration(
            output_token_ids=(20, 21, 22, 23),
            finish_reason="stop",
        )
        self.error = error
        self.requests: list[GenerationRequest] = []
        self.aborted: list[str] = []
        self.loaded = False
        self.stopped = False

    def load(self) -> None:
        self.loaded = True

    async def generate(self, request: GenerationRequest) -> BackendGeneration:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.generation

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    def shutdown(self) -> None:
        self.stopped = True


class ConcurrentBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.peak = 0
        self.release = asyncio.Event()

    async def generate(self, request: GenerationRequest) -> BackendGeneration:
        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.peak == 2:
            self.release.set()
        await self.release.wait()
        self.active -= 1
        return self.generation


class FakeTokenizer:
    def __init__(self) -> None:
        self.truncation_side = "right"
        self.chat_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
        self.encode_calls: list[tuple[str, dict[str, object]]] = []
        self.decode_calls: list[tuple[list[int], dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        self.chat_calls.append((messages, kwargs))
        body = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return f"{body}<assistant>"

    def encode(self, prompt: str, **kwargs: object) -> list[int]:
        self.encode_calls.append((prompt, kwargs))
        return list(range(len(prompt.split())))

    def decode(self, token_ids: object, **kwargs: object) -> str:
        assert isinstance(token_ids, list)
        assert all(isinstance(token_id, int) for token_id in token_ids)
        self.decode_calls.append((cast(list[int], token_ids), kwargs))
        return "decoded answer"


def loaded_replica(backend: FakeBackend) -> ModelReplica:
    replica = ModelReplica(
        DEPLOYMENT,
        backend_factory=lambda _: backend,
        gpu_id_provider=lambda: [3],
    )
    replica.load()
    return replica


def invoke(
    replica: ModelReplica,
    *,
    request_id: str = "request-1",
    max_new_tokens: int = 96,
) -> Any:
    return replica.invoke(
        request_id,
        (1, 2, 3),
        max_new_tokens=max_new_tokens,
        generation=GENERATION,
    )


def test_deployment_key_includes_engine_capacity_but_not_sampling() -> None:
    default = ModelDeploymentConfig.from_node(agent_node())
    sampling_variant = ModelDeploymentConfig.from_node(
        agent_node(do_sample=True, temperature=0.9, enable_thinking=True)
    )
    capacity_variant = ModelDeploymentConfig.from_node(
        agent_node(serving_config=serving(max_num_batched_tokens=2048))
    )

    assert default.model_key == sampling_variant.model_key
    assert default.model_key != capacity_variant.model_key


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "other-model"),
        ("model_path", "/models/other-model"),
    ],
)
def test_each_weight_field_changes_model_key(field: str, value: str) -> None:
    payload = DEPLOYMENT.model_dump()
    payload[field] = value

    assert ModelDeploymentConfig(**payload).model_key != DEPLOYMENT.model_key


def test_deployment_config_is_frozen_strict_and_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ModelDeploymentConfig.model_validate(
            {
                **DEPLOYMENT.model_dump(),
                "model_name": 1,
            }
        )
    with pytest.raises(ValidationError):
        ModelDeploymentConfig.model_validate(
            {
                **DEPLOYMENT.model_dump(),
                "device": "cuda:7",
            }
        )
    with pytest.raises(ValidationError):
        DEPLOYMENT.model_name = "other-model"


def test_load_reports_vllm_runtime_and_ray_gpu() -> None:
    backend = FakeBackend()
    replica = ModelReplica(
        DEPLOYMENT,
        backend_factory=lambda _: backend,
        gpu_id_provider=lambda: ["7"],
    )
    result = replica.load()

    assert backend.loaded is True
    assert result.physical_gpu_id == "7"
    assert result.idle_vram_mb == 123
    assert result.vllm_version == "0.10.2"
    assert result.engine_mode == "V0"
    assert result.attention_backend == "XFORMERS"
    assert result.max_num_seqs == 3


@pytest.mark.parametrize("gpu_ids", [[], [0, 1]])
def test_load_requires_exactly_one_ray_gpu(gpu_ids: list[int]) -> None:
    backend = FakeBackend()
    replica = ModelReplica(
        DEPLOYMENT,
        backend_factory=lambda _: backend,
        gpu_id_provider=lambda: gpu_ids,
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        replica.load()

    assert backend.loaded is False


def test_invoke_fails_fast_before_load() -> None:
    replica = ModelReplica(
        DEPLOYMENT,
        backend_factory=lambda _: FakeBackend(),
        gpu_id_provider=lambda: [0],
    )

    with pytest.raises(RuntimeError, match="load"):
        asyncio.run(invoke(replica))


def test_replica_passes_exact_tokens_and_generation_parameters() -> None:
    backend = FakeBackend()

    async def scenario() -> None:
        replica = loaded_replica(backend)
        result = await invoke(replica)

        assert backend.requests == [
            GenerationRequest(
                request_id="request-1",
                prompt_token_ids=(1, 2, 3),
                max_new_tokens=96,
                do_sample=True,
                temperature=0.7,
            )
        ]
        assert result.output_token_ids == (20, 21, 22, 23)
        assert result.output_tokens == 4
        assert result.finish_reason == "stop"
        assert result.hit_token_limit is False

    asyncio.run(scenario())


def test_finish_reason_length_sets_token_limit() -> None:
    backend = FakeBackend(
        BackendGeneration(
            output_token_ids=(20, 21),
            finish_reason="length",
            queue_time_sec=0.2,
            time_to_first_token_sec=0.4,
        )
    )

    async def scenario() -> None:
        result = await invoke(loaded_replica(backend), max_new_tokens=2)

        assert result.status == "success"
        assert result.hit_token_limit is True
        assert result.queue_time_sec == 0.2
        assert result.time_to_first_token_sec == 0.4
        assert result.replica_inflight_at_start == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("error", "status", "engine_failed"),
    [
        (GenerationOutOfMemoryError("CUDA exhausted"), "oom", False),
        (ValueError("invalid generation"), "failed", False),
        (GenerationEngineFailedError("engine stopped"), "failed", True),
    ],
)
def test_backend_errors_are_normalized(
    error: Exception,
    status: str,
    engine_failed: bool,
) -> None:
    backend = FakeBackend(error=error)

    async def scenario() -> None:
        result = await invoke(loaded_replica(backend))

        assert result.status == status
        assert result.output_token_ids == ()
        assert result.output_tokens == 0
        assert result.engine_failed is engine_failed
        assert result.error_type == type(error).__name__
        assert result.error_message == str(error)

    asyncio.run(scenario())


def test_replica_accepts_overlapping_requests() -> None:
    backend = ConcurrentBackend()

    async def scenario() -> None:
        replica = loaded_replica(backend)
        first, second = await asyncio.gather(
            invoke(replica, request_id="request-1"),
            invoke(replica, request_id="request-2"),
        )

        assert first.status == second.status == "success"
        assert backend.peak == 2
        assert {first.replica_inflight_at_start, second.replica_inflight_at_start} == {
            1,
            2,
        }
        assert replica.get_stats().peak_active_requests == 2

    asyncio.run(scenario())


def test_shutdown_aborts_active_requests_and_is_idempotent() -> None:
    backend = ConcurrentBackend()

    async def scenario() -> None:
        replica = loaded_replica(backend)
        task = asyncio.create_task(invoke(replica))
        while not backend.requests:
            await asyncio.sleep(0)
        await replica.shutdown()
        backend.release.set()
        await task
        await replica.shutdown()

        assert backend.aborted == ["request-1"]
        assert backend.stopped is True
        assert replica.get_stats().stopping is True

    asyncio.run(scenario())


def test_prompt_tokenizer_returns_exact_ids_and_decodes_output() -> None:
    tokenizer = FakeTokenizer()
    execution = ExecutionConfig(
        model_path="/models/test-model",
        use_chat_template=True,
        enable_thinking=True,
        truncation_side="left",
        serving=serving(),
    )
    prompt_tokenizer = PromptTokenizer(
        execution,
        tokenizer_factory=lambda _: tokenizer,
    )

    prompt = prompt_tokenizer.build_prompt("question", system_prompt="rules")
    output = prompt_tokenizer.decode((20, 21))

    assert prompt == PromptEncoding(
        text="<system>rules<user>question<assistant>",
        token_ids=(0,),
    )
    assert tokenizer.truncation_side == "left"
    assert tokenizer.encode_calls[0][1]["truncation"] is False
    assert tokenizer.decode_calls == [
        ([20, 21], {"skip_special_tokens": True})
    ]
    assert output == "decoded answer"


def test_prompt_tokenizer_joins_system_without_chat_template() -> None:
    tokenizer = FakeTokenizer()
    execution = ExecutionConfig(
        model_path="/models/test-model",
        use_chat_template=False,
        serving=serving(),
    )
    prompt_tokenizer = PromptTokenizer(
        execution,
        tokenizer_factory=lambda _: tokenizer,
    )

    prompt = prompt_tokenizer.build_prompt(
        "question with detail",
        system_prompt="rules",
    )

    assert prompt.text == "rules\n\nquestion with detail"
    assert prompt.input_tokens == 4
    assert tokenizer.chat_calls == []
