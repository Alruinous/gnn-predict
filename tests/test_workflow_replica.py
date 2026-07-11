from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from pydantic import ValidationError
from transformers.generation.utils import GenerateDecoderOnlyOutput

from workflow.replica import (
    BackendGeneration,
    GenerationRequest,
    ModelDeploymentConfig,
    ModelReplica,
    ModelReplicaActor,
    PromptTokenizer,
    TransformersBackend,
)
from workflow.schema import AgentNodeConfig, ExecutionConfig


def agent_node(
    *,
    model_name: str = "test-model",
    model_path: str = "/models/test-model",
    dtype: str = "float16",
    do_sample: bool = False,
    temperature: float | None = None,
    enable_thinking: bool = False,
) -> AgentNodeConfig:
    return AgentNodeConfig.model_validate(
        {
            "name": "agent-a",
            "type": "agent",
            "model": {"name": model_name},
            "execution": {
                "model_path": model_path,
                "dtype": dtype,
                "do_sample": do_sample,
                "temperature": temperature,
                "enable_thinking": enable_thinking,
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


DEPLOYMENT = ModelDeploymentConfig(
    model_name="test-model",
    model_path="/models/test-model",
    dtype="float16",
)
GENERATION = ExecutionConfig(
    model_path=DEPLOYMENT.model_path,
    dtype=DEPLOYMENT.dtype,
    do_sample=True,
    temperature=0.7,
)


class FakeBackend:
    def __init__(
        self,
        generation: BackendGeneration | None = None,
        error: Exception | None = None,
    ) -> None:
        self.generation = generation or BackendGeneration(
            output_text="answer",
            output_tokens=4,
        )
        self.error = error
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> BackendGeneration:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.generation


class FakeTokenizer:
    def __init__(self) -> None:
        self.truncation_side = "right"
        self.chat_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
        self.encode_calls: list[tuple[str, dict[str, object]]] = []

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


class FakeEncodedInputs(dict[str, torch.Tensor]):
    def __init__(self) -> None:
        super().__init__(input_ids=torch.tensor([[10, 11]]))
        self.device: str | None = None

    def to(self, device: str) -> FakeEncodedInputs:
        self.device = device
        return self


class FakeInferenceTokenizer:
    def __init__(self) -> None:
        self.encoded = FakeEncodedInputs()
        self.decoded_ids: list[int] | None = None

    def __call__(self, prompt: str, **kwargs: object) -> FakeEncodedInputs:
        return self.encoded

    def decode(self, token_ids: object, **kwargs: object) -> str:
        assert isinstance(token_ids, torch.Tensor)
        self.decoded_ids = token_ids.tolist()
        return "decoded answer"


class FakeTransformersModel:
    def __init__(self) -> None:
        self.generation_kwargs: dict[str, object] = {}

    def generate(
        self,
        **kwargs: object,
    ) -> torch.Tensor | GenerateDecoderOnlyOutput:
        self.generation_kwargs = kwargs
        sequences = torch.tensor([[10, 11, 20, 21, 22]])
        if kwargs.get("return_dict_in_generate") is False:
            return sequences
        return GenerateDecoderOnlyOutput(sequences=cast(torch.LongTensor, sequences))


def loaded_replica(backend: FakeBackend) -> ModelReplica:
    replica = ModelReplica(
        DEPLOYMENT,
        backend_factory=lambda _: backend,
        gpu_id_provider=lambda: [3],
    )
    replica.load()
    return replica


def test_deployment_config_uses_only_weight_compatibility_for_sharing() -> None:
    default = ModelDeploymentConfig.from_node(agent_node())
    request_variant = ModelDeploymentConfig.from_node(
        agent_node(do_sample=True, temperature=0.9, enable_thinking=True)
    )

    assert default == DEPLOYMENT
    assert default.model_key == request_variant.model_key
    assert default.model_key == ModelDeploymentConfig.from_node(agent_node()).model_key


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "other-model"),
        ("model_path", "/models/other-model"),
        ("dtype", "bfloat16"),
    ],
)
def test_each_deployment_field_changes_model_key(field: str, value: str) -> None:
    payload = DEPLOYMENT.model_dump()
    payload[field] = value

    assert ModelDeploymentConfig(**payload).model_key != DEPLOYMENT.model_key


def test_deployment_config_is_frozen_strict_and_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ModelDeploymentConfig.model_validate(
            {
                "model_name": 1,
                "model_path": "/models/test-model",
                "dtype": "float16",
            }
        )
    with pytest.raises(ValidationError):
        ModelDeploymentConfig.model_validate(
            {
                "model_name": "test-model",
                "model_path": "/models/test-model",
                "dtype": "float16",
                "device": "cuda:7",
            }
        )
    with pytest.raises(ValidationError):
        DEPLOYMENT.dtype = "bfloat16"


def test_load_constructs_backend_and_reports_ray_physical_gpu_id() -> None:
    backend = FakeBackend()
    factory_deployments: list[ModelDeploymentConfig] = []

    def backend_factory(deployment: ModelDeploymentConfig) -> FakeBackend:
        factory_deployments.append(deployment)
        return backend

    replica = ModelReplica(
        DEPLOYMENT,
        backend_factory=backend_factory,
        gpu_id_provider=lambda: ["7"],
    )

    result = replica.load()

    assert factory_deployments == [DEPLOYMENT]
    assert result.physical_gpu_id == "7"
    assert result.duration_sec >= 0
    assert result.idle_vram_mb == 0


@pytest.mark.parametrize("gpu_ids", [[], [0, 1]])
def test_load_requires_exactly_one_ray_physical_gpu_id(
    gpu_ids: list[int],
) -> None:
    factory_called = False

    def backend_factory(_: ModelDeploymentConfig) -> FakeBackend:
        nonlocal factory_called
        factory_called = True
        return FakeBackend()

    replica = ModelReplica(
        DEPLOYMENT,
        backend_factory=backend_factory,
        gpu_id_provider=lambda: gpu_ids,
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        replica.load()

    assert factory_called is False


def test_invoke_fails_fast_before_load() -> None:
    replica = ModelReplica(
        DEPLOYMENT,
        backend_factory=lambda _: FakeBackend(),
        gpu_id_provider=lambda: [0],
    )

    with pytest.raises(RuntimeError, match="load"):
        replica.invoke(
            "prompt",
            input_tokens=3,
            max_new_tokens=96,
            generation=GENERATION,
        )


def test_replica_passes_granted_budget_and_all_generation_parameters() -> None:
    backend = FakeBackend()
    result = loaded_replica(backend).invoke(
        "prompt",
        input_tokens=3,
        max_new_tokens=96,
        generation=GENERATION,
    )

    assert backend.requests == [
        GenerationRequest(
            prompt="prompt",
            input_tokens=3,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.7,
        )
    ]
    assert result.output_tokens == 4


@pytest.mark.parametrize(
    ("output_tokens", "max_new_tokens", "hit_token_limit"),
    [(4, 5, False), (4, 4, True), (5, 4, True)],
)
def test_success_report_computes_token_limit_status(
    output_tokens: int,
    max_new_tokens: int,
    hit_token_limit: bool,
) -> None:
    backend = FakeBackend(
        generation=BackendGeneration(
            output_text="answer",
            output_tokens=output_tokens,
        )
    )

    result = loaded_replica(backend).invoke(
        "prompt",
        input_tokens=3,
        max_new_tokens=max_new_tokens,
        generation=GENERATION,
    )

    assert result.status == "success"
    assert result.output_text == "answer"
    assert result.input_tokens == 3
    assert result.output_tokens == output_tokens
    assert result.hit_token_limit is hit_token_limit
    assert result.started_at <= result.finished_at
    assert result.duration_sec >= 0
    assert result.error_type is None
    assert result.error_message is None


def test_cuda_oom_is_normalized_to_oom_report() -> None:
    backend = FakeBackend(error=torch.cuda.OutOfMemoryError("CUDA exhausted"))

    result = loaded_replica(backend).invoke(
        "prompt",
        input_tokens=3,
        max_new_tokens=96,
        generation=GENERATION,
    )

    assert result.status == "oom"
    assert result.output_text is None
    assert result.input_tokens == 3
    assert result.output_tokens == 0
    assert result.hit_token_limit is False
    assert result.error_type == "OutOfMemoryError"
    assert result.error_message == "CUDA exhausted"


def test_other_backend_exception_is_normalized_to_failed_report() -> None:
    backend = FakeBackend(error=ValueError("invalid generation"))

    result = loaded_replica(backend).invoke(
        "prompt",
        input_tokens=3,
        max_new_tokens=96,
        generation=GENERATION,
    )

    assert result.status == "failed"
    assert result.output_text is None
    assert result.input_tokens == 3
    assert result.output_tokens == 0
    assert result.hit_token_limit is False
    assert result.error_type == "ValueError"
    assert result.error_message == "invalid generation"


def test_transformers_backend_forces_tensor_output_and_decodes_new_tokens() -> None:
    tokenizer = FakeInferenceTokenizer()
    model = FakeTransformersModel()
    backend = object.__new__(TransformersBackend)
    setattr(backend, "_tokenizer", tokenizer)
    setattr(backend, "_model", model)

    result = backend.generate(
        GenerationRequest(
            prompt="rendered prompt",
            input_tokens=2,
            max_new_tokens=3,
            do_sample=True,
            temperature=0.7,
        )
    )

    assert model.generation_kwargs["return_dict_in_generate"] is False
    assert tokenizer.encoded.device == "cuda:0"
    assert tokenizer.decoded_ids == [20, 21, 22]
    assert result == BackendGeneration(
        output_text="decoded answer",
        output_tokens=3,
    )


def test_prompt_tokenizer_applies_chat_template_and_counts_final_prompt() -> None:
    tokenizer = FakeTokenizer()
    execution = ExecutionConfig(
        model_path="/models/test-model",
        use_chat_template=True,
        enable_thinking=True,
        truncation_side="left",
    )
    prompt_tokenizer = PromptTokenizer(
        execution,
        tokenizer_factory=lambda _: tokenizer,
    )

    prompt, input_tokens = prompt_tokenizer.build_prompt(
        "question",
        system_prompt="rules",
    )

    assert prompt == "<system>rules<user>question<assistant>"
    assert input_tokens == 1
    assert tokenizer.truncation_side == "left"
    assert tokenizer.chat_calls == [
        (
            [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "question"},
            ],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            },
        )
    ]
    assert tokenizer.encode_calls == [
        (
            prompt,
            {"add_special_tokens": False, "truncation": False},
        )
    ]


def test_prompt_tokenizer_explicitly_joins_system_and_user_without_template() -> None:
    tokenizer = FakeTokenizer()
    execution = ExecutionConfig(
        model_path="/models/test-model",
        use_chat_template=False,
    )
    prompt_tokenizer = PromptTokenizer(
        execution,
        tokenizer_factory=lambda _: tokenizer,
    )

    prompt, input_tokens = prompt_tokenizer.build_prompt(
        "question with detail",
        system_prompt="rules",
    )

    assert prompt == "rules\n\nquestion with detail"
    assert input_tokens == 4
    assert tokenizer.chat_calls == []
    assert tokenizer.encode_calls == [
        (
            prompt,
            {"add_special_tokens": False, "truncation": False},
        )
    ]


def test_prompt_tokenizer_does_not_silently_truncate() -> None:
    tokenizer = FakeTokenizer()
    execution = ExecutionConfig(
        model_path="/models/test-model",
        use_chat_template=False,
        truncation_side="left",
    )
    prompt_tokenizer = PromptTokenizer(
        execution,
        tokenizer_factory=lambda _: tokenizer,
    )

    _, input_tokens = prompt_tokenizer.build_prompt("one two three four")

    assert input_tokens == 4
    assert tokenizer.encode_calls[0][1]["truncation"] is False


def test_replica_actor_declares_one_gpu_without_starting_ray() -> None:
    actor_options = getattr(ModelReplicaActor, "_default_options")

    assert actor_options["num_gpus"] == 1


def test_default_factories_are_deferred_until_objects_are_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def fail(*args: Any, **kwargs: Any) -> Callable[..., object]:
        calls.append(("unexpected", args, kwargs))
        raise AssertionError("default loader was called")

    monkeypatch.setattr("workflow.replica.TransformersBackend", fail)

    ModelReplica(
        DEPLOYMENT,
        backend_factory=lambda _: FakeBackend(),
        gpu_id_provider=lambda: [0],
    )

    assert calls == []
