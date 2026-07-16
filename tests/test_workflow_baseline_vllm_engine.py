from __future__ import annotations

import asyncio
from typing import Any, Literal

import pytest
import ray

from workflow.baseline.vllm_engine import StaticVLLMEngine, StaticVLLMEngineActor
from workflow.replica import (
    BackendGeneration,
    GenerationEngineFailedError,
    GenerationRequest,
    ModelDeploymentConfig,
    ReplicaInferenceResult,
)
from workflow.schema import ExecutionConfig, ServingConfig

SERVING = ServingConfig(
    max_model_len=1024,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
)
DEPLOYMENT = ModelDeploymentConfig(
    model_name="test-model",
    model_path="/models/test-model",
    dtype="float16",
    serving=SERVING,
)
GENERATION = ExecutionConfig(
    model_path=DEPLOYMENT.model_path,
    max_new_tokens=8,
    dtype=DEPLOYMENT.dtype,
    do_sample=False,
    serving=SERVING,
)


class FakeBackend:
    idle_vram_mb = 123.0
    vllm_version = "0.10.2"
    engine_mode: Literal["V0"] = "V0"
    attention_backend: Literal["XFORMERS"] = "XFORMERS"
    block_size = 16
    num_gpu_blocks = 100
    gpu_kv_tokens = 1_600

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.aborted: list[str] = []
        self.loaded = False
        self.stopped = False

    def load(self) -> None:
        self.loaded = True

    async def generate(self, request: GenerationRequest) -> BackendGeneration:
        self.requests.append(request)
        return BackendGeneration(output_token_ids=(10, 11), finish_reason="stop")

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    def shutdown(self) -> None:
        self.stopped = True


class ConcurrentBackend(FakeBackend):
    def __init__(self, expected_requests: int) -> None:
        super().__init__()
        self.expected_requests = expected_requests
        self.active_requests = 0
        self.peak_active_requests = 0
        self.release = asyncio.Event()

    async def generate(self, request: GenerationRequest) -> BackendGeneration:
        self.requests.append(request)
        self.active_requests += 1
        self.peak_active_requests = max(
            self.peak_active_requests,
            self.active_requests,
        )
        if self.active_requests == self.expected_requests:
            self.release.set()
        try:
            await self.release.wait()
            return BackendGeneration(
                output_token_ids=(10, 11),
                finish_reason="stop",
            )
        finally:
            self.active_requests -= 1


class FailingBackend(FakeBackend):
    async def generate(self, request: GenerationRequest) -> BackendGeneration:
        self.requests.append(request)
        raise GenerationEngineFailedError("engine failed")


class BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: GenerationRequest) -> BackendGeneration:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        return BackendGeneration(output_token_ids=(10,), finish_reason="abort")


def loaded_engine(backend: FakeBackend) -> StaticVLLMEngine:
    engine = StaticVLLMEngine(
        DEPLOYMENT,
        backend_factory=lambda _: backend,
        gpu_id_provider=lambda: [2],
    )
    engine.load()
    return engine


async def invoke(
    engine: StaticVLLMEngine,
    request_id: str,
) -> ReplicaInferenceResult:
    return await engine.invoke(
        request_id,
        (1, 2, 3),
        max_new_tokens=8,
        generation=GENERATION,
    )


def concurrent_backend_factory(
    deployment: ModelDeploymentConfig,
) -> ConcurrentBackend:
    assert deployment == DEPLOYMENT
    return ConcurrentBackend(expected_requests=3)


def fake_gpu_ids() -> list[int]:
    return [2]


def test_engine_allows_more_requests_than_max_num_seqs() -> None:
    backend = ConcurrentBackend(expected_requests=3)

    async def scenario() -> None:
        engine = loaded_engine(backend)
        results = await asyncio.gather(
            invoke(engine, "request-1"),
            invoke(engine, "request-2"),
            invoke(engine, "request-3"),
        )

        assert {request.request_id for request in backend.requests} == {
            "request-1",
            "request-2",
            "request-3",
        }
        assert backend.peak_active_requests == 3
        assert sorted(result.replica_inflight_at_start for result in results) == [
            1,
            2,
            3,
        ]
        assert all(result.status == "success" for result in results)
        assert engine.get_stats().active_requests == 0
        assert engine.get_stats().peak_active_requests == 3

    asyncio.run(scenario())


def test_engine_load_reports_kv_capacity() -> None:
    backend = FakeBackend()
    engine = StaticVLLMEngine(
        DEPLOYMENT,
        backend_factory=lambda _: backend,
        gpu_id_provider=lambda: [2],
    )

    result = engine.load()

    assert result.block_size == 16
    assert result.num_gpu_blocks == 100
    assert result.gpu_kv_tokens == 1_600


def test_engine_rejects_output_limit_different_from_workflow() -> None:
    engine = loaded_engine(FakeBackend())

    async def scenario() -> None:
        with pytest.raises(ValueError, match="output limit"):
            await engine.invoke(
                "request-1",
                (1, 2, 3),
                max_new_tokens=4,
                generation=GENERATION,
            )

    asyncio.run(scenario())


def test_actor_forwards_concurrent_requests_without_capacity_rejection(
    ray_session: None,
) -> None:
    actor: Any = StaticVLLMEngineActor.options(num_gpus=0).remote(
        DEPLOYMENT,
        backend_factory=concurrent_backend_factory,
        gpu_id_provider=fake_gpu_ids,
    )
    try:
        ray.get(actor.load.remote())
        results = ray.get(
            [
                actor.invoke.remote(
                    f"request-{index}",
                    (1, 2, 3),
                    max_new_tokens=8,
                    generation=GENERATION,
                )
                for index in range(3)
            ],
            timeout=10,
        )
        stats = ray.get(actor.get_stats.remote())

        assert all(result.status == "success" for result in results)
        assert sorted(result.replica_inflight_at_start for result in results) == [
            1,
            2,
            3,
        ]
        assert stats.active_requests == 0
        assert stats.peak_active_requests == 3
    finally:
        ray.get(actor.shutdown.remote())
        ray.kill(actor)


def test_engine_normalizes_backend_failure_and_clears_active_request() -> None:
    async def scenario() -> None:
        engine = loaded_engine(FailingBackend())
        result = await invoke(engine, "request-failed")

        assert result.status == "failed"
        assert result.engine_failed is True
        assert result.error_type == "GenerationEngineFailedError"
        assert result.error_message == "engine failed"
        assert engine.get_stats().active_requests == 0

    asyncio.run(scenario())


def test_engine_shutdown_aborts_active_requests_and_releases_backend() -> None:
    backend = BlockingBackend()

    async def scenario() -> None:
        engine = loaded_engine(backend)
        task = asyncio.create_task(invoke(engine, "request-active"))
        await backend.started.wait()

        await engine.shutdown()
        backend.release.set()
        result = await task
        await engine.shutdown()

        assert backend.aborted == ["request-active"]
        assert backend.stopped is True
        assert result.status == "cancelled"
        assert engine.get_stats().stopping is True
        assert engine.get_stats().active_requests == 0

    asyncio.run(scenario())
