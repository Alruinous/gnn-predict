from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
import ray

from workflow.actor_support import await_value, dispatch, invoke, resolve


class PlainTarget:
    def greet(self, name: str) -> str:
        return f"hello {name}"


class _RemoteProxy:
    """Mimics a Ray ActorMethod: accessed as an attribute, called via .remote()."""

    def __init__(self, fn: Callable[..., object]) -> None:
        self._fn = fn

    def remote(self, *args: object, **kwargs: object) -> object:
        return self._fn(*args, **kwargs)


class ActorLikeTarget:
    greet = _RemoteProxy(lambda name: f"remote-hello-{name}")


class NoCallableAttribute:
    greet = "not a method"


def test_dispatch_calls_plain_method_directly() -> None:
    assert dispatch(PlainTarget(), "greet", "a") == "hello a"


def test_dispatch_prefers_remote_when_available() -> None:
    assert dispatch(ActorLikeTarget(), "greet", "a") == "remote-hello-a"


def test_dispatch_rejects_non_callable_attribute() -> None:
    with pytest.raises(TypeError, match="not callable"):
        dispatch(NoCallableAttribute(), "greet")


def test_resolve_passes_plain_values_through() -> None:
    assert resolve(42) == 42


def test_resolve_gets_an_object_ref(ray_session: None) -> None:
    ref = ray.put(7)
    assert resolve(ref) == 7


def test_await_value_awaits_a_coroutine() -> None:
    async def coro() -> int:
        return 3

    assert asyncio.run(await_value(coro())) == 3


def test_await_value_awaits_an_object_ref(ray_session: None) -> None:
    ref = ray.put(9)
    assert asyncio.run(await_value(ref)) == 9


def test_await_value_passes_plain_values_through() -> None:
    assert asyncio.run(await_value(5)) == 5


def test_invoke_combines_dispatch_and_await_for_a_plain_method() -> None:
    assert asyncio.run(invoke(PlainTarget(), "greet", "b")) == "hello b"


def test_invoke_combines_dispatch_and_await_for_a_remote_method(
    ray_session: None,
) -> None:
    class RemoteRefTarget:
        greet = _RemoteProxy(lambda: ray.put("remote-and-awaited"))

    assert asyncio.run(invoke(RemoteRefTarget(), "greet")) == "remote-and-awaited"
