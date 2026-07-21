from __future__ import annotations

import inspect

import ray


def dispatch(
    target: object, method_name: str, *args: object, **kwargs: object
) -> object:
    method = getattr(target, method_name)
    remote = getattr(method, "remote", None)
    if callable(remote):
        return remote(*args, **kwargs)
    if callable(method):
        return method(*args, **kwargs)
    raise TypeError(f"method is not callable: {method_name}")


def resolve(value: object, *, timeout: float | None = None) -> object:
    if isinstance(value, ray.ObjectRef):
        return ray.get(value, timeout=timeout)
    return value


async def await_value(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


async def invoke(
    target: object, method_name: str, *args: object, **kwargs: object
) -> object:
    return await await_value(dispatch(target, method_name, *args, **kwargs))
