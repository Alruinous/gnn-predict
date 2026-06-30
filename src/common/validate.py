from __future__ import annotations

from collections.abc import Hashable
from typing import Annotated, TypeVar

from pydantic import Field, StringConstraints

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]

T = TypeVar("T", bound=Hashable)


def ensure_unique[T: Hashable](values: list[T]) -> list[T]:
    if len(values) != len(set(values)):
        raise ValueError("values must be unique")
    return values


UniqueList = Annotated[
    list[T],
    ensure_unique,
]
