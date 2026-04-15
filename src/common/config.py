from __future__ import annotations

import polars as pl
from pydantic import BaseModel


class MetaItem(BaseModel):
    id_fields: list[str]
    target_fields: list[str]


class DataItem(BaseModel):
    varient_name: str
    varient_path: str

    @classmethod
    def from_record(cls, record: pl.DataFrame) -> DataItem:
        assert record.height == 1, "record must have exactly one row"
        varient_name = str(record["varient_name"].item())
        varient_path = str(record["varient_path"].item())
        return cls(
            varient_name=varient_name,
            varient_path=varient_path,
        )
        