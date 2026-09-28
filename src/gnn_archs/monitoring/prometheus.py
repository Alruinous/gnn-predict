from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from prometheus_api_client import PrometheusConnect


class PrometheusQueryAPI(Protocol):
    def instant_query(
        self, query: str, timestamp: float | None = None
    ) -> list[dict[str, Any]]: ...

    def range_query(
        self,
        query: str,
        start_ts: float,
        end_ts: float,
        step_seconds: int,
    ) -> list[dict[str, Any]]: ...


class PrometheusClient:
    def __init__(
        self,
        prometheus_url: str,
        *,
        disable_ssl: bool = True,
        timeout_seconds: int = 30,
    ) -> None:
        self._client = PrometheusConnect(url=prometheus_url, disable_ssl=disable_ssl)
        self._timeout_seconds = timeout_seconds

    def instant_query(
        self, query: str, timestamp: float | None = None
    ) -> list[dict[str, Any]]:
        params = None
        if timestamp is not None:
            params = {"time": f"{timestamp:.6f}"}
        try:
            result = self._client.custom_query(
                query,
                params=params,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(f"Prometheus instant query failed: {query}") from exc
        if not isinstance(result, list):
            raise TypeError(
                "Prometheus instant query returned an unexpected payload type: "
                f"{type(result)!r}"
            )
        return result

    def range_query(
        self,
        query: str,
        start_ts: float,
        end_ts: float,
        step_seconds: int,
    ) -> list[dict[str, Any]]:
        if end_ts < start_ts:
            raise ValueError(f"range query end_ts must be >= start_ts: {query}")
        try:
            result = self._client.custom_query_range(
                query=query,
                start_time=datetime.fromtimestamp(start_ts, tz=timezone.utc),
                end_time=datetime.fromtimestamp(end_ts, tz=timezone.utc),
                step=f"{step_seconds}s",
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(f"Prometheus range query failed: {query}") from exc
        if not isinstance(result, list):
            raise TypeError(
                "Prometheus range query returned an unexpected payload type: "
                f"{type(result)!r}"
            )
        return result
