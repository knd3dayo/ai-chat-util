from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Mapping, Optional


@dataclass(frozen=True)
class RequestHeaders:
    """Inbound request headers container.

    This is used to:
    - capture inbound headers (e.g., Authorization, trace id)
    - safely pass required values to downstream agent runners via per-task env

    Notes:
    - `raw` keys are normalized to lowercase.
    - Avoid persisting secrets; prefer passing via process/container env.
    """

    authorization: Optional[str] = None
    trace_id: Optional[str] = None
    raw: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # dataclass(frozen=True) requires object.__setattr__
        raw = self.raw
        if raw is None:
            raw = {}
        object.__setattr__(self, "raw", raw)

    @classmethod
    def from_mapping(cls, headers: Mapping[str, str]) -> "RequestHeaders":
        normalized = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        authorization = normalized.get("authorization")
        trace_id = (
            normalized.get("x-trace-id")
            or normalized.get("trace-id")
            or normalized.get("trace_id")
        )
        return cls(authorization=authorization, trace_id=trace_id, raw=normalized)

    @classmethod
    def from_values(
        cls,
        *,
        authorization: Optional[str] = None,
        trace_id: Optional[str] = None,
        raw: Optional[Mapping[str, str]] = None,
    ) -> "RequestHeaders":
        normalized = {str(k).lower(): str(v) for k, v in (raw or {}).items()}
        return cls(
            authorization=authorization or normalized.get("authorization"),
            trace_id=trace_id
            or normalized.get("x-trace-id")
            or normalized.get("trace-id")
            or normalized.get("trace_id"),
            raw=normalized,
        )

    def to_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.authorization:
            # Keep both a namespaced key and a generic one for downstream tools.
            env.setdefault("AI_PLATFORM_AUTHORIZATION", self.authorization)
            env.setdefault("AUTHORIZATION", self.authorization)
        if self.trace_id:
            env.setdefault("AI_PLATFORM_TRACE_ID", self.trace_id)
            env.setdefault("TRACE_ID", self.trace_id)
        return env


_current_request_headers: ContextVar[Optional[RequestHeaders]] = ContextVar(
    "ai_platform_current_request_headers", default=None
)


def set_current_request_headers(headers: Optional[RequestHeaders]) -> None:
    _current_request_headers.set(headers)


@contextmanager
def bind_current_request_headers(headers: Optional[RequestHeaders]) -> Iterator[None]:
    """Temporarily bind current request headers for the lifetime of a scope.

    This prevents header values from leaking across unrelated requests/tool calls.
    """

    token = _current_request_headers.set(headers)
    try:
        yield
    finally:
        _current_request_headers.reset(token)


def get_current_request_headers() -> Optional[RequestHeaders]:
    return _current_request_headers.get()
