from __future__ import annotations

import asyncio
from typing import Any

from ai_chat_util.base.llm.llm_mcp_client_util import MCPClientUtil


class _FakeTool:
    def __init__(
        self,
        name: str,
        *,
        response_format: str | None = None,
        coroutine: Any | None = None,
        func: Any | None = None,
    ) -> None:
        self.name = name
        self.response_format = response_format
        if coroutine is not None:
            self.coroutine = coroutine
        if func is not None:
            self.func = func


def test_sync_tool_budget_exceeded_returns_guard_output() -> None:
    state: dict[str, int] = {"used": 0}

    def _func(x: int) -> str:
        return f"ok:{x}"

    tool = _FakeTool("sync_budget", response_format="content", func=_func)

    MCPClientUtil._apply_tool_execution_guards(
        [tool],
        tool_call_state=state,
        tool_call_limit_int=1,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    assert tool.func(1) == "ok:1"
    out2 = tool.func(2)
    assert isinstance(out2, str)
    assert "tool call budget exceeded" in out2


def test_sync_tool_exception_is_converted_to_normal_output() -> None:
    state: dict[str, int] = {"used": 0}

    def _boom() -> str:
        raise ValueError("boom")

    tool = _FakeTool("sync_error", response_format="content_and_artifact", func=_boom)

    MCPClientUtil._apply_tool_execution_guards(
        [tool],
        tool_call_state=state,
        tool_call_limit_int=1,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    out = tool.func()
    assert isinstance(out, tuple)
    assert len(out) == 2
    text, artifact = out
    assert isinstance(text, str)
    assert "ERROR: tool=sync_error failed" in text
    assert isinstance(artifact, dict)
    assert artifact.get("error") == "tool_invocation_failed"
    assert artifact.get("tool") == "sync_error"
    assert state["used"] == 1


def test_async_tool_timeout_retries_then_succeeds() -> None:
    async def _run() -> None:
        state: dict[str, int] = {"used": 0}
        counter: dict[str, int] = {"n": 0}

        async def _coro() -> str:
            counter["n"] += 1
            if counter["n"] == 1:
                # First attempt times out
                await asyncio.sleep(0.05)
                return "late"
            return "ok"

        tool = _FakeTool("async_timeout", response_format="content", coroutine=_coro)

        MCPClientUtil._apply_tool_execution_guards(
            [tool],
            tool_call_state=state,
            tool_call_limit_int=0,
            tool_timeout_seconds_f=0.01,
            tool_timeout_retries_int=1,
        )

        out = await tool.coroutine()
        assert out == "ok"
        # One timeout + one retry => two counted attempts
        assert state["used"] == 2

    asyncio.run(_run())
