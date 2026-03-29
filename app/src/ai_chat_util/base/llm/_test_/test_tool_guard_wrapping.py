from __future__ import annotations

import asyncio
import sqlite3
import sys
import types
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError

fake_client_module = types.ModuleType("langchain_mcp_adapters.client")
fake_client_module.MultiServerMCPClient = object # type: ignore
sys.modules.setdefault("langchain_mcp_adapters.client", fake_client_module)

fake_sessions_module = types.ModuleType("langchain_mcp_adapters.sessions")
fake_sessions_module.Connection = dict[str, Any] # type: ignore
sys.modules.setdefault("langchain_mcp_adapters.sessions", fake_sessions_module)

import ai_chat_util.base.llm.agent as agent_mod
import ai_chat_util.base.llm.llm_mcp_client_util as llm_mcp_client_util_mod
from ai_chat_util.base.llm.agent import AgentBuilder, ToolLimits
from ai_chat_util.base.llm.prompts import CodingAgentPrompts
from ai_chat_util.base.llm.llm_mcp_client_util import MCPClientUtil
from ai_chat_util_base.config.ai_chat_util_mcp_config import MCPServerConfig, MCPServerConfigEntry
from ai_chat_util_base.config.runtime import (
    AiChatUtilConfig,
    FeaturesSection,
    LLMSection,
    LoggingSection,
    MCPSection,
    NetworkSection,
    Office2PDFSection,
)


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

    async def ainvoke(self, payload: Any) -> Any:
        if hasattr(self, "coroutine"):
            return await self.coroutine(**payload)
        if hasattr(self, "func"):
            return self.func(**payload)
        raise AssertionError("No callable configured")

    def invoke(self, payload: Any) -> Any:
        if hasattr(self, "func"):
            return self.func(**payload)
        raise AssertionError("No callable configured")


def test_sync_tool_budget_exceeded_returns_guard_output() -> None:
    state: dict[str, int] = {"used": 0, "general_used": 0, "followup_used": 0, "followup_limit": 0}

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
    assert "tool_call_budget_exceeded" in out2


def test_sync_tool_exception_is_converted_to_normal_output() -> None:
    state: dict[str, int] = {"used": 0, "general_used": 0, "followup_used": 0, "followup_limit": 0}

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
    assert "error=tool_request_invalid" in text
    assert isinstance(artifact, dict)
    assert artifact.get("error") == "tool_request_invalid"
    assert artifact.get("tool") == "sync_error"
    assert state["used"] == 1


def test_classify_tool_error_for_execute_timeout_and_backend_cases() -> None:
    assert ToolLimits.classify_tool_error("execute", asyncio.TimeoutError()) == "tool_timeout"
    assert ToolLimits.classify_tool_error("execute", HTTPException(status_code=400, detail="bad request")) == "execute_request_invalid"
    assert ToolLimits.classify_tool_error("execute", HTTPException(status_code=502, detail="gateway")) == "execute_backend_error"
    assert ToolLimits.classify_tool_error("execute", RuntimeError("boom")) == "execute_invocation_failed"


def test_async_tool_timeout_retries_then_succeeds() -> None:
    async def _run() -> None:
        state: dict[str, int] = {"used": 0, "general_used": 0, "followup_used": 0, "followup_limit": 0}
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


def test_tool_agent_prompt_requires_directory_workspace_and_task_id_before_followups() -> None:
    prompt = CodingAgentPrompts().tool_agent_system_prompt(
        hitl_policy_text="dummy policy",
        agent_name="tool_agent_coding",
        followup_poll_interval_seconds=10.0,
        status_tail_lines=20,
        result_tail_lines=80,
    )

    assert "workspace_path` には、必ず「作業用ディレクトリ」の絶対パス" in prompt
    assert "対象が特定ファイルの場合は、そのファイルパスは prompt 側に含め" in prompt
    assert "task_id を取得できなかった場合は、status/get_result/workspace_path を呼ばず" in prompt


def test_followup_tools_use_separate_budget_from_general_tools() -> None:
    state: dict[str, int] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 2,
    }

    execute_tool = _FakeTool("execute", response_format="content", func=lambda task_id: f"executed:{task_id}")
    status_tool = _FakeTool("status", response_format="content", func=lambda task_id: f"running:{task_id}")

    MCPClientUtil._apply_tool_execution_guards(
        [execute_tool, status_tool],
        tool_call_state=state,
        tool_call_limit_int=1,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    assert execute_tool.func("task-1") == "executed:task-1"
    assert status_tool.func("task-1") == "running:task-1"
    assert status_tool.func("task-2") == "running:task-2"

    execute_out = execute_tool.func("task-2")
    assert isinstance(execute_out, str)
    assert "tool_call_budget_exceeded" in execute_out

    status_out = status_tool.func("task-3")
    assert isinstance(status_out, str)
    assert "tool_call_budget_exceeded" in status_out
    assert state["general_used"] == 1
    assert state["followup_used"] == 2
    assert state["used"] == 3


def test_effective_call_limits_keep_configured_limits_without_explicit_files() -> None:
    assert ToolLimits.effective_call_limits(4, 8, None) == (4, 8)
    assert ToolLimits.effective_call_limits(4, 8, []) == (4, 8)


def test_effective_call_limits_raise_floor_for_explicit_files() -> None:
    assert ToolLimits.effective_call_limits(4, 8, ["/tmp/work/doc.md"]) == (6, 12)


def test_effective_call_limits_preserve_unlimited_values_for_explicit_files() -> None:
    assert ToolLimits.effective_call_limits(0, 0, ["/tmp/work/doc.md"]) == (0, 0)


def test_successful_duplicate_general_tool_call_reuses_cached_result_without_spending_budget() -> None:
    state: dict[str, int] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 0,
    }
    calls = {"n": 0}

    def _func() -> dict[str, str]:
        calls["n"] += 1
        return {"path": "/tmp/ai-chat-util-config.yml"}

    tool = _FakeTool("get_loaded_config_info", response_format="content", func=_func)

    MCPClientUtil._apply_tool_execution_guards(
        [tool],
        tool_call_state=state,
        tool_call_limit_int=1,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    out1 = tool.func()
    out2 = tool.func()

    assert out1 == {"path": "/tmp/ai-chat-util-config.yml"}
    assert out2 == out1
    assert calls["n"] == 1
    assert state["general_used"] == 1
    assert state["used"] == 1


def test_followup_tool_does_not_reuse_cached_result() -> None:
    state: dict[str, int] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 8,
    }
    calls = {"n": 0}

    def _status(task_id: str) -> str:
        calls["n"] += 1
        return f"status:{task_id}:{calls['n']}"

    tool = _FakeTool("status", response_format="content", func=_status)

    MCPClientUtil._apply_tool_execution_guards(
        [tool],
        tool_call_state=state,
        tool_call_limit_int=0,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    out1 = tool.func("task-1")
    out2 = tool.func("task-1")

    assert out1 == "status:task-1:1"
    assert out2 == "status:task-1:2"
    assert calls["n"] == 2
    assert state["followup_used"] == 2
    assert state["used"] == 2


def test_get_result_reuses_cached_result_for_same_task() -> None:
    state: dict[str, int] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 8,
    }
    calls = {"n": 0}

    def _get_result(task_id: str) -> dict[str, str]:
        calls["n"] += 1
        return {"stdout": f"done:{task_id}:{calls['n']}", "stderr": ""}

    tool = _FakeTool("get_result", response_format="content", func=_get_result)

    MCPClientUtil._apply_tool_execution_guards(
        [tool],
        tool_call_state=state,
        tool_call_limit_int=0,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    out1 = tool.func("task-1")
    out2 = tool.func("task-1")

    assert out1 == {"stdout": "done:task-1:1", "stderr": ""}
    assert out2 == out1
    assert calls["n"] == 1
    assert state["followup_used"] == 1
    assert state["used"] == 1


def test_execute_prompt_is_augmented_with_single_explicit_user_file_path() -> None:
    state: dict[str, Any] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 8,
        "explicit_user_file_paths": ["/tmp/work/doc.md"],
    }
    captured: dict[str, Any] = {}

    def _execute(*, req: dict[str, Any]) -> dict[str, str]:
        captured["req"] = req
        return {"task_id": "task-1"}

    tool = _FakeTool("execute", response_format="content", func=_execute)

    MCPClientUtil._apply_tool_execution_guards(
        [tool],
        tool_call_state=state,
        tool_call_limit_int=4,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    out = tool.func(req={"prompt": "Extract all headings from the Markdown file as exact Markdown heading lines.", "workspace_path": "/tmp/work"})

    assert out == {"task_id": "task-1"}
    assert captured["req"]["workspace_path"] == "/tmp/work"
    assert "Target file path: /tmp/work/doc.md" in captured["req"]["prompt"]
    assert "Use only this file as the source of truth" in captured["req"]["prompt"]
    assert "HEADING_LINE_EXACT:" in captured["req"]["prompt"]


def test_execute_prompt_is_not_augmented_when_prompt_already_contains_path() -> None:
    state: dict[str, Any] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 8,
        "explicit_user_file_paths": ["/tmp/work/doc.md"],
    }
    captured: dict[str, Any] = {}

    def _execute(*, req: dict[str, Any]) -> dict[str, str]:
        captured["req"] = req
        return {"task_id": "task-1"}

    tool = _FakeTool("execute", response_format="content", func=_execute)

    MCPClientUtil._apply_tool_execution_guards(
        [tool],
        tool_call_state=state,
        tool_call_limit_int=4,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    original_prompt = "Read /tmp/work/doc.md and extract all headings as exact Markdown lines."
    tool.func(req={"prompt": original_prompt, "workspace_path": "/tmp/work"})

    assert captured["req"]["prompt"] == original_prompt

    def test_execute_top_level_fields_are_moved_into_req() -> None:
        state: dict[str, Any] = {
            "used": 0,
            "general_used": 0,
            "followup_used": 0,
            "followup_limit": 8,
            "explicit_user_file_paths": ["/tmp/work/doc.md"],
        }
        captured: dict[str, Any] = {}

        def _execute(**kwargs: Any) -> dict[str, str]:
            captured["kwargs"] = kwargs
            return {"task_id": "task-1"}

        tool = _FakeTool("execute", response_format="content", func=_execute)

        MCPClientUtil._apply_tool_execution_guards(
            [tool],
            tool_call_state=state,
            tool_call_limit_int=4,
            tool_timeout_seconds_f=0.0,
            tool_timeout_retries_int=0,
        )

        out = tool.func(
            prompt="Extract headings",
            workspace_path="/tmp/work",
            timeout=300,
        )

        assert out == {"task_id": "task-1"}
        assert "timeout" not in captured["kwargs"]
        assert captured["kwargs"]["req"]["timeout"] == 300
        assert captured["kwargs"]["req"]["workspace_path"] == "/tmp/work"
        assert captured["kwargs"]["req"]["prompt"].startswith("Extract headings")


    def test_execute_top_level_fields_are_removed_when_req_already_contains_same_keys() -> None:
        state: dict[str, Any] = {
            "used": 0,
            "general_used": 0,
            "followup_used": 0,
            "followup_limit": 8,
            "explicit_user_file_paths": ["/tmp/work/doc.md"],
        }
        captured: dict[str, Any] = {}

        def _execute(**kwargs: Any) -> dict[str, str]:
            captured["kwargs"] = kwargs
            return {"task_id": "task-1"}

        tool = _FakeTool("execute", response_format="content", func=_execute)

        MCPClientUtil._apply_tool_execution_guards(
            [tool],
            tool_call_state=state,
            tool_call_limit_int=4,
            tool_timeout_seconds_f=0.0,
            tool_timeout_retries_int=0,
        )

        out = tool.func(
            req={"prompt": "Extract headings", "workspace_path": "/tmp/work"},
            workspace_path="/tmp/work",
        )

        assert out == {"task_id": "task-1"}
        assert "workspace_path" not in captured["kwargs"]
        assert captured["kwargs"]["req"]["workspace_path"] == "/tmp/work"



def test_invalid_followup_task_id_is_blocked_after_first_404() -> None:
    state: dict[str, Any] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 8,
    }
    status_calls = {"n": 0}

    def _status(task_id: str) -> str:
        status_calls["n"] += 1
        raise HTTPException(status_code=404, detail="Task not found")

    status_tool = _FakeTool("status", response_format="content", func=_status)

    MCPClientUtil._apply_tool_execution_guards(
        [status_tool],
        tool_call_state=state,
        tool_call_limit_int=4,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    out1 = status_tool.func("task-missing")
    out2 = status_tool.func("task-missing")

    assert isinstance(out1, str)
    assert "invalid_followup_task_id" in out1
    assert "task-missing" in out1
    assert isinstance(out2, str)
    assert "invalid_followup_task_id" in out2
    assert status_calls["n"] == 1
    assert state["followup_used"] == 1


def test_invalid_followup_task_id_404_does_not_log_stack_trace_sync(caplog: pytest.LogCaptureFixture) -> None:
    state: dict[str, Any] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 8,
    }

    def _status(task_id: str) -> str:
        raise HTTPException(status_code=404, detail="Task not found")

    status_tool = _FakeTool("status", response_format="content", func=_status)

    MCPClientUtil._apply_tool_execution_guards(
        [status_tool],
        tool_call_state=state,
        tool_call_limit_int=4,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    with caplog.at_level("INFO"):
        out = status_tool.func("task-missing")

    assert isinstance(out, str)
    assert "invalid_followup_task_id" in out
    assert "Marking follow-up task_id invalid after task-not-found (sync)" in caplog.text
    assert "Tool invocation failed (sync)" not in caplog.text
    assert "Traceback" not in caplog.text


def test_invalid_followup_task_id_404_does_not_log_stack_trace_async(caplog: pytest.LogCaptureFixture) -> None:
    async def _run() -> None:
        state: dict[str, Any] = {
            "used": 0,
            "general_used": 0,
            "followup_used": 0,
            "followup_limit": 8,
        }

        async def _status(task_id: str) -> str:
            raise HTTPException(status_code=404, detail="Task not found")

        status_tool = _FakeTool("status", response_format="content", coroutine=_status)

        MCPClientUtil._apply_tool_execution_guards(
            [status_tool],
            tool_call_state=state,
            tool_call_limit_int=4,
            tool_timeout_seconds_f=0.0,
            tool_timeout_retries_int=1,
        )

        with caplog.at_level("INFO"):
            out = await status_tool.coroutine("task-missing")

        assert isinstance(out, str)
        assert "invalid_followup_task_id" in out

    asyncio.run(_run())

    assert "Marking follow-up task_id invalid after task-not-found: tool=status task_id=task-missing" in caplog.text
    assert "Tool invocation failed: tool=status" not in caplog.text
    assert "Traceback" not in caplog.text


def test_followup_with_stale_task_id_is_blocked_before_invocation() -> None:
    state: dict[str, Any] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 8,
    }
    status_calls = {"n": 0}

    execute_tool = _FakeTool("execute", response_format="content", func=lambda task_id: {"task_id": task_id})

    def _status(task_id: str) -> str:
        status_calls["n"] += 1
        return f"running:{task_id}"

    status_tool = _FakeTool("status", response_format="content", func=_status)

    MCPClientUtil._apply_tool_execution_guards(
        [execute_tool, status_tool],
        tool_call_state=state,
        tool_call_limit_int=4,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    assert execute_tool.func("task-old") == {"task_id": "task-old"}
    assert execute_tool.func("task-new") == {"task_id": "task-new"}

    stale_out = status_tool.func("task-old")
    latest_out = status_tool.func("task-new")

    assert isinstance(stale_out, str)
    assert "stale_followup_task_id" in stale_out
    assert "latest_task_id=task-new" in stale_out
    assert latest_out == "running:task-new"
    assert status_calls["n"] == 1
    assert state["followup_used"] == 1


def test_tool_agent_prompt_instructs_latest_task_id_only_and_no_retry_after_404() -> None:
    prompt = CodingAgentPrompts().tool_agent_system_prompt(
        hitl_policy_text="dummy policy",
        agent_name="tool_agent_coding",
        followup_poll_interval_seconds=10.0,
        status_tail_lines=20,
        result_tail_lines=80,
    )

    assert "最後に成功した execute の戻り task_id" in prompt
    assert "Task not found" in prompt
    assert "同じ task_id での followup を二度と繰り返さないでください" in prompt


def test_duplicate_error_result_is_not_cached_and_still_hits_budget() -> None:
    state: dict[str, int] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 0,
    }
    calls = {"n": 0}

    def _func() -> str:
        calls["n"] += 1
        return "ERROR: temporary failure"

    tool = _FakeTool("get_loaded_config_info", response_format="content", func=_func)

    MCPClientUtil._apply_tool_execution_guards(
        [tool],
        tool_call_state=state,
        tool_call_limit_int=1,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    out1 = tool.func()
    out2 = tool.func()

    assert out1 == "ERROR: temporary failure"
    assert "tool_call_budget_exceeded" in out2
    assert calls["n"] == 1
    assert state["general_used"] == 1


class _FakeMCPClient:
    def __init__(self, _config: Any) -> None:
        self._config = _config

    async def get_tools(self) -> list[Any]:
        return []


class _NamedTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"tool:{name}"
        self.args_schema = {"type": "object"}


class _FakeLangGraphAgent:
    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.system_prompt = system_prompt


class _FakeSupervisorApp:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[Any, Any | None]] = []

    async def ainvoke(self, payload: Any, config: Any | None = None) -> Any:
        self.calls.append((payload, config))
        if not self._responses:
            raise AssertionError("No fake responses left")
        return self._responses.pop(0)


def _build_mcp_config(*server_names: str) -> MCPServerConfig:
    config = MCPServerConfig()
    config.servers = {
        server_name: MCPServerConfigEntry(name=server_name, command="dummy")
        for server_name in server_names
    }
    return config


def _build_runtime_config() -> AiChatUtilConfig:
    return AiChatUtilConfig.model_construct(
        llm=LLMSection.model_construct(
            provider="openai",
            completion_model="gpt-5",
            embedding_model="text-embedding-3-small",
            timeout_seconds=60.0,
            api_key="dummy",
            base_url=None,
            api_version=None,
            extra_headers=None,
        ),
        mcp=MCPSection(),
        features=FeaturesSection(),
        logging=LoggingSection(),
        network=NetworkSection(),
        office2pdf=Office2PDFSection(),
    )


def _patch_agent_creation(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, list[str]]]:
    created: list[tuple[str, str, list[str]]] = []

    monkeypatch.setattr(agent_mod, "MultiServerMCPClient", _FakeMCPClient)

    def _fake_create_agent(_llm: Any, _tools: list[Any], *, system_prompt: str, name: str) -> _FakeLangGraphAgent:
        created.append((name, system_prompt, [str(getattr(tool, "name", "")) for tool in _tools]))
        return _FakeLangGraphAgent(name=name, system_prompt=system_prompt)

    monkeypatch.setattr(agent_mod, "create_agent", _fake_create_agent)
    return created


def test_create_sub_agents_assigns_unique_names_for_mixed_mcp_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_agent_creation(monkeypatch)

    agents = asyncio.run(
        AgentBuilder.create_sub_agents(
            runtime_config=_build_runtime_config(),
            mcp_config=_build_mcp_config("coding-agent", "general-tools"),
            llm=object(),  # type: ignore
            prompts=CodingAgentPrompts(),
            tool_limits=None,
        )
    )

    assert [name for name, _, _ in created] == ["tool_agent_coding", "tool_agent_general"]
    assert [agent.get_agent_name() for agent in agents] == ["tool_agent_coding", "tool_agent_general"]
    assert len({agent.get_agent().name for agent in agents}) == 2
    assert "tool_agent_coding" in created[0][1]
    assert "tool_agent_general" in created[1][1]


def test_create_sub_agents_limits_general_tools_when_coding_agent_route_is_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_agent_creation(monkeypatch)

    class _FakeToolMCPClient:
        def __init__(self, _config: Any) -> None:
            self._config = _config

        async def get_tools(self) -> list[Any]:
            return [
                _NamedTool("get_loaded_config_info"),
                _NamedTool("analyze_files"),
                _NamedTool("analyze_pdf_files"),
            ]

    monkeypatch.setattr(agent_mod, "MultiServerMCPClient", _FakeToolMCPClient)

    agents = asyncio.run(
        AgentBuilder.create_sub_agents(
            runtime_config=_build_runtime_config(),
            mcp_config=_build_mcp_config("coding-agent", "general-tools"),
            llm=object(), # type: ignore
            prompts=CodingAgentPrompts(),
            tool_limits=None,
            include_general_agent=True,
            general_tool_allowlist=["get_loaded_config_info"],
        )
    )

    assert [name for name, _, _ in created] == ["tool_agent_coding", "tool_agent_general"]
    assert [agent.get_agent_name() for agent in agents] == ["tool_agent_coding", "tool_agent_general"]
    assert created[0][2] == ["get_loaded_config_info", "analyze_files", "analyze_pdf_files"]
    assert created[1][2] == ["get_loaded_config_info"]


def test_create_sub_agents_keeps_general_when_coding_agent_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_agent_creation(monkeypatch)

    agents = asyncio.run(
        AgentBuilder.create_sub_agents(
            runtime_config=_build_runtime_config(),
            mcp_config=_build_mcp_config("general-tools"),
            llm=object(), # type: ignore
            prompts=CodingAgentPrompts(),
            tool_limits=None,
            include_general_agent=False,
        )
    )

    assert [name for name, _, _ in created] == ["tool_agent_general"]
    assert [agent.get_agent_name() for agent in agents] == ["tool_agent_general"]


@pytest.mark.parametrize(
    ("server_names", "expected_name"),
    [
        (("coding-agent",), "tool_agent_coding"),
        (("general-tools",), "tool_agent_general"),
    ],
)
def test_create_sub_agents_keeps_stable_name_for_single_group(
    monkeypatch: pytest.MonkeyPatch,
    server_names: tuple[str, ...],
    expected_name: str,
) -> None:
    created = _patch_agent_creation(monkeypatch)

    agents = asyncio.run(
        AgentBuilder.create_sub_agents(
            runtime_config=_build_runtime_config(),
            mcp_config=_build_mcp_config(*server_names),
            llm=object(), # type: ignore
            prompts=CodingAgentPrompts(),
            tool_limits=None,
        )
    )

    assert [name for name, _, _ in created] == [expected_name]
    assert [agent.get_agent_name() for agent in agents] == [expected_name]


def test_supervisor_prompt_lists_dynamic_tool_agent_names() -> None:
    prompt = CodingAgentPrompts().supervisor_system_prompt(
        tools_description="dummy tools",
        supervisor_hitl_policy_text="dummy policy",
        tool_agent_names=["tool_agent_coding", "tool_agent_general"],
    )

    assert "tool_agent_coding, tool_agent_general" in prompt
    assert "ツール実行エージェント" in prompt


def test_tool_limits_default_call_limit_is_raised_for_mixed_scenarios() -> None:
    assert ToolLimits.from_config(_build_runtime_config()).tool_call_limit == 4
    assert ToolLimits.from_config(_build_runtime_config()).followup_tool_call_limit == 8


def test_mcp_client_forces_graceful_completion_after_budget_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_app = _FakeSupervisorApp(
        responses=[
            {
                "output": (
                    "<OUTPUT><TEXT>設定ファイルは /tmp/ai-chat-util-config.yml です。\n"
                    "重要な見出し: Overview, Setup, Troubleshooting</TEXT>"
                    "<RESPONSE_TYPE>complete</RESPONSE_TYPE></OUTPUT>"
                )
            },
        ]
    )

    final_text, resp_type, hitl_kind, hitl_tool, add_in, add_out = asyncio.run(
        MCPClientUtil.force_graceful_completion_after_budget_exhaustion(
            app=fake_app,
            run_trace_id="1234567890abcdef1234567890abcdef",
            recursion_limit=50,
            user_text="ERROR: tool call budget exceeded. error=tool_call_budget_exceeded tool=analyze_files limit=4 used=4.",
        )
    )

    assert resp_type == "complete"
    assert hitl_kind is None
    assert hitl_tool is None
    assert add_in == 0
    assert add_out == 0
    assert len(fake_app.calls) == 1
    second_payload, _second_config = fake_app.calls[0]
    assert "追加のツール実行、同一ツールの再試行、planner_agent への再委譲は行わないでください" in second_payload["messages"][0].content
    assert "/tmp/ai-chat-util-config.yml" in final_text
    assert "Overview" in final_text


def test_extract_successful_tool_evidence_collects_config_path_and_stdout() -> None:
    result = {
        "messages": [
            {"role": "tool", "content": '{"path": "/tmp/ai-chat-util-config.yml", "config": {"ai_chat_util_config": {}}}'},
            {"role": "tool", "content": '{"stdout": "# Overview\n## Setup\n## Troubleshooting", "stderr": null}'},
        ]
    }

    evidence = MCPClientUtil.extract_successful_tool_evidence([result])

    assert evidence["config_path"] == "/tmp/ai-chat-util-config.yml"
    assert evidence["stdout_blocks"] == ["# Overview\n## Setup\n## Troubleshooting"]
    assert evidence["headings"] == ["# Overview", "## Setup", "## Troubleshooting"]


def test_extract_successful_tool_evidence_prefers_heading_exact_lines() -> None:
    result = {
        "messages": [
            {
                "role": "tool",
                "content": '{"stdout": "HEADING_LINE_EXACT: ### 1. 接続成立\nHEADING_LINE_EXACT: ### 2. MCP サーバー化\nHEADING_LINE_EXACT: ### 3. 検証結果", "stderr": null}',
            },
        ]
    }

    evidence = MCPClientUtil.extract_successful_tool_evidence([result])

    assert evidence["headings"] == ["### 1. 接続成立", "### 2. MCP サーバー化", "### 3. 検証結果"]


def test_extract_successful_tool_evidence_prefers_exact_block_over_noisy_fallbacks() -> None:
    result = {
        "messages": [
            {
                "role": "tool",
                "content": '{"stdout": "重要な見出し: 検証目的, R検証手順, 判定基準", "stderr": null}',
            },
            {
                "role": "tool",
                "content": '{"stdout": "HEADING_LINE_EXACT: ### 1. MCP サーバーとしての正常起動\nHEADING_LINE_EXACT: ### 2. スーパーバイザーからの接続成立\nHEADING_LINE_EXACT: ### 3. 委譲と統合の正常系", "stderr": null}',
            },
        ]
    }

    evidence = MCPClientUtil.extract_successful_tool_evidence([result])

    assert evidence["headings"] == [
        "### 1. MCP サーバーとしての正常起動",
        "### 2. スーパーバイザーからの接続成立",
        "### 3. 委譲と統合の正常系",
    ]


def test_extract_successful_tool_evidence_prefers_most_complete_exact_block() -> None:
    result = {
        "messages": [
            {
                "role": "tool",
                "content": '{"stdout": "HEADING_LINE_EXACT: ### 0. 途中経過", "stderr": null}',
            },
            {
                "role": "tool",
                "content": '{"stdout": "HEADING_LINE_EXACT: ### 1. MCP サーバーとしての正常起動\nHEADING_LINE_EXACT: ### 2. スーパーバイザーからの接続成立\nHEADING_LINE_EXACT: ### 3. 委譲と統合の正常系", "stderr": null}',
            },
        ]
    }

    evidence = MCPClientUtil.extract_successful_tool_evidence([result])

    assert evidence["headings"] == [
        "### 1. MCP サーバーとしての正常起動",
        "### 2. スーパーバイザーからの接続成立",
        "### 3. 委譲と統合の正常系",
    ]


def test_extract_successful_tool_evidence_uses_raw_text_when_stdout_is_absent() -> None:
    result = {
        "messages": [
            {
                "role": "tool",
                "content": (
                    "以下が指定された内容のまとめです:\n\n"
                    "1. **設定ファイルの場所**:\n"
                    "   - `/tmp/ai-chat-util-config.yml`\n\n"
                    "2. **重要な見出し**:\n"
                    "   1. **MCP サーバーとしての正常起動**\n"
                    "   2. **スーパーバイザーからの接続成立**\n"
                    "   3. **委譲と統合の正常系**\n"
                ),
            }
        ]
    }

    evidence = MCPClientUtil.extract_successful_tool_evidence([result])

    assert evidence["config_path"] == "/tmp/ai-chat-util-config.yml"
    assert evidence["headings"] == [
        "MCP サーバーとしての正常起動",
        "スーパーバイザーからの接続成立",
        "委譲と統合の正常系",
    ]


def test_extract_successful_tool_evidence_ignores_descriptive_bullets_in_raw_text() -> None:
    result = {
        "messages": [
            {
                "role": "tool",
                "content": (
                    "以下に示す内容は、指定された要求に基づいた結果です。\n\n"
                    "2. **文書からの重要な見出し**:\n"
                    "   1. 検証目的\n"
                    "      - コーディングエージェントのMCPサーバー化の実用性を確認することを目的とします。\n"
                    "   2. 検証範囲と優先確認項目\n"
                    "      - 起動検証と接続確認を中心に構成します。\n"
                    "   3. 役割分担の考え方\n"
                    "      - 各サーバーの役割を切り分けます。\n"
                ),
            }
        ]
    }

    evidence = MCPClientUtil.extract_successful_tool_evidence([result])

    assert evidence["headings"] == [
        "検証目的",
        "検証範囲と優先確認項目",
        "役割分担の考え方",
    ]


def test_extract_successful_tool_evidence_reads_stdout_log_from_artifact_metadata(tmp_path: Path) -> None:
    stdout_path = tmp_path / "stdout.log"
    stdout_path.write_text(
        "HEADING_LINE_EXACT: # コーディングエージェントのMCPサーバー化検証\n"
        "HEADING_LINE_EXACT: ## 検証目的\n"
        "HEADING_LINE_EXACT: ### 1. MCP サーバーとしての正常起動\n",
        encoding="utf-8",
    )

    result = {
        "messages": [
            {
                "role": "tool",
                "content": '{"stdout": null, "stderr": "Read target.md"}',
                "artifact": {
                    "structured_content": {
                        "stdout": None,
                        "stderr": "Read target.md",
                        "workspace_path": tmp_path.as_posix(),
                        "metadata": {
                            "stdout_path": stdout_path.as_posix(),
                        },
                    }
                },
            }
        ]
    }

    evidence = MCPClientUtil.extract_successful_tool_evidence([result])

    assert evidence["stdout_blocks"] == [
        "HEADING_LINE_EXACT: # コーディングエージェントのMCPサーバー化検証\n"
        "HEADING_LINE_EXACT: ## 検証目的\n"
        "HEADING_LINE_EXACT: ### 1. MCP サーバーとしての正常起動"
    ]
    assert evidence["headings"] == [
        "# コーディングエージェントのMCPサーバー化検証",
        "## 検証目的",
        "### 1. MCP サーバーとしての正常起動",
    ]


def test_extract_successful_tool_evidence_extracts_config_path_from_stdout_block() -> None:
    result = {
        "messages": [
            {
                "role": "tool",
                "content": '{"stdout": "設定ファイルの場所: /tmp/ai-chat-util-config.yml\nHEADING_LINE_EXACT: ## 検証目的", "stderr": null}',
            }
        ]
    }

    evidence = MCPClientUtil.extract_successful_tool_evidence([result])

    assert evidence["config_path"] == "/tmp/ai-chat-util-config.yml"
    assert evidence["headings"] == ["## 検証目的"]


def test_collect_checkpoint_write_results_reads_tool_messages(tmp_path: Path) -> None:
    db_path = tmp_path / "langgraph_checkpoints.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "create table writes ("
        "thread_id text not null, "
        "checkpoint_ns text not null default '', "
        "checkpoint_id text not null, "
        "task_id text not null, "
        "idx integer not null, "
        "channel text not null, "
        "type text, "
        "value blob)"
    )
    serde = JsonPlusSerializer()
    payload = [ToolMessage(content='{"stdout": "HEADING_LINE_EXACT: ## 検証目的", "stderr": null}', tool_call_id="call-1")]
    value_type, value_blob = serde.dumps_typed(payload)
    conn.execute(
        "insert into writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) values (?, ?, ?, ?, ?, ?, ?, ?)",
        ("trace-1", "tool_agent_coding:ns", "chk-1", "task-1", 0, "messages", value_type, value_blob),
    )
    conn.commit()
    conn.close()

    results = MCPClientUtil.collect_checkpoint_write_results(
        checkpoint_db_path=db_path,
        run_trace_id="trace-1",
    )

    assert len(results) == 1
    messages = results[0]["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert MCPClientUtil.extract_successful_tool_evidence(results)["headings"] == ["## 検証目的"]


def test_final_text_contradicts_evidence_for_stdout_missing_message() -> None:
    evidence = {
        "config_path": None,
        "stdout_blocks": ["HEADING_LINE_EXACT: ## 検証目的"],
        "headings": ["## 検証目的"],
    }

    assert MCPClientUtil.final_text_contradicts_evidence(
        "stdout に明記された結果が返ってきませんでした。見出しの具体的な内容は不明です。",
        evidence,
    )


def test_final_text_contradicts_evidence_for_failure_preface_with_real_headings() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "stdout_blocks": ["HEADING_LINE_EXACT: ## 検証目的"],
        "headings": ["## 検証目的"],
    }

    assert MCPClientUtil.final_text_contradicts_evidence(
        "指定された Markdown ファイルから見出しを抽出するプロセスで問題が発生しました。再度試行し、正確な見出しの抽出を行いたい場合は、具体的なエラー分析が必要です。",
        evidence,
    )


def test_final_text_contradicts_evidence_when_leading_heading_differs_from_evidence() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "expects_heading_response": True,
        "stdout_blocks": ["HEADING_LINE_EXACT: # コーディングエージェントのMCPサーバー化検証"],
        "headings": [
            "# コーディングエージェントのMCPサーバー化検証",
            "## 検証目的",
        ],
    }

    assert MCPClientUtil.final_text_contradicts_evidence(
        "設定ファイルの場所は /tmp/ai-chat-util-config.yml です。\n- HEADING_LINE_EXACT: # 概要\n- HEADING_LINE_EXACT: ## 検証目的\n文書内の重要な見出し:\n# コーディングエージェントのMCPサーバー化検証\n## 検証目的",
        evidence,
    )


def test_should_prefer_deterministic_evidence_response_for_complete_heading_set() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "expects_heading_response": True,
        "headings": [
            "# コーディングエージェントのMCPサーバー化検証",
            "## 検証目的",
            "### 1. MCP サーバーとしての正常起動",
        ],
    }

    assert MCPClientUtil.should_prefer_deterministic_evidence_response(
        "設定ファイルの場所は /tmp/ai-chat-util-config.yml です。",
        evidence,
    )


def test_evidence_reflection_overrides_negative_final_text() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "expects_heading_response": True,
        "stdout_blocks": ["# Overview\n## Setup\n## Troubleshooting"],
        "headings": ["# Overview", "## Setup", "## Troubleshooting"],
    }

    assert MCPClientUtil.final_text_contradicts_evidence(
        "get_loaded_config_info の正しい情報が取得できなかった。重要な見出しを確認できなかった。",
        evidence,
    )

    fallback = MCPClientUtil.build_evidence_reflected_final_text(evidence)

    assert "設定ファイルの場所: /tmp/ai-chat-util-config.yml" in fallback
    assert "# Overview" in fallback
    assert "## Setup" in fallback
    assert "## Troubleshooting" in fallback


def test_final_text_missing_concrete_evidence_detects_missing_path_and_headings() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "expects_heading_response": True,
        "stdout_blocks": ["# Overview\n## Setup\n## Troubleshooting"],
        "headings": ["# Overview", "## Setup", "## Troubleshooting"],
    }

    assert MCPClientUtil.final_text_missing_concrete_evidence(
        "get_loaded_config_info を使って確認しました。重要な見出しは概要、設定、トラブルシュートです。",
        evidence,
    )


def test_augment_final_text_with_evidence_preserves_text_and_adds_exact_values() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "expects_heading_response": True,
        "stdout_blocks": ["# Overview\n## Setup\n## Troubleshooting"],
        "headings": ["# Overview", "## Setup", "## Troubleshooting"],
    }

    augmented = MCPClientUtil.augment_final_text_with_evidence(
        "確認しました。重要な見出しを以下に示します。",
        evidence,
    )

    assert "確認しました。重要な見出しを以下に示します。" in augmented
    assert "設定ファイルの場所: /tmp/ai-chat-util-config.yml" in augmented
    assert "# Overview" in augmented
    assert "## Setup" in augmented
    assert "## Troubleshooting" in augmented
    assert not MCPClientUtil.final_text_missing_concrete_evidence(augmented, evidence)


def test_augment_final_text_with_evidence_does_not_duplicate_existing_values() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "expects_heading_response": True,
        "stdout_blocks": ["# Overview\n## Setup\n## Troubleshooting"],
        "headings": ["# Overview", "## Setup", "## Troubleshooting"],
    }


def test_build_evidence_reflected_final_text_for_judgment_prompt_does_not_dump_headings() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "expects_heading_response": False,
        "stdout_blocks": ["判断に必要な追加確認事項: 監視設計、ロールバック手順、運用体制"],
        "headings": ["# Overview", "## Setup", "## Troubleshooting"],
    }

    fallback = MCPClientUtil.build_evidence_reflected_final_text(evidence)

    assert "設定ファイルの場所: /tmp/ai-chat-util-config.yml" in fallback
    assert "判断に必要な追加確認事項" in fallback
    assert "# Overview" not in fallback

    original = (
        "設定ファイルの場所: /tmp/ai-chat-util-config.yml\n"
        "文書内の重要な見出し:\n"
        "# Overview\n"
        "## Setup\n"
        "## Troubleshooting"
    )

    augmented = MCPClientUtil.augment_final_text_with_evidence(original, evidence)
    assert "判断に必要な追加確認事項" in augmented
    assert augmented.count("# Overview") == 1


def test_tool_agent_prompt_requires_verbatim_heading_output() -> None:
    prompt = CodingAgentPrompts().tool_agent_system_prompt(
        hitl_policy_text="dummy policy",
        agent_name="tool_agent_coding",
        followup_poll_interval_seconds=2.0,
        status_tail_lines=20,
        result_tail_lines=80,
    )

    assert "HEADING_LINE_EXACT:" in prompt
    assert "### 1. MCP サーバーとしての正常起動" in prompt


def test_supervisor_prompt_requires_coding_agent_when_explicitly_requested() -> None:
    prompt = CodingAgentPrompts().supervisor_system_prompt(
        tools_description="dummy tools",
        supervisor_hitl_policy_text="dummy policy",
        tool_agent_names=["tool_agent_coding", "tool_agent_general"],
    )

    assert "`coding agent` / `coding-agent` / `コーディングエージェント`" in prompt
    assert "通常ツール（例: analyze_files）へ置き換えてはいけません" in prompt
    assert "invalid_followup_task_id" in prompt
    assert "同じ目的の execute やり直しを指示せず" in prompt
    assert "error=execute_request_invalid" in prompt
    assert "error=tool_timeout" in prompt


def test_contains_followup_task_error_signal_detects_guard_errors() -> None:
    assert MCPClientUtil.contains_followup_task_error_signal(
        "ERROR: follow-up task_id is invalid. error=invalid_followup_task_id tool=status task_id=abc"
    )
    assert MCPClientUtil.contains_followup_task_error_signal(
        "ERROR: follow-up task_id is stale. error=stale_followup_task_id tool=get_result task_id=old latest_task_id=new"
    )
    assert not MCPClientUtil.contains_followup_task_error_signal("ERROR: tool_call_budget_exceeded")


def test_explicitly_requests_coding_agent_detects_user_instruction() -> None:
    assert MCPClientUtil.explicitly_requests_coding_agent(
        [
            {
                "role": "user",
                "content": "まず get_loaded_config_info を呼んでから、coding-agent を使ってこの markdown を確認してください。",
            }
        ]
    )


def test_should_run_config_preflight_detects_ordered_request() -> None:
    assert MCPClientUtil.should_run_config_preflight(
        [
            {
                "role": "user",
                "content": "まず get_loaded_config_info を呼んでから、coding-agent を使って確認してください。",
            }
        ]
    )


def test_should_run_config_preflight_ignores_unordered_mentions() -> None:
    assert not MCPClientUtil.should_run_config_preflight(
        [
            {
                "role": "user",
                "content": "get_loaded_config_info というツール名は知っていますが、今は別件です。",
            }
        ]
    )


def test_invalid_followup_task_text_discourages_execute_rerun() -> None:
    text = ToolLimits.invalid_followup_task_text("status", "task-1", None)

    assert "取得済みの stdout/stderr" in text
    assert "execute をやり直さないでください" in text


def test_stale_followup_task_text_discourages_new_execute() -> None:
    text = ToolLimits.stale_followup_task_text("get_result", "task-1", "task-2")

    assert "新しい execute を追加で起こさず" in text


def test_build_config_preflight_message_contains_reuse_instruction() -> None:
    message = MCPClientUtil.build_config_preflight_message(
        {
            "config_path": "/tmp/ai-chat-util-config.yml",
            "text": "{'path': '/tmp/ai-chat-util-config.yml'}",
            "success": True,
        }
    )

    assert message is not None
    assert "path=/tmp/ai-chat-util-config.yml" in message
    assert "再実行しないでください" in message


def test_run_config_preflight_invokes_general_tool_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeToolClient:
        def __init__(self, _config: Any) -> None:
            self._config = _config

        async def get_tools(self) -> list[Any]:
            async def _tool_call() -> dict[str, str]:
                calls.append("get_loaded_config_info")
                return {"path": "/tmp/ai-chat-util-config.yml"}

            return [
                _FakeTool(
                    "get_loaded_config_info",
                    response_format="content",
                    coroutine=_tool_call,
                )
            ]

    monkeypatch.setattr(llm_mcp_client_util_mod, "MultiServerMCPClient", _FakeToolClient)

    runtime_config = _build_runtime_config()
    runtime_config.mcp.coding_agent_endpoint.mcp_server_name = "coding-agent"
    runtime_config.mcp.working_directory = "/tmp"
    runtime_config.mcp.mcp_config_path = None
    runtime_config.features.audit_log_enabled = False
    monkeypatch.setattr(type(runtime_config), "get_mcp_server_config", lambda self: _build_mcp_config("coding-agent", "general-tools"))

    payload = asyncio.run(
        MCPClientUtil.run_config_preflight(
            runtime_config=runtime_config,
            tool_limits=None,
            audit_context=None,
        )
    )

    assert calls == ["get_loaded_config_info"]
    assert payload is not None
    assert payload["config_path"] == "/tmp/ai-chat-util-config.yml"


def test_explicitly_requests_coding_agent_ignores_non_user_mentions() -> None:
    assert not MCPClientUtil.explicitly_requests_coding_agent(
        [
            {
                "role": "assistant",
                "content": "次は coding-agent を使うかもしれません。",
            }
        ]
    )


def test_explicitly_requests_coding_agent_ignores_path_only_match() -> None:
    assert not MCPClientUtil.explicitly_requests_coding_agent(
        [
            {
                "role": "user",
                "content": "この設定と /home/user/docs/02_コーディングエージェントのMCPサーバー化検証.md を踏まえて評価してください。",
            }
        ]
    )


def test_extract_explicit_user_file_paths_returns_existing_files_only(tmp_path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("# Title\n", encoding="utf-8")

    paths = MCPClientUtil.extract_explicit_user_file_paths(
        [{"role": "user", "content": f"Please inspect {target.as_posix()} and summarize headings."}]
    )

    assert paths == [target.resolve().as_posix()]


def test_extract_requested_heading_count_detects_user_constraint() -> None:
    assert MCPClientUtil.extract_requested_heading_count(
        [
            {
                "role": "user",
                "content": "文書内で重要な見出しを 3 点挙げてください。",
            }
        ]
    ) == 3


def test_requests_heading_response_detects_heading_intent() -> None:
    assert MCPClientUtil.requests_heading_response(
        [
            {
                "role": "user",
                "content": "文書内の見出しを抽出してください。",
            }
        ]
    )


def test_requests_heading_response_ignores_judgment_prompt() -> None:
    assert not MCPClientUtil.requests_heading_response(
        [
            {
                "role": "user",
                "content": "この設定と文書だけで本番投入判断に足りるか評価し、不足情報を列挙してください。",
            }
        ]
    )


def test_requests_heading_response_ignores_negative_heading_instruction() -> None:
    assert not MCPClientUtil.requests_heading_response(
        [
            {
                "role": "user",
                "content": "この設定と文書で本番投入判断を評価してください。見出し抽出は不要です。",
            }
        ]
    )


def test_select_headings_for_response_prefers_numbered_heading_block() -> None:
    evidence = {
        "headings": [
            "# コーディングエージェントのMCPサーバー化検証",
            "## 検証目的",
            "### 1. MCP サーバーとしての正常起動",
            "### 2. スーパーバイザーからの接続成立",
            "### 3. 委譲と統合の正常系",
            "## 役割分担の考え方",
        ],
        "requested_heading_count": 3,
    }

    assert MCPClientUtil.select_headings_for_response(evidence) == [
        "### 1. MCP サーバーとしての正常起動",
        "### 2. スーパーバイザーからの接続成立",
        "### 3. 委譲と統合の正常系",
    ]


def test_build_evidence_reflected_final_text_respects_requested_heading_count() -> None:
    fallback = MCPClientUtil.build_evidence_reflected_final_text(
        {
            "config_path": "/tmp/ai-chat-util-config.yml",
            "expects_heading_response": True,
            "headings": [
                "# コーディングエージェントのMCPサーバー化検証",
                "## 検証目的",
                "### 1. MCP サーバーとしての正常起動",
                "### 2. スーパーバイザーからの接続成立",
                "### 3. 委譲と統合の正常系",
                "## 役割分担の考え方",
            ],
            "requested_heading_count": 3,
        }
    )

    assert "設定ファイルの場所: /tmp/ai-chat-util-config.yml" in fallback
    assert "### 1. MCP サーバーとしての正常起動" in fallback
    assert "### 2. スーパーバイザーからの接続成立" in fallback
    assert "### 3. 委譲と統合の正常系" in fallback
    assert "# コーディングエージェントのMCPサーバー化検証" not in fallback
    assert "## 役割分担の考え方" not in fallback


def test_final_text_missing_concrete_evidence_uses_requested_heading_subset() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "expects_heading_response": True,
        "headings": [
            "# コーディングエージェントのMCPサーバー化検証",
            "## 検証目的",
            "### 1. MCP サーバーとしての正常起動",
            "### 2. スーパーバイザーからの接続成立",
            "### 3. 委譲と統合の正常系",
            "## 役割分担の考え方",
        ],
        "requested_heading_count": 3,
    }

    assert not MCPClientUtil.final_text_missing_concrete_evidence(
        "設定ファイルの場所: /tmp/ai-chat-util-config.yml\n"
        "文書内の重要な見出し:\n"
        "### 1. MCP サーバーとしての正常起動\n"
        "### 2. スーパーバイザーからの接続成立\n"
        "### 3. 委譲と統合の正常系",
        evidence,
    )


def test_should_include_general_agent_forced_coding_route_with_explicit_file_is_false() -> None:
    assert not MCPClientUtil.should_include_general_agent(
        force_coding_agent_route=True,
        explicit_user_file_paths=["/tmp/work/doc.md"],
    )


def test_should_include_general_agent_without_explicit_file_is_true() -> None:
    assert MCPClientUtil.should_include_general_agent(
        force_coding_agent_route=True,
        explicit_user_file_paths=None,
    )
    assert MCPClientUtil.should_include_general_agent(
        force_coding_agent_route=False,
        explicit_user_file_paths=["/tmp/work/doc.md"],
    )


def test_get_loaded_runtime_config_path_returns_existing_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config_file = tmp_path / "ai-chat-util-config.yml"
    config_file.write_text("llm: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "ai_chat_util.base.llm.llm_mcp_client_util.get_runtime_config_path",
        lambda: config_file,
    )

    assert MCPClientUtil.get_loaded_runtime_config_path() == config_file.resolve().as_posix()


def test_get_loaded_runtime_config_path_returns_none_for_missing_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    missing = tmp_path / "missing.yml"
    monkeypatch.setattr(
        "ai_chat_util.base.llm.llm_mcp_client_util.get_runtime_config_path",
        lambda: missing,
    )

    assert MCPClientUtil.get_loaded_runtime_config_path() is None


def test_choose_better_config_path_prefers_concrete_file_over_glob(tmp_path) -> None:
    config_file = tmp_path / "ai-chat-util-config.poc.yml"
    config_file.write_text("llm: {}\n", encoding="utf-8")

    assert MCPClientUtil._choose_better_config_path(
        f"{tmp_path.as_posix()}/*.yml",
        config_file.as_posix(),
    ) == config_file.as_posix()


def test_extract_config_path_from_text_ignores_glob_path() -> None:
    assert MCPClientUtil.extract_config_path_from_text(
        "設定ファイルの場所: /tmp/example/*.yml"
    ) is None


def test_extract_successful_tool_evidence_prefers_concrete_path_over_glob() -> None:
    evidence = MCPClientUtil.extract_successful_tool_evidence(
        [
            {
                "messages": [
                    {"role": "tool", "content": '{"path": "/tmp/example/*.yml"}'},
                    {"role": "tool", "content": '{"path": "/tmp/example/ai-chat-util-config.poc.yml"}'},
                ]
            }
        ]
    )

    assert evidence["config_path"] == "/tmp/example/ai-chat-util-config.poc.yml"


def test_extract_markdown_heading_lines_from_files_reads_exact_heading_lines(tmp_path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(
        "# Title\n"
        "text\n"
        "## Section\n"
        "### 1. 単体起動\n",
        encoding="utf-8",
    )

    assert MCPClientUtil.extract_markdown_heading_lines_from_files([doc.as_posix()]) == [
        "# Title",
        "## Section",
        "### 1. 単体起動",
    ]


def test_extract_successful_tool_evidence_ignores_ai_only_heading_claims() -> None:
    evidence = MCPClientUtil.extract_successful_tool_evidence(
        [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "HEADING_LINE_EXACT: ### 1. はじめに\nHEADING_LINE_EXACT: ### 2. 準備",
                    },
                    {
                        "role": "tool",
                        "content": '{"path": "/tmp/ai-chat-util-config.yml"}',
                    },
                ]
            }
        ]
    )

    assert evidence["config_path"] == "/tmp/ai-chat-util-config.yml"
    assert evidence["headings"] == []




def test_collect_checkpoint_results_reads_latest_state_and_history() -> None:
    class _FakeApp:
        async def aget_state(self, config: Any) -> Any:
            return SimpleNamespace(values={"messages": [{"role": "tool", "content": '{"path": "/tmp/a.yml"}'}]})

        async def aget_state_history(self, config: Any, limit: int | None = None):
            yield SimpleNamespace(values={"messages": [{"role": "tool", "content": '{"stdout": "hello"}'}]})

    results = asyncio.run(MCPClientUtil.collect_checkpoint_results(app=_FakeApp(), run_trace_id="abc"))

    assert len(results) == 2
    evidence = MCPClientUtil.extract_successful_tool_evidence(results)
    assert evidence["config_path"] == "/tmp/a.yml"
    assert evidence["stdout_blocks"] == ["hello"]


def test_collect_evidence_results_merges_checkpoint_history_for_headings() -> None:
    workflow_results = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "config_path: /tmp/ai-chat-util-config.yml",
                }
            ]
        }
    ]

    class _FakeApp:
        async def aget_state(self, config: Any) -> Any:
            return SimpleNamespace(
                values={
                    "messages": [
                        {
                            "role": "tool",
                            "content": '{"stdout": "HEADING_LINE_EXACT: ## 概要\\nHEADING_LINE_EXACT: ### MCP サーバー", "stderr": null}',
                        }
                    ]
                }
            )

        async def aget_state_history(self, config: Any, limit: int | None = None):
            if False:
                yield None

    results = asyncio.run(
        MCPClientUtil.collect_evidence_results(
            app=_FakeApp(),
            run_trace_id="abc",
            workflow_results=workflow_results,
        )
    )

    evidence = MCPClientUtil.extract_successful_tool_evidence(results)

    assert evidence["headings"] == ["## 概要", "### MCP サーバー"]

    fallback = MCPClientUtil.build_evidence_reflected_final_text(evidence)
    assert "## 概要" in fallback
    assert "### MCP サーバー" in fallback


def test_build_recursion_limit_fallback_text_prefers_evidence() -> None:
    text = MCPClientUtil.build_recursion_limit_fallback_text(
        "Recursion limit of 50 reached without hitting a stop condition",
        {
            "config_path": "/tmp/ai-chat-util-config.yml",
            "expects_heading_response": True,
            "stdout_blocks": ["# Overview\n## Setup\n## Troubleshooting"],
            "headings": ["# Overview", "## Setup", "## Troubleshooting"],
        },
    )

    assert "再帰上限に到達" in text
    assert "/tmp/ai-chat-util-config.yml" in text
    assert "# Overview" in text
    assert "## Setup" in text
    assert "## Troubleshooting" in text


def test_build_evidence_reflected_final_text_includes_all_headings() -> None:
    fallback = MCPClientUtil.build_evidence_reflected_final_text(
        {
            "config_path": "/tmp/ai-chat-util-config.yml",
            "expects_heading_response": True,
            "headings": [
                "# Overview",
                "## Setup",
                "## Troubleshooting",
                "## Appendix",
            ],
            "stdout_blocks": [],
        }
    )

    assert "# Overview" in fallback
    assert "## Setup" in fallback
    assert "## Troubleshooting" in fallback
    assert "## Appendix" in fallback


def test_should_prefer_deterministic_evidence_response_for_heading_report() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "expects_heading_response": True,
        "headings": ["# Overview", "## Setup"],
        "stdout_blocks": ["HEADING_LINE_EXACT: # Overview\nHEADING_LINE_EXACT: ## Setup"],
    }

    assert MCPClientUtil.should_prefer_deterministic_evidence_response(
        "文書内の見出しは以下です。\n- HEADING_LINE_EXACT: # Wrong",
        evidence,
    )


def test_build_evidence_summary_does_not_mark_heading_extraction_for_evaluation_prompt() -> None:
    summary = MCPClientUtil.build_evidence_summary(
        {
            "config_path": "/tmp/ai-chat-util-config.yml",
            "expects_heading_response": False,
            "headings": ["# Overview", "## Setup"],
            "stdout_blocks": ["判断に必要な追加確認事項: 監視設計"],
        }
    )

    assert "get_loaded_config_info" in summary.successful_tools
    assert "heading_extraction" not in summary.successful_tools


def test_extract_config_path_from_text_returns_yaml_path() -> None:
    assert MCPClientUtil.extract_config_path_from_text(
        "設定ファイルのパスは /tmp/ai-chat-util-config.yml です。"
    ) == "/tmp/ai-chat-util-config.yml"


def test_build_recursion_limit_fallback_text_without_evidence_returns_error() -> None:
    text = MCPClientUtil.build_recursion_limit_fallback_text(
        "Recursion limit of 50 reached without hitting a stop condition",
        {"config_path": None, "stdout_blocks": []},
    )

    assert "ERROR: MCPワークフローが再帰上限に到達したため停止しました。" in text
    assert "Recursion limit of 50 reached without hitting a stop condition" in text
