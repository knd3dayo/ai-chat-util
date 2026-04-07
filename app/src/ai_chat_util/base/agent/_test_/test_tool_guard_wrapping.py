from __future__ import annotations

import asyncio
import importlib
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

import ai_chat_util.base.agent.agent_builder as agent_mod
import ai_chat_util.base.agent.agent_client_util as llm_mcp_client_util_mod
from ai_chat_util.base.agent.agent_builder import AgentBuilder
from ai_chat_util.base.agent.agent_client import AgentClient
from ai_chat_util.base.agent.tool_limits import ToolLimits
from ai_chat_util.base.agent.prompts import CodingAgentPrompts
from ai_chat_util.base.agent.agent_client_util import AgentClientUtil
from ai_chat_util.base.agent.supervisor_support import RouteCandidate, RoutingDecision
from ai_chat_util.common.config.ai_chat_util_mcp_config import MCPServerConfig, MCPServerConfigEntry
from ai_chat_util.common.config.runtime import (
    AiChatUtilConfig,
    FeaturesSection,
    LLMSection,
    LoggingSection,
    MCPSection,
    NetworkSection,
    Office2PDFSection,
)
from ai_chat_util.common.model.ai_chatl_util_models import ChatContent, ChatHistory, ChatMessage, ChatRequest


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

    AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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

        AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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


def test_approval_tool_is_blocked_until_explicitly_approved() -> None:
    state: dict[str, Any] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 0,
        "approval_tools": {"analyze_files"},
        "approved_tools": set(),
        "auto_approve": False,
    }
    calls = {"n": 0}

    def _func(file_path_list: list[str], prompt: str) -> str:
        calls["n"] += 1
        return f"ok:{len(file_path_list)}:{prompt}"

    tool = _FakeTool("analyze_files", response_format="content", func=_func)

    AgentClientUtil._apply_tool_execution_guards(
        [tool],
        tool_call_state=state,
        tool_call_limit_int=1,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    blocked = tool.func(["/tmp/work/doc.md"], "inspect")
    assert isinstance(blocked, str)
    assert "tool_approval_required" in blocked
    assert "APPROVE analyze_files" in blocked
    assert calls["n"] == 0

    state["approved_tools"] = {"analyze_files"}
    allowed = tool.func(["/tmp/work/doc.md"], "inspect")
    assert allowed == "ok:1:inspect"
    assert calls["n"] == 1


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

    AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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

        AgentClientUtil._apply_tool_execution_guards(
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

        AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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

        AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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

    AgentClientUtil._apply_tool_execution_guards(
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
    def __init__(self, name: str, *, description: str | None = None, args_schema: Any | None = None) -> None:
        self.name = name
        self.description = description if description is not None else f"tool:{name}"
        self.args_schema = args_schema if args_schema is not None else {"type": "object"}


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


def test_create_sub_agents_excludes_coding_agent_for_general_route(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_agent_creation(monkeypatch)

    agents = asyncio.run(
        AgentBuilder.create_sub_agents(
            runtime_config=_build_runtime_config(),
            mcp_config=_build_mcp_config("coding-agent", "general-tools"),
            llm=object(), # type: ignore
            prompts=CodingAgentPrompts(),
            tool_limits=None,
            include_coding_agent=False,
            include_general_agent=True,
        )
    )

    assert [name for name, _, _ in created] == ["tool_agent_general"]
    assert [agent.get_agent_name() for agent in agents] == ["tool_agent_general"]


def test_create_sub_agents_keeps_coding_agent_when_general_tools_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_agent_creation(monkeypatch)

    agents = asyncio.run(
        AgentBuilder.create_sub_agents(
            runtime_config=_build_runtime_config(),
            mcp_config=_build_mcp_config("coding-agent"),
            llm=object(), # type: ignore
            prompts=CodingAgentPrompts(),
            tool_limits=None,
            include_coding_agent=False,
            include_general_agent=True,
        )
    )

    assert [name for name, _, _ in created] == ["tool_agent_coding"]
    assert [agent.get_agent_name() for agent in agents] == ["tool_agent_coding"]


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


def test_create_sub_agents_uses_configured_coding_agent_server_key(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_agent_creation(monkeypatch)
    runtime_config = _build_runtime_config()
    runtime_config.mcp.coding_agent_endpoint.mcp_server_name = "deepagent"

    agents = asyncio.run(
        AgentBuilder.create_sub_agents(
            runtime_config=runtime_config,
            mcp_config=_build_mcp_config("deepagent", "general-tools"),
            llm=object(), # type: ignore
            prompts=CodingAgentPrompts(),
            tool_limits=None,
        )
    )

    assert [name for name, _, _ in created] == ["tool_agent_coding", "tool_agent_general"]
    assert [agent.get_agent_name() for agent in agents] == ["tool_agent_coding", "tool_agent_general"]


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


def test_routing_prompt_lists_deep_agent_route() -> None:
    prompt = CodingAgentPrompts().routing_system_prompt()

    assert "deep_agent" in prompt
    assert "DeepAgents" in prompt


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
        AgentClientUtil.force_graceful_completion_after_budget_exhaustion(
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
    assert "追加のツール実行、同一ツールの再試行、追加の再委譲は行わないでください" in second_payload["messages"][0].content
    assert "/tmp/ai-chat-util-config.yml" in final_text
    assert "Overview" in final_text


def test_extract_successful_tool_evidence_collects_config_path_and_stdout() -> None:
    result = {
        "messages": [
            {"role": "tool", "content": '{"path": "/tmp/ai-chat-util-config.yml", "config": {"ai_chat_util_config": {}}}'},
            {"role": "tool", "content": '{"stdout": "# Overview\n## Setup\n## Troubleshooting", "stderr": null}'},
        ]
    }

    evidence = AgentClientUtil.extract_successful_tool_evidence([result])

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

    evidence = AgentClientUtil.extract_successful_tool_evidence([result])

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

    evidence = AgentClientUtil.extract_successful_tool_evidence([result])

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

    evidence = AgentClientUtil.extract_successful_tool_evidence([result])

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

    evidence = AgentClientUtil.extract_successful_tool_evidence([result])

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

    evidence = AgentClientUtil.extract_successful_tool_evidence([result])

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

    evidence = AgentClientUtil.extract_successful_tool_evidence([result])

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

    evidence = AgentClientUtil.extract_successful_tool_evidence([result])

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

    results = AgentClientUtil.collect_checkpoint_write_results(
        checkpoint_db_path=db_path,
        run_trace_id="trace-1",
    )

    assert len(results) == 1
    messages = results[0]["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert AgentClientUtil.extract_successful_tool_evidence(results)["headings"] == ["## 検証目的"]


def test_final_text_contradicts_evidence_for_stdout_missing_message() -> None:
    evidence = {
        "config_path": None,
        "stdout_blocks": ["HEADING_LINE_EXACT: ## 検証目的"],
        "headings": ["## 検証目的"],
    }

    assert AgentClientUtil.final_text_contradicts_evidence(
        "stdout に明記された結果が返ってきませんでした。見出しの具体的な内容は不明です。",
        evidence,
    )


def test_final_text_contradicts_evidence_for_failure_preface_with_real_headings() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "stdout_blocks": ["HEADING_LINE_EXACT: ## 検証目的"],
        "headings": ["## 検証目的"],
    }

    assert AgentClientUtil.final_text_contradicts_evidence(
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

    assert AgentClientUtil.final_text_contradicts_evidence(
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

    assert AgentClientUtil.should_prefer_deterministic_evidence_response(
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

    assert AgentClientUtil.final_text_contradicts_evidence(
        "get_loaded_config_info の正しい情報が取得できなかった。重要な見出しを確認できなかった。",
        evidence,
    )

    fallback = AgentClientUtil.build_evidence_reflected_final_text(evidence)

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

    assert AgentClientUtil.final_text_missing_concrete_evidence(
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

    augmented = AgentClientUtil.augment_final_text_with_evidence(
        "確認しました。重要な見出しを以下に示します。",
        evidence,
    )

    assert "確認しました。重要な見出しを以下に示します。" in augmented
    assert "設定ファイルの場所: /tmp/ai-chat-util-config.yml" in augmented
    assert "# Overview" in augmented
    assert "## Setup" in augmented
    assert "## Troubleshooting" in augmented
    assert not AgentClientUtil.final_text_missing_concrete_evidence(augmented, evidence)


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

    fallback = AgentClientUtil.build_evidence_reflected_final_text(evidence)

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

    augmented = AgentClientUtil.augment_final_text_with_evidence(original, evidence)
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
    assert AgentClientUtil.contains_followup_task_error_signal(
        "ERROR: follow-up task_id is invalid. error=invalid_followup_task_id tool=status task_id=abc"
    )
    assert AgentClientUtil.contains_followup_task_error_signal(
        "ERROR: follow-up task_id is stale. error=stale_followup_task_id tool=get_result task_id=old latest_task_id=new"
    )
    assert not AgentClientUtil.contains_followup_task_error_signal("ERROR: tool_call_budget_exceeded")


def test_explicitly_requests_coding_agent_detects_user_instruction() -> None:
    assert AgentClientUtil.explicitly_requests_coding_agent(
        [
            {
                "role": "user",
                "content": "まず get_loaded_config_info を呼んでから、coding-agent を使ってこの markdown を確認してください。",
            }
        ]
    )


def test_explicitly_requests_deep_agent_detects_user_instruction() -> None:
    assert AgentClientUtil.explicitly_requests_deep_agent(
        [
            {
                "role": "user",
                "content": "deep-agent を使ってこの設定と文書を段階的に調査してください。",
            }
        ]
    )


def test_default_routing_prefers_deep_agent_for_explicit_request_when_enabled() -> None:
    decision = AgentClientUtil._build_default_routing_decision(
        force_coding_agent_route=False,
        force_deep_agent_route=True,
        deep_agent_enabled=True,
        explicit_user_file_paths=[],
        explicit_user_directory_paths=[],
        available_tool_names=["analyze_files", "get_loaded_config_info"],
    )

    assert decision.selected_route == "deep_agent"
    assert decision.reason_code == "route.multi_step_investigation_needed"


def test_default_routing_does_not_select_deep_agent_when_disabled() -> None:
    decision = AgentClientUtil._build_default_routing_decision(
        force_coding_agent_route=False,
        force_deep_agent_route=True,
        deep_agent_enabled=False,
        explicit_user_file_paths=[],
        explicit_user_directory_paths=[],
        available_tool_names=["analyze_files", "get_loaded_config_info"],
    )

    assert decision.selected_route == "general_tool_agent"


def test_default_routing_prefers_general_tool_agent_for_explicit_directory_path() -> None:
    decision = AgentClientUtil._build_default_routing_decision(
        force_coding_agent_route=False,
        force_deep_agent_route=False,
        deep_agent_enabled=True,
        explicit_user_file_paths=[],
        explicit_user_directory_paths=["/tmp/work"],
        available_tool_names=["analyze_files", "get_loaded_config_info"],
    )

    assert decision.selected_route == "general_tool_agent"
    assert decision.reason_code == "route.explicit_directory_path_request"


def test_should_run_config_preflight_detects_ordered_request() -> None:
    assert AgentClientUtil.should_run_config_preflight(
        [
            {
                "role": "user",
                "content": "まず get_loaded_config_info を呼んでから、coding-agent を使って確認してください。",
            }
        ]
    )


def test_should_run_config_preflight_ignores_unordered_mentions() -> None:
    assert not AgentClientUtil.should_run_config_preflight(
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
    message = AgentClientUtil.build_config_preflight_message(
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
        AgentClientUtil.run_config_preflight(
            runtime_config=runtime_config,
            tool_limits=None,
            audit_context=None,
        )
    )

    assert calls == ["get_loaded_config_info"]
    assert payload is not None
    assert payload["config_path"] == "/tmp/ai-chat-util-config.yml"


def test_explicitly_requests_coding_agent_ignores_non_user_mentions() -> None:
    assert not AgentClientUtil.explicitly_requests_coding_agent(
        [
            {
                "role": "assistant",
                "content": "次は coding-agent を使うかもしれません。",
            }
        ]
    )


def test_explicitly_requests_deep_agent_ignores_non_user_mentions() -> None:
    assert not AgentClientUtil.explicitly_requests_deep_agent(
        [
            {
                "role": "assistant",
                "content": "次は deep-agent を使うかもしれません。",
            }
        ]
    )


def test_explicitly_requests_coding_agent_ignores_path_only_match() -> None:
    assert not AgentClientUtil.explicitly_requests_coding_agent(
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

    paths = AgentClientUtil.extract_explicit_user_file_paths(
        [{"role": "user", "content": f"Please inspect {target.as_posix()} and summarize headings."}]
    )

    assert paths == [target.resolve().as_posix()]


def test_extract_explicit_user_directory_paths_resolves_relative_directory_against_working_directory(tmp_path) -> None:
    target_dir = tmp_path / "work"
    target_dir.mkdir()

    paths = AgentClientUtil.extract_explicit_user_directory_paths(
        [{"role": "user", "content": "work ディレクトリを確認してください。"}],
        working_directory=tmp_path.as_posix(),
    )

    assert paths == [target_dir.resolve().as_posix()]


def test_extract_requested_heading_count_detects_user_constraint() -> None:
    assert AgentClientUtil.extract_requested_heading_count(
        [
            {
                "role": "user",
                "content": "文書内で重要な見出しを 3 点挙げてください。",
            }
        ]
    ) == 3


def test_requests_heading_response_detects_heading_intent() -> None:
    assert AgentClientUtil.requests_heading_response(
        [
            {
                "role": "user",
                "content": "文書内の見出しを抽出してください。",
            }
        ]
    )


def test_requests_heading_response_ignores_judgment_prompt() -> None:
    assert not AgentClientUtil.requests_heading_response(
        [
            {
                "role": "user",
                "content": "この設定と文書だけで本番投入判断に足りるか評価し、不足情報を列挙してください。",
            }
        ]
    )


def test_requests_heading_response_ignores_negative_heading_instruction() -> None:
    assert not AgentClientUtil.requests_heading_response(
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

    assert AgentClientUtil.select_headings_for_response(evidence) == [
        "### 1. MCP サーバーとしての正常起動",
        "### 2. スーパーバイザーからの接続成立",
        "### 3. 委譲と統合の正常系",
    ]


def test_build_evidence_reflected_final_text_respects_requested_heading_count() -> None:
    fallback = AgentClientUtil.build_evidence_reflected_final_text(
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

    assert not AgentClientUtil.final_text_missing_concrete_evidence(
        "設定ファイルの場所: /tmp/ai-chat-util-config.yml\n"
        "文書内の重要な見出し:\n"
        "### 1. MCP サーバーとしての正常起動\n"
        "### 2. スーパーバイザーからの接続成立\n"
        "### 3. 委譲と統合の正常系",
        evidence,
    )


def test_should_include_general_agent_forced_coding_route_with_explicit_file_is_false() -> None:
    assert not AgentClientUtil.should_include_general_agent(
        force_coding_agent_route=True,
        explicit_user_file_paths=["/tmp/work/doc.md"],
    )


def test_should_include_general_agent_without_explicit_file_is_true() -> None:
    assert AgentClientUtil.should_include_general_agent(
        force_coding_agent_route=True,
        explicit_user_file_paths=None,
    )
    assert AgentClientUtil.should_include_general_agent(
        force_coding_agent_route=False,
        explicit_user_file_paths=["/tmp/work/doc.md"],
    )


def test_get_loaded_runtime_config_path_returns_existing_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config_file = tmp_path / "ai-chat-util-config.yml"
    config_file.write_text("llm: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "ai_chat_util.base.agent.mcp_client_util.get_runtime_config_path",
        lambda: config_file,
    )

    assert AgentClientUtil.get_loaded_runtime_config_path() == config_file.resolve().as_posix()


def test_get_loaded_runtime_config_path_returns_none_for_missing_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    missing = tmp_path / "missing.yml"
    monkeypatch.setattr(
        "ai_chat_util.base.agent.mcp_client_util.get_runtime_config_path",
        lambda: missing,
    )

    assert AgentClientUtil.get_loaded_runtime_config_path() is None


def test_choose_better_config_path_prefers_concrete_file_over_glob(tmp_path) -> None:
    config_file = tmp_path / "ai-chat-util-config.poc.yml"
    config_file.write_text("llm: {}\n", encoding="utf-8")

    assert AgentClientUtil._choose_better_config_path(
        f"{tmp_path.as_posix()}/*.yml",
        config_file.as_posix(),
    ) == config_file.as_posix()


def test_extract_config_path_from_text_ignores_glob_path() -> None:
    assert AgentClientUtil.extract_config_path_from_text(
        "設定ファイルの場所: /tmp/example/*.yml"
    ) is None


def test_extract_successful_tool_evidence_prefers_concrete_path_over_glob() -> None:
    evidence = AgentClientUtil.extract_successful_tool_evidence(
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

    assert AgentClientUtil.extract_markdown_heading_lines_from_files([doc.as_posix()]) == [
        "# Title",
        "## Section",
        "### 1. 単体起動",
    ]


def test_extract_successful_tool_evidence_ignores_ai_only_heading_claims() -> None:
    evidence = AgentClientUtil.extract_successful_tool_evidence(
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

    results = asyncio.run(AgentClientUtil.collect_checkpoint_results(app=_FakeApp(), run_trace_id="abc"))

    assert len(results) == 2
    evidence = AgentClientUtil.extract_successful_tool_evidence(results)
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
        AgentClientUtil.collect_evidence_results(
            app=_FakeApp(),
            run_trace_id="abc",
            workflow_results=workflow_results,
        )
    )

    evidence = AgentClientUtil.extract_successful_tool_evidence(results)

    assert evidence["headings"] == ["## 概要", "### MCP サーバー"]

    fallback = AgentClientUtil.build_evidence_reflected_final_text(evidence)
    assert "## 概要" in fallback
    assert "### MCP サーバー" in fallback


def test_build_recursion_limit_fallback_text_prefers_evidence() -> None:
    text = AgentClientUtil.build_recursion_limit_fallback_text(
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
    fallback = AgentClientUtil.build_evidence_reflected_final_text(
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

    assert AgentClientUtil.should_prefer_deterministic_evidence_response(
        "文書内の見出しは以下です。\n- HEADING_LINE_EXACT: # Wrong",
        evidence,
    )


def test_build_evidence_summary_does_not_mark_heading_extraction_for_evaluation_prompt() -> None:
    summary = AgentClientUtil.build_evidence_summary(
        {
            "config_path": "/tmp/ai-chat-util-config.yml",
            "expects_heading_response": False,
            "headings": ["# Overview", "## Setup"],
            "stdout_blocks": ["判断に必要な追加確認事項: 監視設計"],
        }
    )

    assert "get_loaded_config_info" in summary.successful_tools
    assert "heading_extraction" not in summary.successful_tools


def test_extract_successful_tool_evidence_ignores_synthetic_headings_by_file_block() -> None:
    evidence = AgentClientUtil.extract_successful_tool_evidence(
        [
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": '{"stdout": "## Headings by File\n### File 1: `01_doc.md`\n### File 2: `02_doc.md`", "stderr": null}',
                    }
                ]
            }
        ]
    )

    assert evidence["headings"] == []


def test_extract_successful_tool_evidence_parses_heading_table_rows() -> None:
    evidence = AgentClientUtil.extract_successful_tool_evidence(
        [
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": '{"stdout": "## 共通見出し 3 点\n| # | 見出し | 該当ファイル |\n|---|--------|---------------|\n| 1 | **検証目的** | 全ファイル |\n| 2 | **検証手順** / **検証シナリオ** | 全ファイル |\n| 3 | **判定基準** | 01, 02, 03 |", "stderr": null}',
                    }
                ]
            }
        ]
    )

    assert evidence["headings"] == ["検証目的", "検証手順 / 検証シナリオ", "判定基準"]


def test_build_evidence_reflected_final_text_includes_tool_catalog_when_requested() -> None:
    text = AgentClientUtil.build_evidence_reflected_final_text(
        {
            "expects_tool_catalog_response": True,
            "tool_catalog": [
                {
                    "agent_name": "tool_agent_general",
                    "tool_names": ["get_loaded_config_info", "analyze_files"],
                }
            ],
        }
    )

    assert "supervisor が参照した利用可能ツール一覧:" in text
    assert "tool_agent_general: get_loaded_config_info, analyze_files" in text


def test_final_text_missing_concrete_evidence_detects_missing_tool_catalog_listing() -> None:
    assert AgentClientUtil.final_text_missing_concrete_evidence(
        "利用可能ツールは確認しました。",
        {
            "expects_tool_catalog_response": True,
            "tool_catalog": [
                {
                    "agent_name": "tool_agent_general",
                    "tool_names": ["get_loaded_config_info", "analyze_files"],
                }
            ],
        },
    )


def test_requests_tool_catalog_response_detects_tool_list_intent() -> None:
    assert AgentClientUtil.requests_tool_catalog_response(
        [
            {
                "role": "user",
                "content": "supervisor が見ている利用可能ツール一覧を教えてください。",
            }
        ]
    )


def test_requests_tool_catalog_response_detects_detailed_tool_list_intent() -> None:
    assert AgentClientUtil.requests_tool_catalog_response(
        [
            {
                "role": "user",
                "content": "利用可能な MCP ツールの名称、説明、主要な引数を一覧で示してください。通常ツールと coding agent 系ツールを分けて整理してください。",
            }
        ]
    )


def test_requests_tool_catalog_details_detects_description_and_args_request() -> None:
    assert AgentClientUtil.requests_tool_catalog_details(
        [
            {
                "role": "user",
                "content": "利用可能な MCP ツールの名称、説明、主要な引数を一覧で示してください。",
            }
        ]
    )


def test_requests_tool_catalog_response_does_not_match_general_tool_summary_prompt() -> None:
    assert not AgentClientUtil.requests_tool_catalog_response(
        [
            {
                "role": "user",
                "content": "現在読み込まれている設定ファイルの場所と利用可能な解析系ツールを簡潔に説明してください",
            }
        ]
    )


def test_requests_tool_catalog_response_does_not_match_exact_shared_general_prompt() -> None:
    assert not AgentClientUtil.requests_tool_catalog_response(
        [
            {
                "role": "user",
                "content": "必ず MCP ツールで設定情報を確認してから、現在読み込まれている設定ファイルの場所と利用可能な解析系ツールを簡潔に説明してください。",
            }
        ]
    )


def test_requests_tool_catalog_response_does_not_match_evaluation_prompt() -> None:
    assert not AgentClientUtil.requests_tool_catalog_response(
        [
            {
                "role": "user",
                "content": "この設定と文書だけで本番投入判断に足りるかを答え、足りない場合は不足情報を挙げて必要な追加確認を示してください",
            }
        ]
    )


def test_requests_tool_catalog_response_does_not_match_exact_shared_evaluation_prompt() -> None:
    assert not AgentClientUtil.requests_tool_catalog_response(
        [
            {
                "role": "user",
                "content": "作業対象は /home/user/source/repos/ai-platform-poc です。まず MCP ツールで現在の設定情報を確認し、その後 coding agent を使って docs/11_検証 配下を調査してください。この設定と文書だけで本番投入判断に足りるかを答え、足りない場合は不足情報を挙げて必要な追加確認を示してください。",
            }
        ]
    )


def test_requests_tool_catalog_response_does_not_match_hitl_prompt() -> None:
    assert not AgentClientUtil.requests_tool_catalog_response(
        [
            {
                "role": "user",
                "content": "本番投入してよいか判断してください。不足している場合は追加で確認すべき点を 3 つまで挙げ、どれがユーザー判断事項かを明示してください",
            }
        ]
    )


def test_requests_tool_catalog_response_does_not_match_exact_shared_hitl_prompt() -> None:
    assert not AgentClientUtil.requests_tool_catalog_response(
        [
            {
                "role": "user",
                "content": "作業対象は /home/user/source/repos/ai-platform-poc です。MCP ツールで関連設定を確認したうえで、本番投入してよいか判断してください。判断に必要な前提が不足している場合は、追加で確認すべき点を 3 つまで挙げ、どれがユーザー判断事項かを明示してください。",
            }
        ]
    )


def test_requests_tool_catalog_details_does_not_match_general_supervisor_prompts() -> None:
    prompts = [
        "現在読み込まれている設定ファイルの場所と利用可能な解析系ツールを簡潔に説明してください",
        "この設定と文書だけで本番投入判断に足りるかを答え、足りない場合は不足情報を挙げて必要な追加確認を示してください",
        "本番投入してよいか判断してください。不足している場合は追加で確認すべき点を 3 つまで挙げ、どれがユーザー判断事項かを明示してください",
        "必ず MCP ツールで設定情報を確認してから、現在読み込まれている設定ファイルの場所と利用可能な解析系ツールを簡潔に説明してください。",
        "作業対象は /home/user/source/repos/ai-platform-poc です。まず MCP ツールで現在の設定情報を確認し、その後 coding agent を使って docs/11_検証 配下を調査してください。この設定と文書だけで本番投入判断に足りるかを答え、足りない場合は不足情報を挙げて必要な追加確認を示してください。",
        "作業対象は /home/user/source/repos/ai-platform-poc です。MCP ツールで関連設定を確認したうえで、本番投入してよいか判断してください。判断に必要な前提が不足している場合は、追加で確認すべき点を 3 つまで挙げ、どれがユーザー判断事項かを明示してください。",
    ]

    for prompt in prompts:
        assert not AgentClientUtil.requests_tool_catalog_details(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )


def test_build_tool_catalog_response_text_formats_agent_names() -> None:
    text = AgentClientUtil.build_tool_catalog_response_text(
        {
            "coding_agent": [
                {"name": "execute", "description": "tool:execute", "primary_args": ["prompt", "workspace_path"]},
                {"name": "status", "description": "tool:status", "primary_args": ["task_id"]},
                {"name": "get_result", "description": "tool:get_result", "primary_args": ["task_id"]},
            ],
            "general_tool_agent": [
                {"name": "get_loaded_config_info", "description": "tool:get_loaded_config_info", "primary_args": []},
                {"name": "analyze_files", "description": "tool:analyze_files", "primary_args": ["file_path_list", "prompt"]},
            ],
        }
    )

    assert "supervisor が参照した利用可能ツール一覧:" in text
    assert "tool_agent_coding: execute, status, get_result" in text
    assert "tool_agent_general: get_loaded_config_info, analyze_files" in text


def test_build_tool_catalog_response_text_includes_details_when_requested() -> None:
    text = AgentClientUtil.build_tool_catalog_response_text(
        {
            "coding_agent": [
                {"name": "execute", "description": "非同期でタスクを実行する", "primary_args": ["prompt", "workspace_path", "timeout"]},
            ],
            "general_tool_agent": [
                {"name": "analyze_files", "description": "複数ファイルを解析する", "primary_args": ["file_path_list", "prompt", "detail"]},
            ],
        },
        include_details=True,
    )

    assert "### tool_agent_coding" in text
    assert "1. execute" in text
    assert "説明: 非同期でタスクを実行する" in text
    assert "主要な引数: prompt, workspace_path, timeout" in text
    assert "### tool_agent_general" in text
    assert "analyze_files" in text


def test_build_route_tool_catalog_payload_includes_route_backend_metadata() -> None:
    runtime_config = _build_runtime_config()
    runtime_config.mcp.coding_agent_endpoint.mcp_server_name = "deepagent"

    payload = AgentClientUtil.build_route_tool_catalog_payload(
        {
            "coding_agent": [
                {"name": "execute", "description": "tool:execute", "primary_args": ["prompt"]},
            ],
            "general_tool_agent": [
                {"name": "analyze_files", "description": "tool:analyze_files", "primary_args": ["file_path_list", "prompt"]},
            ],
        },
        runtime_config=runtime_config,
    )

    assert payload["tool_agent_names"] == ["tool_agent_coding", "tool_agent_general"]
    assert payload["route_backends"]["coding_agent"] == {
        "agent_name": "tool_agent_coding",
        "agent_family": "coding_agent",
        "selected_server_key": "deepagent",
        "server_keys": ["deepagent"],
        "backend_kind": "mcp_async_task",
    }
    assert payload["tool_catalog"][0]["agent_name"] == "tool_agent_coding"
    assert payload["tool_catalog"][0]["selected_server_key"] == "deepagent"
    assert payload["tool_catalog"][0]["agent_family"] == "coding_agent"


def test_build_available_routes_text_includes_visible_tools() -> None:
    text = AgentClientUtil._build_available_routes_text(
        has_coding_agent=True,
        has_deep_agent=True,
        has_general_agent=True,
        route_tool_catalog={
            "coding_agent": ["execute", "status", "get_result"],
            "deep_agent": ["analyze_files", "get_loaded_config_info"],
            "general_tool_agent": ["get_loaded_config_info", "analyze_files"],
        },
    )

    assert "visible_tools: execute, status, get_result" in text
    assert "deep_agent" in text
    assert "visible_tools: get_loaded_config_info, analyze_files" in text


def test_build_routing_context_text_includes_route_tool_catalog() -> None:
    text = AgentClientUtil._build_routing_context_text(
        force_coding_agent_route=False,
        force_deep_agent_route=False,
        explicit_user_file_paths=[],
        explicit_user_directory_paths=["/tmp/work"],
        routing_mode="structured",
        preferred_coding_route="deep_agent",
        route_tool_catalog={
            "coding_agent": ["execute", "status"],
            "deep_agent": ["analyze_files"],
            "general_tool_agent": ["get_loaded_config_info"],
        },
    )

    assert "coding_agent_tools=execute, status" in text
    assert "preferred_coding_route=deep_agent" in text
    assert "explicit_user_directory_paths=/tmp/work" in text
    assert "general_tool_agent_tools=get_loaded_config_info" in text


def test_decide_route_returns_default_for_explicit_directory_path_in_structured_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_config = _build_runtime_config()
    runtime_config.features.routing_mode = "structured"
    runtime_config.features.enable_deep_agent = True
    monkeypatch.setattr(llm_mcp_client_util_mod, "deepagents_available", lambda: True)
    monkeypatch.setattr(type(runtime_config), "get_mcp_server_config", lambda self: _build_mcp_config("coding-agent", "general-tools"))
    monkeypatch.setattr(
        AgentClientUtil,
        "create_llm",
        classmethod(lambda cls, runtime_config: (_ for _ in ()).throw(AssertionError("LLM should not be called"))),
    )

    decision = asyncio.run(
        AgentClientUtil.decide_route(
            runtime_config=runtime_config,
            prompts=CodingAgentPrompts(),
            messages=[{"role": "user", "content": "work ディレクトリを確認してください。"}],
            force_coding_agent_route=False,
            force_deep_agent_route=False,
            explicit_user_file_paths=[],
            explicit_user_directory_paths=["/tmp/work"],
            available_tool_names=["analyze_files", "get_loaded_config_info"],
            route_tool_catalog={
                "general_tool_agent": ["analyze_files", "get_loaded_config_info"],
                "deep_agent": ["analyze_files", "get_loaded_config_info"],
            },
            audit_context=None,
        )
    )

    assert decision.selected_route == "general_tool_agent"
    assert decision.reason_code == "route.explicit_directory_path_request"


def test_extract_task_id_from_tool_result_supports_text_part_list() -> None:
    task_id = AgentClientUtil._extract_task_id_from_tool_result(
        [{"type": "text", "text": '{"task_id":"task-123"}'}]
    )

    assert task_id == "task-123"


def test_resolve_route_tool_catalog_splits_coding_and_general_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCatalogClient:
        def __init__(self, config: Any) -> None:
            self._config = config

        async def get_tools(self) -> list[Any]:
            server_names = tuple(self._config.keys()) if isinstance(self._config, dict) else ()
            if server_names == ("coding-agent",):
                return [_NamedTool("execute"), _NamedTool("status")]
            if server_names == ("general-tools",):
                return [_NamedTool("get_loaded_config_info"), _NamedTool("analyze_files")]
            if server_names == ("coding-agent", "general-tools"):
                return [
                    _NamedTool("execute"),
                    _NamedTool("status"),
                    _NamedTool("get_result"),
                    _NamedTool("get_loaded_config_info"),
                    _NamedTool("analyze_files"),
                ]
            return []

    monkeypatch.setattr(llm_mcp_client_util_mod, "MultiServerMCPClient", _FakeCatalogClient)
    monkeypatch.setattr(llm_mcp_client_util_mod, "deepagents_available", lambda: True)
    runtime_config = _build_runtime_config()
    runtime_config.features.enable_deep_agent = True
    monkeypatch.setattr(type(runtime_config), "get_mcp_server_config", lambda self: _build_mcp_config("coding-agent", "general-tools"))

    catalog = asyncio.run(
        AgentClientUtil.resolve_route_tool_catalog(
            runtime_config=runtime_config,
        )
    )

    assert catalog == {
        "coding_agent": ["execute", "status"],
        "deep_agent": ["get_loaded_config_info", "analyze_files"],
        "general_tool_agent": ["get_loaded_config_info", "analyze_files"],
    }


def test_resolve_route_tool_inventory_collects_description_and_primary_args(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCatalogClient:
        def __init__(self, config: Any) -> None:
            self._config = config

        async def get_tools(self) -> list[Any]:
            server_names = tuple(self._config.keys()) if isinstance(self._config, dict) else ()
            if server_names == ("coding-agent",):
                return [
                    _NamedTool(
                        "execute",
                        description="非同期でタスク実行",
                        args_schema={
                            "type": "object",
                            "properties": {
                                "req": {
                                    "type": "object",
                                    "properties": {
                                        "prompt": {"type": "string"},
                                        "workspace_path": {"type": "string"},
                                    },
                                    "required": ["prompt", "workspace_path"],
                                }
                            },
                            "required": ["req"],
                        },
                    )
                ]
            if server_names == ("general-tools",):
                return [
                    _NamedTool(
                        "analyze_files",
                        description="複数ファイルを解析する",
                        args_schema={
                            "type": "object",
                            "properties": {
                                "file_path_list": {"type": "array"},
                                "prompt": {"type": "string"},
                                "detail": {"type": "string"},
                            },
                        },
                    )
                ]
            if server_names == ("coding-agent", "general-tools"):
                return [
                    _NamedTool(
                        "execute",
                        description="非同期でタスク実行",
                        args_schema={
                            "type": "object",
                            "properties": {
                                "req": {
                                    "type": "object",
                                    "properties": {
                                        "prompt": {"type": "string"},
                                        "workspace_path": {"type": "string"},
                                    },
                                    "required": ["prompt", "workspace_path"],
                                }
                            },
                            "required": ["req"],
                        },
                    ),
                    _NamedTool(
                        "analyze_files",
                        description="複数ファイルを解析する",
                        args_schema={
                            "type": "object",
                            "properties": {
                                "file_path_list": {"type": "array"},
                                "prompt": {"type": "string"},
                                "detail": {"type": "string"},
                            },
                        },
                    ),
                ]
            return []

    monkeypatch.setattr(llm_mcp_client_util_mod, "MultiServerMCPClient", _FakeCatalogClient)
    monkeypatch.setattr(llm_mcp_client_util_mod, "deepagents_available", lambda: True)
    runtime_config = _build_runtime_config()
    runtime_config.features.enable_deep_agent = True
    monkeypatch.setattr(type(runtime_config), "get_mcp_server_config", lambda self: _build_mcp_config("coding-agent", "general-tools"))

    inventory = asyncio.run(
        AgentClientUtil.resolve_route_tool_inventory(
            runtime_config=runtime_config,
        )
    )

    assert inventory == {
        "coding_agent": [
            {
                "name": "execute",
                "description": "非同期でタスク実行",
                "primary_args": ["prompt", "workspace_path"],
            }
        ],
        "general_tool_agent": [
            {
                "name": "analyze_files",
                "description": "複数ファイルを解析する",
                "primary_args": ["file_path_list", "prompt", "detail"],
            }
        ],
        "deep_agent": [
            {
                "name": "analyze_files",
                "description": "複数ファイルを解析する",
                "primary_args": ["file_path_list", "prompt", "detail"],
            }
        ],
    }


def test_resolve_route_tool_inventory_uses_configured_coding_agent_server_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCatalogClient:
        def __init__(self, config: dict[str, Any]) -> None:
            self._config = config

        async def get_tools(self) -> list[Any]:
            servers = sorted(self._config.keys())
            if servers == ["deepagent"]:
                return [_NamedTool("execute", description="run deepagent task")]
            if servers == ["general-tools"]:
                return [_NamedTool("analyze_files", description="analyze files")]
            return []

    monkeypatch.setattr(llm_mcp_client_util_mod, "MultiServerMCPClient", _FakeCatalogClient)
    runtime_config = _build_runtime_config()
    runtime_config.mcp.coding_agent_endpoint.mcp_server_name = "deepagent"
    monkeypatch.setattr(type(runtime_config), "get_mcp_server_config", lambda self: _build_mcp_config("deepagent", "general-tools"))

    inventory = asyncio.run(AgentClientUtil.resolve_route_tool_inventory(runtime_config=runtime_config))

    assert inventory == {
        "coding_agent": [
            {
                "name": "execute",
                "description": "run deepagent task",
                "primary_args": [],
            }
        ],
        "general_tool_agent": [
            {
                "name": "analyze_files",
                "description": "analyze files",
                "primary_args": [],
            }
        ],
    }


def test_mcp_client_chat_emits_deep_agent_audit_events(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_events: list[dict[str, Any]] = []

    original_llm_client_module = sys.modules.get("ai_chat_util.base.llm.llm_client")
    sys.modules.pop("ai_chat_util.base.agent.mcp_client", None)

    stub_llm_client_module = types.ModuleType("ai_chat_util.base.llm.llm_client")

    class _StubMessageFactoryBase:
        pass

    class _StubMessageFactory(_StubMessageFactoryBase):
        def __init__(self, config: Any) -> None:
            self.config = config

    setattr(stub_llm_client_module, "LLMMessageContentFactoryBase", _StubMessageFactoryBase)
    setattr(stub_llm_client_module, "LLMMessageContentFactory", _StubMessageFactory)
    sys.modules["ai_chat_util.base.llm.llm_client"] = stub_llm_client_module

    llm_mcp_client_mod = importlib.import_module("ai_chat_util.base.agent.mcp_client")
    MCPClient = llm_mcp_client_mod.MCPClient

    if original_llm_client_module is not None:
        sys.modules["ai_chat_util.base.llm.llm_client"] = original_llm_client_module
    else:
        sys.modules.pop("ai_chat_util.base.llm.llm_client", None)

    class _FakeAuditContext:
        def emit(self, event_type: str, **kwargs: Any) -> None:
            recorded_events.append({"event_type": event_type, **kwargs})

    async def _fake_create_sqlite_checkpointer(
        cls: type[AgentClientUtil],
        checkpoint_db_path: Path | None,
        *,
        exit_stack: Any,
    ) -> None:
        return None

    async def _fake_resolve_route_tool_inventory(
        cls: type[AgentClientUtil],
        *,
        runtime_config: AiChatUtilConfig,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "coding_agent": [{"name": "execute", "description": "run coding job", "primary_args": ["prompt"]}],
            "general_tool_agent": [{"name": "analyze_files", "description": "analyze files", "primary_args": ["file_path_list", "prompt"]}],
            "deep_agent": [
                {"name": "analyze_files", "description": "analyze files", "primary_args": ["file_path_list", "prompt"]},
                {"name": "get_loaded_config_info", "description": "show config", "primary_args": []},
            ],
        }

    async def _fake_decide_route(
        cls: type[AgentClientUtil],
        *,
        runtime_config: AiChatUtilConfig,
        prompts: Any,
        messages: Any,
        force_coding_agent_route: bool,
        force_deep_agent_route: bool,
        explicit_user_file_paths: Any,
        explicit_user_directory_paths: Any,
        available_tool_names: list[str],
        route_tool_catalog: dict[str, list[str]],
        audit_context: Any,
    ) -> RoutingDecision:
        candidate = RouteCandidate(
            route_name="deep_agent",
            score=0.98,
            reason_code="route.multi_step_investigation_needed",
            tool_hints=["analyze_files"],
            blocking_issues=[],
        )
        return RoutingDecision(
            selected_route="deep_agent",
            candidate_routes=[candidate],
            reason_code="route.multi_step_investigation_needed",
            confidence=0.98,
            missing_information=[],
            next_action="execute_selected_route",
            requires_hitl=False,
            requires_clarification=False,
            notes="explicit deep-agent test",
        )

    async def _fake_create_deep_agent_workflow(
        cls: type[AgentClientUtil],
        runtime_config: AiChatUtilConfig,
        *,
        checkpointer: Any | None = None,
        tool_limits: ToolLimits | None = None,
        explicit_user_file_paths: Any = None,
        explicit_user_directory_paths: Any = None,
        audit_context: Any | None = None,
    ) -> tuple[_FakeSupervisorApp, list[str]]:
        response = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "<RESPONSE_TYPE>complete</RESPONSE_TYPE><TEXT>deep agent response</TEXT>",
                }
            ]
        }
        return _FakeSupervisorApp([response]), ["analyze_files", "get_loaded_config_info"]

    async def _fake_collect_evidence_results(
        cls: type[AgentClientUtil],
        *,
        app: Any,
        run_trace_id: str,
        workflow_results: Any,
        checkpoint_db_path: Path | None = None,
    ) -> list[Any]:
        if isinstance(workflow_results, list):
            return list(workflow_results)
        return [workflow_results]

    async def _fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_mcp_client_mod, "create_audit_context", lambda *args, **kwargs: _FakeAuditContext())
    monkeypatch.setattr(llm_mcp_client_util_mod, "deepagents_available", lambda: True)
    monkeypatch.setattr(AgentClientUtil, "create_llm", classmethod(lambda cls, runtime_config: object()))
    monkeypatch.setattr(AgentClientUtil, "_create_sqlite_checkpointer", classmethod(_fake_create_sqlite_checkpointer))
    monkeypatch.setattr(AgentClientUtil, "resolve_route_tool_inventory", classmethod(_fake_resolve_route_tool_inventory))
    monkeypatch.setattr(AgentClientUtil, "decide_route", classmethod(_fake_decide_route))
    monkeypatch.setattr(AgentClientUtil, "create_deep_agent_workflow", classmethod(_fake_create_deep_agent_workflow))
    monkeypatch.setattr(AgentClientUtil, "collect_evidence_results", classmethod(_fake_collect_evidence_results))
    monkeypatch.setattr(AgentClientUtil, "collect_checkpoint_write_results", classmethod(lambda cls, checkpoint_db_path, run_trace_id: []))
    monkeypatch.setattr(AgentClientUtil, "final_text_contradicts_evidence", classmethod(lambda cls, user_text, evidence: False))
    monkeypatch.setattr(AgentClientUtil, "final_text_missing_concrete_evidence", classmethod(lambda cls, user_text, evidence: False))
    monkeypatch.setattr(llm_mcp_client_mod.asyncio, "sleep", _fake_sleep)

    runtime_config = _build_runtime_config()
    runtime_config.features.enable_deep_agent = True

    chat_request = ChatRequest(
        chat_history=ChatHistory(
            messages=[
                ChatMessage(
                    role="user",
                    content=[ChatContent(params={"type": "text", "text": "deep_agent で調査してください"})],
                )
            ]
        )
    )

    response = asyncio.run(MCPClient(runtime_config).chat(chat_request))

    assert response.status == "completed"
    assert response.output == "deep agent response"

    route_decided_event = next(event for event in recorded_events if event["event_type"] == "route_decided")
    assert route_decided_event["route_name"] == "deep_agent"
    assert route_decided_event["payload"]["route_tool_catalog"]["deep_agent"] == [
        "analyze_files",
        "get_loaded_config_info",
    ]

    tool_catalog_event = next(event for event in recorded_events if event["event_type"] == "tool_catalog_resolved")
    assert tool_catalog_event["route_name"] == "deep_agent"
    assert tool_catalog_event["payload"]["tool_catalog"] == [
        {"agent_name": "deep_agent", "tool_names": ["analyze_files", "get_loaded_config_info"]}
    ]

    final_validated_event = next(event for event in recorded_events if event["event_type"] == "final_answer_validated")
    assert final_validated_event["final_status"] == "completed"
    assert final_validated_event["reason_code"] == "audit.validation_passed"


def test_mcp_client_chat_emits_selected_server_key_for_coding_agent_route(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_events: list[dict[str, Any]] = []

    original_llm_client_module = sys.modules.get("ai_chat_util.base.llm.llm_client")
    sys.modules.pop("ai_chat_util.base.agent.mcp_client", None)

    stub_llm_client_module = types.ModuleType("ai_chat_util.base.llm.llm_client")

    class _StubMessageFactoryBase:
        pass

    class _StubMessageFactory(_StubMessageFactoryBase):
        def __init__(self, config: Any) -> None:
            self.config = config

    setattr(stub_llm_client_module, "LLMMessageContentFactoryBase", _StubMessageFactoryBase)
    setattr(stub_llm_client_module, "LLMMessageContentFactory", _StubMessageFactory)
    sys.modules["ai_chat_util.base.llm.llm_client"] = stub_llm_client_module

    llm_mcp_client_mod = importlib.import_module("ai_chat_util.base.agent.mcp_client")
    MCPClient = llm_mcp_client_mod.MCPClient

    try:
        class _FakeAuditContext:
            def emit(self, event_type: str, **kwargs: Any) -> None:
                recorded_events.append({"event_type": event_type, **kwargs})

        async def _fake_create_sqlite_checkpointer(
            cls: type[AgentClientUtil],
            checkpoint_db_path: Path | None,
            *,
            exit_stack: Any,
        ) -> None:
            return None

        async def _fake_resolve_route_tool_inventory(
            cls: type[AgentClientUtil],
            *,
            runtime_config: AiChatUtilConfig,
        ) -> dict[str, list[dict[str, Any]]]:
            return {
                "coding_agent": [{"name": "execute", "description": "run coding job", "primary_args": ["prompt"]}],
                "general_tool_agent": [{"name": "analyze_files", "description": "analyze files", "primary_args": ["file_path_list", "prompt"]}],
            }

        async def _fake_decide_route(
            cls: type[AgentClientUtil],
            *,
            runtime_config: AiChatUtilConfig,
            prompts: Any,
            messages: Any,
            force_coding_agent_route: bool,
            force_deep_agent_route: bool,
            explicit_user_file_paths: Any,
            explicit_user_directory_paths: Any,
            available_tool_names: list[str],
            route_tool_catalog: dict[str, list[str]],
            audit_context: Any,
        ) -> RoutingDecision:
            candidate = RouteCandidate(
                route_name="coding_agent",
                score=0.99,
                reason_code="route.explicit_coding_agent_request",
                tool_hints=["execute"],
                blocking_issues=[],
            )
            return RoutingDecision(
                selected_route="coding_agent",
                candidate_routes=[candidate],
                reason_code="route.explicit_coding_agent_request",
                confidence=0.99,
                missing_information=[],
                next_action="execute_selected_route",
                requires_hitl=False,
                requires_clarification=False,
                notes="configured coding-agent-family backend",
            )

        async def _fake_create_workflow(
            cls: type[AgentClientUtil],
            runtime_config: AiChatUtilConfig,
            prompts: Any,
            *,
            llm: Any | None = None,
            checkpointer: Any | None = None,
            tool_limits: ToolLimits | None = None,
            explicit_user_file_paths: Any = None,
            explicit_user_directory_paths: Any = None,
            routing_decision: Any = None,
            audit_context: Any | None = None,
            config_preflight_payload: Any | None = None,
            force_coding_agent_route: bool = False,
            force_deep_agent_route: bool = False,
            expects_heading_response: bool = False,
            expects_evaluation_response: bool = False,
        ) -> _FakeSupervisorApp:
            if audit_context is not None:
                audit_context.emit(
                    "tool_catalog_resolved",
                    route_name="coding_agent",
                    payload=AgentClientUtil.build_route_tool_catalog_payload(
                        {
                            "coding_agent": [{"name": "execute", "description": "run coding job", "primary_args": ["prompt"]}],
                            "general_tool_agent": [{"name": "analyze_files", "description": "analyze files", "primary_args": ["file_path_list", "prompt"]}],
                        },
                        runtime_config=runtime_config,
                    ),
                )
            response = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "<RESPONSE_TYPE>complete</RESPONSE_TYPE><TEXT>coding agent response</TEXT>",
                    }
                ]
            }
            return _FakeSupervisorApp([response])

        async def _fake_collect_evidence_results(
            cls: type[AgentClientUtil],
            *,
            app: Any,
            run_trace_id: str,
            workflow_results: Any,
            checkpoint_db_path: Path | None = None,
        ) -> list[Any]:
            if isinstance(workflow_results, list):
                return list(workflow_results)
            return [workflow_results]

        async def _fake_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(llm_mcp_client_mod, "create_audit_context", lambda *args, **kwargs: _FakeAuditContext())
        monkeypatch.setattr(AgentClientUtil, "create_llm", classmethod(lambda cls, runtime_config: object()))
        monkeypatch.setattr(AgentClientUtil, "_create_sqlite_checkpointer", classmethod(_fake_create_sqlite_checkpointer))
        monkeypatch.setattr(AgentClientUtil, "resolve_route_tool_inventory", classmethod(_fake_resolve_route_tool_inventory))
        monkeypatch.setattr(AgentClientUtil, "decide_route", classmethod(_fake_decide_route))
        monkeypatch.setattr(AgentClientUtil, "create_workflow", classmethod(_fake_create_workflow))
        monkeypatch.setattr(AgentClientUtil, "collect_evidence_results", classmethod(_fake_collect_evidence_results))
        monkeypatch.setattr(AgentClientUtil, "collect_checkpoint_write_results", classmethod(lambda cls, checkpoint_db_path, run_trace_id: []))
        monkeypatch.setattr(AgentClientUtil, "final_text_contradicts_evidence", classmethod(lambda cls, user_text, evidence: False))
        monkeypatch.setattr(AgentClientUtil, "final_text_missing_concrete_evidence", classmethod(lambda cls, user_text, evidence: False))
        monkeypatch.setattr(llm_mcp_client_mod.asyncio, "sleep", _fake_sleep)

        runtime_config = _build_runtime_config()
        runtime_config.mcp.coding_agent_endpoint.mcp_server_name = "deepagent"
        monkeypatch.setattr(type(runtime_config), "get_mcp_server_config", lambda self: _build_mcp_config("deepagent", "general-tools"))

        chat_request = ChatRequest(
            chat_history=ChatHistory(
                messages=[
                    ChatMessage(
                        role="user",
                        content=[ChatContent(params={"type": "text", "text": "複数ステップで調査してください"})],
                    )
                ]
            )
        )

        response = asyncio.run(MCPClient(runtime_config).chat(chat_request))

        assert response.status == "completed"
        assert response.output == "coding agent response"

        route_decided_event = next(event for event in recorded_events if event["event_type"] == "route_decided")
        assert route_decided_event["route_name"] == "coding_agent"
        assert route_decided_event["payload"]["selected_route_backend"] == {
            "agent_name": "tool_agent_coding",
            "agent_family": "coding_agent",
            "selected_server_key": "deepagent",
            "server_keys": ["deepagent"],
            "backend_kind": "mcp_async_task",
        }

        tool_catalog_event = next(event for event in recorded_events if event["event_type"] == "tool_catalog_resolved")
        assert tool_catalog_event["route_name"] == "coding_agent"
        assert tool_catalog_event["payload"]["route_backends"]["coding_agent"]["selected_server_key"] == "deepagent"
        assert tool_catalog_event["payload"]["tool_catalog"][0]["agent_name"] == "tool_agent_coding"
        assert tool_catalog_event["payload"]["tool_catalog"][0]["selected_server_key"] == "deepagent"
    finally:
        if original_llm_client_module is not None:
            sys.modules["ai_chat_util.base.llm.llm_client"] = original_llm_client_module
        else:
            sys.modules.pop("ai_chat_util.base.llm.llm_client", None)


def test_agent_client_pauses_when_approval_required_evidence_is_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_chat_util.base.agent.agent_client as agent_client_mod

    recorded_events: list[dict[str, Any]] = []

    class _FakeAuditContext:
        def emit(self, event_type: str, **kwargs: Any) -> None:
            recorded_events.append({"event_type": event_type, **kwargs})

    async def _fake_create_sqlite_checkpointer(
        cls: type[AgentClientUtil],
        checkpoint_db_path: Path | None,
        *,
        exit_stack: Any,
    ) -> None:
        return None

    async def _fake_resolve_route_tool_inventory(
        cls: type[AgentClientUtil],
        *,
        runtime_config: AiChatUtilConfig,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "general_tool_agent": [
                {"name": "analyze_files", "description": "analyze files", "primary_args": ["file_path_list", "prompt"]}
            ]
        }

    async def _fake_decide_route(
        cls: type[AgentClientUtil],
        *,
        runtime_config: AiChatUtilConfig,
        prompts: Any,
        messages: Any,
        force_coding_agent_route: bool,
        force_deep_agent_route: bool,
        explicit_user_file_paths: Any,
        explicit_user_directory_paths: Any,
        available_tool_names: list[str],
        route_tool_catalog: dict[str, list[str]],
        audit_context: Any,
    ) -> RoutingDecision:
        candidate = RouteCandidate(
            route_name="general_tool_agent",
            score=0.97,
            reason_code="route.local_file_investigation",
            tool_hints=["analyze_files"],
            blocking_issues=[],
        )
        return RoutingDecision(
            selected_route="general_tool_agent",
            candidate_routes=[candidate],
            reason_code="route.local_file_investigation",
            confidence=0.97,
            missing_information=[],
            next_action="execute_selected_route",
            requires_hitl=False,
            requires_clarification=False,
            notes="approval regression test",
        )

    async def _fake_create_workflow(
        cls: type[AgentClientUtil],
        runtime_config: AiChatUtilConfig,
        prompts: Any,
        *,
        checkpointer: Any | None = None,
        tool_limits: ToolLimits | None = None,
        force_coding_agent_route: bool = False,
        force_deep_agent_route: bool = False,
        explicit_user_file_paths: Any = None,
        explicit_user_directory_paths: Any = None,
        approved_tool_names: Any = None,
        routing_decision: RoutingDecision | None = None,
        audit_context: Any | None = None,
        expects_heading_response: bool = False,
        expects_evaluation_response: bool = False,
    ) -> _FakeSupervisorApp:
        response = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "<RESPONSE_TYPE>complete</RESPONSE_TYPE><TEXT>work ディレクトリを確認しました。</TEXT>",
                }
            ]
        }
        return _FakeSupervisorApp([response])

    async def _fake_collect_evidence_results(
        cls: type[AgentClientUtil],
        *,
        app: Any,
        run_trace_id: str,
        workflow_results: Any,
        checkpoint_db_path: Path | None = None,
    ) -> list[Any]:
        return [
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": (
                            "ERROR: tool approval required. error=tool_approval_required tool=analyze_files. "
                            "このツールの実行には承認が必要です。続行する場合は 'APPROVE analyze_files'、拒否する場合は 'REJECT analyze_files' と入力してください。"
                        ),
                    }
                ]
            }
        ]

    async def _fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(agent_client_mod, "create_audit_context", lambda *args, **kwargs: _FakeAuditContext())
    monkeypatch.setattr(AgentClientUtil, "_create_sqlite_checkpointer", classmethod(_fake_create_sqlite_checkpointer))
    monkeypatch.setattr(AgentClientUtil, "resolve_route_tool_inventory", classmethod(_fake_resolve_route_tool_inventory))
    monkeypatch.setattr(AgentClientUtil, "decide_route", classmethod(_fake_decide_route))
    monkeypatch.setattr(AgentClientUtil, "create_workflow", classmethod(_fake_create_workflow))
    monkeypatch.setattr(AgentClientUtil, "collect_evidence_results", classmethod(_fake_collect_evidence_results))
    monkeypatch.setattr(AgentClientUtil, "collect_checkpoint_write_results", classmethod(lambda cls, checkpoint_db_path, run_trace_id: []))
    monkeypatch.setattr(AgentClientUtil, "final_text_contradicts_evidence", classmethod(lambda cls, user_text, evidence: False))
    monkeypatch.setattr(AgentClientUtil, "final_text_missing_concrete_evidence", classmethod(lambda cls, user_text, evidence: False))
    monkeypatch.setattr(agent_client_mod.asyncio, "sleep", _fake_sleep)

    runtime_config = _build_runtime_config()
    runtime_config.features.sufficiency_check_enabled = True
    runtime_config.features.hitl_approval_tools = ["analyze_files"]

    chat_request = ChatRequest(
        chat_history=ChatHistory(
            messages=[
                ChatMessage(
                    role="user",
                    content=[ChatContent(params={"type": "text", "text": "work ディレクトリを確認してください"})],
                )
            ]
        )
    )

    response = asyncio.run(AgentClient(runtime_config).chat(chat_request))

    assert response.status == "paused"
    assert response.hitl is not None
    assert response.hitl.kind == "approval"
    assert response.output == (
        "ツール analyze_files の実行には承認が必要です。\n"
        "続行する場合は 'APPROVE analyze_files'、拒否する場合は 'REJECT analyze_files' と入力してください。"
    )

    sufficiency_event = next(event for event in recorded_events if event["event_type"] == "sufficiency_judged")
    assert sufficiency_event["payload"]["reason_code"] == "sufficiency.approval_required"
    assert sufficiency_event["payload"]["requires_approval"] is True

    hitl_event = next(event for event in recorded_events if event["event_type"] == "hitl_requested")
    assert hitl_event["approval_status"] == "requested"
    assert hitl_event["final_status"] == "paused"

    final_event = next(event for event in recorded_events if event["event_type"] == "final_answer_validated")
    assert final_event["final_status"] == "paused"
    assert final_event["reason_code"] == "sufficiency.approval_required"


def test_deepagent_mcp_client_forces_deep_route_without_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_events: list[dict[str, Any]] = []

    original_llm_client_module = sys.modules.get("ai_chat_util.base.llm.llm_client")
    sys.modules.pop("ai_chat_util.base.agent.mcp_client", None)

    stub_llm_client_module = types.ModuleType("ai_chat_util.base.llm.llm_client")

    class _StubMessageFactoryBase:
        pass

    class _StubMessageFactory(_StubMessageFactoryBase):
        def __init__(self, config: Any) -> None:
            self.config = config

    setattr(stub_llm_client_module, "LLMMessageContentFactoryBase", _StubMessageFactoryBase)
    setattr(stub_llm_client_module, "LLMMessageContentFactory", _StubMessageFactory)
    sys.modules["ai_chat_util.base.llm.llm_client"] = stub_llm_client_module

    llm_mcp_client_mod = importlib.import_module("ai_chat_util.base.agent.mcp_client")
    DeepAgentMCPClient = llm_mcp_client_mod.DeepAgentMCPClient

    if original_llm_client_module is not None:
        sys.modules["ai_chat_util.base.llm.llm_client"] = original_llm_client_module
    else:
        sys.modules.pop("ai_chat_util.base.llm.llm_client", None)

    class _FakeAuditContext:
        def emit(self, event_type: str, **kwargs: Any) -> None:
            recorded_events.append({"event_type": event_type, **kwargs})

    async def _fake_create_sqlite_checkpointer(
        cls: type[AgentClientUtil],
        checkpoint_db_path: Path | None,
        *,
        exit_stack: Any,
    ) -> None:
        return None

    async def _fake_resolve_route_tool_inventory(
        cls: type[AgentClientUtil],
        *,
        runtime_config: AiChatUtilConfig,
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "coding_agent": [{"name": "execute", "description": "run coding job", "primary_args": ["prompt"]}],
            "general_tool_agent": [{"name": "analyze_files", "description": "analyze files", "primary_args": ["file_path_list", "prompt"]}],
            "deep_agent": [
                {"name": "analyze_files", "description": "analyze files", "primary_args": ["file_path_list", "prompt"]},
                {"name": "get_loaded_config_info", "description": "show config", "primary_args": []},
            ],
        }

    async def _fake_decide_route(
        cls: type[AgentClientUtil],
        *,
        runtime_config: AiChatUtilConfig,
        prompts: Any,
        messages: Any,
        force_coding_agent_route: bool,
        force_deep_agent_route: bool,
        explicit_user_file_paths: Any,
        explicit_user_directory_paths: Any,
        available_tool_names: list[str],
        route_tool_catalog: dict[str, list[str]],
        audit_context: Any,
    ) -> RoutingDecision:
        assert force_coding_agent_route is False
        assert force_deep_agent_route is True
        candidate = RouteCandidate(
            route_name="deep_agent",
            score=0.99,
            reason_code="route.multi_step_investigation_needed",
            tool_hints=["analyze_files"],
            blocking_issues=[],
        )
        return RoutingDecision(
            selected_route="deep_agent",
            candidate_routes=[candidate],
            reason_code="route.multi_step_investigation_needed",
            confidence=0.99,
            missing_information=[],
            next_action="execute_selected_route",
            requires_hitl=False,
            requires_clarification=False,
            notes="forced deep-agent test",
        )

    async def _fake_create_deep_agent_workflow(
        cls: type[AgentClientUtil],
        runtime_config: AiChatUtilConfig,
        *,
        checkpointer: Any | None = None,
        tool_limits: ToolLimits | None = None,
        explicit_user_file_paths: Any = None,
        explicit_user_directory_paths: Any = None,
        audit_context: Any | None = None,
    ) -> tuple[_FakeSupervisorApp, list[str]]:
        response = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "<RESPONSE_TYPE>complete</RESPONSE_TYPE><TEXT>forced deep agent response</TEXT>",
                }
            ]
        }
        return _FakeSupervisorApp([response]), ["analyze_files", "get_loaded_config_info"]

    async def _fake_collect_evidence_results(
        cls: type[AgentClientUtil],
        *,
        app: Any,
        run_trace_id: str,
        workflow_results: Any,
        checkpoint_db_path: Path | None = None,
    ) -> list[Any]:
        if isinstance(workflow_results, list):
            return list(workflow_results)
        return [workflow_results]

    async def _fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_mcp_client_mod, "create_audit_context", lambda *args, **kwargs: _FakeAuditContext())
    monkeypatch.setattr(llm_mcp_client_util_mod, "deepagents_available", lambda: True)
    monkeypatch.setattr(AgentClientUtil, "create_llm", classmethod(lambda cls, runtime_config: object()))
    monkeypatch.setattr(AgentClientUtil, "_create_sqlite_checkpointer", classmethod(_fake_create_sqlite_checkpointer))
    monkeypatch.setattr(AgentClientUtil, "resolve_route_tool_inventory", classmethod(_fake_resolve_route_tool_inventory))
    monkeypatch.setattr(AgentClientUtil, "decide_route", classmethod(_fake_decide_route))
    monkeypatch.setattr(AgentClientUtil, "create_deep_agent_workflow", classmethod(_fake_create_deep_agent_workflow))
    monkeypatch.setattr(AgentClientUtil, "collect_evidence_results", classmethod(_fake_collect_evidence_results))
    monkeypatch.setattr(AgentClientUtil, "collect_checkpoint_write_results", classmethod(lambda cls, checkpoint_db_path, run_trace_id: []))
    monkeypatch.setattr(AgentClientUtil, "final_text_contradicts_evidence", classmethod(lambda cls, user_text, evidence: False))
    monkeypatch.setattr(AgentClientUtil, "final_text_missing_concrete_evidence", classmethod(lambda cls, user_text, evidence: False))
    monkeypatch.setattr(llm_mcp_client_mod.asyncio, "sleep", _fake_sleep)

    runtime_config = _build_runtime_config()
    runtime_config.features.enable_deep_agent = True

    chat_request = ChatRequest(
        chat_history=ChatHistory(
            messages=[
                ChatMessage(
                    role="user",
                    content=[ChatContent(params={"type": "text", "text": "設定を確認して状況を整理してください"})],
                )
            ]
        )
    )

    response = asyncio.run(DeepAgentMCPClient(runtime_config).chat(chat_request))

    assert response.status == "completed"
    assert response.output == "forced deep agent response"

    route_decided_event = next(event for event in recorded_events if event["event_type"] == "route_decided")
    assert route_decided_event["route_name"] == "deep_agent"
    assert route_decided_event["payload"]["forced_route"] == "deep_agent"


def test_run_direct_coding_agent_heading_rescue_returns_headings(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeExecuteTool:
        name = "execute"

        async def ainvoke(self, payload: Any) -> Any:
            assert payload["req"]["workspace_path"] == "/tmp/workspace"
            assert "HEADING_LINE_EXACT" in payload["req"]["prompt"]
            return [{"type": "text", "text": '{"task_id":"task-123"}'}]

    class _FakeStatusTool:
        name = "status"

        async def ainvoke(self, payload: Any) -> Any:
            assert payload["task_id"] == "task-123"
            return [{"type": "text", "text": '{"status":"exited","sub_status":"completed"}'}]

    class _FakeResultTool:
        name = "get_result"

        async def ainvoke(self, payload: Any) -> Any:
            assert payload["task_id"] == "task-123"
            assert payload["tail"] is None
            return [
                {
                    "type": "text",
                    "text": '{"stdout":"HEADING_LINE_EXACT: ## 検証目的\\nHEADING_LINE_EXACT: ## 検証手順\\nHEADING_LINE_EXACT: ## 判定基準","stderr":null}',
                }
            ]

    class _FakeCatalogClient:
        def __init__(self, _config: Any) -> None:
            pass

        async def get_tools(self) -> list[Any]:
            return [_FakeExecuteTool(), _FakeStatusTool(), _FakeResultTool()]

    monkeypatch.setattr(llm_mcp_client_util_mod, "MultiServerMCPClient", _FakeCatalogClient)
    runtime_config = _build_runtime_config()
    runtime_config.mcp.working_directory = "/tmp/workspace"
    monkeypatch.setattr(type(runtime_config), "get_mcp_server_config", lambda self: _build_mcp_config("coding-agent"))

    evidence = asyncio.run(
        AgentClientUtil.run_direct_coding_agent_heading_rescue(
            runtime_config=runtime_config,
            messages=[{"role": "user", "content": "作業対象は /tmp/workspace です。必ず coding agent を使って docs/11_検証 配下の Markdown を調査し、共通している見出しを 3 点に整理してください。"}],
            run_trace_id="trace-1",
            requested_heading_count=3,
        )
    )

    assert evidence["headings"] == ["## 検証目的", "## 検証手順", "## 判定基準"]
    assert evidence["latest_task_id"] == "task-123"


def test_extract_config_path_from_text_returns_yaml_path() -> None:
    assert AgentClientUtil.extract_config_path_from_text(
        "設定ファイルのパスは /tmp/ai-chat-util-config.yml です。"
    ) == "/tmp/ai-chat-util-config.yml"


def test_build_recursion_limit_fallback_text_without_evidence_returns_error() -> None:
    text = AgentClientUtil.build_recursion_limit_fallback_text(
        "Recursion limit of 50 reached without hitting a stop condition",
        {"config_path": None, "stdout_blocks": []},
    )

    assert "ERROR: MCPワークフローが再帰上限に到達したため停止しました。" in text
    assert "Recursion limit of 50 reached without hitting a stop condition" in text
