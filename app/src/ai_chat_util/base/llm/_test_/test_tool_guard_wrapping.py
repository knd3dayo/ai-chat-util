from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langgraph.errors import GraphRecursionError

fake_client_module = types.ModuleType("langchain_mcp_adapters.client")
fake_client_module.MultiServerMCPClient = object
sys.modules.setdefault("langchain_mcp_adapters.client", fake_client_module)

fake_sessions_module = types.ModuleType("langchain_mcp_adapters.sessions")
fake_sessions_module.Connection = dict[str, Any]
sys.modules.setdefault("langchain_mcp_adapters.sessions", fake_sessions_module)

import ai_chat_util.base.llm.agent as agent_mod
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
    assert isinstance(artifact, dict)
    assert artifact.get("error") == "tool_invocation_failed"
    assert artifact.get("tool") == "sync_error"
    assert state["used"] == 1


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


def test_invalid_followup_task_id_is_blocked_after_first_404() -> None:
    state: dict[str, Any] = {
        "used": 0,
        "general_used": 0,
        "followup_used": 0,
        "followup_limit": 8,
    }
    status_calls = {"n": 0}

    execute_tool = _FakeTool("execute", response_format="content", func=lambda: {"task_id": "task-valid"})

    def _status(task_id: str) -> str:
        status_calls["n"] += 1
        raise HTTPException(status_code=404, detail="Task not found")

    status_tool = _FakeTool("status", response_format="content", func=_status)

    MCPClientUtil._apply_tool_execution_guards(
        [execute_tool, status_tool],
        tool_call_state=state,
        tool_call_limit_int=4,
        tool_timeout_seconds_f=0.0,
        tool_timeout_retries_int=0,
    )

    execute_result = execute_tool.func()
    assert execute_result == {"task_id": "task-valid"}

    out1 = status_tool.func("task-missing")
    out2 = status_tool.func("task-missing")

    assert isinstance(out1, str)
    assert "invalid_followup_task_id" in out1
    assert "task-missing" in out1
    assert "task-valid" in out1
    assert isinstance(out2, str)
    assert "invalid_followup_task_id" in out2
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


def _patch_agent_creation(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    created: list[tuple[str, str]] = []

    monkeypatch.setattr(agent_mod, "MultiServerMCPClient", _FakeMCPClient)

    def _fake_create_agent(_llm: Any, _tools: list[Any], *, system_prompt: str, name: str) -> _FakeLangGraphAgent:
        created.append((name, system_prompt))
        return _FakeLangGraphAgent(name=name, system_prompt=system_prompt)

    monkeypatch.setattr(agent_mod, "create_agent", _fake_create_agent)
    return created


def test_create_sub_agents_assigns_unique_names_for_mixed_mcp_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_agent_creation(monkeypatch)

    agents = asyncio.run(
        AgentBuilder.create_sub_agents(
            runtime_config=_build_runtime_config(),
            mcp_config=_build_mcp_config("coding-agent", "general-tools"),
            llm=object(),
            prompts=CodingAgentPrompts(),
            tool_limits=None,
        )
    )

    assert [name for name, _ in created] == ["tool_agent_coding", "tool_agent_general"]
    assert [agent.get_agent_name() for agent in agents] == ["tool_agent_coding", "tool_agent_general"]
    assert len({agent.get_agent().name for agent in agents}) == 2
    assert "tool_agent_coding" in created[0][1]
    assert "tool_agent_general" in created[1][1]


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
            llm=object(),
            prompts=CodingAgentPrompts(),
            tool_limits=None,
        )
    )

    assert [name for name, _ in created] == [expected_name]
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


def test_evidence_reflection_overrides_negative_final_text() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "stdout_blocks": ["# Overview\n## Setup\n## Troubleshooting"],
        "headings": ["# Overview", "## Setup", "## Troubleshooting"],
    }

    assert MCPClientUtil.final_text_contradicts_evidence(
        "get_loaded_config_info の正しい情報が取得できなかった。重要な見出しを確認できなかった。",
        evidence,
    )

    fallback = MCPClientUtil.build_evidence_reflected_final_text(evidence)

    assert "設定ファイルの場所: /tmp/ai-chat-util-config.yml" in fallback
    assert "- # Overview" in fallback
    assert "- ## Setup" in fallback
    assert "- ## Troubleshooting" in fallback


def test_final_text_missing_concrete_evidence_detects_missing_path_and_headings() -> None:
    evidence = {
        "config_path": "/tmp/ai-chat-util-config.yml",
        "stdout_blocks": ["# Overview\n## Setup\n## Troubleshooting"],
        "headings": ["# Overview", "## Setup", "## Troubleshooting"],
    }

    assert MCPClientUtil.final_text_missing_concrete_evidence(
        "get_loaded_config_info を使って確認しました。重要な見出しは概要、設定、トラブルシュートです。",
        evidence,
    )


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


def test_build_recursion_limit_fallback_text_prefers_evidence() -> None:
    text = MCPClientUtil.build_recursion_limit_fallback_text(
        "Recursion limit of 50 reached without hitting a stop condition",
        {
            "config_path": "/tmp/ai-chat-util-config.yml",
            "stdout_blocks": ["# Overview\n## Setup\n## Troubleshooting"],
            "headings": ["# Overview", "## Setup", "## Troubleshooting"],
        },
    )

    assert "再帰上限に到達" in text
    assert "/tmp/ai-chat-util-config.yml" in text
    assert "# Overview" in text
    assert "## Setup" in text
    assert "## Troubleshooting" in text


def test_build_recursion_limit_fallback_text_without_evidence_returns_error() -> None:
    text = MCPClientUtil.build_recursion_limit_fallback_text(
        "Recursion limit of 50 reached without hitting a stop condition",
        {"config_path": None, "stdout_blocks": []},
    )

    assert "ERROR: MCPワークフローが再帰上限に到達したため停止しました。" in text
    assert "Recursion limit of 50 reached without hitting a stop condition" in text
