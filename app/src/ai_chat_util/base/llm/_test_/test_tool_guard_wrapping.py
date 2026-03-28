from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

fake_client_module = types.ModuleType("langchain_mcp_adapters.client")
fake_client_module.MultiServerMCPClient = object
sys.modules.setdefault("langchain_mcp_adapters.client", fake_client_module)

fake_sessions_module = types.ModuleType("langchain_mcp_adapters.sessions")
fake_sessions_module.Connection = dict[str, Any]
sys.modules.setdefault("langchain_mcp_adapters.sessions", fake_sessions_module)

import ai_chat_util.base.llm.agent as agent_mod
from ai_chat_util.base.llm.agent import AgentBuilder
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


class _FakeMCPClient:
    def __init__(self, _config: Any) -> None:
        self._config = _config

    async def get_tools(self) -> list[Any]:
        return []


class _FakeLangGraphAgent:
    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.system_prompt = system_prompt


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
