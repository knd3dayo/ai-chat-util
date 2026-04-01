from __future__ import annotations

import asyncio
from typing import Any

from ai_chat_util.api.api_server import router
from ai_chat_util.base.agent.agent_client_factory import AgentFactory
from ai_chat_util.core.app import run_deepagent_chat, run_deepagent_batch_chat, run_deepagent_batch_chat_from_excel
from ai_chat_util.cli.__main__ import build_parser
from ai_chat_util.common.model.ai_chatl_util_models import ChatContent, ChatHistory, ChatMessage, ChatRequest, ChatResponse
import ai_chat_util.mcp.mcp_server as mcp_server_mod


def _build_chat_request(text: str = "test prompt") -> ChatRequest:
    return ChatRequest(
        chat_history=ChatHistory(
            messages=[
                ChatMessage(
                    role="user",
                    content=[ChatContent(params={"type": "text", "text": text})],
                )
            ]
        )
    )


def test_run_deepagent_chat_uses_deepagent_factory(monkeypatch) -> None:
    called: dict[str, Any] = {"factory": 0}

    class _FakeClient:
        async def chat(self, chat_request: ChatRequest, **kwargs: Any) -> Any:
            return ChatResponse(status="completed", messages=[])

    def _fake_create_deepagent_client() -> _FakeClient:
        called["factory"] += 1
        return _FakeClient()

    monkeypatch.setattr(AgentFactory, "create_deepagent_client", _fake_create_deepagent_client)

    response = asyncio.run(run_deepagent_chat(_build_chat_request()))

    assert called["factory"] == 1
    assert response.status == "completed"


def test_cli_parser_accepts_run_deepagent_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["run_deepagent_chat", "-p", "inspect config"])

    assert args.command == "run_deepagent_chat"
    assert args.prompt == "inspect config"


def test_run_deepagent_batch_chat_uses_deepagent_batch_client(monkeypatch) -> None:
    called: dict[str, Any] = {"init": 0, "requests": None, "concurrency": None}

    class _FakeBatchClient:
        def __init__(self) -> None:
            called["init"] += 1

        async def run_batch_chat(self, chat_requests: list[ChatRequest], concurrency: int) -> list[tuple[int, ChatResponse]]:
            called["requests"] = chat_requests
            called["concurrency"] = concurrency
            return [(i, ChatResponse(status="completed", messages=[])) for i, _ in enumerate(chat_requests)]

    monkeypatch.setattr("ai_chat_util.core.app.DeepAgentBatchClient", _FakeBatchClient)

    response = asyncio.run(run_deepagent_batch_chat([_build_chat_request("a"), _build_chat_request("b")], concurrency=3))

    assert called["init"] == 1
    assert len(called["requests"]) == 2
    assert called["concurrency"] == 3
    assert len(response) == 2
    assert all(item.status == "completed" for item in response)


def test_run_deepagent_batch_chat_from_excel_uses_deepagent_batch_client(monkeypatch) -> None:
    called: dict[str, Any] = {"init": 0, "args": None}

    class _FakeBatchClient:
        def __init__(self) -> None:
            called["init"] += 1

        async def run_batch_chat_from_excel(self, *args: Any) -> None:
            called["args"] = args

    monkeypatch.setattr("ai_chat_util.core.app.DeepAgentBatchClient", _FakeBatchClient)

    asyncio.run(
        run_deepagent_batch_chat_from_excel(
            prompt="inspect",
            input_excel_path="input.xlsx",
            output_excel_path="output.xlsx",
            content_column="content",
            file_path_column="file_path",
            output_column="output",
            detail="auto",
            concurrency=7,
        )
    )

    assert called["init"] == 1
    assert called["args"] == ("inspect", "input.xlsx", "output.xlsx", "content", "file_path", "output", "auto", 7)


def test_cli_parser_accepts_run_deepagent_batch_chat_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["run_deepagent_batch_chat", "-p", "inspect config", "-i", "input.xlsx"])

    assert args.command == "run_deepagent_batch_chat"
    assert args.prompt == "inspect config"
    assert args.input_excel_path == "input.xlsx"


def test_cli_parser_accepts_deepagent_batch_chat_alias() -> None:
    parser = build_parser()
    args = parser.parse_args(["deepagent_batch_chat", "-p", "inspect config", "-i", "input.xlsx"])

    assert args.command == "deepagent_batch_chat"
    assert args.prompt == "inspect config"
    assert args.input_excel_path == "input.xlsx"


def test_api_router_registers_run_deepagent_route() -> None:
    route_paths = {route.path for route in router.routes}

    assert "/run_deepagent_chat" in route_paths
    assert "/run_deepagent_batch_chat" in route_paths
    assert "/deepagent_batch_chat" in route_paths
    assert "/run_deepagent_batch_chat_from_excel" in route_paths
    assert "/deepagent_batch_chat_from_excel" in route_paths


def test_prepare_mcp_registers_run_deepagent_tool() -> None:
    registered_names: list[str] = []

    class _FakeMCP:
        def tool(self):
            def _decorator(func):
                registered_names.append(func.__name__)
                return func

            return _decorator

    mcp_server_mod.prepare_mcp(_FakeMCP(), "run_deepagent_chat")

    assert registered_names == ["run_deepagent_chat"]


def test_prepare_mcp_registers_run_deepagent_batch_tools() -> None:
    registered_names: list[str] = []

    class _FakeMCP:
        def tool(self):
            def _decorator(func):
                registered_names.append(func.__name__)
                return func

            return _decorator

    mcp_server_mod.prepare_mcp(_FakeMCP(), "run_deepagent_batch_chat,run_deepagent_batch_chat_from_excel")

    assert registered_names == ["run_deepagent_batch_chat", "run_deepagent_batch_chat_from_excel"]


def test_prepare_mcp_registers_deepagent_batch_alias_tools() -> None:
    registered_names: list[str] = []

    class _FakeMCP:
        def tool(self):
            def _decorator(func):
                registered_names.append(func.__name__)
                return func

            return _decorator

    mcp_server_mod.prepare_mcp(_FakeMCP(), "deepagent_batch_chat,deepagent_batch_chat_from_excel")

    assert registered_names == ["deepagent_batch_chat", "deepagent_batch_chat_from_excel"]