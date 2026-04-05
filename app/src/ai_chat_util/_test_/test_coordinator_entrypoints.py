from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ai_chat_util.api.api_server import router
from ai_chat_util.base.agent.agent_client_factory import AgentFactory
from ai_chat_util.cli.__main__ import build_parser
from ai_chat_util.common.config.runtime import AiChatUtilConfig, FeaturesSection, LLMSection, LoggingSection, MCPSection, NetworkSection, Office2PDFSection
from ai_chat_util.common.model.ai_chatl_util_models import ChatContent, ChatHistory, ChatMessage, ChatRequest, ChatRequestContext, ChatResponse
from ai_chat_util.core.app import run_coordinated_chat
import ai_chat_util.mcp.mcp_server as mcp_server_mod


def _build_chat_request(
    text: str = "test prompt",
    *,
    context: ChatRequestContext | None = None,
    trace_id: str | None = None,
) -> ChatRequest:
    return ChatRequest(
        trace_id=trace_id,
        chat_history=ChatHistory(
            messages=[
                ChatMessage(
                    role="user",
                    content=[ChatContent(params={"type": "text", "text": text})],
                )
            ]
        ),
        chat_request_context=context or ChatRequestContext(),
    )


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


def test_run_coordinated_chat_selects_workflow_when_definition_is_present(monkeypatch) -> None:
    runtime_config = _build_runtime_config()
    runtime_config.features.type_selection_mode = "deterministic"

    captured: dict[str, Any] = {}

    class _FakeWorkflowChatClient:
        def __init__(
            self,
            workflow_file_path: str,
            *,
            runtime_config: AiChatUtilConfig | None = None,
            max_node_visits: int = 8,
            plan_mode: bool = False,
            durable: bool = True,
            session_store: Any | None = None,
        ) -> None:
            captured["workflow_file_path"] = workflow_file_path
            captured["max_node_visits"] = max_node_visits
            captured["plan_mode"] = plan_mode
            captured["durable"] = durable

        async def chat(self, chat_request: ChatRequest, **kwargs: Any) -> ChatResponse:
            captured["trace_id"] = chat_request.trace_id
            return ChatResponse(status="completed", trace_id=chat_request.trace_id, messages=[])

    monkeypatch.setattr("ai_chat_util.core.app.get_runtime_config", lambda: runtime_config)
    monkeypatch.setattr("ai_chat_util.core.app.WorkflowChatClient", _FakeWorkflowChatClient)

    response = asyncio.run(
        run_coordinated_chat(
            _build_chat_request(
                "この定義済み workflow を実行してください",
                context=ChatRequestContext(
                    workflow_file_path="/tmp/sample-workflow.md",
                    workflow_plan_mode=True,
                    workflow_durable=False,
                    workflow_max_node_visits=5,
                ),
            )
        )
    )

    assert response.status == "completed"
    assert captured["workflow_file_path"] == "/tmp/sample-workflow.md"
    assert captured["max_node_visits"] == 5
    assert captured["plan_mode"] is True
    assert captured["durable"] is False
    assert len(captured["trace_id"] or "") == 32


def test_run_coordinated_chat_selects_coding_agent_for_explicit_request(monkeypatch) -> None:
    runtime_config = _build_runtime_config()
    runtime_config.features.type_selection_mode = "deterministic"

    called: dict[str, int] = {"coding": 0, "supervisor": 0, "deep": 0}

    class _FakeClient:
        async def chat(self, chat_request: ChatRequest, **kwargs: Any) -> ChatResponse:
            return ChatResponse(status="completed", trace_id=chat_request.trace_id, messages=[])

    monkeypatch.setattr("ai_chat_util.core.app.get_runtime_config", lambda: runtime_config)
    monkeypatch.setattr(AgentFactory, "create_codingagent_client", lambda llm_config=None: called.__setitem__("coding", called["coding"] + 1) or _FakeClient())
    monkeypatch.setattr(AgentFactory, "create_mcp_client", lambda llm_config=None: called.__setitem__("supervisor", called["supervisor"] + 1) or _FakeClient())
    monkeypatch.setattr(AgentFactory, "create_deepagent_client", lambda llm_config=None: called.__setitem__("deep", called["deep"] + 1) or _FakeClient())

    response = asyncio.run(run_coordinated_chat(_build_chat_request("coding agent を使って調査してください")))

    assert response.status == "completed"
    assert called == {"coding": 1, "supervisor": 0, "deep": 0}


def test_run_coordinated_chat_writes_cross_type_audit_event(monkeypatch, tmp_path: Path) -> None:
    audit_path = tmp_path / "cross_type_audit.jsonl"
    runtime_config = _build_runtime_config()
    runtime_config.features.type_selection_mode = "deterministic"
    runtime_config.features.audit_log_enabled = True
    runtime_config.features.audit_log_path = str(audit_path)

    class _FakeClient:
        async def chat(self, chat_request: ChatRequest, **kwargs: Any) -> ChatResponse:
            return ChatResponse(status="completed", trace_id=chat_request.trace_id, messages=[])

    monkeypatch.setattr("ai_chat_util.core.app.get_runtime_config", lambda: runtime_config)
    monkeypatch.setattr(AgentFactory, "create_mcp_client", lambda llm_config=None: _FakeClient())

    response = asyncio.run(
        run_coordinated_chat(
            _build_chat_request(
                "設定だけ確認してください",
                trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            )
        )
    )

    assert response.status == "completed"
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    cross_type_event = next(event for event in events if event["event_type"] == "cross_type_route_decided")
    assert cross_type_event["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert cross_type_event["route_name"] == "supervisor"
    assert cross_type_event["payload"]["policy_snapshot"]["type_selection_mode"] == "deterministic"


def test_api_router_registers_coordinated_chat_route() -> None:
    route_paths = {route.path for route in router.routes}

    assert "/coordinated_chat" in route_paths


def test_cli_parser_accepts_coordinated_chat_command() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "coordinated_chat",
        "-p",
        "型選択してください",
        "--workflow-file",
        "sample.md",
        "--predictability",
        "high",
        "--exploration-level",
        "medium",
    ])

    assert args.command == "coordinated_chat"
    assert args.prompt == "型選択してください"
    assert args.workflow_file == "sample.md"
    assert args.predictability == "high"
    assert args.exploration_level == "medium"


def test_prepare_mcp_registers_run_coordinated_tool() -> None:
    registered_names: list[str] = []

    class _FakeMCP:
        def tool(self):
            def _decorator(func):
                registered_names.append(func.__name__)
                return func

            return _decorator

    mcp_server_mod.prepare_mcp(_FakeMCP(), "run_coordinated_chat")

    assert registered_names == ["run_coordinated_chat"]


def test_run_coordinated_chat_returns_clarification_for_ambiguous_cross_type(monkeypatch) -> None:
    runtime_config = _build_runtime_config()
    runtime_config.features.type_selection_mode = "deterministic"
    runtime_config.features.type_selection_ambiguity_gap = 0.1

    monkeypatch.setattr("ai_chat_util.core.app.get_runtime_config", lambda: runtime_config)

    response = asyncio.run(
        run_coordinated_chat(
            _build_chat_request(
                "この要求を適切な型で処理してください",
                context=ChatRequestContext(
                    workflow_file_path="/tmp/sample-workflow.md",
                    exploration_level="high",
                ),
            )
        )
    )

    assert response.status == "paused"
    assert response.hitl is not None
    assert response.hitl.source == "coordinator:routing"
    assert "型選択が競合しています" in response.output
