from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_chat_util.api.api_server import router
from ai_chat_util.base.agent.agent_client import AgentClient
from ai_chat_util.base.agent.agent_client_util import AgentClientUtil
from ai_chat_util.base.agent.prompts import CodingAgentPrompts
from ai_chat_util.base.agent.supervisor_support import RouteCandidate, RoutingDecision
from ai_chat_util.cli.__main__ import build_parser
from ai_chat_util.common.config.runtime import AiChatUtilConfig, FeaturesSection, LLMSection, LoggingSection, MCPSection, NetworkSection, Office2PDFSection
from ai_chat_util.common.model.ai_chatl_util_models import ChatContent, ChatHistory, ChatMessage, ChatRequest, ChatRequestContext, ChatResponse
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
        chat_request_context=context,
    )


def _build_runtime_config() -> AiChatUtilConfig:
    runtime_config = AiChatUtilConfig.model_construct(
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
    runtime_config.features.routing_mode = "legacy"
    runtime_config.features.type_selection_mode = "deterministic"
    return runtime_config


def test_default_routing_selects_workflow_backend_when_workflow_file_is_present() -> None:
    runtime_config = _build_runtime_config()

    decision = AgentClientUtil._build_default_routing_decision(
        runtime_config=runtime_config,
        force_coding_agent_route=False,
        force_deep_agent_route=False,
        deep_agent_enabled=False,
        workflow_file_path="/tmp/sample-workflow.md",
        predictability="high",
        approval_frequency=None,
        exploration_level=None,
        has_side_effects=None,
        available_tool_names=[],
    )

    assert decision.selected_route == "workflow_backend"
    assert decision.reason_code == "route.workflow_definition_available"


def test_agent_client_dispatches_to_workflow_backend(monkeypatch) -> None:
    runtime_config = _build_runtime_config()
    captured: dict[str, Any] = {}

    async def _fake_resolve_route_tool_inventory(*, runtime_config: AiChatUtilConfig) -> dict[str, list[dict[str, Any]]]:
        return {}

    async def _fake_decide_route(**kwargs: Any) -> RoutingDecision:
        return RoutingDecision(
            selected_route="workflow_backend",
            candidate_routes=[
                RouteCandidate(
                    route_name="workflow_backend",
                    score=0.95,
                    reason_code="route.workflow_definition_available",
                    tool_hints=["/tmp/sample-workflow.md"],
                    blocking_issues=[],
                )
            ],
            reason_code="route.workflow_definition_available",
            confidence=0.95,
            next_action="execute_selected_route",
            missing_information=[],
            requires_hitl=False,
            requires_clarification=False,
            notes="workflow backend selected",
        )

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
            return ChatResponse(status="completed", trace_id=chat_request.trace_id, hitl=None, messages=[])

    monkeypatch.setattr(AgentClientUtil, "resolve_route_tool_inventory", _fake_resolve_route_tool_inventory)
    monkeypatch.setattr(AgentClientUtil, "decide_route", _fake_decide_route)
    monkeypatch.setattr("ai_chat_util.base.agent.agent_client.WorkflowChatClient", _FakeWorkflowChatClient)

    client = AgentClient(runtime_config)
    response = asyncio.run(
        client.chat(
            _build_chat_request(
                "この workflow を実行してください",
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


def test_api_router_does_not_register_coordinated_chat_route() -> None:
    route_paths = {route.path for route in router.routes}

    assert "/agent_chat" in route_paths
    assert "/coordinated_chat" not in route_paths


def test_cli_parser_accepts_agent_chat_workflow_options() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "agent_chat",
        "-p",
        "workflow を実行してください",
        "--workflow-file",
        "sample.md",
        "--predictability",
        "high",
        "--exploration-level",
        "medium",
    ])

    assert args.command == "agent_chat"
    assert args.workflow_file == "sample.md"
    assert args.predictability == "high"
    assert args.exploration_level == "medium"


def test_cli_parser_rejects_coordinated_chat_command() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["coordinated_chat", "-p", "obsolete"])


def test_prepare_mcp_rejects_removed_run_coordinated_tool() -> None:
    class _FakeMCP:
        def tool(self):
            def _decorator(func):
                return func

            return _decorator

    with pytest.raises(ValueError, match="Unknown tool"):
        mcp_server_mod.prepare_mcp(_FakeMCP(), "run_coordinated_chat")