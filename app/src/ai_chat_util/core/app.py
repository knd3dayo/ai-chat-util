from pathlib import Path
from typing import Annotated, Any, Literal

import uuid

from pydantic import Field

from ai_chat_util.base.agent import DeepAgentBatchClient, MCPBatchClient
from ai_chat_util.base.agent.agent_client_util import AgentClientUtil
from ai_chat_util.base.agent.agent_client_factory import AgentFactory
from ai_chat_util.base.agent.supervisor_support import RouteCandidate, RoutingDecision, create_audit_context
from ai_chat_util.base.batch import BatchClient
from ai_chat_util.base.chat import AbstractChatClient, LLMMessageContentFactory, LLMMessageContentFactoryBase, create_llm_client
from ai_chat_util.common.config.runtime import AiChatUtilConfig, get_runtime_config
from ai_chat_util.common.model.ai_chatl_util_models import ChatContent, ChatHistory, ChatMessage, ChatRequest, ChatRequestContext, ChatResponse, HitlRequest, WebRequestModel
from ai_chat_util.common.model.request_headers import get_current_request_headers
from ai_chat_util.workflow import WorkflowExecutionResponse, WorkflowSessionStore, execute_workflow_markdown
from ai_chat_util.workflow.chat_client import WorkflowChatClient


_CROSS_TYPE_ROUTE_WORKFLOW = "workflow"
_CROSS_TYPE_ROUTE_SUPERVISOR = "supervisor"
_CROSS_TYPE_ROUTE_AUTONOMOUS = "autonomous"


def _resolve_chat_trace_id(chat_request: ChatRequest) -> str:
    normalized = str(chat_request.trace_id or "").strip().lower()
    if normalized:
        return normalized
    headers = get_current_request_headers()
    header_trace_id = str(getattr(headers, "trace_id", "") or "").strip().lower()
    if header_trace_id:
        return header_trace_id
    return uuid.uuid4().hex


def _extract_chat_messages_as_payload(chat_request: ChatRequest) -> list[dict[str, Any]]:
    return [message.model_dump() for message in chat_request.chat_history.messages]


def _resolve_autonomous_backend(
        runtime_config: AiChatUtilConfig,
        *,
        explicit_coding_request: bool,
        explicit_deep_request: bool,
) -> str:
    if explicit_deep_request and bool(getattr(runtime_config.features, "enable_deep_agent", False)):
        return "deep_agent"
    if explicit_coding_request:
        return "coding_agent"
    preferred_route = str(getattr(runtime_config.features, "preferred_coding_route", "coding_agent") or "coding_agent").strip().lower()
    if preferred_route == "deep_agent" and bool(getattr(runtime_config.features, "enable_deep_agent", False)):
        return "deep_agent"
    return "coding_agent"


def _build_cross_type_policy_snapshot(runtime_config: AiChatUtilConfig) -> dict[str, Any]:
    features = runtime_config.features
    return {
        "type_selection_mode": getattr(features, "type_selection_mode", "disabled"),
        "type_selection_default_route": getattr(features, "type_selection_default_route", "supervisor"),
        "type_selection_workflow_requires_definition": getattr(features, "type_selection_workflow_requires_definition", True),
        "type_selection_prefer_workflow_when_definition_available": getattr(features, "type_selection_prefer_workflow_when_definition_available", True),
        "type_selection_workflow_on_high_predictability": getattr(features, "type_selection_workflow_on_high_predictability", True),
        "type_selection_workflow_on_high_approval_frequency": getattr(features, "type_selection_workflow_on_high_approval_frequency", True),
        "type_selection_workflow_on_side_effects": getattr(features, "type_selection_workflow_on_side_effects", True),
        "type_selection_autonomous_on_explicit_coding_request": getattr(features, "type_selection_autonomous_on_explicit_coding_request", True),
        "type_selection_autonomous_on_explicit_deep_request": getattr(features, "type_selection_autonomous_on_explicit_deep_request", True),
        "type_selection_autonomous_on_high_exploration": getattr(features, "type_selection_autonomous_on_high_exploration", True),
        "type_selection_require_clarification_on_missing_workflow_definition": getattr(features, "type_selection_require_clarification_on_missing_workflow_definition", True),
        "type_selection_ambiguity_gap": getattr(features, "type_selection_ambiguity_gap", 0.1),
        "preferred_coding_route": getattr(features, "preferred_coding_route", "coding_agent"),
        "enable_deep_agent": getattr(features, "enable_deep_agent", False),
    }


def _merge_chat_request_context(
        current: ChatRequestContext | None,
        fallback: ChatRequestContext | None,
) -> ChatRequestContext | None:
    if current is not None:
        return current
    if fallback is None:
        return None
    return fallback.model_copy(deep=True)


def _decide_cross_type_route(
        chat_request: ChatRequest,
        runtime_config: AiChatUtilConfig,
) -> tuple[RoutingDecision, str | None, dict[str, Any]]:
    features = runtime_config.features
    request_context = chat_request.chat_request_context
    messages_payload = _extract_chat_messages_as_payload(chat_request)
    workflow_file_path = str(getattr(request_context, "workflow_file_path", "") or "").strip() or None
    predictability = str(getattr(request_context, "predictability", "") or "").strip().lower() or None
    approval_frequency = str(getattr(request_context, "approval_frequency", "") or "").strip().lower() or None
    exploration_level = str(getattr(request_context, "exploration_level", "") or "").strip().lower() or None
    has_side_effects = getattr(request_context, "has_side_effects", None)
    explicit_coding_request = AgentClientUtil.explicitly_requests_coding_agent(messages_payload)
    explicit_deep_request = AgentClientUtil.explicitly_requests_deep_agent(messages_payload)
    autonomous_backend = _resolve_autonomous_backend(
        runtime_config,
        explicit_coding_request=explicit_coding_request,
        explicit_deep_request=explicit_deep_request,
    )

    request_hints = {
        "workflow_file_path": workflow_file_path,
        "predictability": predictability,
        "approval_frequency": approval_frequency,
        "exploration_level": exploration_level,
        "has_side_effects": has_side_effects,
        "explicit_coding_request": explicit_coding_request,
        "explicit_deep_request": explicit_deep_request,
    }

    selection_mode = str(getattr(features, "type_selection_mode", "disabled") or "disabled").strip().lower()
    if selection_mode != "deterministic":
        selected_route = str(getattr(features, "type_selection_default_route", _CROSS_TYPE_ROUTE_SUPERVISOR) or _CROSS_TYPE_ROUTE_SUPERVISOR).strip().lower()
        if selected_route not in {_CROSS_TYPE_ROUTE_SUPERVISOR, _CROSS_TYPE_ROUTE_AUTONOMOUS}:
            selected_route = _CROSS_TYPE_ROUTE_SUPERVISOR
        selected_backend = autonomous_backend if selected_route == _CROSS_TYPE_ROUTE_AUTONOMOUS else None
        candidate = RouteCandidate(
            route_name=selected_route,
            score=0.5,
            reason_code="cross_type.disabled_default_route",
            tool_hints=[selected_backend] if selected_backend else [],
        )
        return (
            RoutingDecision(
                selected_route=selected_route,
                candidate_routes=[candidate],
                reason_code="cross_type.disabled_default_route",
                confidence=0.5,
                next_action="execute_selected_route",
                notes="type selection is disabled; default route was used",
            ),
            selected_backend,
            request_hints,
        )

    candidates: list[RouteCandidate] = []
    if explicit_deep_request and bool(getattr(features, "type_selection_autonomous_on_explicit_deep_request", True)):
        candidates.append(
            RouteCandidate(
                route_name=_CROSS_TYPE_ROUTE_AUTONOMOUS,
                score=1.0,
                reason_code="cross_type.explicit_deep_agent_request",
                tool_hints=["deep_agent"],
            )
        )
    elif explicit_coding_request and bool(getattr(features, "type_selection_autonomous_on_explicit_coding_request", True)):
        candidates.append(
            RouteCandidate(
                route_name=_CROSS_TYPE_ROUTE_AUTONOMOUS,
                score=1.0,
                reason_code="cross_type.explicit_coding_agent_request",
                tool_hints=["coding_agent"],
            )
        )

    workflow_reasons: list[str] = []
    workflow_score = 0.0
    workflow_requires_definition = bool(getattr(features, "type_selection_workflow_requires_definition", True))
    workflow_eligible = bool(workflow_file_path) or not workflow_requires_definition
    if workflow_eligible and workflow_file_path and bool(getattr(features, "type_selection_prefer_workflow_when_definition_available", True)):
        workflow_score = max(workflow_score, 0.95)
        workflow_reasons.append("workflow_definition_available")
    if workflow_eligible and predictability == "high" and bool(getattr(features, "type_selection_workflow_on_high_predictability", True)):
        workflow_score = max(workflow_score, 0.9)
        workflow_reasons.append("high_predictability")
    if workflow_eligible and approval_frequency == "high" and bool(getattr(features, "type_selection_workflow_on_high_approval_frequency", True)):
        workflow_score = max(workflow_score, 0.88)
        workflow_reasons.append("high_approval_frequency")
    if workflow_eligible and has_side_effects is True and bool(getattr(features, "type_selection_workflow_on_side_effects", True)):
        workflow_score = max(workflow_score, 0.92)
        workflow_reasons.append("side_effects_present")
    if workflow_score > 0.0:
        candidates.append(
            RouteCandidate(
                route_name=_CROSS_TYPE_ROUTE_WORKFLOW,
                score=workflow_score,
                reason_code="cross_type.workflow_candidate",
                tool_hints=[workflow_file_path] if workflow_file_path else [],
                blocking_issues=[] if workflow_file_path or not workflow_requires_definition else ["workflow_definition_required"],
            )
        )

    if exploration_level == "high" and bool(getattr(features, "type_selection_autonomous_on_high_exploration", True)):
        candidates.append(
            RouteCandidate(
                route_name=_CROSS_TYPE_ROUTE_AUTONOMOUS,
                score=0.87,
                reason_code="cross_type.high_exploration",
                tool_hints=[autonomous_backend],
            )
        )

    candidates.append(
        RouteCandidate(
            route_name=_CROSS_TYPE_ROUTE_SUPERVISOR,
            score=0.6,
            reason_code="cross_type.supervisor_default",
            tool_hints=["structured_routing"],
        )
    )
    candidates.sort(key=lambda item: item.score, reverse=True)
    selected_candidate = candidates[0]
    selected_backend = autonomous_backend if selected_candidate.route_name == _CROSS_TYPE_ROUTE_AUTONOMOUS else None

    notes: list[str] = []
    if workflow_reasons:
        notes.append("workflow=" + ",".join(workflow_reasons))
    if explicit_coding_request:
        notes.append("explicit_coding_request=true")
    if explicit_deep_request:
        notes.append("explicit_deep_request=true")
    if exploration_level:
        notes.append(f"exploration_level={exploration_level}")

    missing_information: list[str] = []
    requires_clarification = False
    requires_hitl = False
    ambiguity_gap = float(getattr(features, "type_selection_ambiguity_gap", 0.1) or 0.1)
    if (
        workflow_score <= 0.0
        and workflow_requires_definition
        and bool(getattr(features, "type_selection_require_clarification_on_missing_workflow_definition", True))
        and (
            predictability == "high"
            or approval_frequency == "high"
            or has_side_effects is True
        )
    ):
        requires_clarification = True
        requires_hitl = True
        missing_information.append("WF型が適する可能性があります。workflow_file_path を指定するか、SV型/自律型で続行してよいか指定してください。")

    if len(candidates) >= 2:
        score_gap = candidates[0].score - candidates[1].score
        if score_gap < ambiguity_gap:
            requires_clarification = True
            requires_hitl = True
            missing_information.append(
                f"型選択が競合しています。{candidates[0].route_name} と {candidates[1].route_name} のどちらを優先するか指定してください。"
            )

    return (
        RoutingDecision(
            selected_route=selected_candidate.route_name,
            candidate_routes=candidates,
            reason_code=selected_candidate.reason_code,
            confidence=selected_candidate.score,
            next_action="execute_selected_route",
            missing_information=missing_information,
            requires_hitl=requires_hitl,
            requires_clarification=requires_clarification,
            notes="; ".join(notes) if notes else None,
        ),
        selected_backend,
        request_hints,
    )


class CoordinatorChatClient(AbstractChatClient):
    def __init__(
            self,
            *,
            runtime_config: AiChatUtilConfig | None = None,
            default_request_context: ChatRequestContext | None = None,
    ) -> None:
        self.runtime_config = runtime_config or get_runtime_config()
        self.message_factory = LLMMessageContentFactory(config=self.runtime_config)
        self.default_request_context = default_request_context.model_copy(deep=True) if default_request_context is not None else None

    async def simple_chat(self, prompt: str) -> str:
        request = ChatRequest(
            chat_history=ChatHistory(
                messages=[
                    ChatMessage(
                        role="user",
                        content=[ChatContent(params={"type": "text", "text": prompt})],
                    )
                ]
            ),
            chat_request_context=self.default_request_context.model_copy(deep=True) if self.default_request_context is not None else None,
        )
        response = await self.chat(request)
        return response.output

    async def chat(self, chat_request: ChatRequest, **kwargs: Any) -> ChatResponse:
        effective_request = chat_request.model_copy(deep=True)
        effective_request.chat_request_context = _merge_chat_request_context(
            effective_request.chat_request_context,
            self.default_request_context,
        )
        return await run_coordinated_chat(effective_request)

    def get_message_factory(self) -> LLMMessageContentFactoryBase:
        return self.message_factory

    def get_config(self) -> AiChatUtilConfig | None:
        return self.runtime_config


def _resolve_workflow_trace_id(trace_id: str = "") -> str:
    normalized = str(trace_id or "").strip()
    if normalized:
        return normalized
    headers = get_current_request_headers()
    header_trace_id = str(getattr(headers, "trace_id", "") or "").strip()
    return header_trace_id


def _is_workflow_approval_text(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    return normalized in {"approve", "approved", "yes", "y"} or normalized.startswith("approve ")


async def run_chat(
        chat_request: Annotated[ChatRequest, Field(description="Chat request object")],
) -> Annotated[ChatResponse, Field(description="List of related articles from Wikipedia")]:
    """
    This function processes a chat request with the standard LLM client.
    """
    client = create_llm_client()
    return await client.chat(chat_request)


async def run_agent_chat(
        chat_request: Annotated[ChatRequest, Field(description="Chat request object")],
) -> Annotated[ChatResponse, Field(description="Agent chat response")]:
    """
    This function processes a chat request with the MCP-backed agent client.
    """
    client = AgentFactory.create_mcp_client()
    return await client.chat(chat_request)


async def run_coordinated_chat(
        chat_request: Annotated[ChatRequest, Field(description="Chat request object")],
) -> Annotated[ChatResponse, Field(description="Coordinator-selected chat response")]:
    """
    This function selects WF / SV / autonomous execution based on deterministic policy hints.
    """
    runtime_config = get_runtime_config()
    run_trace_id = _resolve_chat_trace_id(chat_request)
    current_request_headers = get_current_request_headers()
    audit_context = create_audit_context(
        runtime_config,
        run_trace_id,
        request_headers=current_request_headers,
    )
    routing_decision, autonomous_backend, request_hints = _decide_cross_type_route(chat_request, runtime_config)
    audit_context.emit(
        "cross_type_route_decided",
        route_name=routing_decision.selected_route,
        reason_code=routing_decision.reason_code,
        confidence=routing_decision.confidence,
        target_system=autonomous_backend or routing_decision.selected_route,
        action_kind="route",
        payload={
            "candidate_routes": [candidate.model_dump(mode="json") for candidate in routing_decision.candidate_routes],
            "request_hints": request_hints,
            "policy_snapshot": _build_cross_type_policy_snapshot(runtime_config),
            "selected_backend": autonomous_backend,
        },
    )

    if routing_decision.requires_clarification and not bool(getattr(chat_request, "auto_approve", False)):
        prompt_text = "\n".join(routing_decision.missing_information) if routing_decision.missing_information else (routing_decision.notes or "型選択の確認が必要です。")
        audit_context.emit(
            "cross_type_route_clarification_requested",
            route_name=routing_decision.selected_route,
            reason_code=routing_decision.reason_code,
            confidence=routing_decision.confidence,
            action_kind="ask_user",
            payload={
                "missing_information": list(routing_decision.missing_information),
                "request_hints": request_hints,
                "selected_backend": autonomous_backend,
            },
            final_status="paused",
        )
        return ChatResponse(
            status="paused",
            trace_id=run_trace_id,
            hitl=HitlRequest(
                kind="input",
                prompt=prompt_text,
                action_id=uuid.uuid4().hex,
                source="coordinator:routing",
            ),
            messages=[
                ChatMessage(
                    role="assistant",
                    content=[ChatContent(params={"type": "text", "text": prompt_text})],
                )
            ],
            input_tokens=0,
            output_tokens=0,
        )

    routed_request = chat_request.model_copy(deep=True)
    routed_request.trace_id = run_trace_id
    request_context = routed_request.chat_request_context

    if routing_decision.selected_route == _CROSS_TYPE_ROUTE_WORKFLOW:
        workflow_file_path = str(getattr(request_context, "workflow_file_path", "") or "").strip()
        if not workflow_file_path:
            raise ValueError("WF型が選択されましたが workflow_file_path が指定されていません。")
        client = WorkflowChatClient(
            workflow_file_path,
            runtime_config=runtime_config,
            max_node_visits=int(getattr(request_context, "workflow_max_node_visits", 8) or 8),
            plan_mode=bool(getattr(request_context, "workflow_plan_mode", False)),
            durable=bool(getattr(request_context, "workflow_durable", True)),
        )
        return await client.chat(routed_request)

    if routing_decision.selected_route == _CROSS_TYPE_ROUTE_AUTONOMOUS:
        if autonomous_backend == "deep_agent":
            client = AgentFactory.create_deepagent_client(runtime_config)
        else:
            client = AgentFactory.create_codingagent_client(runtime_config)
        return await client.chat(routed_request)

    client = AgentFactory.create_mcp_client(runtime_config)
    return await client.chat(routed_request)


async def run_deepagent_chat(
        chat_request: Annotated[ChatRequest, Field(description="Chat request object")],
) -> Annotated[ChatResponse, Field(description="DeepAgent chat response")]:
    """
    This function processes a chat request with the MCP-backed DeepAgent client.
    """
    client = AgentFactory.create_deepagent_client()
    return await client.chat(chat_request)


async def run_simple_chat(
        prompt: Annotated[str, Field(description="Prompt for the chat")],
) -> Annotated[str, Field(description="Chat response from the LLM")]:
    """
    This function processes a simple chat with the specified prompt and returns the chat response.
    """
    llm_client = create_llm_client()
    response = await llm_client.simple_chat(prompt)
    return response


async def run_simple_batch_chat(
        prompt: Annotated[str, Field(description="Prompt for the batch chat")],
        messages: Annotated[list[str], Field(description="List of messages for the batch chat")],
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[str], Field(description="List of chat responses from batch processing")]:
    """
    This function processes a simple batch chat with the specified prompt and messages, and returns the list of chat responses.
    """
    batch_client = BatchClient()
    results = await batch_client.run_simple_batch_chat(prompt, messages, concurrency)
    return results


async def run_batch_chat(
        chat_requests: Annotated[list[ChatRequest], Field(description="List of chat histories for batch processing")],
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[ChatResponse], Field(description="List of chat responses from batch processing")]:
    """
    This function processes a batch of chat histories with the standard LLM client.
    """
    batch_client = BatchClient()
    results = await batch_client.run_batch_chat(chat_requests, concurrency)
    return [response for _, response in results]


async def run_agent_batch_chat(
        chat_requests: Annotated[list[ChatRequest], Field(description="List of chat histories for agent batch processing")],
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[ChatResponse], Field(description="List of agent chat responses from batch processing")]:
    """
    This function processes a batch of chat histories with the MCP-backed agent client.
    """
    batch_client = MCPBatchClient()
    results = await batch_client.run_batch_chat(chat_requests, concurrency)
    return [response for _, response in results]


async def run_deepagent_batch_chat(
        chat_requests: Annotated[list[ChatRequest], Field(description="List of chat histories for DeepAgent batch processing")],
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[ChatResponse], Field(description="List of DeepAgent chat responses from batch processing")]:
    """
    This function processes a batch of chat histories with the MCP-backed DeepAgent client.
    """
    batch_client = DeepAgentBatchClient()
    results = await batch_client.run_batch_chat(chat_requests, concurrency)
    return [response for _, response in results]


async def run_batch_chat_from_excel(
        prompt: Annotated[str, Field(description="Prompt for the batch chat")],
        input_excel_path: Annotated[str, Field(description="Path to the input Excel file")],
        output_excel_path: Annotated[str, Field(description="Path to the output Excel file")]="output.xlsx",
        content_column: Annotated[str, Field(description="Name of the column containing input messages")]="content",
        file_path_column: Annotated[str, Field(description="Name of the column containing file paths")]="file_path",
        output_column: Annotated[str, Field(description="Name of the column to store output responses")]="output",
        detail: Annotated[str, Field(description="Detail level for file analysis. e.g., 'low', 'high', 'auto'")]= "auto",
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=16,
) -> None:
    """
    This function reads chat histories from an Excel file, processes them in batch with the standard LLM client, and writes the responses to a new Excel file.
    """
    batch_client = BatchClient()
    await batch_client.run_batch_chat_from_excel(
        prompt,
        input_excel_path,
        output_excel_path,
        content_column,
        file_path_column,
        output_column,
        detail,
        concurrency,
    )


async def run_agent_batch_chat_from_excel(
        prompt: Annotated[str, Field(description="Prompt for the agent batch chat")],
        input_excel_path: Annotated[str, Field(description="Path to the input Excel file")],
        output_excel_path: Annotated[str, Field(description="Path to the output Excel file")]="output.xlsx",
        content_column: Annotated[str, Field(description="Name of the column containing input messages")]="content",
        file_path_column: Annotated[str, Field(description="Name of the column containing file paths")]="file_path",
        output_column: Annotated[str, Field(description="Name of the column to store output responses")]="output",
        detail: Annotated[str, Field(description="Detail level for file analysis. e.g., 'low', 'high', 'auto'")]= "auto",
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=16,
) -> None:
    """
    This function reads chat histories from an Excel file, processes them in batch with the MCP-backed agent client, and writes the responses to a new Excel file.
    """
    batch_client = MCPBatchClient()
    await batch_client.run_batch_chat_from_excel(
        prompt,
        input_excel_path,
        output_excel_path,
        content_column,
        file_path_column,
        output_column,
        detail,
        concurrency,
    )


async def run_deepagent_batch_chat_from_excel(
        prompt: Annotated[str, Field(description="Prompt for the DeepAgent batch chat")],
        input_excel_path: Annotated[str, Field(description="Path to the input Excel file")],
        output_excel_path: Annotated[str, Field(description="Path to the output Excel file")]="output.xlsx",
        content_column: Annotated[str, Field(description="Name of the column containing input messages")]="content",
        file_path_column: Annotated[str, Field(description="Name of the column containing file paths")]="file_path",
        output_column: Annotated[str, Field(description="Name of the column to store output responses")]="output",
        detail: Annotated[str, Field(description="Detail level for file analysis. e.g., 'low', 'high', 'auto'")]= "auto",
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=16,
) -> None:
    """
    This function reads chat histories from an Excel file, processes them in batch with the MCP-backed DeepAgent client, and writes the responses to a new Excel file.
    """
    batch_client = DeepAgentBatchClient()
    await batch_client.run_batch_chat_from_excel(
        prompt,
        input_excel_path,
        output_excel_path,
        content_column,
        file_path_column,
        output_column,
        detail,
        concurrency,
    )


async def run_mermaid_workflow_from_file(
        workflow_file_path: Annotated[str, Field(description="Path to a Markdown file containing exactly one mermaid block")],
        message: Annotated[str, Field(description="Initial input text passed into the workflow")]= "",
        max_node_visits: Annotated[int, Field(description="Loop safety limit for a single node")]=8,
        plan_mode: Annotated[bool, Field(description="If true, prepare updated markdown and pause for approval instead of executing")]=False,
        approved_markdown: Annotated[str, Field(description="Updated markdown approved by the client. When set, execution uses this markdown directly")]= "",
    durable: Annotated[bool, Field(description="If true, persist workflow execution state so HITL pauses can be resumed")]=False,
    resume_value: Annotated[str, Field(description="Resume text for a paused durable workflow")]= "",
    enable_tool_approval_nodes: Annotated[bool, Field(description="If true, inject dry-run and approval nodes for update tools")]=False,
    trace_id: Annotated[str, Field(description="Optional trace_id used as the workflow thread id")]= "",
) -> Annotated[WorkflowExecutionResponse, Field(description="Workflow execution response including plan/HITL status")]:
    """
    Execute a Markdown-defined WF workflow using LangGraph.
    """
    workflow_path = Path(workflow_file_path).expanduser().resolve()
    markdown = workflow_path.read_text(encoding="utf-8")
    return await execute_workflow_markdown(
        markdown,
        message=message,
        max_node_visits=max_node_visits,
        plan_mode=plan_mode,
        approved_markdown=approved_markdown,
        durable=durable,
        resume_value=resume_value,
        enable_tool_approval_nodes=enable_tool_approval_nodes,
        thread_id=_resolve_workflow_trace_id(trace_id),
    )


async def run_durable_workflow_from_file(
        workflow_file_path: Annotated[str, Field(description="Path to a Markdown file containing exactly one mermaid block")],
        message: Annotated[str, Field(description="Initial input text passed into the workflow")]= "",
        max_node_visits: Annotated[int, Field(description="Loop safety limit for a single node")]=8,
        plan_mode: Annotated[bool, Field(description="If true, prepare updated markdown and pause for approval instead of executing")]=False,
        trace_id: Annotated[str, Field(description="Optional trace_id used as the durable workflow thread id")]= "",
) -> Annotated[WorkflowExecutionResponse, Field(description="Durable workflow execution response")]:
    """
    Execute a Markdown-defined WF workflow using durable pause/resume semantics.
    """
    return await run_mermaid_workflow_from_file(
        workflow_file_path=workflow_file_path,
        message=message,
        max_node_visits=max_node_visits,
        plan_mode=plan_mode,
        durable=True,
        enable_tool_approval_nodes=True,
        trace_id=trace_id,
    )


async def resume_durable_workflow(
        resume_value: Annotated[str, Field(description="Approval or answer text used to resume a paused workflow")],
        trace_id: Annotated[str, Field(description="trace_id of the paused durable workflow")]= "",
) -> Annotated[WorkflowExecutionResponse, Field(description="Resumed durable workflow response")]:
    """
    Resume a paused durable workflow using its trace_id.
    """
    effective_trace_id = _resolve_workflow_trace_id(trace_id)
    if not effective_trace_id:
        raise ValueError("trace_id is required to resume a durable workflow")

    session_store = WorkflowSessionStore.from_runtime_config()
    session = session_store.load(effective_trace_id)
    if session is None:
        raise ValueError(f"Paused workflow session was not found for trace_id={effective_trace_id}")

    if session.phase == "plan":
        if not _is_workflow_approval_text(resume_value):
            session_store.delete(effective_trace_id)
            return WorkflowExecutionResponse(
                status="completed",
                final_output="Workflow execution was not approved.",
                prepared_markdown=session.prepared_markdown,
                thread_id=effective_trace_id,
            )
        response = await execute_workflow_markdown(
            session.original_markdown,
            message=session.message,
            approved_markdown=session.prepared_markdown,
            max_node_visits=session.max_node_visits,
            durable=True,
            enable_tool_approval_nodes=True,
            thread_id=effective_trace_id,
        )
    else:
        response = await execute_workflow_markdown(
            session.original_markdown,
            message=session.message,
            approved_markdown=session.prepared_markdown,
            max_node_visits=session.max_node_visits,
            resume_value=resume_value,
            durable=True,
            enable_tool_approval_nodes=True,
            thread_id=effective_trace_id,
        )

    if response.status == "paused":
        session_store.save(session.model_copy(update={"phase": "plan" if getattr(response.hitl, "source", "") == "workflow:plan" else "graph"}))
    else:
        session_store.delete(effective_trace_id)
    return response