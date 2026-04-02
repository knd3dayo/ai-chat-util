from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import contextlib
import uuid

import asyncio
from langgraph.errors import GraphRecursionError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from ai_chat_util.common.config.runtime import (
    AiChatUtilConfig,
    get_runtime_config,
)
from ai_chat_util.common.model.ai_chatl_util_models import ChatRequest, ChatResponse, ChatMessage, ChatContent, ChatHistory, HitlRequest
from ai_chat_util.common.model.request_headers import get_current_request_headers
from ai_chat_util.base.llm.abstract_llm_client import AbstractLLMClient
from ai_chat_util.common.config.runtime import get_runtime_config, AiChatUtilConfig, CodingAgentUtilConfig
from ai_chat_util.base.llm.llm_client import LLMMessageContentFactoryBase, LLMMessageContentFactory
from .prompts import CodingAgentPrompts
from .supervisor_support import create_audit_context

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

from .mcp_client_util import MCPClientUtil
from .tool_limits import ToolLimits

class MCPClient(AbstractLLMClient):
    def __init__(self, runtime_config: AiChatUtilConfig):
        self.runtime_config = runtime_config
        self.message_factory = LLMMessageContentFactory(config=runtime_config)


    @staticmethod
    def _chat_messages_to_langchain(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
        """Convert internal ChatMessage list into LangChain BaseMessage list.

        Supports both text-only and multi-part OpenAI-style content.
        """

        def _payload_for_message(msg: ChatMessage) -> str | list[dict[str, Any]]:
            dumped = [c.model_dump() for c in msg.content]

            # If it's all text parts, collapse to a plain string.
            text_parts: list[str] = []
            all_text = True
            for part in dumped:
                if part.get("type") != "text":
                    all_text = False
                    break
                text_parts.append(str(part.get("text") or ""))

            if all_text:
                return "".join(text_parts)
            return dumped

        lc_messages: list[BaseMessage] = []
        for msg in messages:
            role = (msg.role or "").lower()
            payload = _payload_for_message(msg)

            if role in {"user", "human"}:
                lc_messages.append(HumanMessage(content=cast(Any, payload)))
            elif role in {"assistant", "ai"}:
                lc_messages.append(AIMessage(content=cast(Any, payload)))
            elif role == "system":
                # SystemMessage should be text; best-effort collapse.
                if isinstance(payload, str):
                    system_text = payload
                else:
                    system_text = "".join(
                        str(p.get("text") or "") for p in payload if isinstance(p, dict) and p.get("type") == "text"
                    )
                lc_messages.append(SystemMessage(content=system_text))
            else:
                # Unknown roles: treat as human input.
                lc_messages.append(HumanMessage(content=payload if isinstance(payload, str) else cast(Any, payload)))

        return lc_messages

    async def simple_chat(self, prompt: str) -> str:
        chat_request = ChatRequest(
            chat_history=ChatHistory(
                messages=[
                    ChatMessage(
                        role="user",
                        content=[ChatContent(params={"type": "text", "text": prompt})],
                    )
                ]
            )
        )
        response = await self.chat(chat_request)
        return response.output

    async def chat(self, chat_request: ChatRequest, **kwargs) -> ChatResponse:

 
        trace_id = getattr(chat_request, "trace_id", None)
        current_request_headers = get_current_request_headers()
        # LangGraph checkpoint key is named `thread_id`. This project standardizes on `trace_id`.
        # If not provided, generate a W3C-compatible 32-hex trace_id.
        run_trace_id = (trace_id or uuid.uuid4().hex).lower()
        audit_context = create_audit_context(
            self.runtime_config,
            run_trace_id,
            request_headers=current_request_headers,
        )
        pending_response: dict[str, Any] | None = None
        pending_workflow_results: list[Any] = []
        checkpoint_db_path = MCPClientUtil._default_checkpoint_db_path(self.runtime_config)
        

        async with contextlib.AsyncExitStack() as exit_stack:

            checkpointer = await MCPClientUtil._create_sqlite_checkpointer(
                checkpoint_db_path,
                exit_stack=exit_stack,
            )

            prompts = CodingAgentPrompts()
            # LLM + MCP ツールでエージェントを作成
            tool_limits = ToolLimits.from_config(self.runtime_config)

            recursion_limit = tool_limits.tool_recursion_limit
            auto_approve = tool_limits.auto_approve
            max_retries = tool_limits.max_retries
            lc_messages = self._chat_messages_to_langchain(chat_request.chat_history.messages)
            if not lc_messages:
                raise ValueError("chat_request.chat_history.messages が空です。")

            audit_context.emit(
                "request_received",
                action_kind="read",
                payload={
                    "message_count": len(lc_messages),
                    "auto_approve": auto_approve,
                    "has_authorization": bool(
                        current_request_headers is not None and current_request_headers.authorization
                    ),
                },
            )

            force_coding_agent_route = MCPClientUtil.explicitly_requests_coding_agent(lc_messages)
            force_deep_agent_route = MCPClientUtil.explicitly_requests_deep_agent(lc_messages)
            explicit_user_file_paths = MCPClientUtil.extract_explicit_user_file_paths(lc_messages)
            requested_heading_count = MCPClientUtil.extract_requested_heading_count(lc_messages)
            expects_heading_response = MCPClientUtil.requests_heading_response(lc_messages)
            expects_evaluation_response = MCPClientUtil.requests_evaluation_response(lc_messages)
            expects_tool_catalog_response = MCPClientUtil.requests_tool_catalog_response(lc_messages)
            expects_tool_catalog_details = MCPClientUtil.requests_tool_catalog_details(lc_messages)
            workflow_messages = list(lc_messages)
            config_preflight_payload: dict[str, Any] | None = None
            if force_deep_agent_route and not MCPClientUtil.deep_agent_route_enabled(self.runtime_config):
                response_text = (
                    "deep_agent route が要求されましたが、現在は利用できません。\n"
                    "- features.enable_deep_agent を有効にしてください。\n"
                    "- deepagents パッケージが未導入ならインストールしてください。"
                )
                return ChatResponse(
                    status=cast(Any, "completed"),
                    trace_id=run_trace_id,
                    hitl=None,
                    messages=[
                        ChatMessage(
                            role="assistant",
                            content=[ChatContent(params={"type": "text", "text": response_text})],
                        )
                    ],
                    input_tokens=0,
                    output_tokens=0,
                )
            route_tool_inventory = await MCPClientUtil.resolve_route_tool_inventory(
                runtime_config=self.runtime_config,
            )
            route_tool_catalog = {
                route_name: [
                    str(tool.get("name") or "").strip()
                    for tool in tools
                    if isinstance(tool, Mapping) and str(tool.get("name") or "").strip()
                ]
                for route_name, tools in route_tool_inventory.items()
            }
            route_backend_metadata = MCPClientUtil.build_route_backend_metadata(
                route_tool_inventory=route_tool_inventory,
                runtime_config=self.runtime_config,
            )
            available_tool_names = [
                tool_name
                for tool_names in route_tool_catalog.values()
                for tool_name in tool_names
            ]
            routing_decision = await MCPClientUtil.decide_route(
                runtime_config=self.runtime_config,
                prompts=prompts,
                messages=lc_messages,
                force_coding_agent_route=force_coding_agent_route,
                force_deep_agent_route=force_deep_agent_route,
                explicit_user_file_paths=explicit_user_file_paths,
                available_tool_names=available_tool_names,
                route_tool_catalog=route_tool_catalog,
                audit_context=audit_context,
            )
            audit_context.emit(
                "route_decided",
                route_name=routing_decision.selected_route,
                reason_code=routing_decision.reason_code,
                confidence=routing_decision.confidence,
                target_system=routing_decision.selected_route,
                action_kind="route",
                payload={
                    "candidate_routes": [candidate.model_dump(mode="json") for candidate in routing_decision.candidate_routes],
                    "next_action": routing_decision.next_action,
                    "explicit_user_file_paths": list(explicit_user_file_paths),
                    "route_tool_catalog": route_tool_catalog,
                    "route_backends": route_backend_metadata,
                    "selected_route_backend": route_backend_metadata.get(routing_decision.selected_route),
                },
            )
            if expects_tool_catalog_response and route_tool_catalog:
                tool_catalog_payload = MCPClientUtil.build_route_tool_catalog_payload(
                    route_tool_inventory,
                    runtime_config=self.runtime_config,
                )
                audit_context.emit(
                    "tool_catalog_resolved",
                    route_name=routing_decision.selected_route,
                    payload=tool_catalog_payload,
                )
                response_text = MCPClientUtil.build_tool_catalog_response_text(
                    route_tool_inventory,
                    include_details=expects_tool_catalog_details,
                )
                evidence = {
                    "expects_tool_catalog_response": True,
                    "tool_catalog": tool_catalog_payload["tool_catalog"],
                }
                if bool(getattr(self.runtime_config.features, "sufficiency_check_enabled", False)):
                    summary = MCPClientUtil.build_evidence_summary(evidence)
                    final_sufficiency = MCPClientUtil.judge_sufficiency(
                        response_text=response_text,
                        resp_type="complete",
                        hitl_kind=None,
                        evidence_summary=summary,
                    )
                    audit_context.emit(
                        "final_answer_validated",
                        reason_code=final_sufficiency.reason_code,
                        confidence=final_sufficiency.confidence,
                        payload=final_sufficiency.model_dump(mode="json"),
                        final_status="completed",
                    )
                return ChatResponse(
                    status=cast(Any, "completed"),
                    trace_id=run_trace_id,
                    hitl=None,
                    messages=[
                        ChatMessage(
                            role="assistant",
                            content=[ChatContent(params={"type": "text", "text": response_text})],
                        )
                    ],
                    input_tokens=0,
                    output_tokens=0,
                )
            if routing_decision.requires_clarification and not auto_approve:
                prompt_text = "\n".join(routing_decision.missing_information) if routing_decision.missing_information else (routing_decision.notes or "続行前に確認が必要です。")
                hitl = HitlRequest(
                    kind=cast(Any, "input"),
                    prompt=prompt_text,
                    action_id=str(uuid.uuid4()),
                    source="routing",
                )
                audit_context.emit(
                    "hitl_requested",
                    reason_code="hitl.user_input_requested",
                    route_name=routing_decision.selected_route,
                    confidence=routing_decision.confidence,
                    target_system=routing_decision.selected_route,
                    approval_status="requested",
                    payload={"kind": "input", "source": "routing", "reason_code": routing_decision.reason_code},
                    final_status="paused",
                )
                return ChatResponse(
                    status=cast(Any, "paused"),
                    trace_id=run_trace_id,
                    hitl=hitl,
                    messages=[
                        ChatMessage(
                            role="assistant",
                            content=[ChatContent(params={"type": "text", "text": prompt_text})],
                        )
                    ],
                    input_tokens=0,
                    output_tokens=0,
                )
            if routing_decision.selected_route in {"coding_agent", "deep_agent"} and MCPClientUtil.should_run_config_preflight(lc_messages):
                config_preflight_payload = await MCPClientUtil.run_config_preflight(
                    runtime_config=self.runtime_config,
                    tool_limits=tool_limits,
                    audit_context=audit_context,
                )
                preflight_message = MCPClientUtil.build_config_preflight_message(config_preflight_payload or {})
                if preflight_message:
                    workflow_messages.append(SystemMessage(content=preflight_message))
                    audit_context.emit(
                        "preflight_applied",
                        tool_name="get_loaded_config_info",
                        payload={
                            "success": bool((config_preflight_payload or {}).get("success", False)),
                            "config_path": (config_preflight_payload or {}).get("config_path"),
                        },
                    )
            app = await MCPClientUtil.create_workflow(
                self.runtime_config,
                prompts=prompts,
                checkpointer=checkpointer,
                tool_limits=tool_limits,
                force_coding_agent_route=force_coding_agent_route,
                force_deep_agent_route=force_deep_agent_route,
                explicit_user_file_paths=explicit_user_file_paths,
                routing_decision=routing_decision,
                audit_context=audit_context,
                expects_heading_response=expects_heading_response,
                expects_evaluation_response=expects_evaluation_response,
            )

            # 実行
            workflow_results: list[Any] = []

            try:
                result = await app.ainvoke(
                    {"messages": workflow_messages},
                    config={"configurable": {"thread_id": run_trace_id}, "recursion_limit": recursion_limit},
                )
                workflow_results.append(result)
            except GraphRecursionError as e:
                logger.warning("MCP supervisor hit recursion limit: trace_id=%s", run_trace_id, exc_info=True)
                checkpoint_results = await MCPClientUtil.collect_checkpoint_results(
                    app=app,
                    run_trace_id=run_trace_id,
                )
                workflow_results.extend(checkpoint_results)
                evidence = MCPClientUtil.extract_successful_tool_evidence(workflow_results)
                evidence = MCPClientUtil.merge_preflight_evidence(evidence, config_preflight_payload)
                msg = str(e).strip() or type(e).__name__
                user_text = MCPClientUtil.build_recursion_limit_fallback_text(msg, evidence)
                return ChatResponse(
                    status=cast(Any, "completed"),
                    trace_id=run_trace_id,
                    hitl=None,
                    messages=[
                        ChatMessage(
                            role="assistant",
                            content=[ChatContent(params={"type": "text", "text": user_text})],
                        )
                    ],
                    input_tokens=0,
                    output_tokens=0,
                )
            except Exception as e:
                # Ensure we terminate with a user-visible message (avoid apparent hangs).
                logger.exception("MCP supervisor workflow failed: trace_id=%s", run_trace_id)
                msg = str(e).strip()
                if not msg:
                    msg = type(e).__name__
                user_text = (
                    "ERROR: MCPワークフローが失敗しました。\n"
                    f"- trace_id: {run_trace_id}\n"
                    f"- error: {msg}\n"
                    "対処: ログ（chat_timeout_*.log / mcp_server.log）を確認してください。"
                )
                return ChatResponse(
                    status=cast(Any, "completed"),
                    trace_id=run_trace_id,
                    hitl=None,
                    messages=[
                        ChatMessage(
                            role="assistant",
                            content=[ChatContent(params={"type": "text", "text": user_text})],
                        )
                    ],
                    input_tokens=0,
                    output_tokens=0,
                )
            logger.debug("Extracting output and usage from agent result: %s", result)

            output_text, input_tokens, output_tokens = MCPClientUtil._extract_output_and_usage(result)

            resp_type, extracted_text, hitl_kind, hitl_tool = MCPClientUtil._parse_supervisor_xml(output_text)
            user_text = extracted_text or output_text

            # Fallback: if the model ignored/misused the XML contract, infer HITL from plain text.
            inferred_kind, inferred_tool = MCPClientUtil._infer_hitl_from_plain_text(user_text)

            if inferred_kind is not None:
                # If the model asked a question but didn't specify HITL_KIND, still tag it.
                if resp_type == "question" and not hitl_kind:
                    hitl_kind = inferred_kind
                    hitl_tool = hitl_tool or inferred_tool
                # If the model incorrectly returned complete, force question so CLI can pause.
                if resp_type != "question":
                    resp_type = "question"
                    hitl_kind = inferred_kind
                    hitl_tool = inferred_tool

            budget_exhausted = (
                MCPClientUtil.contains_tool_budget_exceeded_signal(output_text)
                or MCPClientUtil.contains_tool_budget_exceeded_signal(user_text)
            )
            if budget_exhausted:
                logger.warning(
                    "MCP supervisor hit tool call budget; requesting graceful completion without additional tools: trace_id=%s",
                    run_trace_id,
                )
                user_text, resp_type, hitl_kind, hitl_tool, add_in, add_out = await MCPClientUtil.force_graceful_completion_after_budget_exhaustion(
                    app=app,
                    run_trace_id=run_trace_id,
                    recursion_limit=recursion_limit,
                    user_text=user_text,
                )
                input_tokens += add_in
                output_tokens += add_out

            # AUTO_APPROVE: if we still get a question, try to push the supervisor to complete without pausing.
            if auto_approve and resp_type == "question" and max_retries > 0:
                for attempt in range(1, max_retries + 1):
                    directive = (
                        "AUTO_APPROVE モードです。ユーザーに追加確認できません。\n"
                        "直前に提示した質問/承認要求は、あなた自身で合理的に仮定して解決し、作業を完了してください。\n"
                        "不確実性や仮定は TEXT に明記してください。\n"
                        "必ず <RESPONSE_TYPE>complete</RESPONSE_TYPE> を返し、question を返さないでください。\n"
                        f"(attempt {attempt}/{max_retries})\n"
                        f"直前の質問: {user_text}"
                    )
                    result = await app.ainvoke(
                        {"messages": [HumanMessage(content=directive)]},
                        config={"configurable": {"thread_id": run_trace_id}, "recursion_limit": recursion_limit},
                    )
                    workflow_results.append(result)
                    output_text, add_in, add_out = MCPClientUtil._extract_output_and_usage(result)
                    input_tokens += add_in
                    output_tokens += add_out

                    resp_type, extracted_text, hitl_kind, hitl_tool = MCPClientUtil._parse_supervisor_xml(output_text)
                    user_text = extracted_text or output_text
                    if resp_type != "question":
                        break

            if auto_approve and resp_type == "question":
                # Final attempt: even if you cannot fully answer, do not ask.
                final_directive = (
                    "AUTO_APPROVE モードです。ユーザーに追加確認できません。\n"
                    "あなたが自力で完了できない場合でも、質問はせず、現時点でできる範囲の回答と限界（不足情報/前提）を説明して完了してください。\n"
                    "必ず <RESPONSE_TYPE>complete</RESPONSE_TYPE> を返し、question を返さないでください。\n"
                    f"直前の質問: {user_text}"
                )
                result = await app.ainvoke(
                    {"messages": [HumanMessage(content=final_directive)]},
                    config={"configurable": {"thread_id": run_trace_id}, "recursion_limit": recursion_limit},
                )
                workflow_results.append(result)
                output_text, add_in, add_out = MCPClientUtil._extract_output_and_usage(result)
                resp_type, extracted_text, hitl_kind, hitl_tool = MCPClientUtil._parse_supervisor_xml(output_text)

                # 後続処理で使用する。
                input_tokens += add_in
                output_tokens += add_out
                user_text = extracted_text or output_text

            evidence_results = await MCPClientUtil.collect_evidence_results(
                app=app,
                run_trace_id=run_trace_id,
                workflow_results=workflow_results,
                checkpoint_db_path=checkpoint_db_path,
            )
            evidence = MCPClientUtil.extract_successful_tool_evidence(evidence_results)
            evidence = MCPClientUtil.merge_preflight_evidence(evidence, config_preflight_payload)
            evidence = dict(evidence)
            evidence["expects_heading_response"] = expects_heading_response
            evidence["expects_tool_catalog_response"] = expects_tool_catalog_response
            evidence["tool_catalog"] = [
                {
                    "agent_name": route_name,
                    "tool_names": list(tool_names),
                }
                for route_name, tool_names in route_tool_catalog.items()
            ]
            evidence_summary = MCPClientUtil.build_evidence_summary(evidence)
            if requested_heading_count is not None:
                evidence["requested_heading_count"] = requested_heading_count
            followup_task_error_detected = (
                MCPClientUtil.contains_followup_task_error_signal(output_text)
                or MCPClientUtil.contains_followup_task_error_signal(user_text)
            )
            contradicts_evidence = MCPClientUtil.final_text_contradicts_evidence(user_text, evidence)
            missing_concrete_evidence = MCPClientUtil.final_text_missing_concrete_evidence(user_text, evidence)
            if followup_task_error_detected:
                fallback_text = MCPClientUtil.build_evidence_reflected_final_text(evidence)
                if fallback_text:
                    logger.warning(
                        "Supervisor output contained invalid/stale follow-up task signal; forcing evidence-based completion: trace_id=%s",
                        run_trace_id,
                    )
                    user_text = fallback_text
                    resp_type = "complete"
                    hitl_kind = None
                    hitl_tool = None
                    contradicts_evidence = False
                    missing_concrete_evidence = False
            if not contradicts_evidence and missing_concrete_evidence:
                augmented_text = MCPClientUtil.augment_final_text_with_evidence(user_text, evidence)
                if (
                    augmented_text
                    and not MCPClientUtil.final_text_contradicts_evidence(augmented_text, evidence)
                    and not MCPClientUtil.final_text_missing_concrete_evidence(augmented_text, evidence)
                ):
                    logger.info(
                        "Augmented supervisor final text with concrete tool evidence: trace_id=%s",
                        run_trace_id,
                    )
                    user_text = augmented_text
                    missing_concrete_evidence = False

            if contradicts_evidence or missing_concrete_evidence:
                logger.warning(
                    "Supervisor final text did not faithfully reflect successful tool evidence; applying evidence-based fallback: trace_id=%s",
                    run_trace_id,
                )
                fallback_text = MCPClientUtil.build_evidence_reflected_final_text(evidence)
                if fallback_text:
                    user_text = fallback_text
                    resp_type = "complete"
                    hitl_kind = None
                    hitl_tool = None

            if bool(getattr(self.runtime_config.features, "sufficiency_check_enabled", False)):
                sufficiency_decision = MCPClientUtil.judge_sufficiency(
                    response_text=user_text,
                    resp_type=resp_type,
                    hitl_kind=hitl_kind,
                    evidence_summary=evidence_summary,
                )
                audit_context.emit(
                    "sufficiency_judged",
                    reason_code=sufficiency_decision.reason_code,
                    confidence=sufficiency_decision.confidence,
                    action_kind="validate",
                    payload=sufficiency_decision.model_dump(mode="json"),
                )

            status: str = "completed"
            hitl: HitlRequest | None = None
            if resp_type == "question":
                if auto_approve:
                    # Do not pause; return best-effort completion message (avoid asking).
                    status = "completed"
                    hitl = None
                    user_text = (
                        "追加確認が必要な状況でしたが、auto_approve が有効なため pause せずに処理を終了します。\n"
                        "現時点で確定できない点/不足情報（参考）:\n"
                        f"- {user_text}\n"
                        "上記が提供されれば、より正確に継続できます。"
                    )
                else:
                    status = "paused"
                    kind = "approval" if hitl_kind == "approval" else "input"
                    hitl = HitlRequest(
                        kind=cast(Any, kind),
                        prompt=user_text,
                        action_id=str(uuid.uuid4()),
                        source=("supervisor" + (f":{hitl_tool}" if hitl_tool else "")),
                    )
                    audit_context.emit(
                        "hitl_requested",
                        reason_code=("hitl.tool_approval_requested" if kind == "approval" else "hitl.user_input_requested"),
                        approval_status=("requested" if kind == "approval" else None),
                        payload={"kind": kind, "source": hitl.source},
                        final_status=status,
                    )

            pending_workflow_results = list(workflow_results)
            pending_response = {
                "status": cast(Any, status),
                "trace_id": run_trace_id,
                "hitl": hitl,
                "messages": [
                    ChatMessage(
                        role="assistant",
                        content=[ChatContent(params={"type": "text", "text": user_text})],
                    )
                ],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        if pending_response is None:
            raise RuntimeError("MCP workflow finished without producing a response")

        postclose_evidence_results: list[Any] = list(pending_workflow_results)
        postclose_evidence: Mapping[str, Any] = {}
        for attempt in range(15):
            write_results = MCPClientUtil.collect_checkpoint_write_results(
                checkpoint_db_path=checkpoint_db_path,
                run_trace_id=run_trace_id,
            )
            postclose_evidence_results = list(pending_workflow_results)
            postclose_evidence_results.extend(write_results)
            postclose_evidence = MCPClientUtil.extract_successful_tool_evidence(postclose_evidence_results)

            headings = postclose_evidence.get("headings")
            has_headings = isinstance(headings, Sequence) and any(isinstance(v, str) and str(v).strip() for v in headings)
            has_config_path = isinstance(postclose_evidence.get("config_path"), str) and bool(str(postclose_evidence.get("config_path")).strip())
            if has_headings and has_config_path:
                break
            if attempt == 14:
                break
            await asyncio.sleep(0.2)

        response_message = pending_response["messages"][0]
        response_text = ""
        if response_message.content:
            response_text = str(response_message.content[0].params.get("text") or "")

        if not postclose_evidence.get("config_path"):
            config_path_from_text = MCPClientUtil.extract_config_path_from_text(response_text)
            if config_path_from_text:
                postclose_evidence = dict(postclose_evidence)
                postclose_evidence["config_path"] = config_path_from_text

        if not postclose_evidence.get("config_path"):
            runtime_config_path = MCPClientUtil.get_loaded_runtime_config_path()
            if runtime_config_path:
                postclose_evidence = dict(postclose_evidence)
                postclose_evidence["config_path"] = runtime_config_path
        else:
            runtime_config_path = MCPClientUtil.get_loaded_runtime_config_path()
            better_config_path = MCPClientUtil._choose_better_config_path(
                cast(str | None, postclose_evidence.get("config_path")),
                runtime_config_path,
            )
            if better_config_path and better_config_path != postclose_evidence.get("config_path"):
                postclose_evidence = dict(postclose_evidence)
                postclose_evidence["config_path"] = better_config_path

        exact_file_headings = MCPClientUtil.extract_markdown_heading_lines_from_files(explicit_user_file_paths)
        if expects_heading_response and len(exact_file_headings) >= 3:
            postclose_evidence = dict(postclose_evidence)
            postclose_evidence["headings"] = exact_file_headings
        if requested_heading_count is not None:
            postclose_evidence = dict(postclose_evidence)
            postclose_evidence["requested_heading_count"] = requested_heading_count
        postclose_evidence = dict(postclose_evidence)
        postclose_evidence["expects_heading_response"] = expects_heading_response
        postclose_evidence["expects_tool_catalog_response"] = expects_tool_catalog_response
        postclose_evidence["tool_catalog"] = [
            {
                "agent_name": route_name,
                "tool_names": list(tool_names),
            }
            for route_name, tool_names in route_tool_catalog.items()
        ]

        if routing_decision.selected_route == "coding_agent" and expects_heading_response and not (postclose_evidence.get("headings") or []):
            rescue_evidence = await MCPClientUtil.run_direct_coding_agent_heading_rescue(
                runtime_config=self.runtime_config,
                messages=lc_messages,
                run_trace_id=run_trace_id,
                requested_heading_count=requested_heading_count or 3,
            )
            rescue_headings = rescue_evidence.get("headings")
            if isinstance(rescue_headings, Sequence) and any(isinstance(v, str) and str(v).strip() for v in rescue_headings):
                audit_context.emit(
                    "coding_agent_heading_rescued",
                    route_name="coding_agent",
                    payload={
                        "heading_count": len([v for v in rescue_headings if isinstance(v, str) and str(v).strip()]),
                        "latest_task_id": rescue_evidence.get("latest_task_id"),
                    },
                )
                merged_postclose_evidence = dict(postclose_evidence)
                merged_postclose_evidence.update({
                    "stdout_blocks": rescue_evidence.get("stdout_blocks") or postclose_evidence.get("stdout_blocks") or [],
                    "headings": rescue_headings,
                    "raw_texts": rescue_evidence.get("raw_texts") or postclose_evidence.get("raw_texts") or [],
                })
                if rescue_evidence.get("config_path") and not merged_postclose_evidence.get("config_path"):
                    merged_postclose_evidence["config_path"] = rescue_evidence.get("config_path")
                if rescue_evidence.get("latest_task_id"):
                    merged_postclose_evidence["latest_task_id"] = rescue_evidence.get("latest_task_id")
                postclose_evidence = merged_postclose_evidence

        contradicts_evidence = MCPClientUtil.final_text_contradicts_evidence(response_text, postclose_evidence)
        missing_concrete_evidence = MCPClientUtil.final_text_missing_concrete_evidence(response_text, postclose_evidence)
        logger.info(
            "Post-close evidence check: trace_id=%s contradicts=%s missing=%s headings=%s config_path=%s",
            run_trace_id,
            contradicts_evidence,
            missing_concrete_evidence,
            len(postclose_evidence.get("headings") or []),
            postclose_evidence.get("config_path"),
        )

        if MCPClientUtil.should_prefer_deterministic_evidence_response(response_text, postclose_evidence):
            fallback_text = MCPClientUtil.build_evidence_reflected_final_text(postclose_evidence)
            if fallback_text:
                logger.info(
                    "Using deterministic evidence response from post-close evidence: trace_id=%s",
                    run_trace_id,
                )
                response_message.content[0].params["text"] = fallback_text
                pending_response["status"] = cast(Any, "completed")
                pending_response["hitl"] = None
                response_text = fallback_text
                contradicts_evidence = False
                missing_concrete_evidence = False

        if not contradicts_evidence and missing_concrete_evidence:
            augmented_text = MCPClientUtil.augment_final_text_with_evidence(response_text, postclose_evidence)
            if (
                augmented_text
                and not MCPClientUtil.final_text_contradicts_evidence(augmented_text, postclose_evidence)
                and not MCPClientUtil.final_text_missing_concrete_evidence(augmented_text, postclose_evidence)
            ):
                logger.info(
                    "Augmented final text with post-close checkpoint write evidence: trace_id=%s",
                    run_trace_id,
                )
                response_message.content[0].params["text"] = augmented_text
                missing_concrete_evidence = False
                response_text = augmented_text

        if contradicts_evidence or missing_concrete_evidence:
            fallback_text = MCPClientUtil.build_evidence_reflected_final_text(postclose_evidence)
            if fallback_text:
                logger.warning(
                    "Applied post-close evidence-based fallback from checkpoint writes: trace_id=%s",
                    run_trace_id,
                )
                response_message.content[0].params["text"] = fallback_text
                pending_response["status"] = cast(Any, "completed")
                pending_response["hitl"] = None

        if bool(getattr(self.runtime_config.features, "sufficiency_check_enabled", False)):
            postclose_summary = MCPClientUtil.build_evidence_summary(postclose_evidence)
            final_sufficiency = MCPClientUtil.judge_sufficiency(
                response_text=str(response_message.content[0].params.get("text") or ""),
                resp_type="question" if pending_response.get("status") == "paused" else "complete",
                hitl_kind=(getattr(pending_response.get("hitl"), "kind", None) if pending_response.get("hitl") is not None else None),
                evidence_summary=postclose_summary,
            )
            audit_context.emit(
                "final_answer_validated",
                reason_code=final_sufficiency.reason_code,
                confidence=final_sufficiency.confidence,
                action_kind="validate",
                payload=final_sufficiency.model_dump(mode="json"),
                final_status=cast(str | None, pending_response.get("status")),
            )
        else:
            audit_context.emit(
                "final_answer_validated",
                reason_code="audit.validation_passed",
                action_kind="validate",
                payload={
                    "status": pending_response.get("status"),
                    "has_hitl": pending_response.get("hitl") is not None,
                },
                final_status=cast(str | None, pending_response.get("status")),
            )

        return ChatResponse(**pending_response)

    def get_message_factory(self) -> LLMMessageContentFactoryBase:
        '''
        LLMClientが使用するChatMessageFactoryを返す.
        '''
        return self.message_factory

    def get_config(self) -> AiChatUtilConfig | None:
        '''
        LLMClientの設定を返す.
        '''
        return self.runtime_config

if __name__ == "__main__":
    runtime_config = get_runtime_config()  # ここは適宜、実際の設定に合わせて初期化してください
    chat_request = ChatRequest(chat_history=ChatHistory(messages=[ChatMessage(role="user", content=[ChatContent(params={"type": "text", "text": "3 と 5 を足して"})])]))
    asyncio.run(MCPClient(runtime_config).chat(chat_request))