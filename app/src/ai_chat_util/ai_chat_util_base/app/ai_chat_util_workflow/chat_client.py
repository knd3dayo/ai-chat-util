from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ai_chat_util.ai_chat_util_base.core.chat.core import AbstractChatClient, LLMMessageContentFactory, LLMMessageContentFactoryBase
from ai_chat_util.ai_chat_util_base.core.common.config.runtime import AiChatUtilConfig, get_runtime_config
from ai_chat_util.ai_chat_util_base.core.chat.model import ChatContent, ChatHistory, ChatMessage, ChatRequest, ChatResponse
from ai_chat_util.ai_chat_util_base.app.ai_chat_util_workflow.session_store import WorkflowSessionRecord, WorkflowSessionStore
from ai_chat_util.ai_chat_util_base.app.ai_chat_util_workflow.workflow.runner import execute_workflow_markdown


class WorkflowChatClient(AbstractChatClient):
    def __init__(
        self,
        workflow_file_path: str,
        *,
        runtime_config: AiChatUtilConfig | None = None,
        max_node_visits: int = 8,
        plan_mode: bool = False,
        durable: bool = True,
        session_store: WorkflowSessionStore | None = None,
    ) -> None:
        self.runtime_config = runtime_config or get_runtime_config()
        self.message_factory = LLMMessageContentFactory(config=self.runtime_config)
        self.workflow_file_path = str(Path(workflow_file_path).expanduser().resolve())
        self.max_node_visits = max_node_visits
        self.plan_mode = plan_mode
        self.durable = durable
        self.session_store = session_store or WorkflowSessionStore.from_runtime_config(self.runtime_config)

    async def simple_chat(self, prompt: str) -> str:
        request = ChatRequest(
            chat_history=ChatHistory(
                messages=[
                    ChatMessage(role="user", content=[ChatContent(params={"type": "text", "text": prompt})])
                ]
            )
        )
        response = await self.chat(request)
        return response.output

    async def chat(self, chat_request: ChatRequest, **kwargs: Any) -> ChatResponse:
        trace_id = (chat_request.trace_id or uuid.uuid4().hex).lower()
        user_text = self._extract_latest_user_text(chat_request)
        markdown = Path(self.workflow_file_path).read_text(encoding="utf-8")
        session = self.session_store.load(trace_id)

        if session is not None:
            if session.phase == "plan":
                if self._is_approved(user_text):
                    response = await execute_workflow_markdown(
                        session.original_markdown or markdown,
                        message=session.message,
                        runtime_config=self.runtime_config,
                        max_node_visits=session.max_node_visits,
                        approved_markdown=session.prepared_markdown,
                        thread_id=trace_id,
                        durable=self.durable,
                        enable_tool_approval_nodes=self.durable,
                    )
                    self._update_session(trace_id, response, original_markdown=session.original_markdown or markdown, message=session.message)
                    return self._to_chat_response(response, trace_id=trace_id)

                self.session_store.delete(trace_id)
                return ChatResponse(
                    status="completed",
                    trace_id=trace_id,
                    hitl=None,
                    messages=[
                        ChatMessage(
                            role="assistant",
                            content=[ChatContent(params={"type": "text", "text": "Workflow execution was not approved."})],
                        )
                    ],
                    input_tokens=0,
                    output_tokens=0,
                )

            response = await execute_workflow_markdown(
                session.original_markdown or markdown,
                message=session.message,
                runtime_config=self.runtime_config,
                max_node_visits=session.max_node_visits,
                approved_markdown=session.prepared_markdown,
                thread_id=trace_id,
                resume_value=user_text,
                durable=True,
                enable_tool_approval_nodes=True,
            )
            self._update_session(trace_id, response, original_markdown=session.original_markdown or markdown, message=session.message)
            return self._to_chat_response(response, trace_id=trace_id)

        response = await execute_workflow_markdown(
            markdown,
            message=user_text,
            runtime_config=self.runtime_config,
            max_node_visits=self.max_node_visits,
            plan_mode=self.plan_mode,
            thread_id=trace_id,
            durable=self.durable,
            enable_tool_approval_nodes=self.durable,
        )
        self._update_session(trace_id, response, original_markdown=markdown, message=user_text)
        return self._to_chat_response(response, trace_id=trace_id)

    def _update_session(self, trace_id: str, response: Any, *, original_markdown: str, message: str) -> None:
        if getattr(response, "status", "completed") != "paused":
            self.session_store.delete(trace_id)
            return
        hitl = getattr(response, "hitl", None)
        source = str(getattr(hitl, "source", "") or "")
        phase = "plan" if source == "workflow:plan" else "graph"
        self.session_store.save(
            WorkflowSessionRecord(
                trace_id=trace_id,
                phase=phase,
                workflow_file_path=self.workflow_file_path,
                original_markdown=original_markdown,
                prepared_markdown=str(getattr(response, "prepared_markdown", "") or ""),
                message=message,
                max_node_visits=self.max_node_visits,
            )
        )

    def get_message_factory(self) -> LLMMessageContentFactoryBase:
        return self.message_factory

    def get_config(self) -> AiChatUtilConfig | None:
        return self.runtime_config

    @staticmethod
    def _extract_latest_user_text(chat_request: ChatRequest) -> str:
        messages = chat_request.chat_history.messages
        for message in reversed(messages):
            if (message.role or "").lower() != "user":
                continue
            text_parts = [
                str(content.params.get("text") or "")
                for content in message.content
                if content.params.get("type") == "text"
            ]
            text = "".join(text_parts).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _is_approved(user_text: str) -> bool:
        normalized = (user_text or "").strip().lower()
        return normalized in {"approve", "approved", "yes", "y"} or normalized.startswith("approve ")

    @staticmethod
    def _to_chat_response(response: Any, *, trace_id: str) -> ChatResponse:
        return ChatResponse(
            status=response.status,
            trace_id=trace_id,
            hitl=response.hitl,
            messages=[
                ChatMessage(
                    role="assistant",
                    content=[ChatContent(params={"type": "text", "text": response.final_output or response.prepared_markdown})],
                )
            ],
            input_tokens=0,
            output_tokens=0,
        )