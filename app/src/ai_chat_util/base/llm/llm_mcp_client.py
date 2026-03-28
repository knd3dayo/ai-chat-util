from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import contextlib
import uuid

import asyncio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from ai_chat_util_base.config.runtime import (
    AiChatUtilConfig,
    get_runtime_config,
)
from ai_chat_util_base.model.ai_chatl_util_models import ChatRequest, ChatResponse, ChatMessage, ChatContent, ChatHistory, HitlRequest
from .abstract_llm_client import AbstractLLMClient
from ai_chat_util_base.config.runtime import get_runtime_config, AiChatUtilConfig, CodingAgentUtilConfig
from .llm_client import LLMMessageContentFactoryBase, LLMMessageContentFactory
from .prompts import CodingAgentPrompts, PromptsBase

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

from .llm_mcp_client_util import MCPClientUtil
from .agent import ToolLimits

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
        # LangGraph checkpoint key is named `thread_id`. This project standardizes on `trace_id`.
        # If not provided, generate a W3C-compatible 32-hex trace_id.
        run_trace_id = (trace_id or uuid.uuid4().hex).lower()
        

        async with contextlib.AsyncExitStack() as exit_stack:

            checkpointer = await MCPClientUtil._create_sqlite_checkpointer(
                MCPClientUtil._default_checkpoint_db_path(self.runtime_config),
                exit_stack=exit_stack,
            )

            prompts = CodingAgentPrompts()
            # LLM + MCP ツールでエージェントを作成
            tool_limits = ToolLimits.from_config(self.runtime_config)

            recursion_limit = tool_limits.tool_recursion_limit
            auto_approve = tool_limits.auto_approve
            max_retries = tool_limits.max_retries
            app = await MCPClientUtil.create_workflow(
                self.runtime_config,
                prompts=prompts,
                checkpointer=checkpointer,
                tool_limits=tool_limits,
            )

            # 実行
            lc_messages = self._chat_messages_to_langchain(chat_request.chat_history.messages)
            if not lc_messages:
                raise ValueError("chat_request.chat_history.messages が空です。")

            try:
                result = await app.ainvoke(
                    {"messages": lc_messages},
                    config={"configurable": {"thread_id": run_trace_id}, "recursion_limit": recursion_limit},
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
                output_text, add_in, add_out = MCPClientUtil._extract_output_and_usage(result)
                resp_type, extracted_text, hitl_kind, hitl_tool = MCPClientUtil._parse_supervisor_xml(output_text)

                # 後続処理で使用する。
                input_tokens += add_in
                output_tokens += add_out
                user_text = extracted_text or output_text

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

            return ChatResponse(
                status=cast(Any, status),
                trace_id=run_trace_id,
                hitl=hitl,
                messages=[
                    ChatMessage(
                        role="assistant",
                        content=[ChatContent(params={"type": "text", "text": user_text})],
                    )
                ],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

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