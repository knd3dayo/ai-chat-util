from __future__ import annotations

from typing import Any, Mapping, Sequence, cast
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLMRouter
from litellm.router import Router
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..config.mcp_config import MCPConfigParser
from ..config.runtime import AiChatUtilConfig, get_runtime_config, get_runtime_config_path
from ..util.file_path_resolver import resolve_existing_file_path
from ..model.models import ChatRequest, ChatResponse, ChatMessage, ChatContent, ChatHistory

import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class MCPClient:
    def __init__(self, runtime_config: AiChatUtilConfig):
        self.runtime_config = runtime_config
        mcp_config_path = (
            self.runtime_config.paths.mcp_config_path
            or self.runtime_config.paths.mcp_server_config_file_path
        )
        if not mcp_config_path:
            logger.warning(
                "MCP 設定ファイルパスが未設定です。config.yml の paths.mcp_config_path（または互換の paths.mcp_server_config_file_path）を設定してください。"
            )
            self.mcp_config = None
        else:
            # config.yml からの相対パスも解決できるよう、設定ファイルのディレクトリも探索対象に入れる
            config_dir = str(get_runtime_config_path().parent)
            resolved = resolve_existing_file_path(
                mcp_config_path,
                working_directory=self.runtime_config.paths.working_directory,
                extra_search_dirs=[config_dir],
            ).resolved_path

            self.config_parser = MCPConfigParser(resolved)
            # 2. LangChain用設定の生成
            self.mcp_config = self.config_parser.to_langchain_config()

    async def simple_chat(self, message: str) -> str:
        chat_request = ChatRequest(
            chat_history=ChatHistory(
                messages=[
                    ChatMessage(
                        role="user",
                        content=[ChatContent(params={"type": "text", "text": message})],
                    )
                ]
            )
        )
        response = await self.chat(chat_request)
        return response.output


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


    @staticmethod
    def _stringify_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # OpenAI-style multi-part content.
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            return "".join(parts)
        return str(content)


    @classmethod
    def _extract_output_and_usage(cls, result: Any) -> tuple[str, int, int]:
        """Best-effort extract output text + token usage from agent result."""

        def _usage_from_ai_message(ai: AIMessage) -> tuple[int, int]:
            # LangChain standard
            usage_meta = getattr(ai, "usage_metadata", None)
            if isinstance(usage_meta, Mapping):
                in_tok = int(usage_meta.get("input_tokens", 0) or 0)
                out_tok = int(usage_meta.get("output_tokens", 0) or 0)
                if in_tok or out_tok:
                    return in_tok, out_tok

            # Provider-specific (LiteLLM/OpenAI adapters often stash here)
            resp_meta = getattr(ai, "response_metadata", None)
            if isinstance(resp_meta, Mapping):
                usage = resp_meta.get("usage") or resp_meta.get("token_usage") or {}
                if isinstance(usage, Mapping):
                    in_tok = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                    out_tok = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
                    return in_tok, out_tok

            return 0, 0

        # 1) direct string
        if isinstance(result, str):
            return result, 0, 0

        # 2) dict-like payloads
        if isinstance(result, Mapping):
            out = result.get("output")
            if isinstance(out, str) and out.strip():
                return out, 0, 0

            msgs = result.get("messages")
            if isinstance(msgs, Sequence):
                # Prefer last AI message
                last_ai: AIMessage | None = None
                for m in reversed(list(msgs)):
                    if isinstance(m, AIMessage):
                        last_ai = m
                        break
                    # Sometimes messages are plain dicts
                    if isinstance(m, Mapping) and (m.get("role") == "assistant"):
                        content = m.get("content")
                        return cls._stringify_message_content(content), 0, 0

                if last_ai is not None:
                    text = cls._stringify_message_content(last_ai.content)
                    in_tok, out_tok = _usage_from_ai_message(last_ai)
                    return text, in_tok, out_tok

        # 3) fallback
        return str(result), 0, 0
    

    async def chat(self, chat_request: ChatRequest) -> ChatResponse:
        # 3. MultiServerMCPClientの初期化
        # この内部で各サーバーへの接続（stdioの起動やSSEのセッション開始）が行われます
        allowed_langchain_tools = []
        if self.mcp_config:
            client = MultiServerMCPClient(self.mcp_config)
            # LangChainのツールリストを取得
            langchain_tools = await client.get_tools()
        
            # (オプション) allowedToolsによるフィルタリングが必要な場合
            allowed_map = self.config_parser.get_allowed_tools_map()
            # ここで langchain_tools をフィルタリングするロジックを挟めます
            for tool in langchain_tools:
                tool_name = tool.name
                if tool_name in allowed_map:
                    allowed_langchain_tools.append(tool)
                else:
                    print(f"Tool {tool_name} is not in the allowed tools list and will be skipped.")
            # あとはこれを LangChain の Agent や LLM (bind_tools) に渡すだけ！
            # example: 
            # llm_with_tools = ChatOpenAI().bind_tools(langchain_tools)
            
            print(f"Loaded {len(allowed_langchain_tools)} tools from MCP servers.")

        # LLM + MCP ツールでエージェントを作成
        litellm_router = Router(model_list=self.runtime_config.llm.create_litellm_model_list())
        llm = ChatLiteLLMRouter(router=litellm_router, model_name=self.runtime_config.llm.completion_model)
        agent = create_agent(llm, allowed_langchain_tools)

        # 実行
        lc_messages = self._chat_messages_to_langchain(chat_request.chat_history.messages)
        if not lc_messages:
            raise ValueError("chat_request.chat_history.messages が空です。")

        result = await agent.ainvoke({"messages": lc_messages})  # type: ignore
        output_text, input_tokens, output_tokens = self._extract_output_and_usage(result)

        return ChatResponse(
            messages=[
                ChatMessage(
                    role="assistant",
                    content=[ChatContent(params={"type": "text", "text": output_text})],
                )
            ],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

if __name__ == "__main__":
    runtime_config = get_runtime_config()  # ここは適宜、実際の設定に合わせて初期化してください
    chat_request = ChatRequest(chat_history=ChatHistory(messages=[ChatMessage(role="user", content=[ChatContent(params={"type": "text", "text": "3 と 5 を足して"})])]))
    asyncio.run(MCPClient(runtime_config).chat(chat_request))