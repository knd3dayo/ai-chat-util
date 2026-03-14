from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLMRouter
from litellm.router import Router
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import BaseChatModel

from ..config.mcp_config import MCPConfigParser
from ..config.runtime import (
    CONFIG_ENV_VAR,
    AiChatUtilConfig,
    get_runtime_config,
    get_runtime_config_path,
)
from ..util.file_path_resolver import resolve_existing_file_path
from ..model.models import ChatRequest, ChatResponse, ChatMessage, ChatContent, ChatHistory
from .llm_factory import LLMFactory
import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class MCPClientUtil:
    @classmethod
    def create_mcp_config(cls, runtime_config: AiChatUtilConfig, mcp_config_path: str| None) -> tuple[dict, MCPConfigParser|None]:
        if not mcp_config_path:
            logger.warning(
                "MCP 設定ファイルパスが未設定です。config.yml の paths.mcp_config_path（または互換の paths.mcp_server_config_file_path）を設定してください。"
            )
            return {}, None
        else:
            # config.yml からの相対パスも解決できるよう、設定ファイルのディレクトリも探索対象に入れる
            config_dir = str(get_runtime_config_path().parent)
            resolved = resolve_existing_file_path(
                mcp_config_path,
                working_directory=runtime_config.paths.working_directory,
                extra_search_dirs=[config_dir],
            ).resolved_path

            config_parser = MCPConfigParser(resolved)
            # 2. LangChain用設定の生成
            mcp_config = config_parser.to_langchain_config()

            # Ensure the MCP server subprocess can resolve the same config.yml as this process.
            # When stdio servers are launched with a different working directory (e.g., `uv --directory`),
            # a relative AI_CHAT_UTIL_CONFIG like "config.yml" can break. Pass an absolute path.
            runtime_config_path = str(get_runtime_config_path())
            if mcp_config:
                for _name, conn in mcp_config.items():
                    if isinstance(conn, Mapping) and conn.get("transport") == "stdio":
                        conn_dict = cast(dict[str, Any], conn)
                        env = conn_dict.get("env")
                        if env is None:
                            env = {}
                        if isinstance(env, Mapping) and CONFIG_ENV_VAR not in env:
                            env2 = dict(env)
                            env2[CONFIG_ENV_VAR] = runtime_config_path
                            conn_dict["env"] = env2

            return mcp_config, config_parser

    @classmethod
    async def get_allowed_tools(cls, mcp_config: dict[str, Any] | None, config_parser: MCPConfigParser | None) -> list[Any]:
        allowed_langchain_tools = []
        if not mcp_config:
            logger.warning("MCP 設定が見つからないため、ツールはロードされません。")
            return allowed_langchain_tools
        
        client = MultiServerMCPClient(mcp_config)
        # LangChainのツールリストを取得
        langchain_tools = await client.get_tools()
    
        # (オプション) allowedToolsによるフィルタリングが必要な場合
        if not config_parser:
            logger.info("MCP 設定はありますが、allowedToolsの情報が見つからないため、全てのツールを許可します。")
            return langchain_tools
        allowed_map = config_parser.get_allowed_tools_map()
        # If no server specifies allowedTools (all None), allow everything.
        allowed_names: set[str] | None = None
        for _server_name, names in allowed_map.items():
            if names is None:
                continue
            if allowed_names is None:
                allowed_names = set()
            allowed_names.update(names)

        for tool in langchain_tools:
            tool_name = tool.name
            if allowed_names is None or tool_name in allowed_names:
                allowed_langchain_tools.append(tool)
            else:
                logger.debug("Tool %s is not in allowedTools; skipped", tool_name)
        # あとはこれを LangChain の Agent や LLM (bind_tools) に渡すだけ！
        # example: 
        # llm_with_tools = ChatOpenAI().bind_tools(langchain_tools)
        
        logger.info("Loaded %d tools from MCP servers.", len(allowed_langchain_tools))
        return allowed_langchain_tools
    
    @classmethod
    async def create_workflow(cls, mcp_config: dict[str, Any]| None, config_parser: MCPConfigParser| None, llm: BaseChatModel) -> CompiledStateGraph:

        # LLM + MCP ツールでエージェントを作成
        allowed_langchain_tools = await MCPClientUtil.get_allowed_tools(mcp_config, config_parser)

        # ツール実行用のエージェント
        # システムプロンプトで役割分担を指示する例。実際のプロンプトは用途に応じて調整してください。
        tool_agent_system_prompt = (
            "あなたは複数のツールにアクセスできる有能なアシスタントです。"
            "スーパーバイザーを助けるために、必要に応じてツールを使用してください。"
            "利用可能なツールのみを使用してください。"
            """
            出力フォーマットはXML形式で、以下のルールに従ってください。
            <OUTPUT>
                <TEXT>スーパーバイザーへの返答テキスト（必要に応じて）</TEXT>
                <RESPONSE_TYPE>complete|question|reject</RESPONSE_TYPE>
            </OUTPUT>
            - complete: 指示完了。スーパーバイザーへの返答テキストをTEXTに入れてください。
            - question: スーパーバイザーへの質問。スーパーバイザーに確認が必要な場合は、TEXTに質問内容を入れてこのタイプで返してください。
            - reject: 指示拒否。実行できない指示があった場合は、このタイプで返してください。TEXTは任意ですが、拒否理由などがあれば入れてください。
            """
        )
        tool_agent = create_agent(
            llm,
            allowed_langchain_tools,
            system_prompt=tool_agent_system_prompt,
            name="tool_agent",
        )

        tools_description = "\n".join(f"- {tool.name}: {tool.description}" for tool in allowed_langchain_tools)
        logger.info("Allowed tools:\n%s", tools_description)
        # Plannerエージェントはユーザからの指示を受け取り、計画を立ててtool_agentに指示を出す役割
        # システムプロンプトで役割分担を指示する例。実際のプロンプトは用途に応じて調整してください。
        planner_agent_system_prompt = (
            "あなたはプランナー（計画立案）エージェントです。"
            "スーパーバイザーの指示を受け取り、実行計画を作成し、必要に応じてツール実行エージェントへ指示してください。"
            f"利用可能なツールは以下の通りです:\n{tools_description}\n"
            """
            出力フォーマットはXML形式で、以下のルールに従ってください。
            <OUTPUT>
                <TEXT>スーパーバイザーへの返答テキスト（必要に応じて）</TEXT>
                <RESPONSE_TYPE>complete|question|reject</RESPONSE_TYPE>
            </OUTPUT>
            - complete: 指示完了。スーパーバイザーへの返答テキストをTEXTに入れてください。
            - question: スーパーバイザーへの質問。スーパーバイザーに確認が必要な場合は、TEXTに質問内容を入れてこのタイプで返してください。
            - reject: 指示拒否。実行できない指示があった場合は、このタイプで返してください。TEXTは任意ですが、拒否理由などがあれば入れてください。
            """
        )
        planner_agent = create_agent(
            llm,
            [],
            system_prompt=planner_agent_system_prompt,
            name="planner_agent",
        )  # ツールは持たないシンプルなエージェント


        workflow = create_supervisor(
            [planner_agent, tool_agent],
            model=llm,
            prompt=(
                "あなたはチームのスーパーバイザーです。"
                "planner_agent（計画）と tool_agent（ツール実行）の各エージェントを管理し、スーパーバイザーの目的を達成してください。"
                "計画が必要なら planner_agent を使い、具体的な実行が必要なら tool_agent を使ってください。"
                "あなたが解決できない問題であっても、まずは各エージェントに指示を出してみてください。"
                """
                各エージェントからの出力フォーマットはXML形式です。
                <OUTPUT>
                    <TEXT>スーパーバイザーへの返答テキスト（必要に応じて）</TEXT>
                    <RESPONSE_TYPE>complete|question|reject</RESPONSE_TYPE>
                </OUTPUT>
                - complete: 指示完了。スーパーバイザーへの返答テキストをTEXTに入れてください。
                - question: スーパーバイザーへの質問。スーパーバイザーに確認が必要な場合は、TEXTに質問内容を入れてこのタイプで返してください。
                - reject: 指示拒否。実行できない指示があった場合は、このタイプで返してください。TEXTは任意ですが、拒否理由などがあれば入れてください。
                <RESPONSE_TYPE>がquestion、rejectの場合は、あなたが回答可能な場合はその各エージェントに追加の指示を出すこともできます。
                """
            )
        )

        # Compile and run
        graph = workflow.compile(name="mcp_supervisor")

        return graph

class MCPClient:
    def __init__(self, runtime_config: AiChatUtilConfig):
        self.runtime_config = runtime_config
        mcp_config_path = (
            self.runtime_config.paths.mcp_config_path
            or self.runtime_config.paths.mcp_server_config_file_path
        )
        self.mcp_config, self.config_parser = MCPClientUtil.create_mcp_config(runtime_config, mcp_config_path)

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

        # LLM + MCP ツールでエージェントを作成
        litellm_router = Router(model_list=self.runtime_config.llm.create_litellm_model_list())
        llm = ChatLiteLLMRouter(router=litellm_router, model_name=self.runtime_config.llm.completion_model)

        app = await MCPClientUtil.create_workflow(self.mcp_config, self.config_parser, llm)

        # 実行
        lc_messages = self._chat_messages_to_langchain(chat_request.chat_history.messages)
        if not lc_messages:
            raise ValueError("chat_request.chat_history.messages が空です。")

        result = await app.ainvoke({"messages": lc_messages}) 
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