from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import contextlib
import re
import uuid
from pathlib import Path

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLMRouter
from litellm.router import Router
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import BaseChatModel
from .prompts import Prompts
try:
    # Async checkpointer for LangGraph when using app.ainvoke()/astream()
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except Exception:  # pragma: no cover
    AsyncSqliteSaver = None  # type: ignore[assignment]

from ..config.mcp_config import MCPConfigParser
from ..config.runtime import (
    CONFIG_ENV_VAR,
    AiChatUtilConfig,
    get_runtime_config,
    get_runtime_config_path,
)
from ..util.file_path_resolver import resolve_existing_file_path
from ..model.models import ChatRequest, ChatResponse, ChatMessage, ChatContent, ChatHistory, HitlRequest
import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


def _parse_supervisor_xml(output_text: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract RESPONSE_TYPE and TEXT (and optional HITL metadata) from the XML-ish output."""

    text = output_text or ""
    m_type = re.search(r"<RESPONSE_TYPE>\s*(.*?)\s*</RESPONSE_TYPE>", text, flags=re.DOTALL | re.IGNORECASE)
    m_text = re.search(r"<TEXT>\s*(.*?)\s*</TEXT>", text, flags=re.DOTALL | re.IGNORECASE)
    m_kind = re.search(r"<HITL_KIND>\s*(.*?)\s*</HITL_KIND>", text, flags=re.DOTALL | re.IGNORECASE)
    m_tool = re.search(r"<HITL_TOOL>\s*(.*?)\s*</HITL_TOOL>", text, flags=re.DOTALL | re.IGNORECASE)

    resp_type = m_type.group(1).strip().lower() if m_type else None
    payload_text = m_text.group(1).strip() if m_text else None
    hitl_kind = m_kind.group(1).strip().lower() if m_kind else None
    hitl_tool = m_tool.group(1).strip() if m_tool else None
    return resp_type, payload_text, hitl_kind, hitl_tool


def _infer_hitl_from_plain_text(text: str) -> tuple[str | None, str | None]:
    """Best-effort HITL inference when agents don't follow the XML contract.

    Some models may ignore the required XML output and return a plain Japanese
    approval prompt that still contains guidance like "APPROVE analyze_image_files".
    In that case, we infer an approval HITL so the CLI can show `HITL>`.
    """

    t = (text or "").strip()
    if not t:
        return None, None

    # Common approval guidance pattern.
    m = re.search(r"\bAPPROVE\s+([A-Za-z0-9_\-:.]+)", t)
    if m:
        return "approval", m.group(1)

    # Generic HITL hint without tool name.
    if "APPROVE" in t or "REJECT" in t or "承認" in t:
        return "approval", None

    return None, None


def _default_checkpoint_db_path(runtime_config: AiChatUtilConfig) -> Path:
    """Pick a stable per-config SQLite path for LangGraph checkpoints."""

    base = runtime_config.paths.working_directory
    if base:
        root = Path(base).expanduser()
    else:
        root = get_runtime_config_path().parent
    p = (root / ".ai_chat_util" / "langgraph_checkpoints.sqlite").resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


async def _create_sqlite_checkpointer(db_path: Path, *, exit_stack: contextlib.AsyncExitStack) -> Any | None:
    """Create a SQLite checkpointer compatible with async LangGraph execution.

    When running async graphs (app.ainvoke/astream), LangGraph requires AsyncSqliteSaver.
    If it's unavailable, we disable checkpointing (return None) to avoid crashing.
    """

    if AsyncSqliteSaver is None:
        logger.warning(
            "AsyncSqliteSaver が利用できないため、LangGraph のチェックポイントを無効化します。"
            "（対処: langgraph-checkpoint-sqlite と aiosqlite をインストール）"
        )
        return None

    last_err: Exception | None = None
    # AsyncSqliteSaver expects a filesystem path (it passes this into aiosqlite.connect()).
    for conn in (str(db_path),):
        try:
            cm_or_saver = AsyncSqliteSaver.from_conn_string(conn)
            # Some versions return an async context manager.
            if hasattr(cm_or_saver, "__aenter__") and hasattr(cm_or_saver, "__aexit__"):
                return await exit_stack.enter_async_context(cm_or_saver)
            return cm_or_saver
        except Exception as e:
            last_err = e
            continue

    logger.warning(
        "SQLite checkpointer の初期化に失敗したため、チェックポイントを無効化します。db_path=%s",
        db_path,
        exc_info=last_err,
    )
    return None

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
    async def create_workflow(
        cls,
        mcp_config: dict[str, Any] | None,
        config_parser: MCPConfigParser | None,
        llm: BaseChatModel,
        *,
        checkpointer: Any | None = None,
        hitl_approval_tools: Sequence[str] | None = None,
        auto_approve: bool = False,
        tool_call_limit: int | None = None,
        tool_timeout_seconds: float | None = None,
        tool_timeout_retries: int | None = None,
    ) -> CompiledStateGraph:

        # LLM + MCP ツールでエージェントを作成
        allowed_langchain_tools = await MCPClientUtil.get_allowed_tools(mcp_config, config_parser)

        # Safety valves: cap tool calls and hard-timeout tool execution.
        # This enforces termination even if prompts are ignored.
        try:
            tool_call_limit_int = int(tool_call_limit) if tool_call_limit is not None else 0
        except (TypeError, ValueError):
            tool_call_limit_int = 0
        if tool_call_limit_int < 0:
            tool_call_limit_int = 0

        try:
            tool_timeout_seconds_f = float(tool_timeout_seconds) if tool_timeout_seconds is not None else 0.0
        except (TypeError, ValueError):
            tool_timeout_seconds_f = 0.0
        if tool_timeout_seconds_f < 0:
            tool_timeout_seconds_f = 0.0

        try:
            tool_timeout_retries_int = int(tool_timeout_retries) if tool_timeout_retries is not None else 0
        except (TypeError, ValueError):
            tool_timeout_retries_int = 0
        tool_timeout_retries_int = max(0, min(5, tool_timeout_retries_int))

        tool_calls_used = 0

        def _is_timeout_exception(err: BaseException) -> bool:
            if isinstance(err, asyncio.TimeoutError):
                return True
            if isinstance(err, RuntimeError) and "タイムアウト" in str(err):
                return True
            return False

        def _tool_error_text(tool_name: str, err: BaseException) -> str:
            err_type = type(err).__name__
            msg = str(err).strip()
            if msg:
                return f"ERROR: tool={tool_name} failed ({err_type}): {msg}"
            return f"ERROR: tool={tool_name} failed ({err_type})"

        async def _run_tool_with_guards(tool_name: str, orig_coro: Any, *args: Any, **kwargs: Any) -> str:
            nonlocal tool_calls_used

            attempts = tool_timeout_retries_int + 1
            last_err: BaseException | None = None

            for attempt in range(1, attempts + 1):
                if tool_call_limit_int and tool_calls_used >= tool_call_limit_int:
                    logger.warning(
                        "Tool call budget exceeded: tool=%s used=%s limit=%s",
                        tool_name,
                        tool_calls_used,
                        tool_call_limit_int,
                    )
                    return (
                        "ERROR: tool call budget exceeded. "
                        f"limit={tool_call_limit_int} used={tool_calls_used}. "
                        "同一入力でツールが繰り返し実行されたため中断しました。"
                    )

                tool_calls_used += 1
                try:
                    if tool_timeout_seconds_f and tool_timeout_seconds_f > 0:
                        # Give the tool a small cushion so inner timeouts can surface as normal output.
                        timeout = tool_timeout_seconds_f
                        return cast(str, await asyncio.wait_for(orig_coro(*args, **kwargs), timeout=timeout))
                    return cast(str, await orig_coro(*args, **kwargs))
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    last_err = e
                    if _is_timeout_exception(e) and attempt < attempts:
                        logger.warning(
                            "Tool timeout; retrying: tool=%s attempt=%s/%s",
                            tool_name,
                            attempt,
                            attempts,
                        )
                        continue

                    # Convert tool exceptions into normal tool output to avoid retry loops.
                    logger.exception(
                        "Tool invocation failed: tool=%s attempt=%s/%s",
                        tool_name,
                        attempt,
                        attempts,
                    )
                    return _tool_error_text(tool_name, e)

            if last_err is not None:
                return _tool_error_text(tool_name, last_err)
            return f"ERROR: tool={tool_name} failed (unknown error)"

        if allowed_langchain_tools and (tool_call_limit_int or (tool_timeout_seconds_f and tool_timeout_seconds_f > 0) or tool_timeout_retries_int):
            for tool in allowed_langchain_tools:
                tool_name = getattr(tool, "name", "(unknown)")
                orig_coro = getattr(tool, "coroutine", None)
                if orig_coro is not None:
                    async def _wrapped_coro(*args: Any, __tool_name: str = tool_name, __orig_coro: Any = orig_coro, **kwargs: Any) -> str:
                        return await _run_tool_with_guards(__tool_name, __orig_coro, *args, **kwargs)
                    try:
                        setattr(tool, "coroutine", _wrapped_coro)
                    except Exception:
                        # If the tool object is immutable, we leave it as-is.
                        pass

                orig_func = getattr(tool, "func", None)
                if orig_func is not None:
                    def _wrapped_func(*args: Any, __tool_name: str = tool_name, __orig_func: Any = orig_func, **kwargs: Any) -> str:
                        nonlocal tool_calls_used
                        if tool_call_limit_int and tool_calls_used >= tool_call_limit_int:
                            logger.warning(
                                "Tool call budget exceeded (sync): tool=%s used=%s limit=%s",
                                __tool_name,
                                tool_calls_used,
                                tool_call_limit_int,
                            )
                            return (
                                "ERROR: tool call budget exceeded. "
                                f"limit={tool_call_limit_int} used={tool_calls_used}. "
                                "同一入力でツールが繰り返し実行されたため中断しました。"
                            )
                        tool_calls_used += 1
                        try:
                            return cast(str, __orig_func(*args, **kwargs))
                        except Exception as e:
                            logger.exception("Tool invocation failed (sync): tool=%s", __tool_name)
                            return _tool_error_text(__tool_name, e)
                    try:
                        setattr(tool, "func", _wrapped_func)
                    except Exception:
                        pass

        approval_tools = [t for t in (hitl_approval_tools or []) if isinstance(t, str) and t.strip()]
        approval_tools_text = ", ".join(approval_tools) if approval_tools else "(なし)"

        if auto_approve:
            hitl_policy_text = Prompts.auto_approve_hitl_policy_text(approval_tools_text)
        else:
            hitl_policy_text = Prompts.normal_hitl_policy_text(approval_tools_text)

        # ツール実行用のエージェント
        # システムプロンプトで役割分担を指示する例。実際のプロンプトは用途に応じて調整してください。
        tool_agent_system_prompt = Prompts.tool_agent_system_prompt(hitl_policy_text)
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
        planner_agent_system_prompt = Prompts.tool_agent_user_prompt(tools_description, hitl_policy_text)
        planner_agent = create_agent(
            llm,
            [],
            system_prompt=planner_agent_system_prompt,
            name="planner_agent",
        )  # ツールは持たないシンプルなエージェント


        if auto_approve:
            supervisor_hitl_policy_text = Prompts.supervisor_hitl_policy_text(approval_tools_text)
        else:
            supervisor_hitl_policy_text = Prompts.supervisor_normal_hitl_policy_text(approval_tools_text)

        supervisor_prompt = Prompts.supervisor_system_prompt(tools_description, supervisor_hitl_policy_text)

        # Prefer tool execution agent first to reduce accidental planner-only loops.
        workflow = create_supervisor(
            [tool_agent],
            model=llm,
            prompt=supervisor_prompt,
        )

        # Compile and run
        if checkpointer is not None:
            try:
                graph = workflow.compile(name="mcp_supervisor", checkpointer=checkpointer)
            except TypeError:
                # Some versions may not accept checkpointer; fall back to no persistence.
                graph = workflow.compile(name="mcp_supervisor")
        else:
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

        trace_id = getattr(chat_request, "trace_id", None)
        # LangGraph checkpoint key is named `thread_id`. This project standardizes on `trace_id`.
        # If not provided, generate a W3C-compatible 32-hex trace_id.
        run_trace_id = (trace_id or uuid.uuid4().hex).lower()
        # Safety valve: cap LangGraph step recursion to avoid runaway agent/tool loops.
        # This does not affect normal flows, but prevents excessive repeated handoffs.
        try:
            recursion_limit_raw = int(getattr(self.runtime_config.features, "mcp_recursion_limit", 15) or 15)
        except (TypeError, ValueError):
            recursion_limit_raw = 15
        recursion_limit = max(1, min(200, recursion_limit_raw))

        try:
            tool_call_limit_raw = int(getattr(self.runtime_config.features, "mcp_tool_call_limit", 2) or 2)
        except (TypeError, ValueError):
            tool_call_limit_raw = 2
        tool_call_limit = max(1, min(50, tool_call_limit_raw))

        # Default tool hard timeout to LLM timeout. (Tool calls include file processing + LLM.)
        tool_timeout_cfg = getattr(self.runtime_config.features, "mcp_tool_timeout_seconds", None)
        try:
            tool_timeout_seconds = float(tool_timeout_cfg) if tool_timeout_cfg is not None else float(self.runtime_config.llm.timeout_seconds)
        except (TypeError, ValueError):
            tool_timeout_seconds = float(self.runtime_config.llm.timeout_seconds)
        if tool_timeout_seconds <= 0:
            tool_timeout_seconds = float(self.runtime_config.llm.timeout_seconds)

        try:
            tool_timeout_retries_raw = int(getattr(self.runtime_config.features, "mcp_tool_timeout_retries", 1) or 0)
        except (TypeError, ValueError):
            tool_timeout_retries_raw = 1
        tool_timeout_retries = max(0, min(5, tool_timeout_retries_raw))
        auto_approve = bool(getattr(chat_request, "auto_approve", False))
        try:
            max_retries_raw = int(getattr(chat_request, "auto_approve_max_retries", 0) or 0)
        except (TypeError, ValueError):
            max_retries_raw = 0
        max_retries = max(0, min(10, max_retries_raw))
        async with contextlib.AsyncExitStack() as exit_stack:
            checkpointer = await _create_sqlite_checkpointer(
                _default_checkpoint_db_path(self.runtime_config),
                exit_stack=exit_stack,
            )

            app = await MCPClientUtil.create_workflow(
                self.mcp_config,
                self.config_parser,
                llm,
                checkpointer=checkpointer,
                hitl_approval_tools=getattr(self.runtime_config.features, "hitl_approval_tools", None),
                auto_approve=auto_approve,
                tool_call_limit=tool_call_limit,
                tool_timeout_seconds=(tool_timeout_seconds + 5.0 if tool_timeout_seconds > 0 else 0.0),
                tool_timeout_retries=tool_timeout_retries,
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
            output_text, input_tokens, output_tokens = self._extract_output_and_usage(result)

            resp_type, extracted_text, hitl_kind, hitl_tool = _parse_supervisor_xml(output_text)
            user_text = extracted_text or output_text

            # Fallback: if the model ignored/misused the XML contract, infer HITL from plain text.
            inferred_kind, inferred_tool = _infer_hitl_from_plain_text(user_text)
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
                    output_text, add_in, add_out = self._extract_output_and_usage(result)
                    input_tokens += add_in
                    output_tokens += add_out

                    resp_type, extracted_text, hitl_kind, hitl_tool = _parse_supervisor_xml(output_text)
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
                output_text, add_in, add_out = self._extract_output_and_usage(result)
                input_tokens += add_in
                output_tokens += add_out

                resp_type, extracted_text, hitl_kind, hitl_tool = _parse_supervisor_xml(output_text)
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


if __name__ == "__main__":
    runtime_config = get_runtime_config()  # ここは適宜、実際の設定に合わせて初期化してください
    chat_request = ChatRequest(chat_history=ChatHistory(messages=[ChatMessage(role="user", content=[ChatContent(params={"type": "text", "text": "3 と 5 を足して"})])]))
    asyncio.run(MCPClient(runtime_config).chat(chat_request))