from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import contextlib
import re
from pathlib import Path

import asyncio
from pydantic import BaseModel, ConfigDict, Field, create_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools.structured import StructuredTool
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
    ConfigError,
    AiChatUtilConfig,
    get_runtime_config_path,
)
from ..util.file_path_resolver import resolve_existing_file_path
import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class MCPClientUtil:
    @classmethod
    def _maybe_wrap_req_nested_tool(cls, tool: Any) -> Any:
        """Wrap tools whose schema is `{req: {...}}` so callers can pass flat args.

        This is a generic integration hardening for MCP tools that use a single
        nested `req` field. Many LLMs tend to emit flat kwargs (prompt=..., timeout=...)
        which fails validation for such schemas. We absorb that here without
        coupling to a specific server implementation.
        """

        try:
            if getattr(tool, "name", None) is None:
                return tool

            schema = getattr(tool, "args_schema", None)
            if not isinstance(schema, Mapping):
                return tool

            required = schema.get("required")
            if not isinstance(required, Sequence) or "req" not in set(required):
                return tool

            props = schema.get("properties")
            if not isinstance(props, Mapping):
                return tool

            req_schema = props.get("req")
            if not isinstance(req_schema, Mapping):
                return tool

            # Only wrap the common pattern: a single required top-level `req` object.
            top_required = set(required)
            if top_required != {"req"}:
                return tool

            inner_props = req_schema.get("properties")
            if not isinstance(inner_props, Mapping) or not inner_props:
                return tool

            inner_required_raw = req_schema.get("required")
            inner_required = set(inner_required_raw) if isinstance(inner_required_raw, Sequence) else set()
            inner_keys = [k for k in inner_props.keys() if isinstance(k, str)]
            if not inner_keys:
                return tool

            # Build a permissive schema for the wrapper tool.
            # - Accept either `req={...}` or flat keys.
            # - Allow extra keys to avoid hard failures on harmless hallucinated kwargs.
            def _infer_py_type(json_schema: Any) -> Any:
                if isinstance(json_schema, Mapping):
                    t = json_schema.get("type")
                    if t == "string":
                        return str
                    if t == "integer":
                        return int
                    if t == "number":
                        return float
                    if t == "boolean":
                        return bool
                    if t == "object":
                        return dict[str, Any]
                    if t == "array":
                        return list[Any]
                return Any

            field_defs: dict[str, Any] = {
                "req": (dict[str, Any] | None, Field(default=None, description="Nested request payload")),
            }
            for k in inner_keys:
                js = inner_props.get(k)
                desc = js.get("description") if isinstance(js, Mapping) else None
                py_t = _infer_py_type(js)
                field_defs[k] = (py_t | None, Field(default=None, description=(str(desc) if desc else None)))

            # Avoid reserved pydantic create_model kwargs and help type checkers.
            safe_field_defs = {k: v for k, v in field_defs.items() if not k.startswith("__")}

            WrapperArgs: type[BaseModel] = create_model(  # type: ignore[assignment]
                f"ReqNormalized_{getattr(tool, 'name', 'tool')}",
                __config__=ConfigDict(extra="allow"),
                **cast(Any, safe_field_defs),
            )

            original_tool = tool
            original_response_format = cast(str | None, getattr(original_tool, "response_format", None))

            async def _wrapper_coroutine(**kwargs: Any) -> Any:
                req_in = kwargs.get("req")
                merged: dict[str, Any] = {}

                if isinstance(req_in, Mapping):
                    merged.update(dict(req_in))

                # Flat keys override nested values.
                for k in inner_keys:
                    if k in kwargs and kwargs[k] is not None:
                        merged[k] = kwargs[k]

                # Keep only keys defined by the inner schema.
                normalized = {k: merged.get(k) for k in inner_keys if k in merged and merged.get(k) is not None}

                missing = [k for k in inner_required if k not in normalized]
                if missing:
                    raise ValueError(
                        "Missing required fields for nested `req`: "
                        + ", ".join(sorted(missing))
                        + ". Provide them either inside `req` or as top-level arguments."
                    )

                # Delegate to the original tool with the canonical `{req:{...}}` payload.
                orig_coro = getattr(original_tool, "coroutine", None)
                if orig_coro is None:
                    # Fallback: best-effort via ainvoke (may drop artifact in some versions).
                    return await original_tool.ainvoke({"req": normalized})

                # MCP adapter tools commonly expose `coroutine(runtime=None, **arguments)`
                # and return `(content, artifact)` when response_format='content_and_artifact'.
                return await orig_coro(runtime=None, req=normalized)

            desc = str(getattr(tool, "description", "") or "")
            if desc:
                desc2 = desc.rstrip() + "\n\n(入力は `req` ネスト／フラットどちらでも可。内部で正規化します。)"
            else:
                desc2 = "(入力は `req` ネスト／フラットどちらでも可。内部で正規化します。)"

            wrapped = StructuredTool.from_function(
                func=None,
                coroutine=_wrapper_coroutine,
                name=str(getattr(tool, "name")),
                description=desc2,
                args_schema=WrapperArgs,
                infer_schema=False,
                response_format=cast(Any, original_response_format or "content"),
            )

            return wrapped
        except Exception:
            # If wrapping fails for any reason, fall back to the original tool.
            return tool

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

            # Optional: forward reserved x-mcp-* values from llm.extra_headers into MCP transports.
            # - x-mcp-<Header-Name>: forwarded to HTTP-like transports as headers['<Header-Name>']
            # - x-mcp-env-<ENV_NAME>: forwarded to stdio transports as env['<ENV_NAME>']
            # Values are already resolved (env ref) at runtime init.
            extra = getattr(runtime_config.llm, "extra_headers", None)
            mcp_headers: dict[str, str] = {}
            mcp_env: dict[str, str] = {}
            if isinstance(extra, dict) and extra:
                for raw_key, raw_val in extra.items():
                    if not isinstance(raw_key, str) or not isinstance(raw_val, str):
                        continue
                    key = raw_key.strip()
                    if not key:
                        continue

                    lower = key.lower()
                    if lower.startswith("x-mcp-env-"):
                        env_name = key[len("x-mcp-env-") :].strip()
                        if not env_name:
                            raise ConfigError(
                                "llm.extra_headers の x-mcp-env- プレフィックス指定が不正です（ENV名が空）"
                            )
                        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", env_name):
                            raise ConfigError(
                                "llm.extra_headers の環境変数名が不正です: "
                                f"{env_name} (key={raw_key!r})\n"
                                "対処: x-mcp-env-ENV_NAME の ENV_NAME は [A-Za-z_][A-Za-z0-9_]* を満たしてください。"
                            )
                        mcp_env[env_name] = raw_val
                        continue

                    if lower.startswith("x-mcp-"):
                        header_name = key[len("x-mcp-") :].strip()
                        if not header_name:
                            raise ConfigError(
                                "llm.extra_headers の x-mcp- プレフィックス指定が不正です（ヘッダー名が空）"
                            )
                        mcp_headers[header_name] = raw_val

            if mcp_config and (mcp_headers or mcp_env):
                for _, conn in mcp_config.items():
                    if not isinstance(conn, Mapping):
                        continue

                    transport = conn.get("transport")
                    conn_dict = cast(dict[str, Any], conn)

                    if transport == "stdio" and mcp_env:
                        env = conn_dict.get("env")
                        if env is None:
                            env = {}
                        if isinstance(env, Mapping):
                            env2 = dict(env)
                            # config.yml (runtime_config) takes precedence
                            env2.update(mcp_env)
                            conn_dict["env"] = env2

                    if transport in {"http", "sse", "websocket"} and mcp_headers:
                        headers = conn_dict.get("headers")
                        if headers is None:
                            headers = {}
                        if isinstance(headers, Mapping):
                            headers2 = dict(headers)
                            # config.yml (runtime_config) takes precedence
                            headers2.update(mcp_headers)
                            conn_dict["headers"] = headers2

            # Ensure the MCP server subprocess can resolve the same config.yml as this process.
            # When stdio servers are launched with a different working directory (e.g., `uv --directory`),
            # a relative AI_CHAT_UTIL_CONFIG like "config.yml" can break. Pass an absolute path.
            runtime_config_path = str(get_runtime_config_path())
            if mcp_config:
                for _, conn in mcp_config.items():
                    if isinstance(conn, Mapping) and conn.get("transport") == "stdio":
                        conn_dict = cast(dict[str, Any], conn)
                        env = conn_dict.get("env")
                        if env is None:
                            env = {}
                        if isinstance(env, Mapping):
                            env2 = dict(env)
                            # Always override to an absolute path for stability.
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
        for _, names in allowed_map.items():
            if names is None:
                continue
            if allowed_names is None:
                allowed_names = set()
            allowed_names.update(names)

        for tool in langchain_tools:
            tool_name = tool.name
            if allowed_names is None or tool_name in allowed_names:
                allowed_langchain_tools.append(cls._maybe_wrap_req_nested_tool(tool))
            else:
                logger.debug("Tool %s is not in allowedTools; skipped", tool_name)
        # あとはこれを LangChain の Agent や LLM (bind_tools) に渡すだけ！
        # example: 
        # llm_with_tools = ChatOpenAI().bind_tools(langchain_tools)
        
        logger.info("Loaded %d tools from MCP servers.", len(allowed_langchain_tools))
        return allowed_langchain_tools

    @classmethod
    def _infer_hitl_from_plain_text(cls, text: str) -> tuple[str | None, str | None]:
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


    @classmethod
    def _default_checkpoint_db_path(cls, runtime_config: AiChatUtilConfig) -> Path:
        """Pick a stable per-config SQLite path for LangGraph checkpoints."""

        base = runtime_config.paths.working_directory
        if base:
            root = Path(base).expanduser()
        else:
            root = get_runtime_config_path().parent
        p = (root / ".ai_chat_util" / "langgraph_checkpoints.sqlite").resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


    @classmethod
    async def _create_sqlite_checkpointer(cls, db_path: Path, *, exit_stack: contextlib.AsyncExitStack) -> Any | None:
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


    @classmethod
    def _is_timeout_exception(cls, err: BaseException) -> bool:
        if isinstance(err, asyncio.TimeoutError):
            return True
        if isinstance(err, RuntimeError) and "タイムアウト" in str(err):
            return True
        return False

    @classmethod
    def _tool_error_text(cls, tool_name: str, err: BaseException) -> str:
        err_type = type(err).__name__
        msg = str(err).strip()
        if msg:
            return f"ERROR: tool={tool_name} failed ({err_type}): {msg}"
        return f"ERROR: tool={tool_name} failed ({err_type})"

    @classmethod
    def _prepare_timeout_limits(cls, tool_call_limit: int | None, tool_timeout_seconds: float | None, tool_timeout_retries: int | None) -> tuple[int, float, int]:
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

        return tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int

    @classmethod
    async def agent_question_and_non_approval_response(
        cls, auto_approve: bool, resp_type: str, 
        max_retries: int, user_text: str, run_trace_id: str,
        input_tokens: int, output_tokens: int, 
        recursion_limit: int, app: Any) -> tuple[str, int, int]:
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
                output_text, add_in, add_out = cls._extract_output_and_usage(result)
                input_tokens += add_in
                output_tokens += add_out

                parsed_resp_type, extracted_text, _hitl_kind, _hitl_tool = cls._parse_supervisor_xml(output_text)
                # _parse_supervisor_xml は Optional を返すため、ここでは前回値をフォールバックする。
                resp_type = parsed_resp_type or resp_type
                user_text = extracted_text or output_text
                if resp_type != "question":
                    break

        return user_text, input_tokens, output_tokens


    @classmethod
    def _stringify_message_content(cls, content: Any) -> str:
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
    def _parse_supervisor_xml(cls, output_text: str|None) -> tuple[str | None, str | None, str | None, str | None]:
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

    @classmethod
    def create_tool_agent(
        cls,
        llm: BaseChatModel,
        auto_approve: bool,
        hitl_approval_tools: Sequence[str] | None,
        allowed_langchain_tools: list[Any],
    ) -> Any:
        # ツール実行用のエージェント
        # システムプロンプトで役割分担を指示する例。実際のプロンプトは用途に応じて調整してください。
        approval_tools = [t for t in (hitl_approval_tools or []) if isinstance(t, str) and t.strip()]
        approval_tools_text = ", ".join(approval_tools) if approval_tools else "(なし)"

        if auto_approve:
            hitl_policy_text = Prompts.auto_approve_hitl_policy_text(approval_tools_text)
        else:
            hitl_policy_text = Prompts.normal_hitl_policy_text(approval_tools_text)

        tool_agent_system_prompt = Prompts.tool_agent_system_prompt(hitl_policy_text)
        tool_agent = create_agent(
            llm,
            allowed_langchain_tools,
            system_prompt=tool_agent_system_prompt,
            name="tool_agent",
        )
        return tool_agent

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
        tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = cls._prepare_timeout_limits(
            tool_call_limit, tool_timeout_seconds, tool_timeout_retries
        )

        tool_calls_used = 0


        def _guard_output(payload: str, *, response_format: str | None, artifact: Any | None = None) -> Any:
            """Return tool output compatible with LangChain's tool response_format.

            MCP tools created via langchain-mcp-adapters commonly use
            response_format='content_and_artifact', where LangChain expects a
            (content, artifact) two-tuple. If we return a plain string here,
            LangChain raises ValueError.
            """

            if response_format == "content_and_artifact":
                if artifact is None:
                    artifact = {}
                return (payload, artifact)
            return payload


        async def _run_tool_with_guards(
            tool_name: str,
            orig_coro: Any,
            response_format: str | None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
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
                    text = (
                        "ERROR: tool call budget exceeded. "
                        f"limit={tool_call_limit_int} used={tool_calls_used}. "
                        "同一入力でツールが繰り返し実行されたため中断しました。"
                    )
                    return _guard_output(
                        text,
                        response_format=response_format,
                        artifact={"error": "tool_call_budget_exceeded", "tool": tool_name, "limit": tool_call_limit_int, "used": tool_calls_used},
                    )

                tool_calls_used += 1
                try:
                    if tool_timeout_seconds_f and tool_timeout_seconds_f > 0:
                        # Give the tool a small cushion so inner timeouts can surface as normal output.
                        timeout = tool_timeout_seconds_f
                        return await asyncio.wait_for(orig_coro(*args, **kwargs), timeout=timeout)
                    return await orig_coro(*args, **kwargs)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    last_err = e
                    if cls._is_timeout_exception(e) and attempt < attempts:
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
                    return _guard_output(
                        cls._tool_error_text(tool_name, e),
                        response_format=response_format,
                        artifact={"error": "tool_invocation_failed", "tool": tool_name, "exception": type(e).__name__},
                    )

            if last_err is not None:
                return _guard_output(
                    cls._tool_error_text(tool_name, last_err),
                    response_format=response_format,
                    artifact={"error": "tool_invocation_failed", "tool": tool_name, "exception": type(last_err).__name__},
                )
            return _guard_output(
                f"ERROR: tool={tool_name} failed (unknown error)",
                response_format=response_format,
                artifact={"error": "tool_invocation_failed", "tool": tool_name},
            )

        if allowed_langchain_tools and (tool_call_limit_int or (tool_timeout_seconds_f and tool_timeout_seconds_f > 0) or tool_timeout_retries_int):
            for tool in allowed_langchain_tools:
                tool_name = getattr(tool, "name", "(unknown)")
                tool_response_format = cast(str | None, getattr(tool, "response_format", None))
                orig_coro = getattr(tool, "coroutine", None)
                if orig_coro is not None:
                    async def _wrapped_coro(
                        *args: Any,
                        __tool_name: str = tool_name,
                        __orig_coro: Any = orig_coro,
                        __response_format: str | None = tool_response_format,
                        **kwargs: Any,
                    ) -> Any:
                        return await _run_tool_with_guards(__tool_name, __orig_coro, __response_format, *args, **kwargs)
                    try:
                        setattr(tool, "coroutine", _wrapped_coro)
                    except Exception:
                        # If the tool object is immutable, we leave it as-is.
                        pass

                orig_func = getattr(tool, "func", None)
                if orig_func is not None:
                    def _wrapped_func(
                        *args: Any,
                        __tool_name: str = tool_name,
                        __orig_func: Any = orig_func,
                        __response_format: str | None = tool_response_format,
                        **kwargs: Any,
                    ) -> Any:
                        nonlocal tool_calls_used
                        if tool_call_limit_int and tool_calls_used >= tool_call_limit_int:
                            logger.warning(
                                "Tool call budget exceeded (sync): tool=%s used=%s limit=%s",
                                __tool_name,
                                tool_calls_used,
                                tool_call_limit_int,
                            )
                            text = (
                                "ERROR: tool call budget exceeded. "
                                f"limit={tool_call_limit_int} used={tool_calls_used}. "
                                "同一入力でツールが繰り返し実行されたため中断しました。"
                            )
                            return _guard_output(
                                text,
                                response_format=__response_format,
                                artifact={"error": "tool_call_budget_exceeded", "tool": __tool_name, "limit": tool_call_limit_int, "used": tool_calls_used},
                            )
                        tool_calls_used += 1
                        try:
                            return __orig_func(*args, **kwargs)
                        except Exception as e:
                            logger.exception("Tool invocation failed (sync): tool=%s", __tool_name)
                            return _guard_output(
                                cls._tool_error_text(__tool_name, e),
                                response_format=__response_format,
                                artifact={"error": "tool_invocation_failed", "tool": __tool_name, "exception": type(e).__name__},
                            )
                    try:
                        setattr(tool, "func", _wrapped_func)
                    except Exception:
                        pass

        approval_tools = [t for t in (hitl_approval_tools or []) if isinstance(t, str) and t.strip()]
        approval_tools_text = ", ".join(approval_tools) if approval_tools else "(なし)"

        tools_description = "\n".join(f"## name: {tool.name}\n - description: {tool.description}\n - args_schema: {tool.args_schema}\n" for tool in allowed_langchain_tools)
        logger.info("Allowed tools:\n%s", tools_description)


        # ツール実行用のエージェント
        # システムプロンプトで役割分担を指示する例。実際のプロンプトは用途に応じて調整してください。
        tool_agent = cls.create_tool_agent(llm, auto_approve, hitl_approval_tools, allowed_langchain_tools)

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
