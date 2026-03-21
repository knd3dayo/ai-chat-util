from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import contextlib
import re
from pathlib import Path
from langchain_litellm import ChatLiteLLMRouter
from litellm.router import Router

import asyncio
from pydantic import BaseModel, ConfigDict, Field, create_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools.structured import StructuredTool
from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import BaseChatModel
from .prompts import CodingAgentPrompts, PromptsBase
try:
    # Async checkpointer for LangGraph when using app.ainvoke()/astream()
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except Exception:  # pragma: no cover
    AsyncSqliteSaver = None  # type: ignore[assignment]

from ai_chat_util_base.config.ai_chat_util_mcp_config import MCPServerConfig
from ai_chat_util_base.config.runtime import (
    AiChatUtilConfig,
    AutonomousAgentUtilConfig,
    get_runtime_config_path,
)
import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class ToolLimits(BaseModel):
    tool_call_limit: int = Field(
        default=50,
        description="ツール呼び出し回数の上限。0またはNoneで無制限。安全弁として、マイナス値は0として扱います。",
    )
    tool_timeout_seconds: float = Field(
        default=0,
        description="ツール呼び出しのタイムアウト秒数。0またはNoneで無制限。安全弁として、マイナス値は0として扱います。",
    )
    tool_timeout_retries: int  = Field(
        default=5,
        description="タイムアウト発生時のリトライ回数。0でリトライなし。安全弁として、負の値は0として扱います。過度なリトライを防ぐため、最大5回までに制限します。",
    )
    max_retries: int = Field(
        default=3,
        description="ツール呼び出し失敗時の最大リトライ回数。0でリトライなし。安全弁として、負の値は0として扱います。過度なリトライを防ぐため、最大10回までに制限します。",
    )
    
    auto_approve: bool = Field(
        default=False,
        description="Trueの場合、tool_call_limitやtool_timeout_secondsで定められた制限を超えるツール呼び出しに対しても、ユーザーの明示的な承認なしで自動的に許可します。安全弁として、tool_call_limitやtool_timeout_secondsの値が0（無制限）でない場合にのみ有効になります。",
    )
    tool_recursion_limit: int = Field(
        default=200,
        description="ツール呼び出しの再帰制限。安全弁として、負の値は1として扱います。過度な再帰を防ぐため、最大200回までに制限します。",
    )

    @classmethod
    def from_config(cls, config: AiChatUtilConfig) -> "ToolLimits":
        """Build ToolLimits from runtime config.

        Semantics:
        - 0/None means unlimited (for tool_call_limit/tool_timeout_seconds).
        - Negative values are clamped to 0 (or 1 for recursion_limit).
        """

        # tool_call_limit: 0..50 (0 means unlimited)
        try:
            raw_call_limit = getattr(config.features, "mcp_tool_call_limit", None)
            tool_call_limit_raw = int(raw_call_limit) if raw_call_limit is not None else 2
        except (TypeError, ValueError):
            tool_call_limit_raw = 2
        tool_call_limit = max(0, min(50, tool_call_limit_raw))

        # tool_timeout_seconds:
        # - If explicitly set to 0 => unlimited (do not replace with LLM timeout)
        # - If None => default to LLM timeout for safety
        tool_timeout_cfg = getattr(config.features, "mcp_tool_timeout_seconds", None)
        if tool_timeout_cfg is None:
            try:
                tool_timeout_seconds = float(config.llm.timeout_seconds)
            except (TypeError, ValueError):
                tool_timeout_seconds = 0.0
        else:
            try:
                tool_timeout_seconds = float(tool_timeout_cfg)
            except (TypeError, ValueError):
                try:
                    tool_timeout_seconds = float(config.llm.timeout_seconds)
                except (TypeError, ValueError):
                    tool_timeout_seconds = 0.0
        if tool_timeout_seconds < 0:
            tool_timeout_seconds = 0.0

        # tool_timeout_retries: 0..5
        try:
            raw_timeout_retries = getattr(config.features, "mcp_tool_timeout_retries", None)
            tool_timeout_retries_raw = int(raw_timeout_retries) if raw_timeout_retries is not None else 1
        except (TypeError, ValueError):
            tool_timeout_retries_raw = 1
        tool_timeout_retries = max(0, min(5, tool_timeout_retries_raw))

        auto_approve = bool(getattr(config, "auto_approve", False))
        try:
            raw_max_retries = getattr(config, "auto_approve_max_retries", None)
            max_retries_raw = int(raw_max_retries) if raw_max_retries is not None else 0
        except (TypeError, ValueError):
            max_retries_raw = 0
        max_retries = max(0, min(10, max_retries_raw))

        # recursion limit: 1..200 (negative => 1)
        try:
            raw_recursion = getattr(config, "tool_recursion_limit", 15)
            tool_recursion_limit_raw = int(raw_recursion) if raw_recursion is not None else 15
        except (TypeError, ValueError):
            tool_recursion_limit_raw = 15
        tool_recursion_limit = max(1, min(200, tool_recursion_limit_raw))

        return cls(
            tool_call_limit=tool_call_limit,
            tool_timeout_seconds=tool_timeout_seconds,
            tool_timeout_retries=tool_timeout_retries,
            auto_approve=auto_approve,
            max_retries=max_retries,
            tool_recursion_limit=tool_recursion_limit,
        )

    def guard_params(self) -> tuple[int, float, int]:
        """Normalize limits for guard execution.

        Returns (tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int)
        where 0 means unlimited.
        """

        try:
            tool_call_limit_int = int(self.tool_call_limit)
        except (TypeError, ValueError):
            tool_call_limit_int = 0
        if tool_call_limit_int < 0:
            tool_call_limit_int = 0

        try:
            tool_timeout_seconds_f = float(self.tool_timeout_seconds)
        except (TypeError, ValueError):
            tool_timeout_seconds_f = 0.0
        if tool_timeout_seconds_f < 0:
            tool_timeout_seconds_f = 0.0

        try:
            tool_timeout_retries_int = int(self.tool_timeout_retries)
        except (TypeError, ValueError):
            tool_timeout_retries_int = 0
        tool_timeout_retries_int = max(0, min(5, tool_timeout_retries_int))

        return tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int

    @staticmethod
    def is_timeout_exception(err: BaseException) -> bool:
        if isinstance(err, asyncio.TimeoutError):
            return True
        if isinstance(err, RuntimeError) and "タイムアウト" in str(err):
            return True
        return False

    @staticmethod
    def tool_error_text(tool_name: str, err: BaseException) -> str:
        err_type = type(err).__name__
        msg = str(err).strip()
        if msg:
            return f"ERROR: tool={tool_name} failed ({err_type}): {msg}"
        return f"ERROR: tool={tool_name} failed ({err_type})"


class MCPClientUtil:
    @classmethod
    def _apply_tool_execution_guards(
        cls,
        allowed_langchain_tools: Sequence[Any],
        *,
        tool_call_state: dict[str, int],
        tool_call_limit_int: int,
        tool_timeout_seconds_f: float,
        tool_timeout_retries_int: int,
    ) -> None:
        """Apply safety valves to tool execution by wrapping tool callables.

        This mutates tool objects in-place (best-effort). If a tool is immutable,
        we leave it as-is.

        Guards enforced:
        - Shared tool call budget across all tools in a workflow
        - Async execution timeout + retries
        - Convert tool exceptions into normal tool outputs
        """

        if not allowed_langchain_tools:
            return

        needs_guards = bool(
            tool_call_limit_int
            or (tool_timeout_seconds_f and tool_timeout_seconds_f > 0)
            or tool_timeout_retries_int
        )
        if not needs_guards:
            return

        def _wrap_sync(
            *,
            tool_name: str,
            orig_func: Any,
            response_format: str | None,
        ) -> Any:
            def _wrapped_func(*args: Any, **kwargs: Any) -> Any:
                used = int(tool_call_state.get("used", 0) or 0)
                if used < 0:
                    used = 0
                    tool_call_state["used"] = 0

                if tool_call_limit_int and used >= tool_call_limit_int:
                    logger.warning(
                        "Tool call budget exceeded (sync): tool=%s used=%s limit=%s",
                        tool_name,
                        used,
                        tool_call_limit_int,
                    )
                    text = (
                        "ERROR: tool call budget exceeded. "
                        f"limit={tool_call_limit_int} used={used}. "
                        "同一入力でツールが繰り返し実行されたため中断しました。"
                    )
                    return cls._guard_output(
                        text,
                        response_format=response_format,
                        artifact={
                            "error": "tool_call_budget_exceeded",
                            "tool": tool_name,
                            "limit": tool_call_limit_int,
                            "used": used,
                        },
                    )

                tool_call_state["used"] = used + 1
                try:
                    return orig_func(*args, **kwargs)
                except Exception as e:
                    logger.exception("Tool invocation failed (sync): tool=%s", tool_name)
                    return cls._guard_output(
                        ToolLimits.tool_error_text(tool_name, e),
                        response_format=response_format,
                        artifact={
                            "error": "tool_invocation_failed",
                            "tool": tool_name,
                            "exception": type(e).__name__,
                        },
                    )

            return _wrapped_func

        def _wrap_async(
            *,
            tool_name: str,
            orig_coro: Any,
            response_format: str | None,
        ) -> Any:
            async def _wrapped_coro(*args: Any, **kwargs: Any) -> Any:
                return await cls._run_tool_with_guards(
                    tool_name,
                    orig_coro,
                    response_format,
                    tool_call_state,
                    tool_call_limit_int,
                    tool_timeout_seconds_f,
                    tool_timeout_retries_int,
                    *args,
                    **kwargs,
                )

            return _wrapped_coro

        for tool in allowed_langchain_tools:
            tool_name = str(getattr(tool, "name", "(unknown)") or "(unknown)")
            tool_response_format = cast(str | None, getattr(tool, "response_format", None))

            orig_coro = getattr(tool, "coroutine", None)
            if orig_coro is not None:
                try:
                    setattr(
                        tool,
                        "coroutine",
                        _wrap_async(
                            tool_name=tool_name,
                            orig_coro=orig_coro,
                            response_format=tool_response_format,
                        ),
                    )
                except Exception:
                    # If the tool object is immutable, we leave it as-is.
                    pass

            orig_func = getattr(tool, "func", None)
            if orig_func is not None:
                try:
                    setattr(
                        tool,
                        "func",
                        _wrap_sync(
                            tool_name=tool_name,
                            orig_func=orig_func,
                            response_format=tool_response_format,
                        ),
                    )
                except Exception:
                    pass

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
    def create_mcp_config(cls, runtime_config: AiChatUtilConfig) -> MCPServerConfig|None:
        config = runtime_config.get_mcp_server_config()
        return config

    @classmethod
    async def get_allowed_tools(cls, config_parser: MCPServerConfig | None) -> list[Any]:
        if config_parser is None:
            return []

        allowed_langchain_tools = []
        langchain_config = config_parser.to_langchain_config()
        client = MultiServerMCPClient(langchain_config)
        # LangChainのツールリストを取得
        langchain_tools = await client.get_tools()
    
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

        base = runtime_config.mcp.working_directory
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
    def _guard_output(cls, payload: str, *, response_format: str | None, artifact: Any | None = None) -> Any:
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


    @classmethod
    async def _run_tool_with_guards(
        cls,
        tool_name: str,
        orig_coro: Any,
        response_format: str | None,
        tool_call_state: dict[str, int],
        tool_call_limit_int: int,
        tool_timeout_seconds_f: float,
        tool_timeout_retries_int: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        attempts = tool_timeout_retries_int + 1
        last_err: BaseException | None = None

        used = int(tool_call_state.get("used", 0) or 0)
        if used < 0:
            used = 0
            tool_call_state["used"] = 0

        for attempt in range(1, attempts + 1):
            used = int(tool_call_state.get("used", 0) or 0)
            if used < 0:
                used = 0
                tool_call_state["used"] = 0

            if tool_call_limit_int and used >= tool_call_limit_int:
                logger.warning(
                    "Tool call budget exceeded: tool=%s used=%s limit=%s",
                    tool_name,
                    used,
                    tool_call_limit_int,
                )
                text = (
                    "ERROR: tool call budget exceeded. "
                    f"limit={tool_call_limit_int} used={used}. "
                    "同一入力でツールが繰り返し実行されたため中断しました。"
                )
                return cls._guard_output(
                    text,
                    response_format=response_format,
                    artifact={"error": "tool_call_budget_exceeded", "tool": tool_name, "limit": tool_call_limit_int, "used": used},
                )

            tool_call_state["used"] = used + 1
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
                if ToolLimits.is_timeout_exception(e) and attempt < attempts:
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
                return cls._guard_output(
                    ToolLimits.tool_error_text(tool_name, e),
                    response_format=response_format,
                    artifact={"error": "tool_invocation_failed", "tool": tool_name, "exception": type(e).__name__},
                )

        if last_err is not None:
            return cls._guard_output(
                ToolLimits.tool_error_text(tool_name, last_err),
                response_format=response_format,
                artifact={"error": "tool_invocation_failed", "tool": tool_name, "exception": type(last_err).__name__},
            )
        return cls._guard_output(
            f"ERROR: tool={tool_name} failed (unknown error)",
            response_format=response_format,
            artifact={"error": "tool_invocation_failed", "tool": tool_name},
        )

    @classmethod
    def create_sub_agents(
        cls,
        runtime_config: AiChatUtilConfig,
        config: AutonomousAgentUtilConfig | None,
        llm: BaseChatModel,
        prompts: PromptsBase,
        tool_limits: ToolLimits | None,
        allowed_langchain_tools: list[Any],
    ) -> list[Any]:
        logger.info("Creating sub-agents...")

        # Safety valves: cap tool calls and hard-timeout tool execution.
        # This enforces termination even if prompts are ignored.
        if tool_limits is not None:
            tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = tool_limits.guard_params()
        else:
            tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = (0, 0.0, 0)

        # Shared tool call counter across all tools within this workflow.
        # Using a mutable container avoids invalid `nonlocal` usage across methods.
        tool_call_state: dict[str, int] = {"used": 0}

        cls._apply_tool_execution_guards(
            allowed_langchain_tools,
            tool_call_state=tool_call_state,
            tool_call_limit_int=tool_call_limit_int,
            tool_timeout_seconds_f=tool_timeout_seconds_f,
            tool_timeout_retries_int=tool_timeout_retries_int,
        )


        coding_agent_name = config.endpoint.mcp_server_name if config else None

        hitl_approval_tools = runtime_config.features.hitl_approval_tools or []

        # allowed_langchain_toolsにcoding_agent_nameと一致するツールがあれば、コードエージェントを作成する。
        agents = []
        if coding_agent_name and any(getattr(t, "name", None) == coding_agent_name for t in allowed_langchain_tools):
            logger.info("Creating code agent for MCP server '%s'...", coding_agent_name)
            code_agent = cls.create_code_agent(llm, prompts, tool_limits, hitl_approval_tools, allowed_langchain_tools)
            agents.append(code_agent)
            # allowed_langchain_toolsからcoding_agent_nameと一致するツールを除外して、ツールエージェントを作成する。
            allowed_langchain_tools = [t for t in allowed_langchain_tools if getattr(t, "name", None) != coding_agent_name]

        tool_agent_tool_names = [getattr(t, "name", None) for t in allowed_langchain_tools]
        logger.info(f"Creating tool agent with tools: {tool_agent_tool_names}")
        tool_agent = cls.create_tool_agent(llm, prompts, tool_limits, hitl_approval_tools, allowed_langchain_tools)
        agents.append(tool_agent)
        # 他のサブエージェントも必要に応じてここで作成できます。
        return agents

    @classmethod
    def create_code_agent(
        cls,
        llm: BaseChatModel,
        prompts: PromptsBase,
        tool_limits: ToolLimits | None,
        hitl_approval_tools: Sequence[str] | None,
        allowed_langchain_tools: list[Any],
    ) -> Any:
        # ツール実行用のエージェント
        # システムプロンプトで役割分担を指示する例。実際のプロンプトは用途に応じて調整してください。
        approval_tools = [t for t in (hitl_approval_tools or []) if isinstance(t, str) and t.strip()]
        approval_tools_text = ", ".join(approval_tools) if approval_tools else "(なし)"

        if tool_limits is not None and tool_limits.auto_approve:
            hitl_policy_text = prompts.auto_approve_hitl_policy_text(approval_tools_text)
        else:
            hitl_policy_text = prompts.normal_hitl_policy_text(approval_tools_text)

        tool_agent_system_prompt = prompts.tool_agent_system_prompt(hitl_policy_text)
        tool_agent = create_agent(
            llm,
            allowed_langchain_tools,
            system_prompt=tool_agent_system_prompt,
            name="tool_agent",
        )
        return tool_agent

    @classmethod
    def create_tool_agent(
        cls,
        llm: BaseChatModel,
        prompts: PromptsBase,
        tool_limits: ToolLimits | None,
        hitl_approval_tools: Sequence[str] | None,
        allowed_langchain_tools: list[Any],
    ) -> Any:
        # ツール実行用のエージェント
        # システムプロンプトで役割分担を指示する例。実際のプロンプトは用途に応じて調整してください。
        approval_tools = [t for t in (hitl_approval_tools or []) if isinstance(t, str) and t.strip()]
        approval_tools_text = ", ".join(approval_tools) if approval_tools else "(なし)"

        if tool_limits is not None and tool_limits.auto_approve:
            hitl_policy_text = prompts.auto_approve_hitl_policy_text(approval_tools_text)
        else:
            hitl_policy_text = prompts.normal_hitl_policy_text(approval_tools_text)

        tool_agent_system_prompt = prompts.tool_agent_system_prompt(hitl_policy_text)
        tool_agent = create_agent(
            llm,
            allowed_langchain_tools,
            system_prompt=tool_agent_system_prompt,
            name="tool_agent",
        )
        return tool_agent

    @classmethod
    def create_llm(cls, runtime_config: AiChatUtilConfig) -> BaseChatModel:
        litellm_router = Router(model_list=runtime_config.llm.create_litellm_model_list())
        llm = ChatLiteLLMRouter(router=litellm_router, model_name=runtime_config.llm.completion_model)
        return llm
    
    @classmethod
    async def create_workflow(
        cls,
        runtime_config: AiChatUtilConfig ,
        agent_config: AutonomousAgentUtilConfig | None,
        prompts: PromptsBase,
        *,
        checkpointer: Any | None = None,
        tool_limits: ToolLimits | None = None,
    ) -> CompiledStateGraph:

        # LLM + MCP ツールでエージェントを作成
        llm = cls.create_llm(runtime_config)
        mcp_config = MCPClientUtil.create_mcp_config(runtime_config)
        allowed_langchain_tools = await MCPClientUtil.get_allowed_tools(mcp_config)

        # ツール実行用のエージェント
        # システムプロンプトで役割分担を指示する例。実際のプロンプトは用途に応じて調整してください。
        sub_agents = cls.create_sub_agents(
            runtime_config,
            agent_config,
            llm, prompts, tool_limits, 
            allowed_langchain_tools
            )

        hitl_approval_tools = runtime_config.features.hitl_approval_tools or []
        approval_tools = [t for t in (hitl_approval_tools or []) if isinstance(t, str) and t.strip()]
        approval_tools_text = ", ".join(approval_tools) if approval_tools else "(なし)"

        tools_description = "\n".join(f"## name: {tool.name}\n - description: {tool.description}\n - args_schema: {tool.args_schema}\n" for tool in allowed_langchain_tools)
        logger.info("Allowed tools:\n%s", tools_description)

        if tool_limits is not None and tool_limits.auto_approve:
            supervisor_hitl_policy_text = prompts.supervisor_hitl_policy_text(approval_tools_text)
        else:
            supervisor_hitl_policy_text = prompts.supervisor_normal_hitl_policy_text(approval_tools_text)

        supervisor_prompt = prompts.supervisor_system_prompt(tools_description, supervisor_hitl_policy_text)

        # Prefer tool execution agent first to reduce accidental planner-only loops.
        workflow = create_supervisor(
            sub_agents,
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
