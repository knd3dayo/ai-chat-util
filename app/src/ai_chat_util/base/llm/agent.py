from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import asyncio
import copy
import json
from pathlib import Path
import re
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
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
    CodingAgentUtilConfig,
)
import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class ToolLimits(BaseModel):
    followup_tool_call_limit: int = Field(
        default=8,
        description="status/get_result/workspace_path/cancel のような追跡系ツール専用の上限。0またはNoneで無制限。",
    )
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
            tool_call_limit_raw = int(raw_call_limit) if raw_call_limit is not None else 4
        except (TypeError, ValueError):
            tool_call_limit_raw = 4
        tool_call_limit = max(0, min(50, tool_call_limit_raw))

        try:
            raw_followup_limit = getattr(config.features, "mcp_followup_tool_call_limit", None)
            followup_tool_call_limit_raw = int(raw_followup_limit) if raw_followup_limit is not None else 8
        except (TypeError, ValueError):
            followup_tool_call_limit_raw = 8
        followup_tool_call_limit = max(0, min(50, followup_tool_call_limit_raw))

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
        # NOTE: LangGraph's recursion_limit is passed via app.ainvoke(config={..., "recursion_limit": N}).
        # This project's YAML schema defines it under features.mcp_recursion_limit.
        try:
            raw_recursion = getattr(getattr(config, "features", None), "mcp_recursion_limit", 50)
            tool_recursion_limit_raw = int(raw_recursion) if raw_recursion is not None else 50
        except (TypeError, ValueError):
            tool_recursion_limit_raw = 50
        tool_recursion_limit = max(1, min(200, tool_recursion_limit_raw))

        return cls(
            followup_tool_call_limit=followup_tool_call_limit,
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
    def effective_call_limits(
        tool_call_limit_int: int,
        followup_tool_call_limit_int: int,
        explicit_user_file_paths: Sequence[str] | None = None,
    ) -> tuple[int, int]:
        normalized_paths = [
            str(path).strip()
            for path in (explicit_user_file_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        if not normalized_paths:
            return tool_call_limit_int, followup_tool_call_limit_int

        effective_tool_limit = tool_call_limit_int
        effective_followup_limit = followup_tool_call_limit_int

        if effective_tool_limit > 0:
            effective_tool_limit = max(effective_tool_limit, 6)
        if effective_followup_limit > 0:
            effective_followup_limit = max(effective_followup_limit, 12)

        return effective_tool_limit, effective_followup_limit

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

    @staticmethod
    def tool_budget_exceeded_text(tool_name: str, *, limit: int, used: int) -> str:
        return (
            "ERROR: tool call budget exceeded. "
            f"error=tool_call_budget_exceeded tool={tool_name} limit={limit} used={used}. "
            "これ以上ツールを再試行せず、既に取得済みの結果だけで回答を完了してください。"
        )

    @staticmethod
    def is_followup_tool(tool_name: str) -> bool:
        normalized = (tool_name or "").strip().lower()
        return normalized in {"status", "get_result", "workspace_path", "cancel"}

    @staticmethod
    def is_reusable_followup_tool(tool_name: str) -> bool:
        normalized = (tool_name or "").strip().lower()
        return normalized in {"get_result", "workspace_path"}

    @staticmethod
    def _freeze_for_cache(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {
                str(k): ToolLimits._freeze_for_cache(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [ToolLimits._freeze_for_cache(v) for v in value]
        if isinstance(value, set):
            return sorted(ToolLimits._freeze_for_cache(v) for v in value)
        return repr(value)

    @classmethod
    def build_call_cache_key(cls, tool_name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
        payload = {
            "tool": tool_name,
            "args": cls._freeze_for_cache(list(args)),
            "kwargs": cls._freeze_for_cache(dict(kwargs)),
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def is_cacheable_tool_result(result: Any) -> bool:
        if isinstance(result, str):
            return not result.lstrip().startswith("ERROR:")
        if isinstance(result, tuple) and len(result) == 2:
            artifact = result[1]
            if isinstance(artifact, Mapping) and artifact.get("error"):
                return False
            text = result[0]
            return not (isinstance(text, str) and text.lstrip().startswith("ERROR:"))
        return True

    @classmethod
    def should_cache_tool_result(cls, tool_name: str, result: Any) -> bool:
        normalized = (tool_name or "").strip().lower()
        if normalized == "execute":
            return False
        if cls.is_followup_tool(normalized) and not cls.is_reusable_followup_tool(normalized):
            return False
        return cls.is_cacheable_tool_result(result)

    @staticmethod
    def _extract_single_explicit_user_file_path(tool_call_state: Mapping[str, Any]) -> str | None:
        candidates = tool_call_state.get("explicit_user_file_paths")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
            return None

        normalized = [str(value).strip() for value in candidates if isinstance(value, str) and str(value).strip()]
        if len(normalized) != 1:
            return None
        return normalized[0]

    @staticmethod
    def _contains_explicit_file_path(text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"(?:[A-Za-z]:[\\/]|/)[^\s'\"`]+", text))

    @staticmethod
    def _looks_like_heading_extraction_task(prompt: str) -> bool:
        normalized = (prompt or "").lower()
        if not normalized:
            return False
        return any(token in normalized for token in ("heading", "headings", "見出し", "markdown heading", "heading_line_exact"))

    @classmethod
    def normalize_execute_arguments(
        cls,
        tool_name: str,
        tool_call_state: Mapping[str, Any],
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if (tool_name or "").strip().lower() != "execute":
            return tuple(args), dict(kwargs)

        normalized_kwargs = dict(kwargs)
        req = normalized_kwargs.get("req")
        if isinstance(req, Mapping):
            normalized_req = dict(req)
        else:
            normalized_req = {}

        moved_top_level_fields = False
        for field_name in ("prompt", "workspace_path", "timeout", "task_id", "trace_id"):
            if field_name in normalized_kwargs:
                top_level_value = normalized_kwargs.pop(field_name)
                if field_name not in normalized_req:
                    normalized_req[field_name] = top_level_value
                moved_top_level_fields = True

        if moved_top_level_fields or isinstance(req, Mapping):
            normalized_kwargs["req"] = normalized_req

        target_file_path = cls._extract_single_explicit_user_file_path(tool_call_state)
        if not target_file_path:
            return tuple(args), normalized_kwargs

        if not normalized_req:
            return tuple(args), normalized_kwargs

        prompt = normalized_req.get("prompt")
        workspace_path = normalized_req.get("workspace_path")
        if not isinstance(prompt, str) or not prompt.strip() or target_file_path in prompt or cls._contains_explicit_file_path(prompt):
            return tuple(args), normalized_kwargs
        if not isinstance(workspace_path, str) or not workspace_path.strip():
            return tuple(args), normalized_kwargs

        try:
            workspace_dir = Path(workspace_path).expanduser().resolve()
            file_path = Path(target_file_path).expanduser().resolve()
            file_path.relative_to(workspace_dir)
        except Exception:
            return tuple(args), normalized_kwargs

        prompt_suffix_lines = [
            f"Target file path: {file_path.as_posix()}",
            "Use only this file as the source of truth for the requested extraction.",
            "Do not infer headings or content from any other file.",
        ]
        if cls._looks_like_heading_extraction_task(prompt):
            prompt_suffix_lines.extend(
                [
                    "Read the target file directly and output exact Markdown heading lines from that file.",
                    "For each extracted heading line, print one line in the format: HEADING_LINE_EXACT: <exact heading line>",
                ]
            )

        normalized_req["prompt"] = prompt.rstrip() + "\n\n" + "\n".join(prompt_suffix_lines)
        normalized_kwargs["req"] = normalized_req
        return tuple(args), normalized_kwargs

    @staticmethod
    def clone_cached_tool_result(result: Any) -> Any:
        try:
            return copy.deepcopy(result)
        except Exception:
            return result

    @staticmethod
    def extract_followup_task_id(tool_name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> str | None:
        if not ToolLimits.is_followup_tool(tool_name):
            return None

        task_id = kwargs.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            return task_id.strip()

        if args:
            candidate = args[0]
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        req = kwargs.get("req")
        if isinstance(req, Mapping):
            nested = req.get("task_id")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

        return None

    @staticmethod
    def extract_execute_task_id(result: Any) -> str | None:
        if isinstance(result, Mapping):
            task_id = result.get("task_id")
            if isinstance(task_id, str) and task_id.strip():
                return task_id.strip()

        if hasattr(result, "task_id"):
            task_id = getattr(result, "task_id", None)
            if isinstance(task_id, str) and task_id.strip():
                return task_id.strip()

        if isinstance(result, tuple) and len(result) == 2:
            text, artifact = result
            if isinstance(artifact, Mapping):
                task_id = artifact.get("task_id")
                if isinstance(task_id, str) and task_id.strip():
                    return task_id.strip()
            if isinstance(text, Mapping):
                task_id = text.get("task_id")
                if isinstance(task_id, str) and task_id.strip():
                    return task_id.strip()

        return None

    @staticmethod
    def is_task_not_found_exception(err: BaseException) -> bool:
        if isinstance(err, HTTPException) and int(getattr(err, "status_code", 0) or 0) == 404:
            detail = getattr(err, "detail", None)
            return isinstance(detail, str) and "task not found" in detail.strip().lower()

        normalized = str(err).strip().lower()
        return "404" in normalized and "task not found" in normalized

    @staticmethod
    def invalid_followup_task_text(tool_name: str, task_id: str, latest_task_id: str | None = None) -> str:
        latest_text = ""
        if isinstance(latest_task_id, str) and latest_task_id and latest_task_id != task_id:
            latest_text = f" 最新の成功 execute task_id={latest_task_id} のみを追跡してください。"
        return (
            "ERROR: follow-up task_id is invalid. "
            f"error=invalid_followup_task_id tool={tool_name} task_id={task_id}. "
            "この task_id への status/get_result/workspace_path/cancel の再試行を中止してください。"
            f"{latest_text}"
        )

    @staticmethod
    def stale_followup_task_text(tool_name: str, task_id: str, latest_task_id: str) -> str:
        return (
            "ERROR: follow-up task_id is stale. "
            f"error=stale_followup_task_id tool={tool_name} task_id={task_id} latest_task_id={latest_task_id}. "
            "status/get_result/workspace_path/cancel は最新の成功 execute task_id 1件だけを追跡してください。"
        )

    @staticmethod
    def should_block_non_latest_followup_task(
        tool_name: str,
        followup_task_id: str | None,
        latest_task_id: str | None,
    ) -> bool:
        if not ToolLimits.is_followup_tool(tool_name):
            return False
        if not isinstance(followup_task_id, str) or not followup_task_id:
            return False
        if not isinstance(latest_task_id, str) or not latest_task_id:
            return False
        return followup_task_id != latest_task_id

    @classmethod
    def remember_successful_execute_task_id(cls, tool_call_state: dict[str, Any], result: Any) -> None:
        task_id = cls.extract_execute_task_id(result)
        if not task_id:
            return

        tool_call_state["latest_execute_task_id"] = task_id
        known = cast(list[str], tool_call_state.setdefault("successful_execute_task_ids", []))
        if task_id not in known:
            known.append(task_id)

    @classmethod
    def _resolve_budget_scope(
        cls,
        *,
        tool_name: str,
        tool_call_state: dict[str, Any],
        tool_call_limit_int: int,
    ) -> tuple[str, str, int, int]:
        if cls.is_followup_tool(tool_name):
            limit = int(tool_call_state.get("followup_limit", 0) or 0)
            key = "followup_used"
            scope = "followup"
        else:
            limit = tool_call_limit_int
            key = "general_used"
            scope = "general"

        used = int(tool_call_state.get(key, 0) or 0)
        if used < 0:
            used = 0
            tool_call_state[key] = 0
        if limit < 0:
            limit = 0
        return scope, key, limit, used

    @classmethod
    def _apply_tool_execution_guards(
        cls,
        allowed_langchain_tools: Sequence[Any],
        *,
        tool_call_state: dict[str, Any],
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
            or int(tool_call_state.get("followup_limit", 0) or 0)
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
                args, kwargs = cls.normalize_execute_arguments(tool_name, tool_call_state, args, kwargs)
                used = int(tool_call_state.get("used", 0) or 0)
                if used < 0:
                    used = 0
                    tool_call_state["used"] = 0

                followup_task_id = cls.extract_followup_task_id(tool_name, args, kwargs)
                invalid_task_ids = cast(set[str], tool_call_state.setdefault("invalid_followup_task_ids", set()))
                latest_execute_task_id = cast(str | None, tool_call_state.get("latest_execute_task_id"))
                if cls.should_block_non_latest_followup_task(tool_name, followup_task_id, latest_execute_task_id):
                    logger.info(
                        "Blocking stale follow-up task_id (sync): tool=%s task_id=%s latest=%s",
                        tool_name,
                        followup_task_id,
                        latest_execute_task_id,
                    )
                    return cls._guard_output(
                        cls.stale_followup_task_text(tool_name, cast(str, followup_task_id), cast(str, latest_execute_task_id)),
                        response_format=response_format,
                        artifact={
                            "error": "stale_followup_task_id",
                            "tool": tool_name,
                            "task_id": followup_task_id,
                            "latest_execute_task_id": latest_execute_task_id,
                        },
                    )
                if followup_task_id and followup_task_id in invalid_task_ids:
                    logger.info("Skipping repeated invalid follow-up task_id (sync): tool=%s task_id=%s", tool_name, followup_task_id)
                    return cls._guard_output(
                        cls.invalid_followup_task_text(tool_name, followup_task_id, latest_execute_task_id),
                        response_format=response_format,
                        artifact={
                            "error": "invalid_followup_task_id",
                            "tool": tool_name,
                            "task_id": followup_task_id,
                            "latest_execute_task_id": latest_execute_task_id,
                        },
                    )

                call_cache_key = cls.build_call_cache_key(tool_name, args, kwargs)
                cached_results = cast(dict[str, Any], tool_call_state.setdefault("successful_results", {}))
                if (not cls.is_followup_tool(tool_name) or cls.is_reusable_followup_tool(tool_name)) and call_cache_key in cached_results:
                    logger.info("Reusing cached tool result (sync): tool=%s", tool_name)
                    return cls.clone_cached_tool_result(cached_results[call_cache_key])

                budget_scope, used_key, scope_limit, scope_used = cls._resolve_budget_scope(
                    tool_name=tool_name,
                    tool_call_state=tool_call_state,
                    tool_call_limit_int=tool_call_limit_int,
                )

                if scope_limit and scope_used >= scope_limit:
                    logger.warning(
                        "Tool call budget exceeded (sync): tool=%s scope=%s used=%s limit=%s",
                        tool_name,
                        budget_scope,
                        scope_used,
                        scope_limit,
                    )
                    text = cls.tool_budget_exceeded_text(
                        tool_name,
                        limit=scope_limit,
                        used=scope_used,
                    )
                    return cls._guard_output(
                        text,
                        response_format=response_format,
                        artifact={
                            "error": "tool_call_budget_exceeded",
                            "tool": tool_name,
                            "limit": scope_limit,
                            "used": scope_used,
                            "budget_scope": budget_scope,
                        },
                    )

                tool_call_state["used"] = used + 1
                tool_call_state[used_key] = scope_used + 1
                try:
                    result = orig_func(*args, **kwargs)
                    if (tool_name or "").strip().lower() == "execute":
                        cls.remember_successful_execute_task_id(tool_call_state, result)
                    if cls.should_cache_tool_result(tool_name, result):
                        cached_results[call_cache_key] = cls.clone_cached_tool_result(result)
                    return result
                except Exception as e:
                    logger.exception("Tool invocation failed (sync): tool=%s", tool_name)
                    if followup_task_id and cls.is_task_not_found_exception(e):
                        invalid_task_ids.add(followup_task_id)
                        latest_execute_task_id = cast(str | None, tool_call_state.get("latest_execute_task_id"))
                        return cls._guard_output(
                            cls.invalid_followup_task_text(tool_name, followup_task_id, latest_execute_task_id),
                            response_format=response_format,
                            artifact={
                                "error": "invalid_followup_task_id",
                                "tool": tool_name,
                                "task_id": followup_task_id,
                                "latest_execute_task_id": latest_execute_task_id,
                                "exception": type(e).__name__,
                            },
                        )
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
    async def _run_tool_with_guards(
        cls,
        tool_name: str,
        orig_coro: Any,
        response_format: str | None,
        tool_call_state: dict[str, Any],
        tool_call_limit_int: int,
        tool_timeout_seconds_f: float,
        tool_timeout_retries_int: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        attempts = tool_timeout_retries_int + 1
        last_err: BaseException | None = None
        args, kwargs = cls.normalize_execute_arguments(tool_name, tool_call_state, args, kwargs)
        followup_task_id = cls.extract_followup_task_id(tool_name, args, kwargs)
        invalid_task_ids = cast(set[str], tool_call_state.setdefault("invalid_followup_task_ids", set()))
        latest_execute_task_id = cast(str | None, tool_call_state.get("latest_execute_task_id"))
        if cls.should_block_non_latest_followup_task(tool_name, followup_task_id, latest_execute_task_id):
            logger.info(
                "Blocking stale follow-up task_id: tool=%s task_id=%s latest=%s",
                tool_name,
                followup_task_id,
                latest_execute_task_id,
            )
            return cls._guard_output(
                cls.stale_followup_task_text(tool_name, cast(str, followup_task_id), cast(str, latest_execute_task_id)),
                response_format=response_format,
                artifact={
                    "error": "stale_followup_task_id",
                    "tool": tool_name,
                    "task_id": followup_task_id,
                    "latest_execute_task_id": latest_execute_task_id,
                },
            )
        if followup_task_id and followup_task_id in invalid_task_ids:
            logger.info("Skipping repeated invalid follow-up task_id: tool=%s task_id=%s", tool_name, followup_task_id)
            return cls._guard_output(
                cls.invalid_followup_task_text(tool_name, followup_task_id, latest_execute_task_id),
                response_format=response_format,
                artifact={
                    "error": "invalid_followup_task_id",
                    "tool": tool_name,
                    "task_id": followup_task_id,
                    "latest_execute_task_id": latest_execute_task_id,
                },
            )

        call_cache_key = cls.build_call_cache_key(tool_name, args, kwargs)
        cached_results = cast(dict[str, Any], tool_call_state.setdefault("successful_results", {}))
        if (not cls.is_followup_tool(tool_name) or cls.is_reusable_followup_tool(tool_name)) and call_cache_key in cached_results:
            logger.info("Reusing cached tool result: tool=%s", tool_name)
            return cls.clone_cached_tool_result(cached_results[call_cache_key])

        used = int(tool_call_state.get("used", 0) or 0)
        if used < 0:
            used = 0
            tool_call_state["used"] = 0

        for attempt in range(1, attempts + 1):
            used = int(tool_call_state.get("used", 0) or 0)
            if used < 0:
                used = 0
                tool_call_state["used"] = 0

            budget_scope, used_key, scope_limit, scope_used = cls._resolve_budget_scope(
                tool_name=tool_name,
                tool_call_state=tool_call_state,
                tool_call_limit_int=tool_call_limit_int,
            )

            if scope_limit and scope_used >= scope_limit:
                logger.warning(
                    "Tool call budget exceeded: tool=%s scope=%s used=%s limit=%s",
                    tool_name,
                    budget_scope,
                    scope_used,
                    scope_limit,
                )
                text = cls.tool_budget_exceeded_text(
                    tool_name,
                    limit=scope_limit,
                    used=scope_used,
                )
                return cls._guard_output(
                    text,
                    response_format=response_format,
                    artifact={"error": "tool_call_budget_exceeded", "tool": tool_name, "limit": scope_limit, "used": scope_used, "budget_scope": budget_scope},
                )

            tool_call_state["used"] = used + 1
            tool_call_state[used_key] = scope_used + 1
            try:
                if tool_timeout_seconds_f and tool_timeout_seconds_f > 0:
                    # Give the tool a small cushion so inner timeouts can surface as normal output.
                    timeout = tool_timeout_seconds_f
                    result = await asyncio.wait_for(orig_coro(*args, **kwargs), timeout=timeout)
                else:
                    result = await orig_coro(*args, **kwargs)
                if (tool_name or "").strip().lower() == "execute":
                    cls.remember_successful_execute_task_id(tool_call_state, result)
                if cls.should_cache_tool_result(tool_name, result):
                    cached_results[call_cache_key] = cls.clone_cached_tool_result(result)
                return result
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
                if followup_task_id and cls.is_task_not_found_exception(e):
                    invalid_task_ids.add(followup_task_id)
                    latest_execute_task_id = cast(str | None, tool_call_state.get("latest_execute_task_id"))
                    return cls._guard_output(
                        cls.invalid_followup_task_text(tool_name, followup_task_id, latest_execute_task_id),
                        response_format=response_format,
                        artifact={
                            "error": "invalid_followup_task_id",
                            "tool": tool_name,
                            "task_id": followup_task_id,
                            "latest_execute_task_id": latest_execute_task_id,
                            "exception": type(e).__name__,
                        },
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


class AgentBuilder:

    @staticmethod
    def _slugify_agent_name_part(value: str) -> str:
        normalized = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized or "default"

    @classmethod
    def build_tool_agent_name(
        cls,
        *,
        group_label: str | None = None,
        server_names: Sequence[str] | None = None,
    ) -> str:
        suffix_source = group_label
        if suffix_source is None:
            normalized_servers = [
                cls._slugify_agent_name_part(name)
                for name in (server_names or [])
                if isinstance(name, str) and name.strip()
            ]
            if normalized_servers:
                suffix_source = "_".join(sorted(dict.fromkeys(normalized_servers)))

        suffix = cls._slugify_agent_name_part(suffix_source or "default")
        return f"tool_agent_{suffix}"

    def get_tools(self) -> list[Any]:
        return self.langchain_tools
    
    def get_agent(self) -> Any:
        return self.agent
    
    def get_hitl_approval_tools(self) -> Sequence[str]:
        return self.hitl_approval_tools

    def get_agent_name(self) -> str:
        return self.agent_name

    def get_tools_description(self) -> str:
        tools_description = "\n".join(f"## name: {tool.name}\n - description: {tool.description}\n - args_schema: {tool.args_schema}\n" for tool in self.langchain_tools)
        return tools_description

    async def prepare(
            self,
            runtime_config: AiChatUtilConfig,
            mcp_config: MCPServerConfig ,
            llm: BaseChatModel,
            prompts: PromptsBase,
            tool_limits: ToolLimits | None,
            agent_name: str,
            allowed_tool_names: Sequence[str] | None = None,
            explicit_user_file_paths: Sequence[str] | None = None,
        ):

        # Safety valves: cap tool calls and hard-timeout tool execution.
        # This enforces termination even if prompts are ignored.
        if tool_limits is not None:
            tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = tool_limits.guard_params()
        else:
            tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = (0, 0.0, 0)

        effective_tool_call_limit_int, effective_followup_tool_call_limit_int = ToolLimits.effective_call_limits(
            tool_call_limit_int,
            int(tool_limits.followup_tool_call_limit) if tool_limits is not None else 0,
            explicit_user_file_paths,
        )

        # Shared tool call counter across all tools within this workflow.
        # Using a mutable container avoids invalid `nonlocal` usage across methods.
        tool_call_state: dict[str, Any] = {
            "used": 0,
            "general_used": 0,
            "followup_used": 0,
            "followup_limit": effective_followup_tool_call_limit_int,
            "explicit_user_file_paths": [
                str(path).strip()
                for path in (explicit_user_file_paths or [])
                if isinstance(path, str) and str(path).strip()
            ],
        }

        hitl_approval_tools = runtime_config.features.hitl_approval_tools or []

        agent_client = MultiServerMCPClient(mcp_config.to_langchain_config())
        # LangChainのツールリストを取得
        coding_agent_langchain_tools = await agent_client.get_tools()
        if allowed_tool_names is not None:
            allowed_names = {name for name in allowed_tool_names if isinstance(name, str) and name.strip()}
            coding_agent_langchain_tools = [
                tool for tool in coding_agent_langchain_tools if str(getattr(tool, "name", "")) in allowed_names
            ]

        ToolLimits._apply_tool_execution_guards(
            coding_agent_langchain_tools,
            tool_call_state=tool_call_state,
            tool_call_limit_int=effective_tool_call_limit_int,
            tool_timeout_seconds_f=tool_timeout_seconds_f,
            tool_timeout_retries_int=tool_timeout_retries_int,
        )
        logger.info("Creating code agent for MCP server '%s'...", ", ".join(mcp_config.servers.keys()))

        approval_tools = [t for t in (hitl_approval_tools or []) if isinstance(t, str) and t.strip()]
        approval_tools_text = ", ".join(approval_tools) if approval_tools else "(なし)"

        if tool_limits is not None and tool_limits.auto_approve:
            hitl_policy_text = prompts.auto_approve_hitl_policy_text(approval_tools_text)
        else:
            hitl_policy_text = prompts.normal_hitl_policy_text(approval_tools_text)

        tool_agent_system_prompt = prompts.tool_agent_system_prompt(
            hitl_policy_text,
            agent_name=agent_name,
            followup_poll_interval_seconds=float(getattr(runtime_config.features, "mcp_followup_poll_interval_seconds", 2.0) or 0.0),
            status_tail_lines=int(getattr(runtime_config.features, "mcp_status_tail_lines", 20) or 0),
            result_tail_lines=int(getattr(runtime_config.features, "mcp_get_result_tail_lines", 80) or 0),
        )

        tool_agent = create_agent(
            llm,
            coding_agent_langchain_tools,
            system_prompt=tool_agent_system_prompt,
            name=agent_name,
        )

        self.agent = tool_agent
        self.agent_name = agent_name
        self.langchain_tools = coding_agent_langchain_tools
        self.hitl_approval_tools = hitl_approval_tools

    @classmethod
    async def create_sub_agents(
        cls,
        runtime_config: AiChatUtilConfig,
        mcp_config: MCPServerConfig ,
        llm: BaseChatModel,
        prompts: PromptsBase,
        tool_limits: ToolLimits | None,
        include_general_agent: bool = True,
        general_tool_allowlist: Sequence[str] | None = None,
        explicit_user_file_paths: Sequence[str] | None = None,
    ) -> list[AgentBuilder]:
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

        hitl_approval_tools = runtime_config.features.hitl_approval_tools or []

        coding_agent_server_name = runtime_config.mcp.coding_agent_endpoint.mcp_server_name

        # coding_agent用のサーバー定義を取得
        code_agent_mcp_config = mcp_config.filter(include_name=coding_agent_server_name)
        normal_tools_mcp_config = mcp_config.filter(exclude_name=coding_agent_server_name)
        # allowed_tools_configにcoding_agent_nameと一致するツールがあれば、コードエージェントを作成する。
        agents = []
        if len(code_agent_mcp_config.servers) > 0 :
            logger.info("Creating code agent for MCP server '%s'...", ", ".join(code_agent_mcp_config.servers.keys()))
            code_agent_builder = AgentBuilder()
            await code_agent_builder.prepare(
                runtime_config=runtime_config,
                mcp_config=code_agent_mcp_config,
                llm=llm,
                prompts=prompts,
                tool_limits=tool_limits,
                agent_name=cls.build_tool_agent_name(
                    group_label="coding",
                    server_names=tuple(code_agent_mcp_config.servers.keys()),
                ),
                explicit_user_file_paths=explicit_user_file_paths,
            )
            agents.append(code_agent_builder)

        should_create_general_agent = include_general_agent or len(code_agent_mcp_config.servers) == 0
        if should_create_general_agent and len(normal_tools_mcp_config.servers) > 0:
            logger.info("Creating normal agent for MCP server '%s'...", ", ".join(normal_tools_mcp_config.servers.keys()))
            normal_agent_builder = AgentBuilder()
            await normal_agent_builder.prepare(
                runtime_config=runtime_config,
                mcp_config=normal_tools_mcp_config,
                llm=llm,
                prompts=prompts,
                tool_limits=tool_limits,
                agent_name=cls.build_tool_agent_name(
                    group_label="general",
                    server_names=tuple(normal_tools_mcp_config.servers.keys()),
                ),
                allowed_tool_names=general_tool_allowlist,
                explicit_user_file_paths=explicit_user_file_paths,
            )
            if general_tool_allowlist is None or normal_agent_builder.langchain_tools:
                agents.append(normal_agent_builder)

        # 他のサブエージェントも必要に応じてここで作成できます。
        return agents

    @classmethod
    def get_tools_description_all(cls, agents: list[AgentBuilder]) -> str:
        tools_description = "\n".join(f"## name: {tool.name}\n - description: {tool.description}\n - args_schema: {tool.args_schema}\n" for agent in agents for tool in agent.langchain_tools)
        return tools_description