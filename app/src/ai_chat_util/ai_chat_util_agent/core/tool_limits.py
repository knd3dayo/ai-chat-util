from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import asyncio
import copy
import json
from pathlib import Path
import re

from fastapi import HTTPException
from pydantic import BaseModel, Field

from ai_chat_util.common.config.runtime import AiChatUtilConfig
import ai_chat_util.log.log_settings as log_settings

from .supervisor_support import AuditContext

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
    tool_timeout_retries: int = Field(
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
        """Build ToolLimits from runtime config."""

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
        explicit_user_directory_paths: Sequence[str] | None = None,
    ) -> tuple[int, int]:
        normalized_file_paths = [
            str(path).strip()
            for path in (explicit_user_file_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        normalized_directory_paths = [
            str(path).strip()
            for path in (explicit_user_directory_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        if not normalized_file_paths and not normalized_directory_paths:
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
        error_code = ToolLimits.classify_tool_error(tool_name, err)
        err_type = type(err).__name__
        msg = str(err).strip()
        if msg:
            return f"ERROR: tool={tool_name} failed ({err_type}). error={error_code}: {msg}"
        return f"ERROR: tool={tool_name} failed ({err_type}). error={error_code}"

    @staticmethod
    def classify_tool_error(tool_name: str, err: BaseException) -> str:
        normalized_tool = (tool_name or "").strip().lower()
        if ToolLimits.is_timeout_exception(err):
            return "tool_timeout"
        if isinstance(err, HTTPException):
            status_code = int(getattr(err, "status_code", 0) or 0)
            if 400 <= status_code < 500:
                return "execute_request_invalid" if normalized_tool == "execute" else "tool_request_invalid"
            if status_code >= 500:
                return "execute_backend_error" if normalized_tool == "execute" else "tool_backend_error"
        if isinstance(err, ValueError):
            return "execute_request_invalid" if normalized_tool == "execute" else "tool_request_invalid"
        if normalized_tool == "execute":
            return "execute_invocation_failed"
        return "tool_invocation_failed"

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
    def tool_action_kind(tool_name: str) -> str:
        normalized = (tool_name or "").strip().lower()
        if normalized in {"convert_office_files_to_pdf", "convert_pdf_files_to_images", "execute", "cancel"}:
            return "write"
        if normalized in {"status", "get_result", "workspace_path"}:
            return "control"
        if normalized in {"healthz", "get_loaded_config_info"}:
            return "inspect"
        return "read"

    @staticmethod
    def tool_target_system(tool_name: str) -> str:
        normalized = (tool_name or "").strip().lower()
        if normalized in {"execute", "status", "get_result", "workspace_path", "cancel", "healthz"}:
            return "coding_agent"
        if normalized.startswith("analyze_") or normalized.startswith("convert_"):
            return "filesystem"
        return "mcp_tool"

    @classmethod
    def tool_resource_identifier(
        cls,
        tool_name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> str | None:
        normalized = (tool_name or "").strip().lower()
        if normalized in {"status", "get_result", "workspace_path", "cancel"}:
            task_id = kwargs.get("task_id")
            if isinstance(task_id, str) and task_id.strip():
                return f"task_id:{task_id.strip()}"

        for key in (
            "file_list",
            "file_path_list",
            "pdf_path_list",
            "office_path_list",
            "output_dir",
            "input_excel_path",
            "output_excel_path",
            "workspace_path",
        ):
            value = kwargs.get(key)
            if isinstance(value, str) and value.strip():
                try:
                    return Path(value).name or value.strip()
                except Exception:
                    return value.strip()
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                names: list[str] = []
                for item in value[:3]:
                    if not isinstance(item, str) or not item.strip():
                        continue
                    try:
                        names.append(Path(item).name or item.strip())
                    except Exception:
                        names.append(item.strip())
                if names:
                    more = "" if len(value) <= 3 else f" (+{len(value) - 3} more)"
                    return ", ".join(names) + more

        if normalized == "execute":
            for key in ("task_id", "trace_id"):
                value = kwargs.get(key)
                if isinstance(value, str) and value.strip():
                    return f"{key}:{value.strip()}"

        return None

    @staticmethod
    def tool_approval_status(tool_name: str, tool_call_state: Mapping[str, Any]) -> str | None:
        approval_tools = cast(set[str], tool_call_state.get("approval_tools", set()))
        if (tool_name or "").strip() in approval_tools:
            return "required"
        return None

    @staticmethod
    def tool_is_approved(tool_name: str, tool_call_state: Mapping[str, Any]) -> bool:
        if bool(tool_call_state.get("auto_approve", False)):
            return True
        approved_tools = cast(set[str], tool_call_state.get("approved_tools", set()))
        normalized = (tool_name or "").strip()
        return "*" in approved_tools or normalized in approved_tools

    @classmethod
    def requires_tool_approval(cls, tool_name: str, tool_call_state: Mapping[str, Any]) -> bool:
        return cls.tool_approval_status(tool_name, tool_call_state) == "required" and not cls.tool_is_approved(tool_name, tool_call_state)

    @staticmethod
    def tool_approval_required_text(tool_name: str) -> str:
        normalized = (tool_name or "").strip() or "TOOL_NAME"
        return (
            f"ERROR: tool approval required. error=tool_approval_required tool={normalized}. "
            f"このツールの実行には承認が必要です。続行する場合は 'APPROVE {normalized}'、拒否する場合は 'REJECT {normalized}' と入力してください。"
        )

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

        prompt = normalized_req.get("prompt")
        if isinstance(prompt, str) and cls._looks_like_heading_extraction_task(prompt):
            cast(dict[str, Any], tool_call_state)["expects_heading_output"] = True

        target_file_path = cls._extract_single_explicit_user_file_path(tool_call_state)
        if not target_file_path:
            return tuple(args), normalized_kwargs
        if not normalized_req:
            return tuple(args), normalized_kwargs

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
    def normalize_followup_arguments(
        tool_name: str,
        tool_call_state: Mapping[str, Any],
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        normalized_kwargs = dict(kwargs)
        if (tool_name or "").strip().lower() == "get_result" and bool(tool_call_state.get("expects_heading_output")):
            normalized_kwargs["tail"] = None
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
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray, Mapping)):
            for item in result:
                if isinstance(item, Mapping):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        try:
                            parsed = json.loads(text_value)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, Mapping):
                            task_id = parsed.get("task_id")
                            if isinstance(task_id, str) and task_id.strip():
                                return task_id.strip()
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
            " 取得済みの stdout/stderr や既存の証拠で回答できるなら、その内容で完了してください。"
            " このエラーを理由に同じ目的の execute をやり直さないでください。"
            f"{latest_text}"
        )

    @staticmethod
    def stale_followup_task_text(tool_name: str, task_id: str, latest_task_id: str) -> str:
        return (
            "ERROR: follow-up task_id is stale. "
            f"error=stale_followup_task_id tool={tool_name} task_id={task_id} latest_task_id={latest_task_id}. "
            "status/get_result/workspace_path/cancel は最新の成功 execute task_id 1件だけを追跡してください。"
            " 古い task_id を補うために新しい execute を追加で起こさず、最新 task_id か取得済み結果で収束してください。"
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

        def _emit_tool_event(
            event_type: str,
            *,
            tool_name: str,
            args: Sequence[Any] | None = None,
            kwargs: Mapping[str, Any] | None = None,
            payload: Mapping[str, Any] | None = None,
            reason_code: str | None = None,
        ) -> None:
            audit_context = tool_call_state.get("audit_context")
            if not isinstance(audit_context, AuditContext):
                return
            call_args = args or ()
            call_kwargs = kwargs or {}
            audit_context.emit(
                event_type,
                agent_name=cast(str | None, tool_call_state.get("agent_name")),
                tool_name=tool_name,
                reason_code=reason_code,
                target_system=cls.tool_target_system(tool_name),
                action_kind=cls.tool_action_kind(tool_name),
                resource_identifier=cls.tool_resource_identifier(tool_name, call_args, call_kwargs),
                approval_status=cls.tool_approval_status(tool_name, tool_call_state),
                payload=dict(payload or {}),
            )

        def _wrap_sync(*, tool_name: str, orig_func: Any, response_format: str | None) -> Any:
            def _wrapped_func(*args: Any, **kwargs: Any) -> Any:
                args, kwargs = cls.normalize_execute_arguments(tool_name, tool_call_state, args, kwargs)
                args, kwargs = cls.normalize_followup_arguments(tool_name, tool_call_state, args, kwargs)
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
                    _emit_tool_event(
                        "tool_result_received",
                        tool_name=tool_name,
                        args=args,
                        kwargs=kwargs,
                        payload={"success": True, "cached": True},
                    )
                    return cls.clone_cached_tool_result(cached_results[call_cache_key])

                _emit_tool_event(
                    "tool_selected",
                    tool_name=tool_name,
                    args=args,
                    kwargs=kwargs,
                    payload={"args_count": len(args), "kwargs_keys": sorted(str(key) for key in kwargs.keys())},
                )

                if cls.requires_tool_approval(tool_name, tool_call_state):
                    _emit_tool_event(
                        "tool_result_received",
                        tool_name=tool_name,
                        args=args,
                        kwargs=kwargs,
                        reason_code="hitl.tool_approval_required",
                        payload={"success": False, "approval_required": True, "blocked": True},
                    )
                    return cls._guard_output(
                        cls.tool_approval_required_text(tool_name),
                        response_format=response_format,
                        artifact={
                            "error": "tool_approval_required",
                            "tool": tool_name,
                            "approval_required": True,
                        },
                    )

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
                    return cls._guard_output(
                        cls.tool_budget_exceeded_text(tool_name, limit=scope_limit, used=scope_used),
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
                    _emit_tool_event(
                        "tool_result_received",
                        tool_name=tool_name,
                        args=args,
                        kwargs=kwargs,
                        payload={"success": True, "cached": False},
                    )
                    return result
                except Exception as err:
                    if followup_task_id and cls.is_task_not_found_exception(err):
                        logger.info(
                            "Marking follow-up task_id invalid after task-not-found (sync): tool=%s task_id=%s",
                            tool_name,
                            followup_task_id,
                        )
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
                                "exception": type(err).__name__,
                            },
                        )
                    logger.exception("Tool invocation failed (sync): tool=%s", tool_name)
                    error_code = cls.classify_tool_error(tool_name, err)
                    _emit_tool_event(
                        "tool_result_received",
                        tool_name=tool_name,
                        args=args,
                        kwargs=kwargs,
                        reason_code="sufficiency.tool_result_error_only",
                        payload={"success": False, "exception": type(err).__name__, "error": error_code},
                    )
                    return cls._guard_output(
                        cls.tool_error_text(tool_name, err),
                        response_format=response_format,
                        artifact={"error": error_code, "tool": tool_name, "exception": type(err).__name__},
                    )

            return _wrapped_func

        def _wrap_async(*, tool_name: str, orig_coro: Any, response_format: str | None) -> Any:
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
        args, kwargs = cls.normalize_followup_arguments(tool_name, tool_call_state, args, kwargs)
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
        audit_context = tool_call_state.get("audit_context")
        if (not cls.is_followup_tool(tool_name) or cls.is_reusable_followup_tool(tool_name)) and call_cache_key in cached_results:
            logger.info("Reusing cached tool result: tool=%s", tool_name)
            if isinstance(audit_context, AuditContext):
                audit_context.emit(
                    "tool_result_received",
                    agent_name=cast(str | None, tool_call_state.get("agent_name")),
                    tool_name=tool_name,
                    target_system=cls.tool_target_system(tool_name),
                    action_kind=cls.tool_action_kind(tool_name),
                    resource_identifier=cls.tool_resource_identifier(tool_name, args, kwargs),
                    approval_status=cls.tool_approval_status(tool_name, tool_call_state),
                    payload={"success": True, "cached": True},
                )
            return cls.clone_cached_tool_result(cached_results[call_cache_key])
        if isinstance(audit_context, AuditContext):
            audit_context.emit(
                "tool_selected",
                agent_name=cast(str | None, tool_call_state.get("agent_name")),
                tool_name=tool_name,
                target_system=cls.tool_target_system(tool_name),
                action_kind=cls.tool_action_kind(tool_name),
                resource_identifier=cls.tool_resource_identifier(tool_name, args, kwargs),
                approval_status=cls.tool_approval_status(tool_name, tool_call_state),
                payload={"args_count": len(args), "kwargs_keys": sorted(str(key) for key in kwargs.keys())},
            )

        if cls.requires_tool_approval(tool_name, tool_call_state):
            if isinstance(audit_context, AuditContext):
                audit_context.emit(
                    "tool_result_received",
                    agent_name=cast(str | None, tool_call_state.get("agent_name")),
                    tool_name=tool_name,
                    reason_code="hitl.tool_approval_required",
                    target_system=cls.tool_target_system(tool_name),
                    action_kind=cls.tool_action_kind(tool_name),
                    resource_identifier=cls.tool_resource_identifier(tool_name, args, kwargs),
                    approval_status=cls.tool_approval_status(tool_name, tool_call_state),
                    payload={"success": False, "approval_required": True, "blocked": True},
                )
            return cls._guard_output(
                cls.tool_approval_required_text(tool_name),
                response_format=response_format,
                artifact={
                    "error": "tool_approval_required",
                    "tool": tool_name,
                    "approval_required": True,
                },
            )

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
                return cls._guard_output(
                    cls.tool_budget_exceeded_text(tool_name, limit=scope_limit, used=scope_used),
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
                if tool_timeout_seconds_f and tool_timeout_seconds_f > 0:
                    result = await asyncio.wait_for(orig_coro(*args, **kwargs), timeout=tool_timeout_seconds_f)
                else:
                    result = await orig_coro(*args, **kwargs)
                if (tool_name or "").strip().lower() == "execute":
                    cls.remember_successful_execute_task_id(tool_call_state, result)
                if cls.should_cache_tool_result(tool_name, result):
                    cached_results[call_cache_key] = cls.clone_cached_tool_result(result)
                if isinstance(audit_context, AuditContext):
                    audit_context.emit(
                        "tool_result_received",
                        agent_name=cast(str | None, tool_call_state.get("agent_name")),
                        tool_name=tool_name,
                        target_system=cls.tool_target_system(tool_name),
                        action_kind=cls.tool_action_kind(tool_name),
                        resource_identifier=cls.tool_resource_identifier(tool_name, args, kwargs),
                        approval_status=cls.tool_approval_status(tool_name, tool_call_state),
                        payload={"success": True, "cached": False},
                    )
                return result
            except asyncio.CancelledError:
                raise
            except Exception as err:
                last_err = err
                if cls.is_timeout_exception(err) and attempt < attempts:
                    logger.warning(
                        "Tool timeout; retrying: tool=%s attempt=%s/%s",
                        tool_name,
                        attempt,
                        attempts,
                    )
                    continue

                if followup_task_id and cls.is_task_not_found_exception(err):
                    logger.info(
                        "Marking follow-up task_id invalid after task-not-found: tool=%s task_id=%s",
                        tool_name,
                        followup_task_id,
                    )
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
                            "exception": type(err).__name__,
                        },
                    )
                logger.exception(
                    "Tool invocation failed: tool=%s attempt=%s/%s",
                    tool_name,
                    attempt,
                    attempts,
                )
                error_code = cls.classify_tool_error(tool_name, err)
                if isinstance(audit_context, AuditContext):
                    audit_context.emit(
                        "tool_result_received",
                        agent_name=cast(str | None, tool_call_state.get("agent_name")),
                        tool_name=tool_name,
                        reason_code="sufficiency.tool_result_error_only",
                        target_system=cls.tool_target_system(tool_name),
                        action_kind=cls.tool_action_kind(tool_name),
                        resource_identifier=cls.tool_resource_identifier(tool_name, args, kwargs),
                        approval_status=cls.tool_approval_status(tool_name, tool_call_state),
                        payload={"success": False, "exception": type(err).__name__, "error": error_code},
                    )
                return cls._guard_output(
                    cls.tool_error_text(tool_name, err),
                    response_format=response_format,
                    artifact={"error": error_code, "tool": tool_name, "exception": type(err).__name__},
                )

        if last_err is not None:
            error_code = cls.classify_tool_error(tool_name, last_err)
            return cls._guard_output(
                cls.tool_error_text(tool_name, last_err),
                response_format=response_format,
                artifact={"error": error_code, "tool": tool_name, "exception": type(last_err).__name__},
            )
        return cls._guard_output(
            f"ERROR: tool={tool_name} failed (unknown error)",
            response_format=response_format,
            artifact={"error": "tool_invocation_failed", "tool": tool_name},
        )

    @classmethod
    def _guard_output(cls, payload: str, *, response_format: str | None, artifact: Any | None = None) -> Any:
        if response_format == "content_and_artifact":
            if artifact is None:
                artifact = {}
            return (payload, artifact)
        return payload