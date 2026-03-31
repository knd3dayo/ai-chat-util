from __future__ import annotations

from typing import Any, Mapping, Sequence, cast
import asyncio
import copy
import json
from pathlib import Path
import re
from fastapi import HTTPException
from pydantic import BaseModel, Field
import ai_chat_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


from typing import Any, Mapping, Sequence, cast
import asyncio
import copy
import json
from pathlib import Path
import re
from fastapi import HTTPException
from pydantic import BaseModel, Field
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
            def from_config(cls, config):
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
        @classmethod
        def from_config(cls, config):
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
                from __future__ import annotations

                from typing import Any, Mapping, Sequence, cast
                import asyncio
                import copy
                import json
                from pathlib import Path
                import re
                from fastapi import HTTPException
                from pydantic import BaseModel, Field
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
                    # --- ここから下にメソッド定義を追記（from_config, guard_params, ...） ---
        description="ツール呼び出しのタイムアウト秒数。0またはNoneで無制限。安全弁として、マイナス値は0として扱います。",
