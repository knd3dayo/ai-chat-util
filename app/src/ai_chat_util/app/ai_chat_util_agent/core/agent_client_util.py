from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence, cast

import ast
import contextlib
import json
import re
import sqlite3
from pathlib import Path
from langchain_litellm import ChatLiteLLMRouter
from litellm.router import Router
from langchain_mcp_adapters.client import MultiServerMCPClient

from pydantic import BaseModel, ConfigDict, Field, create_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool
from langgraph_supervisor import create_supervisor
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import BaseChatModel
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from .deep_agent_support import build_deep_agent_system_prompt, deepagents_available, require_create_deep_agent
from .prompts_base import PromptsBase
from .agent_builder import AgentBuilder
from .tool_limits import ToolLimits
from .supervisor_support import AuditContext, EvidenceSummary, RoutingDecision, RouteCandidate, SufficiencyDecision

try:
    # Async checkpointer for LangGraph when using app.ainvoke()/astream()
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except Exception:  # pragma: no cover
    AsyncSqliteSaver = None  # type: ignore[assignment]

from ai_chat_util.ai_chat_util_base.core.common.config.ai_chat_util_mcp_config import MCPServerConfig
from ai_chat_util.ai_chat_util_base.core.common.config.runtime import (
    AiChatUtilConfig,
    CodingAgentUtilConfig,
    get_runtime_config_path,
)
import ai_chat_util.ai_chat_util_base.core.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class AgentClientUtil:

    _TOOL_CATALOG_DETAIL_PATTERNS: tuple[str, ...] = (
        r"ツールの名称",
        r"名称[、,/]?説明",
        r"主要な引数",
        r"引数を一覧",
        r"arguments?\b",
    )

    _TOOL_CATALOG_REFERENCE_PATTERNS: tuple[str, ...] = (
        r"利用可能(?:な)?(?:\s+MCP)?\s*ツール",
        r"MCP\s*ツール",
        r"tool\s+catalog",
        r"available\s+tools",
        r"tool\s+list",
        r"tool_catalog_resolved",
        r"使えるツール",
    )

    _TOOL_CATALOG_REQUEST_PATTERNS: tuple[str, ...] = (
        r"ツール一覧",
        r"一覧(?:で|を|に|表示|化|提示)?",
        r"列挙",
        r"agent\s*名ごと",
        r"agent\s*ごと",
        r"分けて整理",
        r"名称",
        r"主要な引数",
        r"tool\s+catalog",
        r"tool\s+list",
    )

    _GENERAL_TOOLS_ALLOWED_WITH_EXPLICIT_CODING_AGENT: tuple[str, ...] = (
        "get_loaded_config_info",
    )

    _EXPLICIT_CODING_AGENT_PATTERNS: tuple[str, ...] = (
        r"\bcoding(?:[\s\-_]+agent)(?:\s+mcp)?\b",
        r"コーディング[\s\-]*エージェント",
    )

    _EXPLICIT_DEEP_AGENT_PATTERNS: tuple[str, ...] = (
        r"\bdeep(?:[\s\-_]+agent)s?\b",
        r"ディープ[\s\-]*エージェント",
    )

    _DEEP_AGENT_EXCLUDED_TOOL_NAMES: tuple[str, ...] = (
        "execute",
        "status",
        "get_result",
        "workspace_path",
        "cancel",
    )

    _PATH_LIKE_PATTERNS: tuple[str, ...] = (
        r"(?:[A-Za-z]:[\\/]|/)[^\s'\"`]+",
    )

    _ALLOWED_ROUTE_REASON_CODES: tuple[str, ...] = (
        "route.explicit_coding_agent_request",
        "route.explicit_file_path_request",
        "route.explicit_directory_path_request",
        "route.workflow_definition_available",
        "route.workflow_definition_missing",
        "route.general_tool_sufficient",
        "route.multi_step_investigation_needed",
        "route.tool_not_available",
        "route.missing_required_argument",
        "route.low_confidence_route",
        "route.direct_answer_possible",
        "route.reject_unsupported_request",
    )

    @classmethod
    def explicitly_requests_coding_agent(cls, messages: Sequence[Any]) -> bool:
        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            normalized_text = text
            for pattern in cls._PATH_LIKE_PATTERNS:
                normalized_text = re.sub(pattern, " ", normalized_text)

            if any(re.search(pattern, normalized_text, flags=re.IGNORECASE) for pattern in cls._EXPLICIT_CODING_AGENT_PATTERNS):
                return True

        return False

    @classmethod
    def explicitly_requests_deep_agent(cls, messages: Sequence[Any]) -> bool:
        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            normalized_text = text
            for pattern in cls._PATH_LIKE_PATTERNS:
                normalized_text = re.sub(pattern, " ", normalized_text)

            if any(re.search(pattern, normalized_text, flags=re.IGNORECASE) for pattern in cls._EXPLICIT_DEEP_AGENT_PATTERNS):
                return True

        return False

    @classmethod
    def deep_agent_route_enabled(cls, runtime_config: AiChatUtilConfig) -> bool:
        return bool(getattr(runtime_config.features, "enable_deep_agent", False)) and deepagents_available()

    @classmethod
    def _is_deep_agent_tool_allowed(cls, tool_name: str) -> bool:
        normalized = str(tool_name or "").strip().lower()
        return normalized not in cls._DEEP_AGENT_EXCLUDED_TOOL_NAMES

    @classmethod
    def extract_explicit_user_file_paths(cls, messages: Sequence[Any]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        patterns = (
            r"(?P<path>/[^\s'\"`]+)",
            r"(?P<path>[A-Za-z]:[\\/][^\s'\"`]+)",
        )

        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    raw_path = match.group("path").strip().rstrip(",.);:]>}")
                    try:
                        resolved = Path(raw_path).expanduser().resolve()
                    except Exception:
                        continue
                    if not resolved.is_file():
                        continue
                    normalized = resolved.as_posix()
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    candidates.append(normalized)

        return candidates

    @classmethod
    def _append_existing_directory_candidate(
        cls,
        candidates: list[str],
        seen: set[str],
        raw_path: str,
        *,
        working_directory: str | None = None,
    ) -> None:
        normalized_raw_path = str(raw_path or "").strip().strip("'\"`").rstrip(",.);:]>}")
        if not normalized_raw_path:
            return

        try:
            candidate_path = Path(normalized_raw_path).expanduser()
            if not candidate_path.is_absolute():
                base_directory = str(working_directory or "").strip()
                if not base_directory:
                    return
                candidate_path = Path(base_directory).expanduser().resolve() / candidate_path
            resolved = candidate_path.resolve()
        except Exception:
            return

        if not resolved.is_dir():
            return

        normalized = resolved.as_posix()
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    @classmethod
    def extract_explicit_user_directory_paths(
        cls,
        messages: Sequence[Any],
        *,
        working_directory: str | None = None,
    ) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        patterns = (
            r"(?P<path>/[^\s'\"`]+)",
            r"(?P<path>[A-Za-z]:[\\/][^\s'\"`]+)",
        )
        relative_directory_patterns = (
            r"(?P<path>(?:\./|\.\./|~/)?[A-Za-z0-9][A-Za-z0-9._/\-]*)\s*(?:ディレクトリ|フォルダ|directory|folder)(?=(?:\s|[、。,.!?:]|を|に|で|配下|内|$))",
        )

        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    cls._append_existing_directory_candidate(
                        candidates,
                        seen,
                        match.group("path"),
                    )

            for pattern in relative_directory_patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    cls._append_existing_directory_candidate(
                        candidates,
                        seen,
                        match.group("path"),
                        working_directory=working_directory,
                    )

        return candidates

    @classmethod
    def extract_explicit_approval_tool_names(cls, messages: Sequence[Any]) -> list[str]:
        approved_tools: list[str] = []
        seen: set[str] = set()

        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            for match in re.finditer(r"\bAPPROVE(?:\s+([A-Za-z0-9_\-:.]+))?", text, flags=re.IGNORECASE):
                tool_name = str(match.group(1) or "*").strip()
                if not tool_name or tool_name in seen:
                    continue
                seen.add(tool_name)
                approved_tools.append(tool_name)

        return approved_tools

    @classmethod
    def should_include_general_agent(
        cls,
        *,
        force_coding_agent_route: bool,
        explicit_user_file_paths: Sequence[str] | None = None,
    ) -> bool:
        normalized_paths = [
            str(path).strip()
            for path in (explicit_user_file_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        if force_coding_agent_route and normalized_paths:
            return False
        return True

    @classmethod
    def extract_requested_heading_count(cls, messages: Sequence[Any]) -> int | None:
        patterns = (
            r"見出し(?:を|は)?\s*(\d+)\s*点",
            r"重要な見出し(?:を|は)?\s*(\d+)\s*点",
            r"(\d+)\s*点(?:の見出し|挙げて)",
            r"(\d+)\s*つ(?:の見出し)?",
            r"(\d+)\s*個(?:の見出し)?",
        )

        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            for pattern in patterns:
                match = re.search(pattern, text)
                if not match:
                    continue
                try:
                    count = int(match.group(1))
                except (TypeError, ValueError):
                    continue
                if count > 0:
                    return count

        return None

    @classmethod
    def requests_heading_response(cls, messages: Sequence[Any]) -> bool:
        patterns = (
            r"見出し",
            r"heading",
            r"heading_line_exact",
            r"markdown\s+heading",
            r"節名",
            r"タイトルを抽出",
        )
        negative_patterns = (
            r"見出し(?:抽出)?は不要",
            r"見出し(?:抽出)?はいらない",
            r"見出し(?:抽出)?は不要です",
            r"heading(?:\s+extraction)?\s+(?:is\s+)?not\s+needed",
            r"no\s+heading(?:\s+extraction)?",
            r"不要な見出し抽出",
        )

        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in negative_patterns):
                continue

            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
                return True

        return False

    @classmethod
    def requests_evaluation_response(cls, messages: Sequence[Any]) -> bool:
        patterns = (
            r"本番投入",
            r"投入判断",
            r"可否",
            r"足りるか",
            r"不足情報",
            r"追加確認",
            r"確認事項",
            r"評価してください",
            r"judge",
            r"evaluate",
            r"readiness",
        )

        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
                return True

        return False

    @classmethod
    def requests_tool_catalog_response(cls, messages: Sequence[Any]) -> bool:
        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            has_tool_reference = any(
                re.search(pattern, text, flags=re.IGNORECASE)
                for pattern in cls._TOOL_CATALOG_REFERENCE_PATTERNS
            )
            has_catalog_request = any(
                re.search(pattern, text, flags=re.IGNORECASE)
                for pattern in cls._TOOL_CATALOG_REQUEST_PATTERNS
            )

            if has_tool_reference and has_catalog_request:
                return True

        return False

    @classmethod
    def requests_tool_catalog_details(cls, messages: Sequence[Any]) -> bool:
        for message in messages:
            is_user_message = False
            content: Any | None = None

            if isinstance(message, HumanMessage):
                is_user_message = True
                content = message.content
            elif isinstance(message, BaseMessage):
                continue
            elif isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    is_user_message = True
                    content = message.get("content")

            if not is_user_message:
                continue

            text = cls._stringify_message_content(content)
            if not text:
                continue

            has_tool_reference = any(
                re.search(pattern, text, flags=re.IGNORECASE)
                for pattern in cls._TOOL_CATALOG_REFERENCE_PATTERNS
            )
            has_detail_request = any(
                re.search(pattern, text, flags=re.IGNORECASE)
                for pattern in cls._TOOL_CATALOG_DETAIL_PATTERNS
            )

            if has_tool_reference and has_detail_request:
                return True

        return False

    @classmethod
    async def collect_checkpoint_results(
        cls,
        *,
        app: Any,
        run_trace_id: str,
        history_limit: int = 32,
    ) -> list[Any]:
        config = {"configurable": {"thread_id": run_trace_id}}
        results: list[Any] = []

        try:
            state = await app.aget_state(config)
            values = getattr(state, "values", None)
            if values is not None:
                results.append(values)
        except Exception:
            logger.debug("Failed to load latest graph state for trace_id=%s", run_trace_id, exc_info=True)

        try:
            async for snapshot in app.aget_state_history(config, limit=history_limit):
                values = getattr(snapshot, "values", None)
                if values is not None:
                    results.append(values)
        except Exception:
            logger.debug("Failed to load graph state history for trace_id=%s", run_trace_id, exc_info=True)

        return results

    @classmethod
    def collect_checkpoint_write_results(
        cls,
        *,
        checkpoint_db_path: Path | None,
        run_trace_id: str,
        limit: int = 64,
    ) -> list[Any]:
        if checkpoint_db_path is None:
            return []

        try:
            db_path = checkpoint_db_path.expanduser().resolve()
        except Exception:
            return []
        if not db_path.is_file():
            return []

        results: list[Any] = []
        serde = JsonPlusSerializer()

        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "select checkpoint_id, checkpoint_ns, task_id, idx, channel, type, value "
                    "from writes where thread_id = ? order by rowid desc limit ?",
                    (run_trace_id, int(limit)),
                ).fetchall()
        except Exception:
            logger.debug("Failed to load checkpoint writes for trace_id=%s db_path=%s", run_trace_id, db_path, exc_info=True)
            return []

        for checkpoint_id, checkpoint_ns, task_id, idx, channel, value_type, value_blob in rows:
            if str(channel) != "messages":
                continue
            try:
                decoded = serde.loads_typed((value_type, value_blob))
            except Exception:
                logger.debug(
                    "Failed to decode checkpoint write for trace_id=%s checkpoint_id=%s channel=%s",
                    run_trace_id,
                    checkpoint_id,
                    channel,
                    exc_info=True,
                )
                continue
            results.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": checkpoint_ns,
                    "task_id": task_id,
                    "idx": idx,
                    "channel": channel,
                    "messages": decoded if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes, bytearray)) else [decoded],
                }
            )

        return results

    @classmethod
    def _iter_result_messages(cls, result: Any) -> list[Any]:
        if isinstance(result, Mapping):
            msgs = result.get("messages")
            if isinstance(msgs, Sequence) and not isinstance(msgs, (str, bytes, bytearray)):
                return list(msgs)
        return []

    @classmethod
    def _extract_mapping_from_text(cls, text: str) -> Mapping[str, Any] | None:
        stripped = (text or "").strip()
        if not stripped or stripped.startswith("ERROR:"):
            return None

        candidates = [stripped]
        if "```json" in stripped:
            candidates.extend(re.findall(r"```json\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE))
        if "```" in stripped:
            candidates.extend(re.findall(r"```\s*(.*?)\s*```", stripped, flags=re.DOTALL))

        for candidate in candidates:
            body = candidate.strip()
            if not body:
                continue
            try:
                parsed = json.loads(body)
                if isinstance(parsed, Mapping):
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(body)
                if isinstance(parsed, Mapping):
                    return parsed
            except Exception:
                pass

        fallback: dict[str, Any] = {}
        path_match = re.search(r'"path"\s*:\s*"([^\"]+)"', stripped, flags=re.DOTALL)
        if path_match:
            fallback["path"] = path_match.group(1).strip()

        stdout_match = re.search(r'"stdout"\s*:\s*"(.*)"\s*(?:,\s*"stderr"|\}$)', stripped, flags=re.DOTALL)
        if stdout_match:
            stdout_value = stdout_match.group(1)
            stdout_value = stdout_value.replace("\\n", "\n").strip()
            if stdout_value:
                fallback["stdout"] = stdout_value

        if fallback:
            return fallback
        return None

    @classmethod
    def _read_text_if_exists(cls, path_value: Any) -> str | None:
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        try:
            path = Path(path_value).expanduser().resolve()
        except Exception:
            return None
        try:
            if not path.is_file():
                return None
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None
        text = text.strip()
        return text or None

    @staticmethod
    def _looks_like_glob_path(path_value: str) -> bool:
        normalized = (path_value or "").strip()
        if not normalized:
            return False
        return any(token in normalized for token in ("*", "?", "[", "]", "{" , "}"))

    @classmethod
    def _choose_better_config_path(cls, current_path: str | None, candidate_path: str | None) -> str | None:
        def _score(path_value: str | None) -> tuple[int, int]:
            if not isinstance(path_value, str):
                return (0, 0)
            normalized = path_value.strip()
            if not normalized:
                return (0, 0)
            if cls._looks_like_glob_path(normalized):
                return (1, len(normalized))
            try:
                resolved = Path(normalized).expanduser().resolve()
            except Exception:
                return (2, len(normalized))
            if resolved.is_file():
                return (4, len(normalized))
            return (3, len(normalized))

        candidate = candidate_path.strip() if isinstance(candidate_path, str) else ""
        current = current_path.strip() if isinstance(current_path, str) else ""
        if not candidate:
            return current or None
        if not current:
            return candidate
        return candidate if _score(candidate) > _score(current) else current

    @classmethod
    def _extract_stdout_from_artifact_paths(cls, candidate: Mapping[str, Any]) -> str | None:
        metadata = candidate.get("metadata")
        if isinstance(metadata, Mapping):
            stdout_path = cls._read_text_if_exists(metadata.get("stdout_path"))
            if stdout_path:
                return stdout_path

        workspace_path = candidate.get("workspace_path")
        artifacts = candidate.get("artifacts")
        if isinstance(workspace_path, str) and workspace_path.strip() and isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes, bytearray)):
            for artifact_name in artifacts:
                if str(artifact_name).strip() != "stdout.log":
                    continue
                try:
                    artifact_path = Path(workspace_path).expanduser().resolve() / "stdout.log"
                except Exception:
                    continue
                text = cls._read_text_if_exists(artifact_path.as_posix())
                if text:
                    return text

        return None

    @classmethod
    def extract_successful_tool_evidence(cls, results: Sequence[Any] | Any) -> dict[str, Any]:
        items = list(results) if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray, Mapping)) else [results]

        config_path: str | None = None
        stdout_blocks: list[str] = []
        headings: list[str] = []
        raw_texts: list[str] = []

        for result in items:
            for message in cls._iter_result_messages(result):
                text = ""
                artifact: Any | None = None
                is_tool_message = False

                if isinstance(message, ToolMessage):
                    text = cls._stringify_message_content(message.content)
                    artifact = getattr(message, "artifact", None)
                    is_tool_message = True
                elif isinstance(message, AIMessage):
                    text = cls._stringify_message_content(message.content)
                    artifact = getattr(message, "artifact", None)
                elif isinstance(message, Mapping):
                    text = cls._stringify_message_content(message.get("content"))
                    artifact = message.get("artifact")
                    role = str(message.get("role") or "").lower()
                    is_tool_message = role == "tool" or bool(message.get("tool_call_id"))
                else:
                    text = cls._stringify_message_content(getattr(message, "content", None))
                    artifact = getattr(message, "artifact", None)
                    is_tool_message = bool(getattr(message, "tool_call_id", None))

                if text and is_tool_message:
                    raw_texts.append(text)

                mapping_candidates: list[Mapping[str, Any]] = []
                if is_tool_message and isinstance(artifact, Mapping):
                    mapping_candidates.append(artifact)
                    structured_content = artifact.get("structured_content")
                    if isinstance(structured_content, Mapping):
                        mapping_candidates.append(structured_content)
                parsed_from_text = cls._extract_mapping_from_text(text) if is_tool_message else None
                if parsed_from_text is not None:
                    mapping_candidates.append(parsed_from_text)

                for candidate in mapping_candidates:
                    path_value = candidate.get("path")
                    if isinstance(path_value, str) and path_value.strip():
                        config_path = cls._choose_better_config_path(config_path, path_value)

                    stdout_value = candidate.get("stdout")
                    if isinstance(stdout_value, str) and stdout_value.strip() and not stdout_value.lstrip().startswith("ERROR:"):
                        stdout_blocks.append(stdout_value.strip())
                        continue

                    artifact_stdout = cls._extract_stdout_from_artifact_paths(candidate)
                    if artifact_stdout and not artifact_stdout.lstrip().startswith("ERROR:"):
                        stdout_blocks.append(artifact_stdout)

                for stdout_match in re.findall(r"\[stdout\]\s*(.*?)\s*\[/stdout\]", text, flags=re.DOTALL | re.IGNORECASE):
                    value = stdout_match.strip()
                    if value and not value.lstrip().startswith("ERROR:"):
                        stdout_blocks.append(value)

        def _extract_path_candidate(block: str) -> str | None:
            patterns = (
                r"(/[\w.\-~/一-龠ぁ-んァ-ヶー_%]+\.ya?ml)",
                r"(/[^\s'`\"]+\.ya?ml)",
            )
            for pattern in patterns:
                match = re.search(pattern, block)
                if match:
                    value = match.group(1).strip()
                    if value:
                        return value
            return None

        def _strip_markdown_emphasis(value: str) -> str:
            text = value.strip()
            if len(text) >= 4 and text.startswith("**") and text.endswith("**"):
                text = text[2:-2].strip()
            if len(text) >= 2 and text.startswith("`") and text.endswith("`"):
                return text[1:-1].strip()
            text = text.replace("**", "").replace("__", "")
            return text

        def _extract_heading_candidates(block: str) -> dict[str, list[str]]:
            exact_lines: list[str] = []
            markdown_lines: list[str] = []
            fallback_lines: list[str] = []
            for line in block.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                exact_line_match = re.match(r"^HEADING_LINE_EXACT\s*:\s*(.+)$", stripped, flags=re.IGNORECASE)
                if exact_line_match:
                    value = exact_line_match.group(1)
                    if value:
                        exact_lines.append(value)
                    continue
                exact_match = re.match(r"^HEADING_EXACT\s*:\s*(.+)$", stripped, flags=re.IGNORECASE)
                if exact_match:
                    value = exact_match.group(1).strip()
                    if value:
                        exact_lines.append(value)
                    continue
                if re.match(r"^#{1,6}\s+.+$", stripped):
                    markdown_lines.append(stripped)
                    continue
                numbered_match = re.match(r"^\d+[.)]\s+(.+)$", stripped)
                if numbered_match:
                    value = _strip_markdown_emphasis(numbered_match.group(1).strip())
                    if value and not value.endswith((":", "：")):
                        fallback_lines.append(value)
                        continue
                bullet_match = re.match(r"^(?:[-*])\s+(.+)$", stripped)
                if bullet_match:
                    value = bullet_match.group(1).strip()
                    if value.startswith("**") and value.endswith("**"):
                        normalized = _strip_markdown_emphasis(value)
                        if normalized and not normalized.endswith((":", "：")):
                            fallback_lines.append(normalized)
                    continue
                quote_chars = {'"', "'", "「", "」", "“", "”"}
                if len(stripped) >= 2 and stripped[0] in quote_chars and stripped[-1] in quote_chars:
                    value = stripped[1:-1].strip()
                    if value:
                        fallback_lines.append(value)

                table_match = re.match(r"^\|\s*\d+\s*\|\s*(.+?)\s*\|\s*.+\|$", stripped)
                if table_match:
                    value = _strip_markdown_emphasis(table_match.group(1).strip())
                    if value:
                        fallback_lines.append(value)

            label_match = re.search(r"重要な見出し\s*[:：]\s*(.+)", block)
            if label_match:
                tail = label_match.group(1).strip()
                if tail:
                    for part in re.split(r"\s*,\s*|\s*、\s*|\s*\|\s*", tail):
                        value = part.strip()
                        if value:
                            fallback_lines.append(value)
            return {
                "exact": exact_lines,
                "markdown": markdown_lines,
                "fallback": fallback_lines,
            }

        def _dedupe(values: Sequence[str]) -> list[str]:
            items: list[str] = []
            seen: set[str] = set()
            for value in values:
                normalized = str(value).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                items.append(normalized)
            return items

        def _is_synthetic_heading(value: str) -> bool:
            normalized = str(value).strip()
            if not normalized:
                return True
            lowered = normalized.lower()
            if re.match(r"^#{1,6}\s+headings by file$", lowered):
                return True
            if re.match(r"^#{1,6}\s+共通見出し\s*\d+\s*点$", normalized):
                return True
            if re.match(r"^#{1,6}\s+common\s+headings?\b", lowered):
                return True
            if re.match(r"^#{1,6}\s+file\s+\d+\s*:", lowered):
                return True
            if re.search(r"`[^`]+\.md`", normalized, flags=re.IGNORECASE):
                return True
            return False

        def _is_usable_heading_block(block: Sequence[str]) -> bool:
            normalized_block = [str(value).strip() for value in block if str(value).strip()]
            if not normalized_block:
                return False
            synthetic_count = sum(1 for value in normalized_block if _is_synthetic_heading(value))
            return synthetic_count < len(normalized_block)

        deduped_stdout: list[str] = []
        seen_stdout: set[str] = set()
        exact_heading_blocks: list[list[str]] = []
        markdown_heading_blocks: list[list[str]] = []
        fallback_heading_blocks: list[list[str]] = []
        for stdout_text in stdout_blocks:
            if stdout_text not in seen_stdout:
                seen_stdout.add(stdout_text)
                deduped_stdout.append(stdout_text)

            if not config_path:
                stdout_path = _extract_path_candidate(stdout_text)
                if stdout_path:
                    config_path = cls._choose_better_config_path(config_path, stdout_path)

            candidates = _extract_heading_candidates(stdout_text)
            exact_values = _dedupe(candidates["exact"])
            markdown_values = _dedupe(candidates["markdown"])
            fallback_values = _dedupe(candidates["fallback"])

            if exact_values:
                exact_heading_blocks.append(exact_values)
                continue
            if markdown_values:
                markdown_heading_blocks.append(markdown_values)
            if fallback_values:
                fallback_heading_blocks.append(fallback_values)

        def _pick_best_heading_block(blocks: Sequence[list[str]]) -> list[str]:
            if not blocks:
                return []
            usable_blocks = [block for block in blocks if _is_usable_heading_block(block)]
            if not usable_blocks:
                return []
            return list(max(enumerate(usable_blocks), key=lambda item: (len(item[1]), item[0]))[1])

        headings = _pick_best_heading_block(exact_heading_blocks)
        if not headings:
            headings = _pick_best_heading_block(markdown_heading_blocks)
        if not headings:
            headings = _pick_best_heading_block(fallback_heading_blocks)

        if not config_path or not headings:
            raw_exact_blocks: list[list[str]] = []
            raw_markdown_blocks: list[list[str]] = []
            raw_fallback_blocks: list[list[str]] = []

            for raw_text in raw_texts:
                raw_path = _extract_path_candidate(raw_text)
                if raw_path:
                    config_path = cls._choose_better_config_path(config_path, raw_path)

                candidates = _extract_heading_candidates(raw_text)
                exact_values = _dedupe(candidates["exact"])
                markdown_values = _dedupe(candidates["markdown"])
                fallback_values = _dedupe(candidates["fallback"])

                if exact_values:
                    raw_exact_blocks.append(exact_values)
                    continue
                if markdown_values:
                    raw_markdown_blocks.append(markdown_values)
                if fallback_values:
                    raw_fallback_blocks.append(fallback_values)

            if not headings:
                headings = _pick_best_heading_block(raw_exact_blocks)
            if not headings:
                headings = _pick_best_heading_block(raw_markdown_blocks)
            if not headings:
                headings = _pick_best_heading_block(raw_fallback_blocks)

        return {
            "config_path": config_path,
            "stdout_blocks": deduped_stdout,
            "headings": headings,
            "raw_texts": raw_texts,
        }

    @classmethod
    def _extract_task_id_from_tool_result(cls, result: Any) -> str | None:
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray, Mapping)):
            for item in result:
                if isinstance(item, Mapping):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        mapping = cls._extract_mapping_from_text(text_value)
                        if isinstance(mapping, Mapping):
                            task_id = mapping.get("task_id")
                            if isinstance(task_id, str) and task_id.strip():
                                return task_id.strip()
        if isinstance(result, Mapping):
            task_id = result.get("task_id")
            if isinstance(task_id, str) and task_id.strip():
                return task_id.strip()
        return None

    @classmethod
    def build_coding_agent_heading_rescue_prompt(
        cls,
        *,
        user_request_text: str,
        requested_heading_count: int,
    ) -> str:
        count = requested_heading_count if requested_heading_count > 0 else 3
        lines = [user_request_text.strip()]
        lines.extend(
            [
                "",
                "Requirements:",
                "- Investigate only the Markdown files relevant to the user request.",
                f"- Return exactly {count} common headings.",
                "- Output each heading on its own line in the format: HEADING_LINE_EXACT: <exact Markdown heading line>",
                "- Do not summarize in a table.",
                "- Do not paraphrase the heading text.",
                "- Preserve the exact Markdown heading line from the source files.",
            ]
        )
        return "\n".join(lines).strip()

    @classmethod
    async def run_direct_coding_agent_heading_rescue(
        cls,
        *,
        runtime_config: AiChatUtilConfig,
        messages: Sequence[Any],
        run_trace_id: str,
        requested_heading_count: int,
    ) -> dict[str, Any]:
        user_request_text = cls._extract_user_request_text(messages)
        if not user_request_text.strip():
            return {}

        workspace_candidates = cls.extract_explicit_user_directory_paths(messages)
        workspace_path = workspace_candidates[0] if workspace_candidates else str(runtime_config.mcp.working_directory or "")
        if not workspace_path:
            return {}

        mcp_config = runtime_config.get_mcp_server_config().filter(include_name=runtime_config.mcp.coding_agent_endpoint.mcp_server_name)
        if len(mcp_config.servers) == 0:
            return {}

        def _status_mapping(value: Any) -> Mapping[str, Any] | None:
            if isinstance(value, Mapping):
                return value
            if hasattr(value, "model_dump"):
                try:
                    dumped = value.model_dump(mode="python")
                except TypeError:
                    dumped = value.model_dump()
                if isinstance(dumped, Mapping):
                    return dumped
            text = cls._stringify_message_content(value)
            parsed = cls._extract_mapping_from_text(text)
            return parsed if isinstance(parsed, Mapping) else None

        def _build_rescue_evidence(status_value: Any, result_value: Any) -> dict[str, Any]:
            status_mapping = _status_mapping(status_value)
            return cls.extract_successful_tool_evidence(
                [
                    {
                        "messages": [
                            {
                                "role": "tool",
                                "content": cls._stringify_message_content(status_value),
                                "artifact": status_mapping,
                            },
                            {
                                "role": "tool",
                                "content": cls._stringify_message_content(result_value),
                            },
                        ]
                    }
                ]
            )

        client = MultiServerMCPClient(mcp_config.to_langchain_config())
        tools = await client.get_tools()
        tool_map = {
            str(getattr(tool, "name", "")).strip(): tool
            for tool in tools
            if str(getattr(tool, "name", "")).strip()
        }
        execute_tool = tool_map.get("execute")
        status_tool = tool_map.get("status")
        result_tool = tool_map.get("get_result")
        if execute_tool is None or status_tool is None or result_tool is None:
            return {}

        rescue_prompt = cls.build_coding_agent_heading_rescue_prompt(
            user_request_text=user_request_text,
            requested_heading_count=requested_heading_count,
        )
        execute_result = await execute_tool.ainvoke(
            {
                "req": {
                    "prompt": rescue_prompt,
                    "workspace_path": workspace_path,
                    "timeout": 300,
                    "trace_id": run_trace_id,
                }
            }
        )
        task_id = cls._extract_task_id_from_tool_result(execute_result)
        if not task_id:
            return {}

        latest_status_result: Any | None = None
        for _ in range(30):
            status_result = await status_tool.ainvoke({"task_id": task_id, "tail": 40, "wait_seconds": 1})
            latest_status_result = status_result
            status_text = cls._stringify_message_content(status_result)
            if "completed" in status_text or "failed" in status_text or "timeout" in status_text or "cancelled" in status_text or '"sub_status":"completed"' in status_text:
                break

        result = await result_tool.ainvoke({"task_id": task_id, "tail": None, "wait_seconds": 0})
        evidence = _build_rescue_evidence(latest_status_result, result)
        if not (evidence.get("headings") or []):
            for _ in range(5):
                await asyncio.sleep(1)
                latest_status_result = await status_tool.ainvoke({"task_id": task_id, "tail": 40, "wait_seconds": 0})
                result = await result_tool.ainvoke({"task_id": task_id, "tail": None, "wait_seconds": 0})
                evidence = _build_rescue_evidence(latest_status_result, result)
                if evidence.get("headings"):
                    break
        evidence = dict(evidence)
        evidence["latest_task_id"] = task_id
        if requested_heading_count > 0:
            evidence["requested_heading_count"] = requested_heading_count
        return evidence

    @classmethod
    async def collect_evidence_results(
        cls,
        *,
        app: Any,
        run_trace_id: str,
        workflow_results: Sequence[Any] | Any,
        checkpoint_db_path: Path | None = None,
    ) -> list[Any]:
        items = list(workflow_results) if isinstance(workflow_results, Sequence) and not isinstance(workflow_results, (str, bytes, bytearray, Mapping)) else [workflow_results]

        checkpoint_results = await cls.collect_checkpoint_results(
            app=app,
            run_trace_id=run_trace_id,
        )
        if checkpoint_results:
            items.extend(checkpoint_results)

        checkpoint_write_results = cls.collect_checkpoint_write_results(
            checkpoint_db_path=checkpoint_db_path,
            run_trace_id=run_trace_id,
        )
        if checkpoint_write_results:
            items.extend(checkpoint_write_results)
        return items

    @classmethod
    def final_text_contradicts_evidence(cls, user_text: str | None, evidence: Mapping[str, Any]) -> bool:
        text = (user_text or "").strip().lower()
        if not text:
            return bool(evidence.get("config_path") or evidence.get("stdout_blocks"))

        has_evidence = bool(evidence.get("config_path") or evidence.get("stdout_blocks"))
        if not has_evidence:
            return False

        exact_headings = cls.select_headings_for_response(evidence) if cls.expects_heading_response(evidence) else []
        if exact_headings:
            final_heading_candidates: list[str] = []
            for raw_line in (user_text or "").splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                exact_line_match = re.match(r"^(?:[-*]\s+)?HEADING_LINE_EXACT\s*:\s*(.+)$", stripped, flags=re.IGNORECASE)
                if exact_line_match:
                    value = exact_line_match.group(1).strip()
                    if value:
                        final_heading_candidates.append(value)
                        continue
                markdown_match = re.match(r"^(?:[-*]\s+)?(#{1,6}\s+.+)$", stripped)
                if markdown_match:
                    final_heading_candidates.append(markdown_match.group(1).strip())

            if final_heading_candidates and final_heading_candidates[0] != exact_headings[0]:
                return True

        negative_markers = (
            "取得できなかった",
            "確認できなかった",
            "行えませんでした",
            "できませんでした",
            "わかりませんでした",
            "失敗しました",
            "抽出に失敗",
            "取得することができませんでした",
            "返ってきませんでした",
            "不明です",
            "問題が発生しました",
            "追加の結果が必要",
            "再度試行",
            "手動で確認",
            "他の手法での抽出が必要",
        )
        return any(marker in text for marker in negative_markers)

    @classmethod
    def final_text_missing_concrete_evidence(cls, user_text: str | None, evidence: Mapping[str, Any]) -> bool:
        text = (user_text or "").strip()
        if not text:
            if cls.expects_heading_response(evidence):
                return bool(evidence.get("config_path") or evidence.get("headings"))
            if cls.expects_tool_catalog_response(evidence):
                return bool(evidence.get("tool_catalog"))
            return bool(evidence.get("config_path") or evidence.get("stdout_blocks"))

        config_path = evidence.get("config_path")
        if (
            not cls.expects_tool_catalog_response(evidence)
            and isinstance(config_path, str)
            and config_path.strip()
            and config_path.strip() not in text
        ):
            return True

        exact_headings = cls.select_headings_for_response(evidence) if cls.expects_heading_response(evidence) else []
        if exact_headings:
            matched = sum(1 for heading in exact_headings if heading in text)
            if matched < len(exact_headings):
                return True

        if cls.expects_tool_catalog_response(evidence):
            tool_catalog = cast(Sequence[Any], evidence.get("tool_catalog") or [])
            expected_tool_names = [
                str(tool_name).strip()
                for entry in tool_catalog
                if isinstance(entry, Mapping)
                for tool_name in cast(Sequence[Any], entry.get("tool_names") or [])
                if str(tool_name).strip()
            ]
            if expected_tool_names and not any(tool_name in text for tool_name in expected_tool_names):
                return True

        return False

    @staticmethod
    def contains_followup_task_error_signal(text: str | None) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return (
            "invalid_followup_task_id" in normalized
            or "stale_followup_task_id" in normalized
            or "follow-up task_id is invalid" in normalized
            or "follow-up task_id is stale" in normalized
        )

    @classmethod
    def expects_heading_response(cls, evidence: Mapping[str, Any]) -> bool:
        explicit = evidence.get("expects_heading_response")
        if isinstance(explicit, bool):
            return explicit

        requested_count_raw = evidence.get("requested_heading_count")
        try:
            requested_count = int(requested_count_raw) if requested_count_raw is not None else 0
        except (TypeError, ValueError):
            requested_count = 0
        return requested_count > 0

    @classmethod
    def expects_tool_catalog_response(cls, evidence: Mapping[str, Any]) -> bool:
        explicit = evidence.get("expects_tool_catalog_response")
        return bool(explicit) if isinstance(explicit, bool) else False

    @staticmethod
    def _heading_level(heading: str) -> int | None:
        match = re.match(r"^(#{1,6})\s+.+$", heading or "")
        if not match:
            return None
        return len(match.group(1))

    @staticmethod
    def _is_numbered_heading(heading: str) -> bool:
        match = re.match(r"^#{1,6}\s+(.+)$", heading or "")
        if not match:
            return False
        return bool(re.match(r"^\d+[\.．\)]\s*.+$", match.group(1).strip()))

    @classmethod
    def _select_numbered_heading_block(cls, headings: Sequence[str], requested_count: int) -> list[str]:
        for start_index, heading in enumerate(headings):
            if not cls._is_numbered_heading(heading):
                continue
            level = cls._heading_level(heading)
            if level is None:
                continue

            block = [heading]
            for candidate in headings[start_index + 1 :]:
                candidate_level = cls._heading_level(candidate)
                if candidate_level is None:
                    continue
                if candidate_level < level:
                    break
                if candidate_level == level and cls._is_numbered_heading(candidate):
                    block.append(candidate)
                    if len(block) >= requested_count:
                        return block[:requested_count]
                    continue
                if candidate_level == level:
                    break

            if len(block) >= requested_count:
                return block[:requested_count]

        return []

    @classmethod
    def select_headings_for_response(cls, evidence: Mapping[str, Any]) -> list[str]:
        headings = evidence.get("headings")
        exact_headings = [str(v).strip() for v in headings if isinstance(v, str) and str(v).strip()] if isinstance(headings, Sequence) else []
        if not exact_headings:
            return []

        requested_count_raw = evidence.get("requested_heading_count")
        try:
            requested_count = int(requested_count_raw) if requested_count_raw is not None else 0
        except (TypeError, ValueError):
            requested_count = 0

        if requested_count <= 0 or requested_count >= len(exact_headings):
            return exact_headings

        numbered_block = cls._select_numbered_heading_block(exact_headings, requested_count)
        if numbered_block:
            return numbered_block

        return exact_headings[:requested_count]

    @classmethod
    def build_evidence_reflected_final_text(cls, evidence: Mapping[str, Any]) -> str:
        lines: list[str] = []

        config_path = evidence.get("config_path")
        if isinstance(config_path, str) and config_path.strip():
            lines.append(f"設定ファイルの場所: {config_path.strip()}")

        exact_headings = cls.select_headings_for_response(evidence) if cls.expects_heading_response(evidence) else []
        if exact_headings:
            lines.append("文書内の重要な見出し:")
            for heading in exact_headings:
                lines.append(heading)

        if cls.expects_tool_catalog_response(evidence):
            tool_catalog = cast(Sequence[Any], evidence.get("tool_catalog") or [])
            if tool_catalog:
                lines.append("supervisor が参照した利用可能ツール一覧:")
                for entry in tool_catalog:
                    if not isinstance(entry, Mapping):
                        continue
                    agent_name = str(entry.get("agent_name") or "").strip()
                    tool_names = [
                        str(tool_name).strip()
                        for tool_name in cast(Sequence[Any], entry.get("tool_names") or [])
                        if str(tool_name).strip()
                    ]
                    if not agent_name or not tool_names:
                        continue
                    lines.append(f"- {agent_name}: {', '.join(tool_names)}")

        stdout_blocks = evidence.get("stdout_blocks")
        if isinstance(stdout_blocks, Sequence):
            stdout_values = [str(v).strip() for v in stdout_blocks if isinstance(v, str) and v.strip()]
            if stdout_values and not exact_headings:
                lines.append("取得済みの coding-agent 実行結果:")
                lines.append("[stdout]")
                lines.append(stdout_values[-1])
                lines.append("[/stdout]")

        return "\n".join(lines).strip()

    @classmethod
    def _normalize_tool_description(cls, description: Any) -> str:
        text = cls._strip_tool_metadata(description)
        text = re.sub(r"\s+", " ", text).strip()
        return text or "(説明なし)"

    @classmethod
    def _extract_tool_metadata(cls, description: Any) -> dict[str, str]:
        text = str(description or "")
        match = re.search(r"\[MCP_META\](.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return {}

        metadata: dict[str, str] = {}
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            normalized_key = str(key).strip().lower()
            normalized_value = str(value).strip()
            if normalized_key:
                metadata[normalized_key] = normalized_value
        return metadata

    @classmethod
    def _strip_tool_metadata(cls, description: Any) -> str:
        text = str(description or "")
        stripped = re.sub(r"\n?\[MCP_META\].*$", "", text, flags=re.IGNORECASE | re.DOTALL)
        return stripped.strip()

    @classmethod
    def _extract_primary_arg_names(cls, args_schema: Any) -> list[str]:
        if not isinstance(args_schema, Mapping):
            return []

        props = args_schema.get("properties")
        if not isinstance(props, Mapping):
            return []

        req_schema = props.get("req")
        required = args_schema.get("required")
        if (
            isinstance(req_schema, Mapping)
            and isinstance(required, Sequence)
            and set(required) == {"req"}
        ):
            inner_props = req_schema.get("properties")
            if isinstance(inner_props, Mapping):
                return [str(name).strip() for name in inner_props.keys() if str(name).strip()]

        return [str(name).strip() for name in props.keys() if str(name).strip()]

    @classmethod
    def build_route_tool_catalog_payload(
        cls,
        route_tool_inventory: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        runtime_config: AiChatUtilConfig | None = None,
    ) -> dict[str, Any]:
        route_backend_metadata = cls.build_route_backend_metadata(
            route_tool_inventory=route_tool_inventory,
            runtime_config=runtime_config,
        )
        tool_catalog: list[dict[str, Any]] = []
        tool_agent_names: list[str] = []

        for route_name, tools in route_tool_inventory.items():
            normalized_tools = [tool for tool in tools if isinstance(tool, Mapping) and str(tool.get("name") or "").strip()]
            if not normalized_tools:
                continue
            route_name_str = str(route_name).strip()
            backend_metadata = route_backend_metadata.get(route_name_str, {})
            agent_name = str(backend_metadata.get("agent_name") or cls.build_tool_agent_label(route_name_str))
            tool_agent_names.append(agent_name)
            tool_catalog.append(
                {
                    "agent_name": agent_name,
                    "agent_family": backend_metadata.get("agent_family"),
                    "selected_server_key": backend_metadata.get("selected_server_key"),
                    "server_keys": backend_metadata.get("server_keys") or [],
                    "backend_kind": backend_metadata.get("backend_kind"),
                    "tool_names": [str(tool.get("name") or "").strip() for tool in normalized_tools],
                    "tools": [
                        {
                            "name": str(tool.get("name") or "").strip(),
                            "description": cls._normalize_tool_description(tool.get("description")),
                            "primary_args": [
                                str(arg_name).strip()
                                for arg_name in cast(Sequence[Any], tool.get("primary_args") or [])
                                if str(arg_name).strip()
                            ],
                        }
                        for tool in normalized_tools
                    ],
                }
            )

        return {
            "tool_agent_names": tool_agent_names,
            "tool_catalog": tool_catalog,
            "route_backends": route_backend_metadata,
        }

    @classmethod
    def build_tool_catalog_response_text(
        cls,
        route_tool_inventory: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        include_details: bool = False,
    ) -> str:
        lines = ["supervisor が参照した利用可能ツール一覧:"]
        for route_name, tools in route_tool_inventory.items():
            normalized_tools = [tool for tool in tools if isinstance(tool, Mapping) and str(tool.get("name") or "").strip()]
            if not normalized_tools:
                continue
            label = cls.build_tool_agent_label(str(route_name).strip())
            if not include_details:
                lines.append(
                    f"- {label}: {', '.join(str(tool.get('name') or '').strip() for tool in normalized_tools)}"
                )
                continue
            lines.append(f"\n### {label}")
            for index, tool in enumerate(normalized_tools, start=1):
                tool_name = str(tool.get("name") or "").strip()
                description = cls._normalize_tool_description(tool.get("description"))
                primary_args = [
                    str(arg_name).strip()
                    for arg_name in cast(Sequence[Any], tool.get("primary_args") or [])
                    if str(arg_name).strip()
                ]
                tool_metadata = tool.get("tool_metadata") if isinstance(tool.get("tool_metadata"), Mapping) else {}
                lines.append(f"{index}. {tool_name}")
                lines.append(f"   - 説明: {description}")
                lines.append(f"   - 主要な引数: {', '.join(primary_args) if primary_args else 'なし'}")
                if tool_metadata:
                    action_kind = str(tool_metadata.get("action_kind") or "").strip()
                    requires_approval = str(tool_metadata.get("requires_approval") or "").strip().lower()
                    usage_guidance = str(tool_metadata.get("usage_guidance") or "").strip()
                    if action_kind:
                        lines.append(f"   - 操作種別: {action_kind}")
                    if requires_approval:
                        lines.append(f"   - 承認要否: {requires_approval}")
                    if usage_guidance:
                        lines.append(f"   - 利用ガイダンス: {usage_guidance}")
        return "\n".join(lines)

    @classmethod
    def extract_config_path_from_text(cls, text: str | None) -> str | None:
        value = (text or "").strip()
        if not value:
            return None
        match = re.search(r"((?:[A-Za-z]:[\\/]|/)[^\s'\"`]+\.ya?ml)", value)
        if not match:
            return None
        candidate = match.group(1).strip()
        if cls._looks_like_glob_path(candidate):
            return None
        return candidate

    @classmethod
    def should_run_config_preflight(cls, messages: Sequence[Any]) -> bool:
        for message in messages:
            text = ""
            if isinstance(message, HumanMessage):
                text = cls._stringify_message_content(message.content)
            elif isinstance(message, Mapping) and str(message.get("role") or "").lower() in {"user", "human"}:
                text = cls._stringify_message_content(message.get("content"))

            normalized = text.strip().lower()
            if not normalized or "get_loaded_config_info" not in normalized:
                continue
            if any(token in text for token in ("まず", "最初", "先に")):
                return True
            if re.search(r"\b(first|before)\b", normalized):
                return True
        return False

    @classmethod
    def _extract_config_preflight_payload(cls, result: Any) -> dict[str, Any]:
        text_value: str | None = None
        config_path: str | None = None
        artifact_value: Any | None = None

        if isinstance(result, tuple) and len(result) == 2:
            text_value = cls._stringify_message_content(result[0])
            artifact_value = result[1]
        elif isinstance(result, Mapping):
            artifact_value = dict(result)
            text_value = cls._stringify_message_content(result)
        else:
            text_value = cls._stringify_message_content(result)

        if isinstance(artifact_value, Mapping):
            path_value = artifact_value.get("path")
            if isinstance(path_value, str) and path_value.strip():
                config_path = path_value.strip()

        if not config_path:
            config_path = cls.extract_config_path_from_text(text_value)

        return {
            "text": text_value,
            "config_path": config_path,
            "artifact": artifact_value,
            "success": not (isinstance(text_value, str) and text_value.lstrip().startswith("ERROR:")),
        }

    @classmethod
    def get_coding_agent_server_name(cls, runtime_config: AiChatUtilConfig) -> str:
        return str(runtime_config.mcp.coding_agent_endpoint.mcp_server_name).strip()

    @classmethod
    def build_tool_agent_label(cls, route_name: str | None) -> str:
        normalized_route_name = str(route_name or "").strip()
        label_map = {
            "coding_agent": "tool_agent_coding",
            "general_tool_agent": "tool_agent_general",
        }
        return label_map.get(normalized_route_name, normalized_route_name or "tool_agent")

    @classmethod
    def build_route_backend_metadata(
        cls,
        *,
        route_tool_inventory: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        runtime_config: AiChatUtilConfig | None = None,
        workflow_file_path: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        known_routes = {
            str(route_name).strip()
            for route_name in (route_tool_inventory or {}).keys()
            if str(route_name).strip()
        }
        metadata: dict[str, dict[str, Any]] = {}

        if "coding_agent" in known_routes:
            selected_server_key = cls.get_coding_agent_server_name(runtime_config) if runtime_config is not None else None
            metadata["coding_agent"] = {
                "agent_name": cls.build_tool_agent_label("coding_agent"),
                "agent_family": "coding_agent",
                "selected_server_key": selected_server_key,
                "server_keys": [selected_server_key] if selected_server_key else [],
                "backend_kind": "mcp_async_task",
            }

        if "general_tool_agent" in known_routes:
            server_keys: list[str] = []
            if runtime_config is not None:
                general_mcp_config = runtime_config.get_mcp_server_config().filter(
                    exclude_name=cls.get_coding_agent_server_name(runtime_config)
                )
                server_keys = sorted(str(name).strip() for name in general_mcp_config.servers.keys() if str(name).strip())
            metadata["general_tool_agent"] = {
                "agent_name": cls.build_tool_agent_label("general_tool_agent"),
                "agent_family": "general_tool_agent",
                "selected_server_key": server_keys[0] if len(server_keys) == 1 else None,
                "server_keys": server_keys,
                "backend_kind": "mcp_tools",
            }

        if "deep_agent" in known_routes:
            metadata["deep_agent"] = {
                "agent_name": cls.build_tool_agent_label("deep_agent"),
                "agent_family": "deep_agent",
                "selected_server_key": None,
                "server_keys": [],
                "backend_kind": "deepagents",
            }

        if isinstance(workflow_file_path, str) and workflow_file_path.strip():
            metadata["workflow_backend"] = {
                "agent_name": "workflow_backend",
                "agent_family": "workflow_backend",
                "selected_server_key": None,
                "server_keys": [],
                "backend_kind": "workflow_markdown",
                "workflow_file_path": workflow_file_path.strip(),
            }

        for route_name in sorted(known_routes):
            metadata.setdefault(
                route_name,
                {
                    "agent_name": cls.build_tool_agent_label(route_name),
                    "agent_family": route_name,
                    "selected_server_key": None,
                    "server_keys": [],
                    "backend_kind": None,
                },
            )

        return metadata

    @classmethod
    async def run_config_preflight(
        cls,
        *,
        runtime_config: AiChatUtilConfig,
        tool_limits: ToolLimits | None,
        audit_context: AuditContext | None,
    ) -> dict[str, Any] | None:
        mcp_config = runtime_config.get_mcp_server_config()
        coding_agent_server_name = cls.get_coding_agent_server_name(runtime_config)
        general_mcp_config = mcp_config.filter(exclude_name=coding_agent_server_name)
        if len(general_mcp_config.servers) == 0:
            return None

        tool_client = MultiServerMCPClient(general_mcp_config.to_langchain_config())
        tools = [
            cls._maybe_wrap_req_nested_tool(tool)
            for tool in await tool_client.get_tools()
            if str(getattr(tool, "name", "")).strip() == "get_loaded_config_info"
        ]
        if not tools:
            return None

        if tool_limits is not None:
            tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = tool_limits.guard_params()
        else:
            tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = (0, 0.0, 0)

        tool_state: dict[str, Any] = {
            "used": 0,
            "general_used": 0,
            "followup_used": 0,
            "followup_limit": 0,
            "agent_name": "tool_agent_general",
            "audit_context": audit_context,
            "explicit_user_file_paths": [],
        }
        ToolLimits._apply_tool_execution_guards(
            tools,
            tool_call_state=tool_state,
            tool_call_limit_int=tool_call_limit_int,
            tool_timeout_seconds_f=tool_timeout_seconds_f,
            tool_timeout_retries_int=tool_timeout_retries_int,
        )

        tool = tools[0]
        if getattr(tool, "coroutine", None) is not None or hasattr(tool, "ainvoke"):
            result = await tool.ainvoke({})
        elif getattr(tool, "func", None) is not None:
            result = tool.func()
        else:
            return None

        return cls._extract_config_preflight_payload(result)

    @classmethod
    def build_config_preflight_message(cls, preflight_payload: Mapping[str, Any]) -> str | None:
        config_path = preflight_payload.get("config_path")
        text_value = preflight_payload.get("text")
        success = bool(preflight_payload.get("success", False))

        lines = ["[Config Preflight]"]
        if isinstance(config_path, str) and config_path.strip():
            lines.append(f"- get_loaded_config_info 実行済み: path={config_path.strip()}")
            lines.append("- 以後は取得済みの path/config を使い回し、get_loaded_config_info を再実行しないでください。")
        elif isinstance(text_value, str) and text_value.strip():
            status_text = "成功" if success else "失敗"
            lines.append(f"- get_loaded_config_info 実行済み ({status_text})")
            lines.append(f"- result: {text_value.strip()}")
            lines.append("- この preflight は既に実行済みです。同じ確認を繰り返さず、取得済み情報だけで次の手順へ進んでください。")
        else:
            return None
        return "\n".join(lines)

    @classmethod
    def merge_preflight_evidence(cls, evidence: Mapping[str, Any], preflight_payload: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = dict(evidence)
        if not isinstance(preflight_payload, Mapping):
            return merged

        config_path = preflight_payload.get("config_path")
        if isinstance(config_path, str) and config_path.strip():
            merged["config_path"] = cls._choose_better_config_path(cast(str | None, merged.get("config_path")), config_path.strip())

        text_value = preflight_payload.get("text")
        if isinstance(text_value, str) and text_value.strip() and not text_value.lstrip().startswith("ERROR:"):
            stdout_blocks = list(cast(Sequence[Any], merged.get("stdout_blocks") or []))
            if text_value.strip() not in [str(item).strip() for item in stdout_blocks if isinstance(item, str)]:
                stdout_blocks.insert(0, text_value.strip())
                merged["stdout_blocks"] = stdout_blocks

        return merged

    @classmethod
    def get_loaded_runtime_config_path(cls) -> str | None:
        try:
            config_path = get_runtime_config_path().expanduser().resolve()
        except Exception:
            return None
        if not config_path.is_file():
            return None
        return config_path.as_posix()

    @classmethod
    def extract_markdown_heading_lines_from_files(cls, file_paths: Sequence[str] | None) -> list[str]:
        headings: list[str] = []
        seen: set[str] = set()

        for raw_path in file_paths or []:
            try:
                path = Path(str(raw_path)).expanduser().resolve()
            except Exception:
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".markdown"}:
                continue
            text = cls._read_text_if_exists(path.as_posix())
            if not text:
                continue
            for line in text.splitlines():
                normalized_line = line.rstrip("\r")
                if not re.match(r"^#{1,6}\s+\S", normalized_line):
                    continue
                if normalized_line in seen:
                    continue
                seen.add(normalized_line)
                headings.append(normalized_line)

        return headings

    @classmethod
    def should_prefer_deterministic_evidence_response(cls, user_text: str | None, evidence: Mapping[str, Any]) -> bool:
        if not cls.expects_heading_response(evidence):
            return False
        headings = evidence.get("headings")
        exact_headings = [str(v).strip() for v in headings if isinstance(v, str) and str(v).strip()] if isinstance(headings, Sequence) else []
        if not exact_headings:
            return False

        if len(exact_headings) >= 3:
            return True

        text = (user_text or "").strip().lower()
        if not text:
            return True

        if "見出し" in text or "heading" in text or "heading_line_exact" in text:
            return True

        for raw_line in (user_text or "").splitlines():
            stripped = raw_line.strip()
            if re.match(r"^(?:[-*]\s+)?HEADING_LINE_EXACT\s*:", stripped, flags=re.IGNORECASE):
                return True
            if re.match(r"^(?:[-*]\s+)?#{1,6}\s+.+$", stripped):
                return True

        return False

    @classmethod
    def augment_final_text_with_evidence(cls, user_text: str | None, evidence: Mapping[str, Any]) -> str:
        base_text = (user_text or "").strip()
        lines: list[str] = [base_text] if base_text else []

        config_path = evidence.get("config_path")
        if isinstance(config_path, str) and config_path.strip():
            normalized_path = config_path.strip()
            if normalized_path not in base_text:
                lines.append(f"設定ファイルの場所: {normalized_path}")

        exact_headings = cls.select_headings_for_response(evidence) if cls.expects_heading_response(evidence) else []
        if exact_headings:
            missing_headings = [heading for heading in exact_headings if heading not in base_text]
            if missing_headings:
                if "文書内の重要な見出し:" not in base_text:
                    lines.append("文書内の重要な見出し:")
                lines.extend(missing_headings)

        if cls.expects_tool_catalog_response(evidence):
            tool_catalog = cast(Sequence[Any], evidence.get("tool_catalog") or [])
            expected_lines = []
            for entry in tool_catalog:
                if not isinstance(entry, Mapping):
                    continue
                agent_name = str(entry.get("agent_name") or "").strip()
                tool_names = [
                    str(tool_name).strip()
                    for tool_name in cast(Sequence[Any], entry.get("tool_names") or [])
                    if str(tool_name).strip()
                ]
                if agent_name and tool_names:
                    expected_lines.append(f"- {agent_name}: {', '.join(tool_names)}")
            if expected_lines:
                if "supervisor が参照した利用可能ツール一覧:" not in base_text:
                    lines.append("supervisor が参照した利用可能ツール一覧:")
                for line in expected_lines:
                    if line not in base_text:
                        lines.append(line)

        stdout_blocks = evidence.get("stdout_blocks")
        if isinstance(stdout_blocks, Sequence):
            stdout_values = [str(v).strip() for v in stdout_blocks if isinstance(v, str) and v.strip()]
            if stdout_values and not exact_headings and "[stdout]" not in base_text:
                lines.append("取得済みの coding-agent 実行結果:")
                lines.append("[stdout]")
                lines.append(stdout_values[-1])
                lines.append("[/stdout]")

        return "\n".join(line for line in lines if line).strip()

    @classmethod
    def build_recursion_limit_fallback_text(cls, error_text: str, evidence: Mapping[str, Any]) -> str:
        evidence_text = cls.build_evidence_reflected_final_text(evidence)
        prefix = (
            "ワークフローが再帰上限に到達したため、追加のツール実行は停止しました。"
            "既に取得済みの結果だけを返します。"
        )
        if evidence_text:
            return prefix + "\n" + evidence_text
        return (
            "ERROR: MCPワークフローが再帰上限に到達したため停止しました。\n"
            f"- error: {error_text}"
        )

    @classmethod
    def contains_tool_budget_exceeded_signal(cls, text: str | None) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False

        markers = (
            "tool_call_budget_exceeded",
            "tool call budget exceeded",
            "ツール実行の制限",
            "ツール呼び出し回数の上限",
            "既に取得済みの結果だけで回答を完了してください",
        )
        return any(marker in normalized for marker in markers)

    @classmethod
    def build_budget_exhausted_completion_directive(cls, prior_text: str) -> str:
        return (
            "ツール呼び出し予算に到達しました。これ以上ツールを呼び出すことはできません。\n"
            "追加のツール実行、同一ツールの再試行、追加の再委譲は行わないでください。\n"
            "このスレッドで既に取得済みのツール結果だけを使って、回答できる部分をまとめてください。\n"
            "不足している情報があれば、その不足点だけを短く明記してください。\n"
            "必ず <RESPONSE_TYPE>complete</RESPONSE_TYPE> を返してください。\n"
            f"直前の応答: {prior_text}"
        )

    @classmethod
    async def force_graceful_completion_after_budget_exhaustion(
        cls,
        *,
        app: Any,
        run_trace_id: str,
        recursion_limit: int,
        user_text: str,
    ) -> tuple[str, str | None, str | None, str | None, int, int]:
        result = await app.ainvoke(
            {
                "messages": [
                    HumanMessage(content=cls.build_budget_exhausted_completion_directive(user_text))
                ]
            },
            config={"configurable": {"thread_id": run_trace_id}, "recursion_limit": recursion_limit},
        )
        output_text, add_in, add_out = cls._extract_output_and_usage(result)
        resp_type, extracted_text, hitl_kind, hitl_tool = cls._parse_supervisor_xml(output_text)
        final_text = extracted_text or output_text

        if resp_type != "complete":
            logger.warning(
                "MCP supervisor did not complete after budget exhaustion; returning controlled fallback: trace_id=%s resp_type=%s",
                run_trace_id,
                resp_type,
            )
            resp_type = "complete"
            if not final_text.strip():
                final_text = (
                    "ツール呼び出し回数の上限に到達したため、追加の調査は行わずに処理を終了しました。\n"
                    "既に取得済みの結果がある場合は、その結果のみを信頼してください。"
                )

        return final_text, resp_type, hitl_kind, hitl_tool, add_in, add_out

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
        """Backward-compatible wrapper for tool execution guards.

        The implementation lives in ToolLimits to keep agent-related guard logic
        co-located. Some callers/tests still reference MCPClientUtil.
        """

        ToolLimits._apply_tool_execution_guards(
            allowed_langchain_tools,
            tool_call_state=tool_call_state,
            tool_call_limit_int=tool_call_limit_int,
            tool_timeout_seconds_f=tool_timeout_seconds_f,
            tool_timeout_retries_int=tool_timeout_retries_int,
        )

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
    async def get_allowed_tools(cls, input_config: MCPServerConfig | None) -> MCPServerConfig | None:
        if input_config is None:
            return None
        
        return input_config.get_allowed_tools_config()

        allowed_tools = []
    
        allowed_map = input_config.get_allowed_tools_config()
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
                allowed_tools.append(cls._maybe_wrap_req_nested_tool(tool))
            else:
                logger.debug("Tool %s is not in allowedTools; skipped", tool_name)
        
        logger.info("Loaded %d tools from MCP servers.", len(allowed_tools))
        return allowed_tools

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
    def extract_approval_required_tool_name(cls, text: str | None) -> str | None:
        t = (text or "").strip()
        if not t:
            return None

        patterns = (
            r"error=tool_approval_required\s+tool=([A-Za-z0-9_\-:.]+)",
            r"tool=([A-Za-z0-9_\-:.]+)\s+.*error=tool_approval_required",
            r"APPROVE\s+([A-Za-z0-9_\-:.]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, t, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip().rstrip(".,);:]>}")
        if "tool_approval_required" in t or "承認が必要" in t:
            return None
        return None

    @classmethod
    def detect_approval_required_from_evidence(cls, evidence: Mapping[str, Any]) -> str | None:
        text_candidates: list[str] = []
        for key in ("raw_texts", "stdout_blocks"):
            values = evidence.get(key)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                text_candidates.extend(
                    str(value).strip()
                    for value in values
                    if isinstance(value, str) and str(value).strip()
                )

        for text in text_candidates:
            tool_name = cls.extract_approval_required_tool_name(text)
            if tool_name is not None:
                return tool_name
        return None

    @staticmethod
    def build_tool_approval_request_text(tool_name: str | None) -> str:
        normalized = str(tool_name or "").strip()
        if normalized:
            return (
                f"ツール {normalized} の実行には承認が必要です。\n"
                f"続行する場合は 'APPROVE {normalized}'、拒否する場合は 'REJECT {normalized}' と入力してください。"
            )
        return "このツールの実行には承認が必要です。続行する場合は 'APPROVE TOOL_NAME'、拒否する場合は 'REJECT TOOL_NAME' と入力してください。"

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
    def create_llm(cls, runtime_config: AiChatUtilConfig) -> BaseChatModel:
        litellm_router = Router(model_list=runtime_config.llm.create_litellm_model_list())
        llm = ChatLiteLLMRouter(router=litellm_router, model_name=runtime_config.llm.completion_model)
        return llm

    @classmethod
    async def _resolve_deep_agent_tools(
        cls,
        *,
        runtime_config: AiChatUtilConfig,
        tool_limits: ToolLimits | None,
        audit_context: AuditContext | None,
        explicit_user_file_paths: Sequence[str] | None = None,
        explicit_user_directory_paths: Sequence[str] | None = None,
    ) -> list[Any]:
        agent_client = MultiServerMCPClient(runtime_config.get_mcp_server_config().to_langchain_config())
        deep_agent_tools = [
            cls._maybe_wrap_req_nested_tool(tool)
            for tool in await agent_client.get_tools()
            if cls._is_deep_agent_tool_allowed(str(getattr(tool, "name", "")))
        ]

        if tool_limits is not None:
            tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = tool_limits.guard_params()
            followup_limit = int(tool_limits.followup_tool_call_limit)
        else:
            tool_call_limit_int, tool_timeout_seconds_f, tool_timeout_retries_int = (0, 0.0, 0)
            followup_limit = 0

        effective_tool_call_limit_int, effective_followup_tool_call_limit_int = ToolLimits.effective_call_limits(
            tool_call_limit_int,
            followup_limit,
            explicit_user_file_paths,
            explicit_user_directory_paths,
        )
        configured_hitl_approval_tools = [
            str(name).strip()
            for name in (runtime_config.features.hitl_approval_tools or [])
            if isinstance(name, str) and str(name).strip()
        ]
        tool_call_state: dict[str, Any] = {
            "used": 0,
            "general_used": 0,
            "followup_used": 0,
            "followup_limit": effective_followup_tool_call_limit_int,
            "agent_name": "deep_agent",
            "audit_context": audit_context,
            "approval_tools": set(configured_hitl_approval_tools),
            "explicit_user_file_paths": [
                str(path).strip()
                for path in (explicit_user_file_paths or [])
                if isinstance(path, str) and str(path).strip()
            ],
            "explicit_user_directory_paths": [
                str(path).strip()
                for path in (explicit_user_directory_paths or [])
                if isinstance(path, str) and str(path).strip()
            ],
        }
        ToolLimits._apply_tool_execution_guards(
            deep_agent_tools,
            tool_call_state=tool_call_state,
            tool_call_limit_int=effective_tool_call_limit_int,
            tool_timeout_seconds_f=tool_timeout_seconds_f,
            tool_timeout_retries_int=tool_timeout_retries_int,
        )
        return deep_agent_tools

    @classmethod
    async def create_deep_agent_workflow(
        cls,
        runtime_config: AiChatUtilConfig,
        *,
        checkpointer: Any | None = None,
        tool_limits: ToolLimits | None = None,
        explicit_user_file_paths: Sequence[str] | None = None,
        explicit_user_directory_paths: Sequence[str] | None = None,
        audit_context: AuditContext | None = None,
    ) -> tuple[CompiledStateGraph, list[str]]:
        create_deep_agent = require_create_deep_agent()
        llm = cls.create_llm(runtime_config)
        deep_agent_tools = await cls._resolve_deep_agent_tools(
            runtime_config=runtime_config,
            tool_limits=tool_limits,
            audit_context=audit_context,
            explicit_user_file_paths=explicit_user_file_paths,
            explicit_user_directory_paths=explicit_user_directory_paths,
        )
        graph = create_deep_agent(
            model=llm,
            tools=deep_agent_tools,
            system_prompt=build_deep_agent_system_prompt(
                runtime_config.mcp.working_directory,
                explicit_user_file_paths=explicit_user_file_paths,
                explicit_user_directory_paths=explicit_user_directory_paths,
            ),
            checkpointer=checkpointer,
            name="mcp_deep_agent",
        )
        tool_names = [str(getattr(tool, "name", "")).strip() for tool in deep_agent_tools if str(getattr(tool, "name", "")).strip()]
        return graph, tool_names
    
    @classmethod
    async def create_workflow(
        cls,
        runtime_config: AiChatUtilConfig ,
        prompts: PromptsBase,
        *,
        checkpointer: Any | None = None,
        tool_limits: ToolLimits | None = None,
        force_coding_agent_route: bool = False,
        force_deep_agent_route: bool = False,
        explicit_user_file_paths: Sequence[str] | None = None,
        explicit_user_directory_paths: Sequence[str] | None = None,
        approved_tool_names: Sequence[str] | None = None,
        routing_decision: RoutingDecision | None = None,
        audit_context: AuditContext | None = None,
        expects_heading_response: bool = False,
        expects_evaluation_response: bool = False,
    ) -> CompiledStateGraph:

        # LLM + MCP ツールでエージェントを作成
        llm = cls.create_llm(runtime_config)
        mcp_config = runtime_config.get_mcp_server_config()

        if routing_decision is not None and routing_decision.selected_route == "deep_agent":
            graph, deep_agent_tool_names = await cls.create_deep_agent_workflow(
                runtime_config,
                checkpointer=checkpointer,
                tool_limits=tool_limits,
                explicit_user_file_paths=explicit_user_file_paths,
                explicit_user_directory_paths=explicit_user_directory_paths,
                audit_context=audit_context,
            )
            if audit_context is not None:
                audit_context.emit(
                    "tool_catalog_resolved",
                    route_name="deep_agent",
                    payload={
                        "tool_agent_names": ["deep_agent"],
                        "tool_catalog": [{"agent_name": "deep_agent", "tool_names": deep_agent_tool_names}],
                    },
                )
            return graph

        # ツール実行用のエージェント
        # システムプロンプトで役割分担を指示する例。実際のプロンプトは用途に応じて調整してください。
        sub_agents = await AgentBuilder.create_sub_agents(
            runtime_config,
            mcp_config,
            llm, prompts, tool_limits, 
            include_coding_agent=(
                routing_decision.selected_route != "general_tool_agent"
                if routing_decision is not None
                else True
            ),
            include_general_agent=(
                routing_decision.selected_route != "coding_agent"
                if routing_decision is not None
                else cls.should_include_general_agent(
                    force_coding_agent_route=force_coding_agent_route,
                    explicit_user_file_paths=explicit_user_file_paths,
                )
            ),
            general_tool_allowlist=(
                cls._GENERAL_TOOLS_ALLOWED_WITH_EXPLICIT_CODING_AGENT
                if force_coding_agent_route or (routing_decision is not None and routing_decision.selected_route == "coding_agent")
                else None
            ),
            explicit_user_file_paths=explicit_user_file_paths,
            approved_tool_names=approved_tool_names,
            audit_context=audit_context,
            )


        approval_tool_names = AgentBuilder.merge_approval_tool_names(
            *[agent.get_hitl_approval_tools() for agent in sub_agents]
        )
        approval_tools_text = ", ".join(approval_tool_names) if approval_tool_names else "(なし)"
        tools_description = AgentBuilder.get_tools_description_all(sub_agents)
        logger.info("Allowed tools:\n%s", tools_description)
        tool_catalog_payload = {
            "tool_agent_names": [agent.get_agent_name() for agent in sub_agents],
            "tool_catalog": [
                {
                    "agent_name": agent.get_agent_name(),
                    "tool_names": [str(getattr(tool, "name", "")) for tool in agent.get_tools()],
                }
                for agent in sub_agents
            ],
        }
        logger.info(
            "Resolved tool catalog: route=%s catalog=%s",
            (routing_decision.selected_route if routing_decision is not None else "(unspecified)"),
            json.dumps(tool_catalog_payload, ensure_ascii=False),
        )
        if audit_context is not None:
            audit_context.emit(
                "tool_catalog_resolved",
                route_name=(routing_decision.selected_route if routing_decision is not None else None),
                payload=tool_catalog_payload,
            )

        if tool_limits is not None and tool_limits.auto_approve:
            supervisor_hitl_policy_text = prompts.supervisor_hitl_policy_text(approval_tools_text)
        else:
            supervisor_hitl_policy_text = prompts.supervisor_normal_hitl_policy_text(approval_tools_text)

        supervisor_prompt = prompts.supervisor_system_prompt(
            tools_description,
            supervisor_hitl_policy_text,
            tool_agent_names=[agent.get_agent_name() for agent in sub_agents],
            routing_guidance_text=cls._build_supervisor_routing_guidance_text(
                routing_decision=routing_decision,
                force_coding_agent_route=force_coding_agent_route,
                explicit_user_file_paths=explicit_user_file_paths,
                explicit_user_directory_paths=explicit_user_directory_paths,
                expects_heading_response=expects_heading_response,
                expects_evaluation_response=expects_evaluation_response,
            ),
        )

        # Prefer tool execution agent first to reduce accidental planner-only loops.
        workflow = create_supervisor(
            [agent.get_agent() for agent in sub_agents],
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

    @classmethod
    def _build_default_routing_decision(
        cls,
        *,
        runtime_config: AiChatUtilConfig,
        force_coding_agent_route: bool,
        force_deep_agent_route: bool,
        deep_agent_enabled: bool,
        explicit_user_file_paths: Sequence[str] | None = None,
        explicit_user_directory_paths: Sequence[str] | None = None,
        available_tool_names: Sequence[str] | None = None,
        workflow_file_path: str | None = None,
        predictability: str | None = None,
        approval_frequency: str | None = None,
        exploration_level: str | None = None,
        has_side_effects: bool | None = None,
    ) -> RoutingDecision:
        normalized_tools = [
            str(tool_name).strip()
            for tool_name in (available_tool_names or [])
            if isinstance(tool_name, str) and str(tool_name).strip()
        ]
        has_coding_agent_tools = any(name in {"execute", "status", "get_result", "cancel", "workspace_path"} for name in normalized_tools)
        normalized_paths = [
            str(path).strip()
            for path in (explicit_user_file_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        normalized_directories = [
            str(path).strip()
            for path in (explicit_user_directory_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        normalized_workflow_file_path = str(workflow_file_path or "").strip() or None
        normalized_predictability = str(predictability or "").strip().lower() or None
        normalized_approval_frequency = str(approval_frequency or "").strip().lower() or None
        normalized_exploration_level = str(exploration_level or "").strip().lower() or None
        features = runtime_config.features
        workflow_requires_definition = bool(getattr(features, "type_selection_workflow_requires_definition", True))
        workflow_eligible = bool(normalized_workflow_file_path) or not workflow_requires_definition

        if force_deep_agent_route and deep_agent_enabled:
            candidate = RouteCandidate(
                route_name="deep_agent",
                score=1.0,
                reason_code="route.multi_step_investigation_needed",
                tool_hints=[name for name in normalized_tools if cls._is_deep_agent_tool_allowed(name)],
            )
            return RoutingDecision(
                selected_route="deep_agent",
                candidate_routes=[candidate],
                reason_code="route.multi_step_investigation_needed",
                confidence=1.0,
                next_action="execute_selected_route",
                notes="explicit deep-agent request detected",
            )

        if force_coding_agent_route and has_coding_agent_tools:
            candidate = RouteCandidate(
                route_name="coding_agent",
                score=1.0,
                reason_code="route.explicit_coding_agent_request",
                tool_hints=[name for name in normalized_tools if name in {"execute", "status", "get_result"}],
            )
            return RoutingDecision(
                selected_route="coding_agent",
                candidate_routes=[candidate],
                reason_code="route.explicit_coding_agent_request",
                confidence=1.0,
                next_action="execute_selected_route",
                notes="explicit coding-agent request detected",
            )

        workflow_reasons: list[str] = []
        workflow_score = 0.0
        if workflow_eligible and normalized_workflow_file_path and bool(getattr(features, "type_selection_prefer_workflow_when_definition_available", True)):
            workflow_score = max(workflow_score, 0.95)
            workflow_reasons.append("workflow_definition_available")
        if workflow_eligible and normalized_predictability == "high" and bool(getattr(features, "type_selection_workflow_on_high_predictability", True)):
            workflow_score = max(workflow_score, 0.9)
            workflow_reasons.append("high_predictability")
        if workflow_eligible and normalized_approval_frequency == "high" and bool(getattr(features, "type_selection_workflow_on_high_approval_frequency", True)):
            workflow_score = max(workflow_score, 0.88)
            workflow_reasons.append("high_approval_frequency")
        if workflow_eligible and has_side_effects is True and bool(getattr(features, "type_selection_workflow_on_side_effects", True)):
            workflow_score = max(workflow_score, 0.92)
            workflow_reasons.append("side_effects_present")

        if workflow_score > 0.0 and normalized_workflow_file_path:
            candidate = RouteCandidate(
                route_name="workflow_backend",
                score=workflow_score,
                reason_code="route.workflow_definition_available",
                tool_hints=[normalized_workflow_file_path],
            )
            return RoutingDecision(
                selected_route="workflow_backend",
                candidate_routes=[candidate],
                reason_code="route.workflow_definition_available",
                confidence=workflow_score,
                next_action="execute_selected_route",
                notes="workflow backend selected: " + ", ".join(workflow_reasons),
            )

        if (
            workflow_score <= 0.0
            and workflow_requires_definition
            and bool(getattr(features, "type_selection_require_clarification_on_missing_workflow_definition", True))
            and (
                normalized_predictability == "high"
                or normalized_approval_frequency == "high"
                or has_side_effects is True
            )
        ):
            candidate = RouteCandidate(
                route_name="workflow_backend",
                score=0.65,
                reason_code="route.workflow_definition_missing",
                blocking_issues=["workflow_definition_required"],
            )
            notes: list[str] = ["workflow backend needs workflow_file_path"]
            if normalized_exploration_level:
                notes.append(f"exploration_level={normalized_exploration_level}")
            return RoutingDecision(
                selected_route="workflow_backend",
                candidate_routes=[candidate],
                reason_code="route.workflow_definition_missing",
                confidence=0.65,
                next_action="ask_user",
                requires_hitl=True,
                requires_clarification=True,
                missing_information=["workflow backend を使うには workflow_file_path が必要です。workflow_file_path を指定するか、通常の agent routing で続行するか指定してください。"],
                notes="; ".join(notes),
            )

        if normalized_paths:
            candidate = RouteCandidate(
                route_name="general_tool_agent",
                score=0.8,
                reason_code="route.explicit_file_path_request",
                tool_hints=normalized_tools[:5],
            )
            return RoutingDecision(
                selected_route="general_tool_agent",
                candidate_routes=[candidate],
                reason_code="route.explicit_file_path_request",
                confidence=0.8,
                next_action="execute_selected_route",
                notes="explicit file path detected",
            )

        if normalized_directories:
            candidate = RouteCandidate(
                route_name="general_tool_agent",
                score=0.78,
                reason_code="route.explicit_directory_path_request",
                tool_hints=normalized_tools[:5],
            )
            return RoutingDecision(
                selected_route="general_tool_agent",
                candidate_routes=[candidate],
                reason_code="route.explicit_directory_path_request",
                confidence=0.78,
                next_action="execute_selected_route",
                notes="explicit directory path detected",
            )

        candidate = RouteCandidate(
            route_name="general_tool_agent",
            score=0.6,
            reason_code="route.general_tool_sufficient",
            tool_hints=normalized_tools[:5],
        )
        return RoutingDecision(
            selected_route="general_tool_agent",
            candidate_routes=[candidate],
            reason_code="route.general_tool_sufficient",
            confidence=0.6,
            next_action="execute_selected_route",
            notes="default route selected",
        )

    @classmethod
    def _build_available_routes_text(
        cls,
        *,
        has_coding_agent: bool,
        has_deep_agent: bool,
        has_general_agent: bool,
        has_workflow_backend: bool = False,
        workflow_file_path: str | None = None,
        route_tool_catalog: Mapping[str, Sequence[str]] | None = None,
    ) -> str:
        lines: list[str] = []
        coding_tools = [
            str(name).strip()
            for name in cast(Sequence[Any], (route_tool_catalog or {}).get("coding_agent") or [])
            if isinstance(name, str) and str(name).strip()
        ]
        general_tools = [
            str(name).strip()
            for name in cast(Sequence[Any], (route_tool_catalog or {}).get("general_tool_agent") or [])
            if isinstance(name, str) and str(name).strip()
        ]
        deep_tools = [
            str(name).strip()
            for name in cast(Sequence[Any], (route_tool_catalog or {}).get("deep_agent") or [])
            if isinstance(name, str) and str(name).strip()
        ]
        if has_coding_agent:
            if coding_tools:
                lines.append(
                    "- coding_agent: execute/status/get_result を使う複数ステップ調査向け"
                    f" (visible_tools: {', '.join(coding_tools)})"
                )
            else:
                lines.append("- coding_agent: execute/status/get_result を使う複数ステップ調査向け")
        if has_deep_agent:
            if deep_tools:
                lines.append(
                    "- deep_agent: 深い分解や複数ステップ調査向け。非同期ジョブ系ツールを使わずに完結する経路"
                    f" (visible_tools: {', '.join(deep_tools)})"
                )
            else:
                lines.append("- deep_agent: 深い分解や複数ステップ調査向け。非同期ジョブ系ツールを使わずに完結する経路")
        if has_general_agent:
            if general_tools:
                lines.append(
                    "- general_tool_agent: 一般 MCP ツールで完結する設定確認・単発調査向け"
                    f" (visible_tools: {', '.join(general_tools)})"
                )
            else:
                lines.append("- general_tool_agent: 一般 MCP ツールで完結する設定確認・単発調査向け")
        if has_workflow_backend:
            if isinstance(workflow_file_path, str) and workflow_file_path.strip():
                lines.append(
                    "- workflow_backend: 定義済み workflow をそのまま実行する経路"
                    f" (workflow_file: {workflow_file_path.strip()})"
                )
            else:
                lines.append("- workflow_backend: 定義済み workflow をそのまま実行する経路")
        lines.append("- direct_answer: ツール不要で即答できる場合のみ")
        lines.append("- reject: サポート範囲外または route を決めても安全に進めない場合")
        return "\n".join(lines)

    @classmethod
    def _build_routing_context_text(
        cls,
        *,
        force_coding_agent_route: bool,
        force_deep_agent_route: bool,
        explicit_user_file_paths: Sequence[str] | None,
        explicit_user_directory_paths: Sequence[str] | None,
        routing_mode: str,
        preferred_coding_route: str,
        workflow_file_path: str | None = None,
        predictability: str | None = None,
        approval_frequency: str | None = None,
        exploration_level: str | None = None,
        has_side_effects: bool | None = None,
        route_tool_catalog: Mapping[str, Sequence[str]] | None = None,
    ) -> str:
        normalized_paths = [
            str(path).strip()
            for path in (explicit_user_file_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        normalized_directories = [
            str(path).strip()
            for path in (explicit_user_directory_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        catalog_lines = [
            f"{route_name}_tools=" + (", ".join(str(name).strip() for name in tool_names if str(name).strip()) or "(none)")
            for route_name, tool_names in (route_tool_catalog or {}).items()
        ]
        return "\n".join(
            [
                f"routing_mode={routing_mode}",
                f"force_coding_agent_route={str(force_coding_agent_route).lower()}",
                f"force_deep_agent_route={str(force_deep_agent_route).lower()}",
                f"preferred_coding_route={preferred_coding_route}",
                f"workflow_file_path={str(workflow_file_path or '').strip() or '(none)'}",
                f"predictability={str(predictability or '').strip().lower() or '(none)'}",
                f"approval_frequency={str(approval_frequency or '').strip().lower() or '(none)'}",
                f"exploration_level={str(exploration_level or '').strip().lower() or '(none)'}",
                f"has_side_effects={str(has_side_effects).lower() if has_side_effects is not None else '(none)'}",
                "explicit_user_file_paths=" + (", ".join(normalized_paths) if normalized_paths else "(none)"),
                "explicit_user_directory_paths=" + (", ".join(normalized_directories) if normalized_directories else "(none)"),
                *catalog_lines,
            ]
        )

    @classmethod
    async def resolve_route_tool_inventory(
        cls,
        *,
        runtime_config: AiChatUtilConfig,
    ) -> dict[str, list[dict[str, Any]]]:
        mcp_config = runtime_config.get_mcp_server_config()
        coding_agent_server_name = cls.get_coding_agent_server_name(runtime_config)
        route_configs = {
            "coding_agent": mcp_config.filter(include_name=coding_agent_server_name),
            "general_tool_agent": mcp_config.filter(exclude_name=coding_agent_server_name),
        }

        route_tool_inventory: dict[str, list[dict[str, Any]]] = {}
        for route_name, route_config in route_configs.items():
            if len(route_config.servers) == 0:
                continue
            try:
                client = MultiServerMCPClient(route_config.to_langchain_config())
                tools = await client.get_tools()
                route_tool_inventory[route_name] = [
                    {
                        "name": str(getattr(tool, "name", "")).strip(),
                        "description": cls._normalize_tool_description(getattr(tool, "description", "")),
                        "primary_args": cls._extract_primary_arg_names(getattr(tool, "args_schema", None)),
                        "tool_metadata": cls._extract_tool_metadata(getattr(tool, "description", "")),
                    }
                    for tool in tools
                    if str(getattr(tool, "name", "")).strip()
                ]
            except Exception:
                logger.debug("Failed to resolve route tool catalog for %s", route_name, exc_info=True)

        if cls.deep_agent_route_enabled(runtime_config):
            try:
                client = MultiServerMCPClient(mcp_config.to_langchain_config())
                tools = await client.get_tools()
                route_tool_inventory["deep_agent"] = [
                    {
                        "name": str(getattr(tool, "name", "")).strip(),
                        "description": cls._normalize_tool_description(getattr(tool, "description", "")),
                        "primary_args": cls._extract_primary_arg_names(getattr(tool, "args_schema", None)),
                        "tool_metadata": cls._extract_tool_metadata(getattr(tool, "description", "")),
                    }
                    for tool in tools
                    if str(getattr(tool, "name", "")).strip()
                    and cls._is_deep_agent_tool_allowed(str(getattr(tool, "name", "")))
                ]
            except Exception:
                logger.debug("Failed to resolve route tool catalog for deep_agent", exc_info=True)

        return route_tool_inventory

    @classmethod
    async def resolve_route_tool_catalog(
        cls,
        *,
        runtime_config: AiChatUtilConfig,
    ) -> dict[str, list[str]]:
        route_tool_inventory = await cls.resolve_route_tool_inventory(runtime_config=runtime_config)
        route_tool_catalog: dict[str, list[str]] = {}
        for route_name, tools in route_tool_inventory.items():
            route_tool_catalog[route_name] = [
                str(tool.get("name") or "").strip()
                for tool in tools
                if isinstance(tool, Mapping) and str(tool.get("name") or "").strip()
            ]

        return route_tool_catalog

    @classmethod
    def _build_supervisor_routing_guidance_text(
        cls,
        *,
        routing_decision: RoutingDecision | None,
        force_coding_agent_route: bool,
        explicit_user_file_paths: Sequence[str] | None,
        explicit_user_directory_paths: Sequence[str] | None = None,
        expects_heading_response: bool = False,
        expects_evaluation_response: bool = False,
    ) -> str | None:
        normalized_paths = [
            str(path).strip()
            for path in (explicit_user_file_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        normalized_directories = [
            str(path).strip()
            for path in (explicit_user_directory_paths or [])
            if isinstance(path, str) and str(path).strip()
        ]
        decision = routing_decision
        if decision is None and not force_coding_agent_route and not normalized_paths and not normalized_directories:
            return None

        lines: list[str] = []
        selected_route = decision.selected_route if decision is not None else "coding_agent"
        if selected_route == "coding_agent":
            lines.append("- 初手は coding_agent ルートを優先してください。general_tool_agent へ広げるのは `get_loaded_config_info` のような事前確認が必要な場合だけです。")
            lines.append("- `get_loaded_config_info` を使う場合でも 1 回だけ実行し、取得した path / config を以後の委譲で再利用してください。")
            lines.append("- 設定確認が済んだら、以降は coding-agent 系の execute/status/get_result に進み、同じ設定確認を繰り返さないでください。")
        elif selected_route == "deep_agent":
            lines.append("- 初手は deep_agent ルートを優先してください。非同期ジョブ系の execute/status/get_result は使わず、利用可能なファイル系/MCP ツールだけで完結してください。")
            lines.append("- deep_agent で必要情報を取得できた後は、同じツールを同じ引数で繰り返し呼ばないでください。")
            lines.append("- deep_agent で不足する場合のみ clarification を返し、coding_agent への安易な切り替えは行わないでください。")
        elif selected_route == "general_tool_agent":
            lines.append("- 初手は general_tool_agent で完結するかを優先確認してください。")
            lines.append("- general_tool_agent で必要なツールが見えている場合、coding-agent 系の execute/status/get_result へ切り替えないでください。")
            lines.append("- 一般ツールで必要情報を取得できた後は、同じツールを同じ引数で繰り返し呼ばないでください。")
        elif selected_route == "direct_answer":
            lines.append("- 現時点ではツール不要の即答を優先してください。不足が見つかった場合のみ route を見直してください。")
        elif selected_route == "reject":
            lines.append("- サポート範囲外または安全に進めない要求として扱ってください。")

        if expects_evaluation_response and not expects_heading_response:
            lines.append("- この run は結果評価/投入判断系です。見出し抽出へ逸れず、判断可能な点・不足情報・追加確認事項を整理して complete で返してください。")
            lines.append("- get_loaded_config_info と文書解析系ツールは、同一目的なら成功結果を再利用してください。analyze_files 系の再実行を繰り返さないでください。")

        if normalized_paths:
            lines.append(f"- ユーザーが明示した対象パス: {', '.join(normalized_paths)}")
        if normalized_directories:
            lines.append(f"- ユーザーが明示した対象ディレクトリ: {', '.join(normalized_directories)}")
        if decision is not None and decision.reason_code:
            lines.append(f"- route_reason_code: {decision.reason_code}")
        if decision is not None and decision.notes:
            lines.append(f"- route_notes: {decision.notes}")

        return "\n".join(lines) if lines else None

    @classmethod
    def _extract_user_request_text(cls, messages: Sequence[Any]) -> str:
        chunks: list[str] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                text = cls._stringify_message_content(message.content)
                if text.strip():
                    chunks.append(text.strip())
                continue
            if isinstance(message, Mapping):
                role = str(message.get("role") or "").lower()
                if role in {"user", "human"}:
                    text = cls._stringify_message_content(message.get("content"))
                    if text.strip():
                        chunks.append(text.strip())
        return "\n\n".join(chunks)

    @classmethod
    def _coerce_route_candidate(cls, value: Any) -> RouteCandidate | None:
        if not isinstance(value, Mapping):
            return None
        route_name = str(value.get("route_name") or "").strip()
        reason_code = cls.normalize_route_reason_code(
            str(value.get("reason_code") or "route.general_tool_sufficient").strip(),
            route_name=route_name,
        )
        if not route_name:
            return None
        try:
            score = float(value.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        tool_hints = [str(item).strip() for item in cast(Sequence[Any], value.get("tool_hints") or []) if str(item).strip()]
        blocking_issues = [str(item).strip() for item in cast(Sequence[Any], value.get("blocking_issues") or []) if str(item).strip()]
        return RouteCandidate(
            route_name=route_name,
            score=score,
            reason_code=reason_code,
            tool_hints=tool_hints,
            blocking_issues=blocking_issues,
        )

    @classmethod
    def normalize_route_reason_code(cls, reason_code: str | None, *, route_name: str | None = None) -> str:
        normalized = str(reason_code or "").strip()
        if normalized in cls._ALLOWED_ROUTE_REASON_CODES:
            return normalized

        route = str(route_name or "").strip().lower()
        if route == "coding_agent":
            return "route.multi_step_investigation_needed"
        if route == "deep_agent":
            return "route.multi_step_investigation_needed"
        if route == "workflow_backend":
            return "route.workflow_definition_available"
        if route == "general_tool_agent":
            return "route.general_tool_sufficient"
        if route == "direct_answer":
            return "route.direct_answer_possible"
        if route == "reject":
            return "route.reject_unsupported_request"
        return "route.low_confidence_route"

    @classmethod
    def _parse_routing_decision_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        confidence_threshold: float,
    ) -> RoutingDecision | None:
        selected_route = str(payload.get("selected_route") or "").strip()
        if not selected_route:
            return None
        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason_code = cls.normalize_route_reason_code(
            str(payload.get("reason_code") or "route.low_confidence_route").strip() or "route.low_confidence_route",
            route_name=selected_route,
        )
        missing_information = [
            str(item).strip()
            for item in cast(Sequence[Any], payload.get("missing_information") or [])
            if str(item).strip()
        ]
        candidate_routes = [
            candidate
            for candidate in (
                cls._coerce_route_candidate(item)
                for item in cast(Sequence[Any], payload.get("candidate_routes") or [])
            )
            if candidate is not None
        ]
        next_action = str(payload.get("next_action") or "execute_selected_route").strip() or "execute_selected_route"
        requires_hitl = bool(payload.get("requires_hitl", False))
        requires_clarification = bool(payload.get("requires_clarification", False))
        notes = str(payload.get("notes") or "").strip() or None

        if confidence < confidence_threshold and next_action == "execute_selected_route":
            next_action = "ask_user"
            requires_clarification = True
            requires_hitl = True
            reason_code = "route.low_confidence_route"

        return RoutingDecision(
            selected_route=selected_route,
            candidate_routes=candidate_routes,
            reason_code=reason_code,
            confidence=confidence,
            missing_information=missing_information,
            next_action=next_action,
            requires_hitl=requires_hitl,
            requires_clarification=requires_clarification,
            notes=notes,
        )

    @classmethod
    async def decide_route(
        cls,
        *,
        runtime_config: AiChatUtilConfig,
        prompts: PromptsBase,
        messages: Sequence[Any],
        force_coding_agent_route: bool,
        force_deep_agent_route: bool,
        explicit_user_file_paths: Sequence[str] | None = None,
        explicit_user_directory_paths: Sequence[str] | None = None,
        available_tool_names: Sequence[str] | None = None,
        route_tool_catalog: Mapping[str, Sequence[str]] | None = None,
        audit_context: AuditContext | None = None,
        workflow_file_path: str | None = None,
        predictability: str | None = None,
        approval_frequency: str | None = None,
        exploration_level: str | None = None,
        has_side_effects: bool | None = None,
    ) -> RoutingDecision:
        default_decision = cls._build_default_routing_decision(
            runtime_config=runtime_config,
            force_coding_agent_route=force_coding_agent_route,
            force_deep_agent_route=force_deep_agent_route,
            deep_agent_enabled=cls.deep_agent_route_enabled(runtime_config),
            explicit_user_file_paths=explicit_user_file_paths,
            explicit_user_directory_paths=explicit_user_directory_paths,
            available_tool_names=available_tool_names,
            workflow_file_path=workflow_file_path,
            predictability=predictability,
            approval_frequency=approval_frequency,
            exploration_level=exploration_level,
            has_side_effects=has_side_effects,
        )

        routing_mode = str(getattr(runtime_config.features, "routing_mode", "legacy") or "legacy").strip().lower()
        if (
            routing_mode == "legacy"
            or force_deep_agent_route
            or default_decision.reason_code in {
                "route.explicit_coding_agent_request",
                "route.explicit_file_path_request",
                "route.explicit_directory_path_request",
            }
        ):
            return default_decision

        mcp_config = runtime_config.get_mcp_server_config()
        coding_agent_server_name = cls.get_coding_agent_server_name(runtime_config)
        has_coding_agent = len(mcp_config.filter(include_name=coding_agent_server_name).servers) > 0
        has_deep_agent = cls.deep_agent_route_enabled(runtime_config)
        has_general_agent = len(mcp_config.filter(exclude_name=coding_agent_server_name).servers) > 0
        if not has_coding_agent and not has_deep_agent and not has_general_agent:
            return default_decision

        llm = cls.create_llm(runtime_config)
        user_request_text = cls._extract_user_request_text(messages)
        available_routes_text = cls._build_available_routes_text(
            has_coding_agent=has_coding_agent,
            has_deep_agent=has_deep_agent,
            has_general_agent=has_general_agent,
            has_workflow_backend=bool(str(workflow_file_path or "").strip()),
            workflow_file_path=workflow_file_path,
            route_tool_catalog=route_tool_catalog,
        )
        context_text = cls._build_routing_context_text(
            force_coding_agent_route=force_coding_agent_route,
            force_deep_agent_route=force_deep_agent_route,
            explicit_user_file_paths=explicit_user_file_paths,
            explicit_user_directory_paths=explicit_user_directory_paths,
            routing_mode=routing_mode,
            preferred_coding_route=str(getattr(runtime_config.features, "preferred_coding_route", "coding_agent") or "coding_agent"),
            workflow_file_path=workflow_file_path,
            predictability=predictability,
            approval_frequency=approval_frequency,
            exploration_level=exploration_level,
            has_side_effects=has_side_effects,
            route_tool_catalog=route_tool_catalog,
        )

        try:
            result = await llm.ainvoke(
                [
                    SystemMessage(content=prompts.routing_system_prompt()),
                    HumanMessage(
                        content=prompts.routing_user_prompt(
                            user_request_text=user_request_text,
                            available_routes_text=available_routes_text,
                            context_text=context_text,
                        )
                    ),
                ]
            )
            output_text = cls._stringify_message_content(getattr(result, "content", result))
            payload = cls._extract_mapping_from_text(output_text)
            confidence_threshold = float(getattr(runtime_config.features, "routing_confidence_threshold", 0.6) or 0.6)
            if payload is not None:
                parsed_decision = cls._parse_routing_decision_payload(
                    payload,
                    confidence_threshold=confidence_threshold,
                )
                if parsed_decision is not None:
                    if parsed_decision.selected_route == "workflow_backend" and not str(workflow_file_path or "").strip():
                        return RoutingDecision(
                            selected_route="workflow_backend",
                            candidate_routes=parsed_decision.candidate_routes,
                            reason_code="route.workflow_definition_missing",
                            confidence=parsed_decision.confidence,
                            missing_information=["workflow backend を使うには workflow_file_path が必要です。workflow_file_path を指定するか、別 route で続行するか指定してください。"],
                            next_action="ask_user",
                            requires_hitl=True,
                            requires_clarification=True,
                            notes=parsed_decision.notes,
                        )
                    if audit_context is not None:
                        audit_context.emit(
                            "route_decision_model_output",
                            route_name=parsed_decision.selected_route,
                            reason_code=parsed_decision.reason_code,
                            confidence=parsed_decision.confidence,
                            payload={"raw_output": output_text},
                        )
                    return parsed_decision
        except Exception:
            logger.debug("Structured routing decision failed; falling back to default routing", exc_info=True)

        return default_decision

    @classmethod
    def build_evidence_summary(cls, evidence: Mapping[str, Any]) -> EvidenceSummary:
        headings = [
            str(value).strip()
            for value in cast(Sequence[Any], evidence.get("headings") or [])
            if isinstance(value, str) and str(value).strip()
        ]
        stdout_blocks = [
            str(value).strip()
            for value in cast(Sequence[Any], evidence.get("stdout_blocks") or [])
            if isinstance(value, str) and str(value).strip()
        ]
        raw_texts = [
            str(value).strip()
            for value in cast(Sequence[Any], evidence.get("raw_texts") or [])
            if isinstance(value, str) and str(value).strip()
        ]
        tool_errors = [
            text for text in raw_texts if text.lstrip().startswith("ERROR:")
        ]
        heading_extraction_succeeded = bool(headings) and cls.expects_heading_response(evidence)
        successful_tools = [
            "get_loaded_config_info" if isinstance(evidence.get("config_path"), str) and str(evidence.get("config_path")).strip() else "",
            "heading_extraction" if heading_extraction_succeeded else "",
            "tool_catalog_resolved" if cast(Sequence[Any], evidence.get("tool_catalog") or []) else "",
        ]
        successful_tools = [tool_name for tool_name in successful_tools if tool_name]
        latest_task_id = evidence.get("latest_task_id")
        return EvidenceSummary(
            config_path=cast(str | None, evidence.get("config_path")),
            headings=headings,
            stdout_blocks=stdout_blocks,
            raw_texts=raw_texts,
            tool_errors=tool_errors,
            successful_tools=successful_tools,
            latest_task_id=latest_task_id if isinstance(latest_task_id, str) and latest_task_id.strip() else None,
            has_actionable_evidence=bool(headings or stdout_blocks or evidence.get("config_path")),
        )

    @staticmethod
    def _contains_absence_claim(response_text: str) -> bool:
        normalized = (response_text or "").strip().lower()
        if not normalized:
            return False
        markers = (
            "空です",
            "空でした",
            "空のディレクトリ",
            "見つかりませんでした",
            "見つからない",
            "存在しません",
            "存在しない",
            "ファイルがありません",
            "ファイルが存在しない",
            "ディレクトリは空",
            "not found",
            "no files found",
            "directory is empty",
            "empty directory",
            "does not exist",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _has_substantive_tool_evidence(evidence_summary: EvidenceSummary) -> bool:
        if evidence_summary.headings or evidence_summary.stdout_blocks:
            return True
        for text in evidence_summary.tool_errors:
            normalized = str(text).strip().lower()
            if not normalized:
                continue
            if any(
                marker in normalized
                for marker in (
                    "path not found",
                    "no supported analysis files found",
                    "file not found",
                    "directory",
                    "見つかりません",
                    "存在しません",
                    "空です",
                )
            ):
                return True
        for text in evidence_summary.raw_texts:
            normalized = str(text).strip()
            if not normalized:
                continue
            if normalized in {"[]", "{}", "null", "None"}:
                continue
            if normalized.startswith("ERROR:"):
                continue
            return True
        return False

    @classmethod
    def judge_sufficiency(
        cls,
        *,
        response_text: str,
        resp_type: str | None,
        hitl_kind: str | None,
        evidence_summary: EvidenceSummary,
    ) -> SufficiencyDecision:
        if resp_type == "question" and hitl_kind == "approval":
            return SufficiencyDecision(
                decision="needs_approval",
                reason_code="sufficiency.approval_required",
                confidence=1.0,
                missing_facts=[],
                recommended_next_action="request_approval",
                requires_hitl=True,
                requires_approval=True,
                evidence_summary=evidence_summary,
                user_prompt_hint=response_text or None,
            )
        if resp_type == "question":
            return SufficiencyDecision(
                decision="needs_user_input",
                reason_code="sufficiency.missing_user_context",
                confidence=0.9,
                missing_facts=[response_text] if response_text else [],
                recommended_next_action="ask_user",
                requires_hitl=True,
                requires_approval=False,
                evidence_summary=evidence_summary,
                user_prompt_hint=response_text or None,
            )
        if evidence_summary.tool_errors and not evidence_summary.has_actionable_evidence:
            return SufficiencyDecision(
                decision="needs_more_tool_calls",
                reason_code="sufficiency.tool_result_error_only",
                confidence=0.7,
                missing_facts=["successful tool evidence is unavailable"],
                recommended_next_action="call_more_tools",
                requires_hitl=False,
                requires_approval=False,
                evidence_summary=evidence_summary,
            )
        if cls._contains_absence_claim(response_text) and not cls._has_substantive_tool_evidence(evidence_summary):
            return SufficiencyDecision(
                decision="needs_more_tool_calls",
                reason_code="sufficiency.unverified_absence_claim",
                confidence=0.85,
                missing_facts=["absence claim is not backed by substantive tool evidence"],
                recommended_next_action="call_more_tools",
                requires_hitl=False,
                requires_approval=False,
                evidence_summary=evidence_summary,
            )
        if evidence_summary.has_actionable_evidence or (response_text or "").strip():
            return SufficiencyDecision(
                decision="answerable",
                reason_code="sufficiency.answer_supported_by_evidence",
                confidence=0.8,
                missing_facts=[],
                recommended_next_action="finalize_answer",
                requires_hitl=False,
                requires_approval=False,
                evidence_summary=evidence_summary,
            )
        return SufficiencyDecision(
            decision="needs_more_tool_calls",
            reason_code="sufficiency.insufficient_tool_evidence",
            confidence=0.6,
            missing_facts=["tool evidence is missing"],
            recommended_next_action="call_more_tools",
            requires_hitl=False,
            requires_approval=False,
            evidence_summary=evidence_summary,
        )
