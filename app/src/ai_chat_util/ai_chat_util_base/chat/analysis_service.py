from __future__ import annotations

import asyncio
from pathlib import Path
import time
from collections.abc import Awaitable, Callable

from ai_chat_util.ai_chat_util_base.chat import AbstractChatClient
from ai_chat_util.common.config.runtime import get_runtime_config
from ai_chat_util.ai_chat_util_base.ai_chatl_util_models import ChatResponse
from ai_chat_util.ai_chat_util_base.file_util.model import FileUtilDocument
from ai_chat_util.ai_chat_util_base.file_util.util.file_path_resolver import resolve_existing_path

import ai_chat_util.log.log_settings as log_settings

from .llm_client_util import LLMClientUtil


logger = log_settings.getLogger(__name__)


class AnalysisService:
    _MAX_DIRECTORY_ANALYSIS_FILES = 20
    _MAX_DIRECTORY_ANALYSIS_FILE_BYTES = 200_000
    _SUPPORTED_TEXT_SUFFIXES = {
        ".bat",
        ".c",
        ".cc",
        ".cfg",
        ".conf",
        ".cpp",
        ".cs",
        ".css",
        ".csv",
        ".go",
        ".h",
        ".hpp",
        ".htm",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".log",
        ".md",
        ".markdown",
        ".php",
        ".ps1",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsv",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
    _SUPPORTED_IMAGE_SUFFIXES = {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".webp",
    }
    _SUPPORTED_DOCUMENT_SUFFIXES = {
        ".docx",
        ".pdf",
        ".pptx",
        ".xlsx",
    }

    @staticmethod
    def summarize_path_basenames(paths: list[str], *, limit: int = 5) -> str:
        names: list[str] = []
        for path in paths[:limit]:
            try:
                names.append(path.rsplit("/", 1)[-1])
            except Exception:
                names.append(str(path))
        more = "" if len(paths) <= limit else f" (+{len(paths) - limit} more)"
        return f"[{', '.join(names)}]{more}"

    @staticmethod
    def prompt_len(prompt: str) -> int:
        return len((prompt or "").strip())

    @classmethod
    def _is_supported_analysis_file(cls, path: Path) -> bool:
        suffix = path.suffix.lower()
        return (
            suffix in cls._SUPPORTED_TEXT_SUFFIXES
            or suffix in cls._SUPPORTED_IMAGE_SUFFIXES
            or suffix in cls._SUPPORTED_DOCUMENT_SUFFIXES
        )

    @staticmethod
    def _is_hidden_path(path: Path) -> bool:
        return any(part.startswith(".") for part in path.parts)

    @classmethod
    def _expand_directory_analysis_targets(cls, directory_path: str) -> list[str]:
        directory = Path(directory_path)
        resolved_files: list[str] = []

        for candidate in sorted(directory.rglob("*")):
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue

            if cls._is_hidden_path(candidate.relative_to(directory)):
                continue

            if not cls._is_supported_analysis_file(candidate):
                continue

            try:
                if candidate.stat().st_size > cls._MAX_DIRECTORY_ANALYSIS_FILE_BYTES:
                    continue
            except OSError:
                continue

            resolved_files.append(str(candidate.resolve()))
            if len(resolved_files) >= cls._MAX_DIRECTORY_ANALYSIS_FILES:
                break

        if resolved_files:
            return resolved_files

        raise FileNotFoundError(f"No supported analysis files found in directory: {directory.resolve()}")

    @classmethod
    def resolve_existing_file_paths(cls, file_path_list: list[str]) -> list[str]:
        runtime_config = get_runtime_config()
        resolved: list[str] = []
        seen: set[str] = set()
        for file_path in file_path_list:
            result = resolve_existing_path(
                file_path,
                working_directory=runtime_config.mcp.working_directory,
                allow_directory=True,
            )

            candidate_paths = [result.resolved_path]
            if result.path_kind == "directory":
                candidate_paths = cls._expand_directory_analysis_targets(result.resolved_path)

            for candidate_path in candidate_paths:
                if candidate_path in seen:
                    continue
                seen.add(candidate_path)
                resolved.append(candidate_path)
        return resolved

    @staticmethod
    def tool_timeout_seconds() -> float:
        runtime_config = get_runtime_config()
        tool_timeout_cfg = getattr(runtime_config.features, "mcp_tool_timeout_seconds", None)
        try:
            timeout = (
                float(tool_timeout_cfg)
                if tool_timeout_cfg is not None
                else float(runtime_config.llm.timeout_seconds)
            )
        except (TypeError, ValueError):
            timeout = float(runtime_config.llm.timeout_seconds)
        if timeout <= 0:
            timeout = float(runtime_config.llm.timeout_seconds)
        return timeout

    @staticmethod
    def tool_timeout_retries() -> int:
        runtime_config = get_runtime_config()
        try:
            retries_raw = int(getattr(runtime_config.features, "mcp_tool_timeout_retries", 1) or 0)
        except (TypeError, ValueError):
            retries_raw = 1
        return max(0, min(5, retries_raw))

    @classmethod
    async def run_analysis_tool(
        cls,
        *,
        tool_name: str,
        prompt: str,
        detail: str,
        input_count: int,
        input_kind: str,
        operation: Callable[[], Awaitable[ChatResponse]],
        input_summary: str | None = None,
        use_timeout_retry: bool = False,
        stringify_errors: bool = False,
    ) -> str:
        started = time.perf_counter()
        logger.info(
            "MCP_TOOL_START tool=%s %s=%d detail=%s prompt_len=%d",
            tool_name,
            input_kind,
            input_count,
            detail,
            cls.prompt_len(prompt),
        )
        if input_summary is not None:
            logger.info(
                "MCP_TOOL_INPUT tool=%s %s=%d basenames=%s",
                tool_name,
                input_kind,
                input_count,
                input_summary,
            )

        try:
            if not use_timeout_retry:
                response = await operation()
                return response.output

            timeout = cls.tool_timeout_seconds()
            retries = cls.tool_timeout_retries()
            last_err: Exception | None = None
            for attempt in range(1, retries + 2):
                try:
                    response = await asyncio.wait_for(operation(), timeout=timeout)
                    return response.output
                except asyncio.TimeoutError as exc:
                    last_err = exc
                    logger.warning(
                        "MCP_TOOL_TIMEOUT tool=%s attempt=%s/%s timeout=%ss",
                        tool_name,
                        attempt,
                        retries + 1,
                        timeout,
                    )
                    if attempt <= retries:
                        continue
                    break

            if last_err is not None:
                return (
                    f"ERROR: {tool_name} timed out. "
                    f"timeout={timeout}s retries={retries}. "
                    "同一入力での無限再試行を防ぐため中断しました。"
                )
            return f"ERROR: {tool_name} failed (unknown error)"
        except Exception as exc:
            logger.exception("MCP_TOOL_ERR tool=%s", tool_name)
            if stringify_errors:
                return f"ERROR: {tool_name} failed: {type(exc).__name__}: {str(exc).strip()}"
            raise
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "MCP_TOOL_END tool=%s elapsed_ms=%s",
                tool_name,
                elapsed_ms,
            )

    @classmethod
    async def analyze_image_files(
        cls,
        llm_client: AbstractChatClient,
        file_list: list[str],
        prompt: str,
        detail: str,
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        target_paths = cls.resolve_existing_file_paths(file_list) if resolve_paths else file_list
        return await LLMClientUtil.analyze_image_files(llm_client, target_paths, prompt, detail)

    @classmethod
    async def analyze_pdf_files(
        cls,
        llm_client: AbstractChatClient,
        file_list: list[str],
        prompt: str,
        detail: str = "auto",
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        target_paths = cls.resolve_existing_file_paths(file_list) if resolve_paths else file_list
        return await LLMClientUtil.analyze_pdf_files(llm_client, target_paths, prompt, detail)

    @classmethod
    async def analyze_office_files(
        cls,
        llm_client: AbstractChatClient,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        target_paths = cls.resolve_existing_file_paths(file_path_list) if resolve_paths else file_path_list
        return await LLMClientUtil.analyze_office_files(llm_client, target_paths, prompt, detail)

    @classmethod
    async def analyze_files(
        cls,
        llm_client: AbstractChatClient,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        target_paths = cls.resolve_existing_file_paths(file_path_list) if resolve_paths else file_path_list
        return await LLMClientUtil.analyze_files(llm_client, target_paths, prompt, detail)

    @classmethod
    async def analyze_documents_data(
        cls,
        llm_client: AbstractChatClient,
        document_type_list: list[FileUtilDocument],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        return await LLMClientUtil.analyze_documents_data(
            llm_client,
            document_type_list,
            prompt,
            detail,
        )