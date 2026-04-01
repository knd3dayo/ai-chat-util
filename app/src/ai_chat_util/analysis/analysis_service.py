from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from ai_chat_util.base.llm.abstract_llm_client import AbstractLLMClient
from ai_chat_util.common.config.runtime import get_runtime_config
from ai_chat_util.common.model.ai_chatl_util_models import ChatResponse
from file_util.model import FileUtilDocument
from file_util.util.file_path_resolver import resolve_existing_file_path

import ai_chat_util.log.log_settings as log_settings

from .llm_client_util import LLMClientUtil


logger = log_settings.getLogger(__name__)


class AnalysisService:
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

    @staticmethod
    def resolve_existing_file_paths(file_path_list: list[str]) -> list[str]:
        runtime_config = get_runtime_config()
        resolved: list[str] = []
        for file_path in file_path_list:
            result = resolve_existing_file_path(
                file_path,
                working_directory=runtime_config.mcp.working_directory,
            )
            resolved.append(result.resolved_path)
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
        llm_client: AbstractLLMClient,
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
        llm_client: AbstractLLMClient,
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
        llm_client: AbstractLLMClient,
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
        llm_client: AbstractLLMClient,
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
        llm_client: AbstractLLMClient,
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