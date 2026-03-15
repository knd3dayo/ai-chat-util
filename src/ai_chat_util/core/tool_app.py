from typing import Annotated, Literal
import tempfile
import atexit
import time
import asyncio
from itertools import count
from pathlib import Path

from pydantic import Field
from ai_chat_util.model.models import ChatHistory, ChatResponse, WebRequestModel, ChatRequest, ChatMessage, ChatContent
from ai_chat_util.llm.llm_factory import LLMFactory
from ai_chat_util.llm.llm_client import LLMClientUtil
from ai_chat_util.config.runtime import get_runtime_config
from ai_chat_util.llm.batch_client import LLMBatchClient
from file_util.model import FileUtilDocument
from ai_chat_util.util.file_path_resolver import resolve_existing_file_path
from ai_chat_util.config.runtime import get_runtime_config

import ai_chat_util.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

_TOOL_CALL_SEQ = count(1)


def _summarize_path_basenames(paths: list[str], *, limit: int = 5) -> str:
    names: list[str] = []
    for p in paths[:limit]:
        try:
            names.append(Path(p).name)
        except Exception:
            names.append(str(p))
    more = "" if len(paths) <= limit else f" (+{len(paths) - limit} more)"
    return f"[{', '.join(names)}]{more}"


def _prompt_len(prompt: str) -> int:
    return len((prompt or "").strip())


def _resolve_existing_file_paths(file_path_list: list[str]) -> list[str]:
    """ユーザー入力のパスを、実在するパスへ解決して返す。"""
    llm_config = get_runtime_config()
    resolved: list[str] = []
    for p in file_path_list:
        r = resolve_existing_file_path(p, working_directory=llm_config.paths.working_directory)
        resolved.append(r.resolved_path)
    return resolved

# 複数の画像の分析を行う URLから画像をダウンロードして分析する 
async def analyze_image_urls(
        image_path_urls: Annotated[list[WebRequestModel], Field(description="List of urls to the image files to analyze. e.g., http://path/to/image1.jpg")],
        prompt: Annotated[str, Field(description="Prompt to analyze the images")],
        detail: Annotated[str, Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes multiple images using the specified prompt and returns the analysis result.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_image_urls call_id=%s urls=%d detail=%s prompt_len=%d",
        call_id,
        len(image_path_urls or []),
        detail,
        _prompt_len(prompt),
    )
    llm_client = LLMFactory.create_llm_client()
    try:
        response = await llm_client.analyze_image_urls(image_path_urls, prompt, detail)
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_image_urls call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_image_urls call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )

# 複数の画像の分析を行う
async def analyze_image_files(
        file_list: Annotated[list[str], Field(description="List of absolute paths to the image files to analyze. e.g., [/path/to/image1.jpg, /path/to/image2.jpg]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the images")],
        detail: Annotated[str, Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes multiple images using the specified prompt and returns the analysis result.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_image_files call_id=%s files=%d detail=%s prompt_len=%d",
        call_id,
        len(file_list or []),
        detail,
        _prompt_len(prompt),
    )
    llm_client = LLMFactory.create_llm_client()
    try:
        resolved_paths = _resolve_existing_file_paths(file_list)
        logger.info(
            "MCP_TOOL_INPUT tool=analyze_image_files call_id=%s resolved_files=%d basenames=%s",
            call_id,
            len(resolved_paths),
            _summarize_path_basenames(resolved_paths),
        )
        cfg = get_runtime_config()
        tool_timeout_cfg = getattr(cfg.features, "mcp_tool_timeout_seconds", None)
        try:
            tool_timeout = float(tool_timeout_cfg) if tool_timeout_cfg is not None else float(cfg.llm.timeout_seconds)
        except (TypeError, ValueError):
            tool_timeout = float(cfg.llm.timeout_seconds)
        if tool_timeout <= 0:
            tool_timeout = float(cfg.llm.timeout_seconds)

        try:
            retries_raw = int(getattr(cfg.features, "mcp_tool_timeout_retries", 1) or 0)
        except (TypeError, ValueError):
            retries_raw = 1
        retries = max(0, min(5, retries_raw))

        last_err: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                response = await asyncio.wait_for(
                    llm_client.analyze_image_files(resolved_paths, prompt, detail),
                    timeout=tool_timeout,
                )
                return response.output
            except asyncio.TimeoutError as e:
                last_err = e
                logger.warning(
                    "MCP_TOOL_TIMEOUT tool=analyze_image_files call_id=%s attempt=%s/%s timeout=%ss",
                    call_id,
                    attempt,
                    retries + 1,
                    tool_timeout,
                )
                if attempt <= retries:
                    continue
                break
            except Exception as e:
                # Convert tool exceptions into normal output to avoid agent-level retry loops.
                logger.exception("MCP_TOOL_ERR tool=analyze_image_files call_id=%s", call_id)
                return f"ERROR: analyze_image_files failed: {type(e).__name__}: {str(e).strip()}"

        if last_err is not None:
            return (
                "ERROR: analyze_image_files timed out. "
                f"timeout={tool_timeout}s retries={retries}. "
                "同一入力での無限再試行を防ぐため中断しました。"
            )
        return "ERROR: analyze_image_files failed (unknown error)"
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_image_files call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )


# 複数のPDFの分析を行う URLからPDFをダウンロードして分析する
async def analyze_pdf_urls(
        pdf_path_urls: Annotated[
            list[WebRequestModel],
            Field(
                description="List of URLs to the PDF files to analyze. e.g., http://path/to/document2.pdf"
            ),
        ],
        prompt: Annotated[str, Field(description="Prompt to analyze the PDFs")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
) -> Annotated[str, Field(description="Analysis result of the PDFs")]:
    """
    This function analyzes multiple PDFs using the specified prompt and returns the analysis result.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_pdf_urls call_id=%s urls=%d detail=%s prompt_len=%d",
        call_id,
        len(pdf_path_urls or []),
        detail,
        _prompt_len(prompt),
    )
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = LLMFactory.create_llm_client()
    try:
        path_list = LLMClientUtil.download_files(pdf_path_urls, tmpdir.name)
        response = await llm_client.analyze_pdf_files(path_list, prompt, detail)
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_pdf_urls call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_pdf_urls call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )

# 複数のPDFの分析を行う
async def analyze_pdf_files(
        pdf_path_list: Annotated[list[str], Field(description="List of absolute paths to the PDF files to analyze. e.g., [/path/to/document1.pdf, /path/to/document2.pdf]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the PDFs")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the PDFs")]:
    """
    This function analyzes multiple PDFs using the specified prompt and returns the analysis result.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_pdf_files call_id=%s files=%d detail=%s prompt_len=%d",
        call_id,
        len(pdf_path_list or []),
        detail,
        _prompt_len(prompt),
    )
    llm_client = LLMFactory.create_llm_client()
    try:
        resolved_paths = _resolve_existing_file_paths(pdf_path_list)
        response = await llm_client.analyze_pdf_files(resolved_paths, prompt, detail)
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_pdf_files call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_pdf_files call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )

# 複数のOfficeドキュメントの分析を行う URLからOfficeドキュメントをダウンロードして分析する
async def analyze_office_urls(
        office_path_urls: Annotated[list[WebRequestModel], Field(description="List of urls to the Office files to analyze. e.g., http://path/to/document1.docx")],
        prompt: Annotated[str, Field(description="Prompt to analyze the Office documents")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the Office documents")]:
    """ 
    This function analyzes multiple Office documents using the specified prompt and returns the analysis result.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_office_urls call_id=%s urls=%d detail=%s prompt_len=%d",
        call_id,
        len(office_path_urls or []),
        detail,
        _prompt_len(prompt),
    )
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = LLMFactory.create_llm_client()
    try:
        path_list = LLMClientUtil.download_files(office_path_urls, tmpdir.name)
        response = await llm_client.analyze_office_files(path_list, prompt, detail)
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_office_urls call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_office_urls call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )

async def analyze_office_files(
        office_path_list: Annotated[list[str], Field(description="List of absolute paths to the Office files to analyze. e.g., [/path/to/document1.docx, /path/to/spreadsheet1.xlsx]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the Office documents")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the Office documents")]:
    """
    This function analyzes multiple Office documents using the specified prompt and returns the analysis result.
    """ 
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_office_files call_id=%s files=%d detail=%s prompt_len=%d",
        call_id,
        len(office_path_list or []),
        detail,
        _prompt_len(prompt),
    )
    llm_client = LLMFactory.create_llm_client()
    try:
        resolved_paths = _resolve_existing_file_paths(office_path_list)
        response = await llm_client.analyze_office_files(resolved_paths, prompt, detail=detail)
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_office_files call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_office_files call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )

async def analyze_urls(
        file_path_urls: Annotated[list[WebRequestModel], Field(description="List of urls to the files to analyze. e.g., http://path/to/document1.pdf, http://path/to/image1.jpg")],
        prompt: Annotated[str, Field(description="Prompt to analyze the files")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the files")]:
    """
    This function analyzes multiple files of various formats using the specified prompt and returns the analysis result.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_urls call_id=%s urls=%d detail=%s prompt_len=%d",
        call_id,
        len(file_path_urls or []),
        detail,
        _prompt_len(prompt),
    )
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = LLMFactory.create_llm_client()
    try:
        path_list = LLMClientUtil.download_files(file_path_urls, tmpdir.name)
        response = await llm_client.analyze_files(path_list, prompt, detail)
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_urls call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_urls call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )

async def analyze_files(
        file_path_list: Annotated[list[str], Field(description="List of absolute paths to the files to analyze. e.g., [/path/to/document1.pdf, /path/to/image1.jpg]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the files")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the files")]:
    """
    This function analyzes multiple files of various formats using the specified prompt and returns the analysis result.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_files call_id=%s files=%d detail=%s prompt_len=%d",
        call_id,
        len(file_path_list or []),
        detail,
        _prompt_len(prompt),
    )
    llm_client = LLMFactory.create_llm_client()
    try:
        resolved_paths = _resolve_existing_file_paths(file_path_list)
        response = await llm_client.analyze_files(resolved_paths, prompt, detail=detail)
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_files call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_files call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )

async def analyze_documents_data(
        document_type_list: Annotated[list[FileUtilDocument], Field(description="List of FileUtilDocument objects to analyze.")],
        prompt: Annotated[str, Field(description="Prompt to analyze the documents")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the files")]:
    """
    This function analyzes multiple files of various formats using the specified prompt and returns the analysis result.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_documents_data call_id=%s docs=%d detail=%s prompt_len=%d",
        call_id,
        len(document_type_list or []),
        detail,
        _prompt_len(prompt),
    )
    llm_client = LLMFactory.create_llm_client()
    try:
        response = await llm_client.analyze_documents_data(document_type_list, prompt, detail=detail)
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_documents_data call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_documents_data call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )
