from typing import Annotated, Literal
import tempfile
import atexit
import time
from itertools import count
from pathlib import Path
from typing import Any
from pydantic import Field

from ai_chat_util.analysis import AnalysisService, LLMClientUtil
from ai_chat_util.ai_chat_util_base.chat import create_llm_client
from ai_chat_util.common.config.runtime import get_runtime_config
from ai_chat_util.ai_chat_util_base.ai_chat_util_models import WebRequestModel
from ai_chat_util.ai_chat_util_base.file_util.model import FileUtilDocument
from ai_chat_util.ai_chat_util_base.file_util.util import pdf_util
from ai_chat_util.ai_chat_util_base.file_util.util.downloader import DownLoader
from ai_chat_util.ai_chat_util_base.file_util.util.office2pdf import Office2PDFUtil

import ai_chat_util.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

_TOOL_CALL_SEQ = count(1)


def _resolve_existing_file_paths(file_path_list: list[str]) -> list[str]:
    """ユーザー入力のパスを、実在するパスへ解決して返す。"""
    return AnalysisService.resolve_existing_file_paths(file_path_list)


def _resolve_output_dir(output_dir: str | None) -> Path | None:
    if output_dir is None or not str(output_dir).strip():
        return None

    candidate = Path(output_dir).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    llm_config = get_runtime_config()
    working_directory = llm_config.mcp.working_directory or "."
    return (Path(working_directory).expanduser() / candidate).resolve()


def _plan_office_pdf_outputs(resolved_paths: list[str], output_dir: Path | None) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for office_path in resolved_paths:
        source_path = Path(office_path)
        if output_dir is None:
            pdf_path = source_path.with_suffix(".pdf")
        else:
            pdf_path = output_dir / source_path.with_suffix(".pdf").name
        results.append({
            "source_path": office_path,
            "pdf_path": str(pdf_path),
        })
    return results


def _plan_pdf_image_outputs(resolved_paths: list[str], output_root: Path | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for pdf_path in resolved_paths:
        source_path = Path(pdf_path)
        image_dir = (
            (output_root / f"{source_path.stem}_pages")
            if output_root is not None
            else source_path.with_name(f"{source_path.stem}_pages")
        )
        results.append({
            "source_path": pdf_path,
            "image_dir": str(image_dir),
            "image_file_pattern": f"{source_path.stem}_page_####.png",
        })
    return results


def _get_network_download_options() -> tuple[bool, str | None]:
    cfg = get_runtime_config()
    return cfg.network.requests_verify, cfg.network.ca_bundle


def _get_configured_libreoffice_path() -> str | None:
    return get_runtime_config().office2pdf.libreoffice_path


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
        AnalysisService.prompt_len(prompt),
    )
    llm_client = create_llm_client()
    try:
        response = await LLMClientUtil.analyze_image_urls(llm_client, image_path_urls, prompt, detail)
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


async def analyze_image_files(
        file_list: Annotated[list[str], Field(description="List of absolute paths to the image files to analyze. e.g., [/path/to/image1.jpg, /path/to/image2.jpg]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the images")],
        detail: Annotated[str, Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes multiple images using the specified prompt and returns the analysis result.
    """
    llm_client = create_llm_client()
    resolved_paths = _resolve_existing_file_paths(file_list)
    return await AnalysisService.run_analysis_tool(
        tool_name="analyze_image_files",
        prompt=prompt,
        detail=detail,
        input_count=len(resolved_paths),
        input_kind="files",
        input_summary=AnalysisService.summarize_path_basenames(resolved_paths),
        use_timeout_retry=True,
        stringify_errors=True,
        operation=lambda: AnalysisService.analyze_image_files(
            llm_client,
            resolved_paths,
            prompt,
            detail,
            resolve_paths=False,
        ),
    )


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
        AnalysisService.prompt_len(prompt),
    )
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = create_llm_client()
    try:
        requests_verify, ca_bundle = _get_network_download_options()
        path_list = DownLoader.download_files(
            pdf_path_urls,
            tmpdir.name,
            requests_verify=requests_verify,
            ca_bundle=ca_bundle,
        )
        response = await AnalysisService.analyze_pdf_files(
            llm_client,
            path_list,
            prompt,
            detail,
            resolve_paths=False,
        )
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
    llm_client = create_llm_client()
    resolved_paths = _resolve_existing_file_paths(pdf_path_list)
    return await AnalysisService.run_analysis_tool(
        tool_name="analyze_pdf_files",
        prompt=prompt,
        detail=detail,
        input_count=len(resolved_paths),
        input_kind="files",
        input_summary=AnalysisService.summarize_path_basenames(resolved_paths),
        operation=lambda: AnalysisService.analyze_pdf_files(
            llm_client,
            resolved_paths,
            prompt,
            detail,
            resolve_paths=False,
        ),
    )


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
        AnalysisService.prompt_len(prompt),
    )
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = create_llm_client()
    try:
        requests_verify, ca_bundle = _get_network_download_options()
        path_list = DownLoader.download_files(
            office_path_urls,
            tmpdir.name,
            requests_verify=requests_verify,
            ca_bundle=ca_bundle,
        )
        response = await AnalysisService.analyze_office_files(
            llm_client,
            path_list,
            prompt,
            detail,
            resolve_paths=False,
        )
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
    llm_client = create_llm_client()
    resolved_paths = _resolve_existing_file_paths(office_path_list)
    return await AnalysisService.run_analysis_tool(
        tool_name="analyze_office_files",
        prompt=prompt,
        detail=detail,
        input_count=len(resolved_paths),
        input_kind="files",
        input_summary=AnalysisService.summarize_path_basenames(resolved_paths),
        operation=lambda: AnalysisService.analyze_office_files(
            llm_client,
            resolved_paths,
            prompt,
            detail,
            resolve_paths=False,
        ),
    )


async def convert_office_files_to_pdf(
        office_path_list: Annotated[list[str], Field(description="List of Office file paths to convert to PDF. e.g., [/path/to/document1.docx, /path/to/spreadsheet1.xlsx]")],
        output_dir: Annotated[str | None, Field(description="Optional output directory for generated PDFs. If omitted, PDFs are created next to the source files.")] = None,
    dry_run: Annotated[bool, Field(description="If true, do not write files. Return the planned output paths only.")] = False,
    ) -> Annotated[list[dict[str, str]], Field(description="List of source and generated PDF paths")]:
    """
        Convert Office documents to PDF files and return the generated PDF paths.

        For write-safe usage, call this tool with dry_run=True first to preview the
        output paths. After approval, call it again with dry_run=False.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=convert_office_files_to_pdf call_id=%s files=%d output_dir=%s dry_run=%s",
        call_id,
        len(office_path_list or []),
        output_dir,
        dry_run,
    )
    try:
        resolved_paths = _resolve_existing_file_paths(office_path_list)
        target_dir = _resolve_output_dir(output_dir)
        if dry_run:
            return [{**item, "dry_run": "true"} for item in AnalysisService.convert_office_files_to_pdf(
                resolved_paths,
                output_dir=target_dir,
                dry_run=True,
                libreoffice_path=_get_configured_libreoffice_path(),
                resolve_paths=False,
            )]

        return AnalysisService.convert_office_files_to_pdf(
            resolved_paths,
            output_dir=target_dir,
            dry_run=False,
            libreoffice_path=_get_configured_libreoffice_path(),
            resolve_paths=False,
        )
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=convert_office_files_to_pdf call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=convert_office_files_to_pdf call_id=%s elapsed_ms=%s",
            call_id,
            elapsed_ms,
        )


async def convert_pdf_files_to_images(
        pdf_path_list: Annotated[list[str], Field(description="List of PDF file paths to convert into PNG page images. e.g., [/path/to/document1.pdf, /path/to/document2.pdf]")],
        output_dir: Annotated[str | None, Field(description="Optional output directory root for generated images. If omitted, an adjacent <pdf-stem>_pages directory is created for each PDF.")] = None,
    dry_run: Annotated[bool, Field(description="If true, do not write files. Return the planned output directories and filename pattern only.")] = False,
    ) -> Annotated[list[dict[str, Any]], Field(description="List of source PDF paths and generated image paths")]:
    """
        Convert PDF pages into PNG image files and return the generated image paths.

        For write-safe usage, call this tool with dry_run=True first to preview the
        target image directory and filename pattern. After approval, call it again
        with dry_run=False.
    """
    call_id = next(_TOOL_CALL_SEQ)
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=convert_pdf_files_to_images call_id=%s files=%d output_dir=%s dry_run=%s",
        call_id,
        len(pdf_path_list or []),
        output_dir,
        dry_run,
    )
    try:
        resolved_paths = _resolve_existing_file_paths(pdf_path_list)
        output_root = _resolve_output_dir(output_dir)
        if dry_run:
            return [{**item, "dry_run": True} for item in AnalysisService.convert_pdf_files_to_images(
                resolved_paths,
                output_dir=output_root,
                dry_run=True,
                resolve_paths=False,
            )]

        return AnalysisService.convert_pdf_files_to_images(
            resolved_paths,
            output_dir=output_root,
            dry_run=False,
            resolve_paths=False,
        )
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=convert_pdf_files_to_images call_id=%s", call_id)
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=convert_pdf_files_to_images call_id=%s elapsed_ms=%s",
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
        AnalysisService.prompt_len(prompt),
    )
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = create_llm_client()
    try:
        requests_verify, ca_bundle = _get_network_download_options()
        path_list = DownLoader.download_files(
            file_path_urls,
            tmpdir.name,
            requests_verify=requests_verify,
            ca_bundle=ca_bundle,
        )
        response = await AnalysisService.analyze_files(
            llm_client,
            path_list,
            prompt,
            detail,
            resolve_paths=False,
        )
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


async def detect_log_format_and_search(
        file_path: Annotated[str, Field(description="Path to the log file to inspect")],
        search_terms: Annotated[list[str] | None, Field(description="Optional extra search keywords")] = None,
        sample_line_limit: Annotated[int, Field(description="Maximum number of head lines used for format detection")] = 100,
        match_limit: Annotated[int, Field(description="Maximum number of matched lines returned per pattern")] = 50,
    ) -> Annotated[str, Field(description="JSON string describing detected log format and matched records")]:
    resolved_path = _resolve_existing_file_paths([file_path])[0]
    return AnalysisService.detect_log_format_and_search_from_file(
        resolved_path,
        search_terms=search_terms,
        sample_line_limit=sample_line_limit,
        match_limit=match_limit,
        resolve_paths=False,
    )


async def infer_log_header_pattern(
        file_path: Annotated[str, Field(description="Path to the log file to inspect")],
        sample_line_limit: Annotated[int, Field(description="Maximum number of head lines used for inference")] = 100,
    ) -> Annotated[str, Field(description="JSON string describing inferred header pattern and timestamp slice")]:
    llm_client = create_llm_client()
    resolved_path = _resolve_existing_file_paths([file_path])[0]
    return await AnalysisService.infer_log_header_pattern(
        llm_client,
        resolved_path,
        sample_line_limit,
        resolve_paths=False,
    )


async def extract_log_time_range(
        file_path: Annotated[str, Field(description="Path to the log file to extract from")],
        workspace_path: Annotated[str, Field(description="Workspace root where derived artifacts are written")],
        header_pattern: Annotated[str, Field(description="Regular expression matching each log record header")],
        timestamp_start: Annotated[int, Field(description="Start offset of timestamp text within the header line")],
        timestamp_end: Annotated[int, Field(description="End offset of timestamp text within the header line")],
        range_start: Annotated[str, Field(description="Extraction start timestamp")],
        range_end: Annotated[str, Field(description="Extraction end timestamp")],
        time_format: Annotated[str | None, Field(description="Optional datetime.strptime-compatible timestamp format")] = None,
        output_subdir: Annotated[str, Field(description="Artifacts subdirectory for extracted logs")] = "log_extracts",
        output_filename: Annotated[str | None, Field(description="Optional output filename")] = None,
    ) -> Annotated[str, Field(description="JSON string describing extracted records and output path")]:
    resolved_path = _resolve_existing_file_paths([file_path])[0]
    runtime_config = get_runtime_config()
    return AnalysisService.extract_log_time_range_to_file(
        file_path=resolved_path,
        workspace_path=workspace_path,
        artifacts_subdir=".artifacts",
        header_pattern=header_pattern,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        range_start=range_start,
        range_end=range_end,
        time_format=time_format,
        output_subdir=output_subdir,
        output_filename=output_filename,
        resolve_paths=False,
    )


async def analyze_files(
    file_path_list: Annotated[list[str], Field(description="List of existing file or directory paths to analyze. If the target is a directory, pass that directory path itself. Do not invent child paths or substitute unrelated files. e.g., [/path/to/document1.pdf, /path/to/workdir]")],
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
        Analyze existing files or directories using the specified prompt and return the result.

        Usage rules:
        - Pass only paths that actually exist.
        - If the user asked to inspect a directory, pass the directory path itself.
            This tool resolves supported files inside that directory automatically.
        - Do not invent child paths, placeholder paths, or unrelated substitute files.
    """
    llm_client = create_llm_client()
    resolved_paths = _resolve_existing_file_paths(file_path_list)
    return await AnalysisService.run_analysis_tool(
        tool_name="analyze_files",
        prompt=prompt,
        detail=detail,
        input_count=len(resolved_paths),
        input_kind="files",
        input_summary=AnalysisService.summarize_path_basenames(resolved_paths),
        operation=lambda: AnalysisService.analyze_files(
            llm_client,
            resolved_paths,
            prompt,
            detail,
            resolve_paths=False,
        ),
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
    llm_client = create_llm_client()
    return await AnalysisService.run_analysis_tool(
        tool_name="analyze_documents_data",
        prompt=prompt,
        detail=detail,
        input_count=len(document_type_list or []),
        input_kind="docs",
        operation=lambda: AnalysisService.analyze_documents_data(
            llm_client,
            document_type_list,
            prompt,
            detail=detail,
        ),
    )