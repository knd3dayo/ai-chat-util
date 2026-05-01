from typing import Annotated, Literal
import tempfile
import atexit
import time
from itertools import count
from pathlib import Path
from typing import Any
from pydantic import Field

from .base import _get_network_download_options
from ...util.analyze_file_util.analyze_util import AnalyzePDFUtil
from ai_chat_util.ai_chat_util_base.core.chat.core import create_llm_client
from ai_chat_util.ai_chat_util_base.core.common.config.runtime import get_runtime_config
from ai_chat_util.ai_chat_util_base.core.chat.model import WebRequestModel
from ai_chat_util.ai_chat_util_base.core.analyze_file_util.model import FileUtilDocument
from ai_chat_util.ai_chat_util_base.util.analyze_file_util import pdf_util
from ai_chat_util.ai_chat_util_base.util.analyze_file_util.downloader import DownLoader
from ai_chat_util.ai_chat_util_base.util.analyze_file_util.office2pdf import Office2PDFUtil

import ai_chat_util.ai_chat_util_base.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

def _get_configured_libreoffice_path() -> str | None:
    return get_runtime_config().office2pdf.libreoffice_path

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
    This function analyzes multiple PDFs using the specified prompt 
    and returns the analysis result.
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_pdf_urls urls=%d detail=%s",
        len(pdf_path_urls or []),
        detail,
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
        response = await AnalyzePDFUtil.analyze_pdf_files(
            llm_client,
            path_list,
            prompt,
            detail,
        )
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_pdf_urls")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_pdf_urls elapsed_ms=%s",
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
    response = await AnalyzePDFUtil.analyze_pdf_files(
        prompt=prompt,
        detail=detail,
        llm_client=llm_client,
        file_list=pdf_path_list,)
    return response.output

async def convert_office_files_to_pdf(
        office_path_list: Annotated[list[str], Field(description="List of Office file paths to convert to PDF. e.g., [/path/to/document1.docx, /path/to/spreadsheet1.xlsx]")],
        output_dir: Annotated[str | None, Field(description="Optional output directory for generated PDFs. If omitted, PDFs are created next to the source files.")] = None,
    ) -> Annotated[list[dict[str, str]], Field(description="List of source and generated PDF paths")]:
    """
        Convert Office documents to PDF files and return the generated PDF paths.
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=convert_office_files_to_pdf files=%d output_dir=%s",
        len(office_path_list or []),
        output_dir
    )
    try:
        return AnalyzePDFUtil.convert_office_files_to_pdf(
            office_path_list,
            output_dir=output_dir,
            libreoffice_path=_get_configured_libreoffice_path(),
        )
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=convert_office_files_to_pdf")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=convert_office_files_to_pdf elapsed_ms=%s",
            elapsed_ms,
        )


async def convert_pdf_files_to_images(
        pdf_path_list: Annotated[list[str], Field(description="List of PDF file paths to convert into PNG page images. e.g., [/path/to/document1.pdf, /path/to/document2.pdf]")],
        output_dir: Annotated[str | None, Field(description="Optional output directory root for generated images. If omitted, an adjacent <pdf-stem>_pages directory is created for each PDF.")] = None,
    ) -> Annotated[list[dict[str, Any]], Field(description="List of source PDF paths and generated image paths")]:
    """
        Convert PDF pages into PNG image files and return the generated image paths.
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=convert_pdf_files_to_images files=%d output_dir=%s",
        len(pdf_path_list or []),
        output_dir,
    )
    try:
        return AnalyzePDFUtil.convert_pdf_files_to_images(
            pdf_path_list,
            output_dir=output_dir,
        )
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=convert_pdf_files_to_images")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=convert_pdf_files_to_images elapsed_ms=%s",
            elapsed_ms,
        )


