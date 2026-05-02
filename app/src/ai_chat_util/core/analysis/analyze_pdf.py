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
from ...util.analyze_file_util.office2pdf import (
    LibreOfficeExecOffice2PDFUtil,
    LibreOfficeUnoOffice2PDFUtil,
    Pywin32Office2PDFUtil,
    _build_default_output_path,
)
from ai_chat_util.core.chat import create_llm_client
from ai_chat_util.core.common.config.runtime import get_runtime_config
from ai_chat_util.core.chat.model import WebRequestModel
from ai_chat_util.util.analyze_file_util.downloader import DownLoader

import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

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
        print_orientation: Annotated[Literal["portrait", "landscape"] | None, Field(description="Optional print orientation for generated PDFs")] = None,
    fit_width_pages: Annotated[int | None, Field(description="Optional number of horizontal pages to fit into when using LibreOffice UNO")] = None,
    fit_height_pages: Annotated[int | None, Field(description="Optional number of vertical pages to fit into when using LibreOffice UNO")] = None,

    ) -> Annotated[list[dict[str, str]], Field(description="List of source and generated PDF paths")]:
    """
        Convert Office documents to PDF files and return the generated PDF paths.
        印刷方向、ページ幅に合わせる、ページ高さに合わせるのオプションを指定しない場合は、
        変換前のOfficeファイルのレイアウトをできるだけ維持するように変換されます。
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=convert_office_files_to_pdf files=%d output_dir=%s",
        len(office_path_list or []),
        output_dir
    )
    try:
        config = get_runtime_config()
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        results: list[dict[str, str]] = []
        for office_path in office_path_list:
            resolved_output_path = output_dir if output_dir is not None else _build_default_output_path(office_path)
            if config.office2pdf.method == LibreOfficeExecOffice2PDFUtil.METHOD_NAME:
                pdf_path = LibreOfficeExecOffice2PDFUtil.create_pdf_from_document_file(
                    input_path=office_path,
                    output_path=resolved_output_path,
                    libreoffice_path=config.office2pdf.libreoffice_exec.libreoffice_path,
                )
            elif config.office2pdf.method == LibreOfficeUnoOffice2PDFUtil.METHOD_NAME:
                pdf_path = LibreOfficeUnoOffice2PDFUtil.create_pdf_from_document_file(
                    input_path=office_path,
                    output_path=resolved_output_path,
                    api_url=config.office2pdf.libreoffice_uno.api_url,
                    print_orientation=print_orientation,
                    fit_width_pages=fit_width_pages,
                    fit_height_pages=fit_height_pages,
                )
            elif config.office2pdf.method == Pywin32Office2PDFUtil.METHOD_NAME:
                pdf_path = Pywin32Office2PDFUtil.create_pdf_from_document_file(
                    input_path=office_path,
                    output_path=resolved_output_path,
                    office_path=config.office2pdf.pywin32.office_path,
                    print_orientation=print_orientation,
                    fit_width_pages=fit_width_pages,
                    fit_height_pages=fit_height_pages,
                )
            else:
                raise RuntimeError(f"Unsupported Office2PDF method: {config.office2pdf.method}")
            results.append({"source_path": office_path, "pdf_path": str(pdf_path)})
        return results
    
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


