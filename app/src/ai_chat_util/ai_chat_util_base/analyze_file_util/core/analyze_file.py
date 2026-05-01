from typing import Annotated, Literal
import tempfile
import atexit
import time
from pydantic import Field

from ..util.analyze_util import AnalyzeFileUtil
from .base import _get_network_download_options
from ai_chat_util.ai_chat_util_base.chat.core import create_llm_client
from ai_chat_util.ai_chat_util_base.chat.model import WebRequestModel
from ai_chat_util.ai_chat_util_base.analyze_file_util.model import FileUtilDocument
from ai_chat_util.ai_chat_util_base.analyze_file_util.util.downloader import DownLoader

import ai_chat_util.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


async def analyze_file_urls(
        file_path_urls: Annotated[list[WebRequestModel], Field(description="List of urls to the files to analyze. e.g., http://path/to/document1.docx")],
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
    This function analyzes multiple files (text, image, PDF, Office documents) using the specified prompt and returns the analysis result.
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_file_urls urls=%d detail=%s",
        len(file_path_urls or []),
        detail,
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
        response = await AnalyzeFileUtil.analyze_files(
            llm_client,
            path_list,
            prompt,
            detail,
        )
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_file_urls")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_file_urls elapsed_ms=%s",
            elapsed_ms,
        )


async def analyze_files(
        file_path_list: Annotated[list[str], Field(description="List of absolute paths to the files to analyze. e.g., [/path/to/document1.docx, /path/to/spreadsheet1.xlsx]")],
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
    This function analyzes multiple files (text, image, PDF, Office documents) using the specified prompt and returns the analysis result.
    """ 
    llm_client = create_llm_client()
    response = await AnalyzeFileUtil.analyze_files(
        llm_client,
        file_path_list,
        prompt,
        detail,
    )
    return response.output
