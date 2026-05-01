from typing import Annotated, Literal
import tempfile
import atexit
import time
from pydantic import Field

from ..util.analyze_util import AnalyzeImageUtil
from .base import _get_network_download_options

from ai_chat_util.ai_chat_util_base.chat.core import create_llm_client
from ai_chat_util.ai_chat_util_base.chat.model import WebRequestModel
from ai_chat_util.ai_chat_util_base.analyze_file_util.util.downloader import DownLoader

import ai_chat_util.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

async def analyze_image_urls(
        image_path_urls: Annotated[list[WebRequestModel], Field(description="List of urls to the image files to analyze. e.g., http://path/to/image1.jpg")],
        prompt: Annotated[str, Field(description="Prompt to analyze the images")],
        detail: Annotated[str, Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes multiple images using the specified prompt and returns the analysis result.
    """
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_image_urls urls=%d detail=%s",
        len(image_path_urls or []),
        detail
    )
    llm_client = create_llm_client()

    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = create_llm_client()
    try:
        requests_verify, ca_bundle = _get_network_download_options()
        path_list = DownLoader.download_files(
            image_path_urls,
            tmpdir.name,
            requests_verify=requests_verify,
            ca_bundle=ca_bundle,
        )
        response = await AnalyzeImageUtil.analyze_image_files(llm_client, path_list, prompt, detail)
        return response.output

    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_image_urls")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_image_urls elapsed_ms=%s",
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
    response = await AnalyzeImageUtil.analyze_image_files(
        llm_client, file_list, prompt, detail)
    return response.output
