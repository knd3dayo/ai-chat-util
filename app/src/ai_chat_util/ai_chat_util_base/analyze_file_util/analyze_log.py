from typing import Annotated, Literal
import tempfile
import atexit
import time
from itertools import count
from pathlib import Path
from typing import Any
from pydantic import Field

from ...util.analyze_file_util.analyze_util import AnalyzeLogUtil
from ai_chat_util.ai_chat_util_base.chat.core import create_llm_client
from ai_chat_util.ai_chat_util_base.common.config.runtime import get_runtime_config
from ai_chat_util.ai_chat_util_base.chat.model import WebRequestModel
from ai_chat_util.ai_chat_util_base.analyze_file_util.model import FileUtilDocument
from ai_chat_util.ai_chat_util_base.util.analyze_file_util import pdf_util
from ai_chat_util.util.analyze_file_util.downloader import DownLoader
from ai_chat_util.util.analyze_file_util.office2pdf import Office2PDFUtil

import ai_chat_util.ai_chat_util_base.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)

async def detect_log_format_and_search(
        file_path: Annotated[str, Field(description="Path to the log file to inspect")],
        search_terms: Annotated[list[str] | None, Field(description="Optional extra search keywords")] = None,
        sample_line_limit: Annotated[int, Field(description="Maximum number of head lines used for format detection")] = 100,
        match_limit: Annotated[int, Field(description="Maximum number of matched lines returned per pattern")] = 50,
    ) -> Annotated[str, Field(description="JSON string describing detected log format and matched records")]:
    
    return AnalyzeLogUtil.detect_log_format_and_search_from_file(
        file_path=file_path,
        search_terms=search_terms,
        sample_line_limit=sample_line_limit,
        match_limit=match_limit,
    )


async def infer_log_header_pattern(
        file_path: Annotated[str, Field(description="Path to the log file to inspect")],
        sample_line_limit: Annotated[int, Field(description="Maximum number of head lines used for inference")] = 100,
    ) -> Annotated[str, Field(description="JSON string describing inferred header pattern and timestamp slice")]:
    llm_client = create_llm_client()
    return await AnalyzeLogUtil.infer_log_header_pattern(
        llm_client,
        file_path,
        sample_line_limit,
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
    runtime_config = get_runtime_config()
    return AnalyzeLogUtil.extract_log_time_range_to_file(
        file_path=file_path,
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
    )
