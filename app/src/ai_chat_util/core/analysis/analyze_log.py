from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import Field

from ai_chat_util.core.analysis.analyze_util import AnalyzeLogUtil
from ai_chat_util.core.analysis.model import (
    ExtractLogTimeRangeData,
    InferLogFormatData,
)
from ai_chat_util.core.chat import create_llm_client


async def infer_log_header_pattern(
    file_path: Annotated[str, Field(description="Path to the log file to inspect")],
    sample_line_limit: Annotated[int, Field(description="Maximum number of head lines used for inference")] = 100,
    additional_instructions: Annotated[str | None, Field(description="Optional extra guidance for the LLM, such as filename-based format hints")] = None,
) -> Annotated[InferLogFormatData, Field(description="Structured result describing inferred header pattern, likely log format description, and timestamp format")]:
    llm_client = create_llm_client()
    return await AnalyzeLogUtil.infer_log_header_pattern(
        llm_client,
        file_path,
        sample_line_limit,
        additional_instructions,
    )


async def extract_time_range_from_logfile(
    file_path: Annotated[str, Field(description="Path to the log file to inspect")],
    output_path: Annotated[str, Field(description="Directory where the extracted log file will be written")],
    start_time: Annotated[datetime, Field(description="Inclusive start timestamp for extracted log records")],
    end_time: Annotated[datetime, Field(description="Inclusive end timestamp for extracted log records")],
    sample_line_limit: Annotated[int, Field(description="Maximum number of head lines used for log header inference")] = 100,
    additional_instructions: Annotated[str | None, Field(description="Optional extra guidance for the LLM, such as filename-based format hints")] = None,
) -> Annotated[ExtractLogTimeRangeData, Field(description="Structured result describing the extracted log artifact written for the requested time range")]:
    llm_client = create_llm_client()
    return await AnalyzeLogUtil.extract_time_range_from_logfile(
        llm_client,
        file_path=file_path,
        output_path=output_path,
        start_time=start_time,
        end_time=end_time,
        sample_line_limit=sample_line_limit,
        additional_instructions=additional_instructions,
    )