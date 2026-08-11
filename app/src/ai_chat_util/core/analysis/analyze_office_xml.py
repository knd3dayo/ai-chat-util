"""Office XML 解析ツール公開モジュール。

Office ファイルの内部 XML / フォント / 書式 / 画像情報を抽出し、
LLM に分析させるための公開関数を提供します。
"""

from typing import Annotated
import tempfile
import atexit
import time

from pydantic import Field

from ...util.analyze_file_util.office_xml_analysis import OfficeXmlAnalysisUtil
from ai_chat_util.core.chat import create_llm_client
from ai_chat_util.core.chat.model import ChatHistory, ChatMessage, ChatRequest, ChatResponse

import ai_chat_util.core.log.log_settings as log_settings

logger = log_settings.getLogger(__name__)


class AnalyzeOfficeXmlUtil:
    @classmethod
    async def analyze_office_xml_files(
        cls,
        llm_client,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        report_contents: list = []

        for file_path in file_path_list:
            report = OfficeXmlAnalysisUtil.analyze_office_file(file_path)
            report_text = OfficeXmlAnalysisUtil.format_report(report)
            report_contents.append(
                llm_client.get_message_factory().create_text_content(
                    text=f"[Office XML analysis] {file_path}\n{report_text}"
                )
            )

        chat_message = ChatMessage(role="user", content=[prompt_content] + report_contents)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        response: ChatResponse = await llm_client.chat(chat_request)
        return response


async def analyze_office_xml_files(
        office_path_list: Annotated[list[str], Field(description="List of absolute paths to the Office files to analyze as XML structure. e.g., [/path/to/document1.docx, /path/to/spreadsheet1.xlsx]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the Office XML structure")],
        detail: Annotated[
            str,
            Field(description="Reserved for compatibility with other analyzers. Defaults to auto"),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the Office XML structure")]:
    """Analyse Office XML structure and return an LLM summary."""
    started = time.perf_counter()
    logger.info(
        "MCP_TOOL_START tool=analyze_office_xml_files files=%d detail=%s",
        len(office_path_list or []),
        detail,
    )
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = create_llm_client()
    try:
        response = await AnalyzeOfficeXmlUtil.analyze_office_xml_files(
            llm_client,
            office_path_list,
            prompt,
            detail,
        )
        return response.output
    except Exception:
        logger.exception("MCP_TOOL_ERR tool=analyze_office_xml_files")
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "MCP_TOOL_END tool=analyze_office_xml_files elapsed_ms=%s",
            elapsed_ms,
        )
