from __future__ import annotations

import time
import json
import re
from pathlib import Path
from typing import Any
from datetime import datetime

from ai_chat_util.core.chat import AbstractChatClient
from ai_chat_util.core.chat.model import (
    ChatContent,
    ChatHistory,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    WebRequestModel,
)

from ai_chat_util.core.analysis.model import (
    FileUtilDocument,
    InferLogFormatData,
    LogSearchMatchData,
)
import fitz  # PyMuPDF
import ai_chat_util.core.log.log_settings as log_settings
from .file_util_llm_messages import FileUtilLLMMessages

logger = log_settings.getLogger(__name__)

class AnalyzeImageUtil:
    @classmethod
    async def analyze_image_files(
        cls,
        llm_client: AbstractChatClient,
        file_list: list[str],
        prompt: str,
        detail: str,
    ) -> ChatResponse:
        started = time.perf_counter()
        logger.info(
            "IMAGE_ANALYZE_START images=%d detail=%s prompt_len=%d",
            len(file_list or []),
            detail,
            len((prompt or "").strip()),
        )
        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        image_content_list: list[ChatContent] = []
        encode_started = time.perf_counter()
        total_bytes = 0
        for image_path in file_list:
            doc = FileUtilDocument.from_file(document_path=image_path)
            try:
                total_bytes += len(doc.data or b"")
            except Exception:
                pass
            image_contents = llm_client.get_message_factory()._create_image_content_(doc.identifier, doc.data, detail)
            image_content_list.extend(image_contents)
        logger.info(
            "IMAGE_ENCODE_END images=%d total_bytes=%d elapsed_ms=%d",
            len(file_list or []),
            total_bytes,
            int((time.perf_counter() - encode_started) * 1000),
        )

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_started = time.perf_counter()
        logger.info("IMAGE_CHAT_START")
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        logger.info(
            "IMAGE_CHAT_END elapsed_ms=%d total_elapsed_ms=%d",
            int((time.perf_counter() - chat_started) * 1000),
            int((time.perf_counter() - started) * 1000),
        )
        return chat_response

    @classmethod
    async def analyze_image_urls(
        cls,
        llm_client: AbstractChatClient,
        image_url_list: list[WebRequestModel],
        prompt: str,
        detail: str,
    ) -> ChatResponse:
        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        image_content_list: list[ChatContent] = []
        file_util_llm_messages = FileUtilLLMMessages(llm_client)
        for image_url in image_url_list:
            image_contents = await file_util_llm_messages.create_image_content_from_url_async(
                image_url, detail
            )
            image_content_list.extend(image_contents)

        chat_message = ChatMessage(role="user", content=[prompt_content] + image_content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        return chat_response


class AnalyzePDFUtil:

    @classmethod
    async def analyze_pdf_files(
        cls,
        llm_client: AbstractChatClient,
        file_list: list[str],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        pdf_content_list = []
        config = llm_client.get_config()
        if not config:
            raise ValueError("LLMClientの設定が取得できませんでした。")

        for file_path in file_list:
            if config.features.use_custom_pdf_analyzer:
                logger.info(f"Using custom PDF analyzer for file: {file_path}")
                with open(file_path, "rb") as f:
                    pdf_data = f.read()
                pdf_content = llm_client.get_message_factory()._create_custom_pdf_content_(file_path, pdf_data, detail)
            else:
                logger.info(f"Using standard PDF analyzer for file: {file_path}")
                with open(file_path, "rb") as f:
                    pdf_data = f.read()
                pdf_content = llm_client.get_message_factory()._create_pdf_content_(file_path, pdf_data, detail)
            pdf_content_list.extend(pdf_content)

        chat_message = ChatMessage(role="user", content=[prompt_content] + pdf_content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        return chat_response

    @classmethod
    def convert_pdf_files_to_images(
        cls,
        file_path_list: list[str],
        *,
        output_dir: str | None = None,
        dpi: int = 144,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        output_dir_path = Path(output_dir) if output_dir is not None else None
        for pdf_path in file_path_list:
            path = Path(pdf_path)
            image_dir = (output_dir_path / f"{path.stem}_pages") if output_dir_path is not None else (path.parent / f"{path.stem}_pages")
            with fitz.open(path) as document:
                page_total = len(document)
                image_paths = [str(image_dir / f"{path.stem}_page_{index:04d}.png") for index in range(1, page_total + 1)]
                image_dir.mkdir(parents=True, exist_ok=True)
                for index in range(1, page_total + 1):
                    page = document.load_page(index - 1)
                    pixmap = page.get_pixmap(matrix=matrix)
                    pixmap.save(str(image_dir / f"{path.stem}_page_{index:04d}.png"))
            results.append({"source_path": str(path), "image_dir": str(image_dir), "image_paths": image_paths})
        return results

class AnalyzeOfficeUtil:
    @classmethod
    async def analyze_office_files(
        cls,
        llm_client: AbstractChatClient,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        office_contents: list[ChatContent] = []
        file_util_llm_messages = FileUtilLLMMessages(llm_client)
        for file_path in file_path_list:
            pdf_content = file_util_llm_messages.create_office_content_from_file(
                file_path, detail=detail
            )
            office_contents.extend(pdf_content)

        prompt_content = file_util_llm_messages.create_text_content(text=prompt)

        chat_message = ChatMessage(role="user", content=[prompt_content] + office_contents)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        response: ChatResponse = await llm_client.chat(chat_request)
        return response

class AnalyzeFileUtil:

    @classmethod
    async def analyze_documents_data(
        cls,
        llm_client: AbstractChatClient,
        document_type_list: list[FileUtilDocument],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        file_util_llm_messages = FileUtilLLMMessages(llm_client)
        content_list = []
        for document_type in document_type_list:
            contents = file_util_llm_messages.create_multi_format_content(
                document_type, detail=detail
            )
            content_list.extend(contents)

        prompt_content = file_util_llm_messages.create_text_content(text=prompt)
        chat_message = ChatMessage(role="user", content=[prompt_content] + content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        return chat_response

    @classmethod
    async def analyze_files(
        cls,
        llm_client: AbstractChatClient,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        content_list = []
        skipped_files: list[str] = []
        file_util_llm_messages = FileUtilLLMMessages(llm_client)
        for file_path in file_path_list:
            try:
                contents = file_util_llm_messages.create_multi_format_contents_from_file(
                    file_path, detail=detail
                )
            except ValueError as exc:
                if "Unsupported document type" not in str(exc):
                    raise
                skipped_files.append(file_path)
                logger.info("FILE_ANALYZE_SKIP unsupported=%s", file_path)
                continue
            content_list.extend(contents)

        if not content_list:
            raise ValueError(
                "No supported files were found for analyze_files. "
                f"skipped={len(skipped_files)}"
            )

        if skipped_files:
            logger.info("FILE_ANALYZE_SKIPPED_COUNT count=%d", len(skipped_files))

        prompt_content = llm_client.get_message_factory().create_text_content(text=prompt)
        chat_message = ChatMessage(role="user", content=[prompt_content] + content_list)
        chat_request: ChatRequest = ChatRequest(
            chat_history=ChatHistory(messages=[chat_message]), chat_request_context=None
        )
        chat_response: ChatResponse = await llm_client.chat(chat_request)
        return chat_response


class AnalyzeLogUtil:

    @staticmethod
    def _coerce_json_object(raw_text: str) -> dict[str, Any]:
        candidate = (raw_text or "").strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        return payload

    @classmethod
    def search_log_with_pattern(
        cls,
        text: str,
        infer_log_format_data: InferLogFormatData,
    ) -> list[LogSearchMatchData]:
        lines = text.splitlines()

        compiled = re.compile(infer_log_format_data.header_pattern)
        timestamp_compiled = re.compile(infer_log_format_data.timestamp_format) if infer_log_format_data.timestamp_format else None

        matches: list[LogSearchMatchData] = []
        for line_number, line in enumerate(lines, start=1):
            if compiled.search(line):
                if timestamp_compiled is None:
                    continue
                timestamp_match = timestamp_compiled.search(line)
                if timestamp_match is None:
                    continue
                # datetimeに変換
                iso_timestamp = datetime.strptime(timestamp_match.group(0), infer_log_format_data.timestamp_format) 
                matches.append(
                    LogSearchMatchData(
                        line_number=line_number, 
                        line=line, 
                        timestamp=iso_timestamp
                    )
                )
        return matches

    @classmethod
    async def infer_log_header_pattern(
        cls,
        llm_client: AbstractChatClient,
        file_path: str,
        sample_line_limit: int = 100,
        additional_instructions: str | None = None,
    ) -> InferLogFormatData:
        path = Path(file_path).expanduser().resolve()
        sample_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:sample_line_limit]
        if not sample_lines:
            inferred = InferLogFormatData(
                header_pattern="",
                format_description="",
                timestamp_format="",
                confidence=0.0,
                reason="ログファイルが空です。",
            )
            return inferred

        extra_instructions = (additional_instructions or "").strip()
        prompt = (
            "You analyze log headers. Return JSON only without markdown. "
            "Keys: header_pattern, format_description, timestamp_format, confidence, reason. "
            "header_pattern must match the beginning of each record header line and must include a named capture group (?P<timestamp>...) for the timestamp. "
            "format_description must concisely describe the likely log family or structure, such as log4j plus Java stack trace, syslog-style logs, XML-based Windows Event Log export, or unknown. "
            "Assume a valid log file always has a timestamp on each record header line. "
            "If you cannot identify timestamp-bearing record headers, treat the format as unknown and return an empty header_pattern. "
            "Some log records may span multiple lines, such as a timestamped header followed by Java stack trace lines. "
            "For multi-line logs, treat one record as starting at a line that contains the timestamped header and ending immediately before the next line that contains a timestamped header. "
            "When inferring header_pattern, match only the first line of each record, not stack trace continuation lines. "
            "timestamp_format must be datetime.strptime-compatible when possible.\n\n"
            f"file_path: {path}\n"
            f"sample_line_limit: {sample_line_limit}\n"
            + (f"additional_instructions:\n{extra_instructions}\n\n" if extra_instructions else "")
            + "sample_lines:\n"
            + "\n".join(f"{index + 1:03d}: {line}" for index, line in enumerate(sample_lines))
        )
        raw_response = await llm_client.simple_chat(prompt)
        parsed = cls._coerce_json_object(raw_response)
        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        inferred = InferLogFormatData(
            header_pattern=str(parsed.get("header_pattern") or ""),
            format_description=str(parsed.get("format_description") or ""),
            timestamp_format=str(parsed.get("timestamp_format") or ""),
            confidence=confidence,
            reason=str(parsed.get("reason") or ""),
        )
        return inferred

    @classmethod
    async def extract_time_range_from_logfile(
        cls,
        llm_client: AbstractChatClient,
        file_path: str,
        output_path: str,
        start_time: datetime,
        end_time: datetime,
        sample_line_limit: int = 100,
        search_terms: list[str] | None = None,
        additional_instructions: str | None = None,
    ) -> None:
        path = Path(file_path).expanduser().resolve()
        text = path.read_text(encoding="utf-8", errors="ignore")
        inferred = await cls.infer_log_header_pattern(
            llm_client,
            str(path),
            sample_line_limit,
            additional_instructions,
        )
        search_results = cls.search_log_with_pattern(
            text,
            inferred
        )
        # 指定された時間範囲でフィルタリング
        filtered_matches = LogSearchMatchData.filter_by_timestamp_range(search_results, start_time, end_time)
        # フィルタリングされた最初の行と最後の行のタイムスタンプを抽出 
        # filitered_matchesが空の場合はNoneをセット
        # filtered_matchesが1つの場合はその行のタイムスタンプを
        # first_timestampとlast_timestampの両方にセット
        if filtered_matches:
            first_timestamp = filtered_matches[0].timestamp
            last_timestamp = filtered_matches[-1].timestamp
        else:
            first_timestamp = None
            last_timestamp = None

        # textから指定された時間範囲に該当する行を抜き出す
        extracted_lines = []
        for match in search_results:
            if match.timestamp and start_time <= match.timestamp <= end_time:
                extracted_lines.append(match.line)

        # 抽出された行をoutput_path(ディレクトリ)に書き出す
        # ファイル名は元のファイル名+_+開始時間_終了時間+<.元のファイルの拡張子>とする
        output_filename = f"{path.stem}_{start_time.strftime('%Y%m%d%H%M%S')}_{end_time.strftime('%Y%m%d%H%M%S')}{path.suffix}"
        output_path_obj = Path(output_path).expanduser().resolve() / output_filename
        
        output_path_obj.write_text("\n".join(extracted_lines), encoding="utf-8")

