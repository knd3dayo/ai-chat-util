from __future__ import annotations

import time
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

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
    ExtractLogTimeRangeData,
    FileUtilDocument,
    InferLogFormatData,
    LogSearchMatchData,
)
from ai_chat_util.core.common.config.runtime import get_runtime_config
import ai_chat_util.core.log.log_settings as log_settings
import fitz  # PyMuPDF
from ai_chat_util.util.analyze_file_util.file_util_llm_messages import FileUtilLLMMessages
from ai_chat_util.util.analyze_file_util.office2pdf import (
    LibreOfficeExecOffice2PDFUtil,
    LibreOfficeUnoOffice2PDFUtil,
    Pywin32Office2PDFUtil,
    _build_default_output_path,
)

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
    def convert_office_files_to_pdf(
        cls,
        file_path_list: list[str],
        output_dir: str | None = None,
        libreoffice_path: str | None = None,
    ) -> list[dict[str, str]]:
        planned: list[dict[str, str]] = []
        for office_path in file_path_list:
            source_path = Path(office_path)
            pdf_path = source_path.with_suffix(".pdf") if output_dir is None else Path(output_dir) / source_path.with_suffix(".pdf").name
            planned.append({"source_path": office_path, "pdf_path": str(pdf_path)})


        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        results: list[dict[str, str]] = []
        config = get_runtime_config()
        for office_path, planned_item in zip(file_path_list, planned):
            resolved_output_path = output_dir if output_dir is not None else _build_default_output_path(office_path)
            if config.office2pdf.method == LibreOfficeExecOffice2PDFUtil.METHOD_NAME:
                pdf_path = LibreOfficeExecOffice2PDFUtil.create_pdf_from_document_file(
                    input_path=office_path,
                    output_path=resolved_output_path,
                    libreoffice_path=(libreoffice_path or config.office2pdf.libreoffice_exec.libreoffice_path),
                )
            elif config.office2pdf.method == LibreOfficeUnoOffice2PDFUtil.METHOD_NAME:
                pdf_path = LibreOfficeUnoOffice2PDFUtil.create_pdf_from_document_file(
                    input_path=office_path,
                    output_path=resolved_output_path,
                    api_url=config.office2pdf.libreoffice_uno.api_url,
                )
            elif config.office2pdf.method == Pywin32Office2PDFUtil.METHOD_NAME:
                pdf_path = Pywin32Office2PDFUtil.create_pdf_from_document_file(
                    input_path=office_path,
                    output_path=resolved_output_path,
                    office_path=config.office2pdf.pywin32.office_path,
                )
            else:
                raise RuntimeError(f"Unsupported Office2PDF method: {config.office2pdf.method}")
            results.append({"source_path": planned_item["source_path"], "pdf_path": str(pdf_path)})
        return results

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
            with open(file_path, "rb") as f:
                office_data = f.read()
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
    @classmethod
    def search_log_with_pattern(
        cls,
        text: str,
        infer_log_format_data: InferLogFormatData,
    ) -> list[LogSearchMatchData]:
        header_pattern = (infer_log_format_data.header_pattern or "").strip()
        timestamp_format = (infer_log_format_data.timestamp_format or "").strip()
        if not header_pattern or not timestamp_format:
            return []

        compiled = re.compile(header_pattern)
        lines = text.splitlines()
        matches: list[LogSearchMatchData] = []
        for line_number, line in enumerate(lines, start=1):
            match = compiled.search(line)
            if match is None:
                continue
            timestamp_text = match.groupdict().get("timestamp")
            if not timestamp_text:
                continue
            try:
                parsed_timestamp = datetime.strptime(timestamp_text, timestamp_format)
            except ValueError:
                continue
            matches.append(
                LogSearchMatchData(
                    line_number=line_number,
                    line=line,
                    timestamp=parsed_timestamp,
                )
            )
        return matches

    @classmethod
    def _extract_records_in_time_range(
        cls,
        lines: list[str],
        header_matches: list[LogSearchMatchData],
        start_time: datetime,
        end_time: datetime,
    ) -> tuple[list[str], list[LogSearchMatchData]]:
        filtered_matches = LogSearchMatchData.filter_by_timestamp_range(
            header_matches,
            start_time,
            end_time,
        )
        if not filtered_matches:
            return [], []

        filtered_line_numbers = {match.line_number for match in filtered_matches}
        extracted_lines: list[str] = []
        for index, match in enumerate(header_matches):
            if match.line_number not in filtered_line_numbers:
                continue
            start_index = match.line_number - 1
            next_index = (
                header_matches[index + 1].line_number - 1
                if index + 1 < len(header_matches)
                else len(lines)
            )
            extracted_lines.extend(lines[start_index:next_index])
        return extracted_lines, filtered_matches

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
        header_pattern = inferred.header_pattern
        has_named_timestamp = False
        if header_pattern.strip():
            try:
                has_named_timestamp = "timestamp" in re.compile(header_pattern).groupindex
            except re.error:
                has_named_timestamp = False
        return InferLogFormatData(
            header_pattern=inferred.header_pattern,
            format_description=inferred.format_description,
            timestamp_format=inferred.timestamp_format,
            confidence=inferred.confidence,
            reason=inferred.reason,
        )

    @classmethod
    async def extract_time_range_from_logfile(
        cls,
        llm_client: AbstractChatClient,
        file_path: str,
        output_path: str,
        start_time: datetime,
        end_time: datetime,
        sample_line_limit: int = 100,
        additional_instructions: str | None = None,
    ) -> ExtractLogTimeRangeData:
        path = Path(file_path).expanduser().resolve()
        output_dir = Path(output_path).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        inferred = await cls.infer_log_header_pattern(
            llm_client,
            str(path),
            sample_line_limit,
            additional_instructions,
        )
        header_matches = cls.search_log_with_pattern(text, inferred)
        if not header_matches:
            raise ValueError(
                "Could not infer a usable log header pattern and timestamp format for time-range extraction."
            )

        extracted_lines, filtered_matches = cls._extract_records_in_time_range(
            lines,
            header_matches,
            start_time,
            end_time,
        )

        output_filename = (
            f"{path.stem}_{start_time.strftime('%Y%m%d%H%M%S')}_"
            f"{end_time.strftime('%Y%m%d%H%M%S')}{path.suffix}"
        )
        output_file = output_dir / output_filename
        output_file.write_text("\n".join(extracted_lines), encoding="utf-8")

        first_timestamp = filtered_matches[0].timestamp if filtered_matches else None
        last_timestamp = filtered_matches[-1].timestamp if filtered_matches else None
        return ExtractLogTimeRangeData(
            file_path=str(path),
            output_path=str(output_file),
            header_pattern=inferred.header_pattern,
            timestamp_format=inferred.timestamp_format,
            matched_record_count=len(filtered_matches),
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
        )

