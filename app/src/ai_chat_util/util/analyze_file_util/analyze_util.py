from __future__ import annotations

import time
import json
import re
from pathlib import Path
from datetime import datetime
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

from ai_chat_util.core.analysis.model import FileUtilDocument
import ai_chat_util.core.log.log_settings as log_settings
from .office2pdf import Office2PDFUtil
import fitz  # PyMuPDF
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
    def convert_office_files_to_pdf_by_local_libreoffice(
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
        for office_path, planned_item in zip(file_path_list, planned):
            pdf_path = Office2PDFUtil.create_pdf_from_document_file(
                input_path=office_path,
                output_path=output_dir,
                configured_libreoffice_path=libreoffice_path,
            )
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
    _MAX_DIRECTORY_ANALYSIS_FILES = 20
    _MAX_DIRECTORY_ANALYSIS_FILE_BYTES = 200_000
    _SUPPORTED_TEXT_SUFFIXES = {
        ".bat",
        ".c",
        ".cc",
        ".cfg",
        ".conf",
        ".cpp",
        ".cs",
        ".css",
        ".csv",
        ".go",
        ".h",
        ".hpp",
        ".htm",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".log",
        ".md",
        ".markdown",
        ".php",
        ".ps1",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsv",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
    _SUPPORTED_IMAGE_SUFFIXES = {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".webp",
    }
    _SUPPORTED_DOCUMENT_SUFFIXES = {
        ".docx",
        ".pdf",
        ".pptx",
        ".xlsx",
    }
    _LOG_TIMESTAMP_CANDIDATES = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y%m%d %H:%M:%S",
    )

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

    @staticmethod
    def detect_log_format_from_lines(lines: list[str]) -> dict[str, Any]:
        joined = "\n".join(lines)
        format_scores = {
            "syslog": sum(1 for line in lines if re.match(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+", line)),
            "log4j": sum(1 for line in lines if re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d{3,6})?\s+(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\b", line)),
            "iso8601": sum(1 for line in lines if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", line)),
            "java_stacktrace": sum(1 for line in lines if re.match(r"^\s+at\s+[\w.$_]+\([^\n]+:\d+\)$", line)),
        }
        primary_format = max(format_scores, key=lambda item: format_scores[item]) if any(format_scores.values()) else "unknown"
        has_java_stacktrace = bool(re.search(r"^\s+at\s+[\w.$_]+\([^\n]+:\d+\)$", joined, flags=re.MULTILINE))

        timestamp_regex = r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d{3,6})?"
        if primary_format == "syslog":
            timestamp_regex = r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
        elif primary_format == "iso8601":
            timestamp_regex = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"

        return {
            "primary_format": primary_format,
            "format_scores": format_scores,
            "has_java_stacktrace": has_java_stacktrace,
            "generated_patterns": {
                "timestamp": timestamp_regex,
                "severity": r"\b(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\b",
                "java_exception": r"\b[\w.$]+(?:Exception|Error)\b",
                "java_stack_frame": r"^\s+at\s+[\w.$_]+\([^\n]+:\d+\)$",
            },
        }

    @classmethod
    def search_log_with_patterns(
        cls,
        text: str,
        *,
        generated_patterns: dict[str, str],
        search_terms: list[str] | None,
        match_limit: int,
    ) -> dict[str, list[dict[str, Any]]]:
        results: dict[str, list[dict[str, Any]]] = {}
        lines = text.splitlines()
        named_patterns = {
            "severity": generated_patterns["severity"],
            "java_exception": generated_patterns["java_exception"],
        }
        if search_terms:
            named_patterns["search_terms"] = "|".join(re.escape(term) for term in search_terms if term)

        for name, pattern in named_patterns.items():
            if not pattern:
                continue
            compiled = re.compile(pattern)
            matches: list[dict[str, Any]] = []
            for line_number, line in enumerate(lines, start=1):
                if compiled.search(line):
                    matches.append({"line_number": line_number, "line": line})
                    if len(matches) >= match_limit:
                        break
            results[name] = matches
        return results

    @classmethod
    def parse_timestamp_value(cls, raw_value: str, time_format: str | None = None) -> datetime | None:
        value = raw_value.strip()
        if not value:
            return None
        if time_format:
            try:
                return datetime.strptime(value, time_format)
            except ValueError:
                pass

        normalized = value.replace(",", ".")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass

        for candidate in cls._LOG_TIMESTAMP_CANDIDATES:
            try:
                return datetime.strptime(normalized, candidate)
            except ValueError:
                continue
        return None

    @staticmethod
    def sanitize_output_component(value: str) -> str:
        collapsed = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
        sanitized = collapsed.strip("-._")
        return sanitized or "value"

    @classmethod
    def extract_log_records_in_time_range(
        cls,
        text: str,
        *,
        header_pattern: str,
        timestamp_start: int,
        timestamp_end: int,
        range_start: str,
        range_end: str,
        time_format: str | None = None,
    ) -> dict[str, Any]:
        if timestamp_start < 0 or timestamp_end <= timestamp_start:
            raise ValueError("timestamp_start and timestamp_end must define a valid non-empty slice")

        start_time = cls.parse_timestamp_value(range_start, time_format)
        end_time = cls.parse_timestamp_value(range_end, time_format)
        if start_time is None or end_time is None:
            raise ValueError("range_start and range_end must be parseable timestamps")
        if start_time > end_time:
            raise ValueError("range_start must be earlier than or equal to range_end")

        compiled = re.compile(header_pattern)
        lines = text.splitlines()
        records: list[dict[str, Any]] = []
        current_record: dict[str, Any] | None = None
        unmatched_preamble: list[str] = []

        for line_number, line in enumerate(lines, start=1):
            if compiled.search(line):
                if current_record is not None:
                    records.append(current_record)
                current_record = {
                    "start_line": line_number,
                    "lines": [line],
                    "header_line": line,
                }
                continue
            if current_record is not None:
                current_record["lines"].append(line)
            else:
                unmatched_preamble.append(line)

        if current_record is not None:
            records.append(current_record)

        extracted_records: list[dict[str, Any]] = []
        parse_failures: list[dict[str, Any]] = []
        for record in records:
            header_line = str(record["header_line"])
            timestamp_text = header_line[timestamp_start:timestamp_end]
            parsed_timestamp = cls.parse_timestamp_value(timestamp_text, time_format)
            if parsed_timestamp is None:
                parse_failures.append(
                    {
                        "start_line": record["start_line"],
                        "timestamp_text": timestamp_text,
                        "header_line": header_line,
                    }
                )
                continue
            if start_time <= parsed_timestamp <= end_time:
                extracted_records.append(
                    {
                        "start_line": record["start_line"],
                        "end_line": record["start_line"] + len(record["lines"]) - 1,
                        "timestamp": parsed_timestamp.isoformat(),
                        "text": "\n".join(record["lines"]),
                    }
                )

        return {
            "total_lines": len(lines),
            "matched_record_count": len(extracted_records),
            "record_count": len(records),
            "parse_failure_count": len(parse_failures),
            "parse_failures": parse_failures[:10],
            "records": extracted_records,
            "unmatched_preamble_line_count": len(unmatched_preamble),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
        }

    @classmethod
    async def infer_log_header_pattern(
        cls,
        llm_client: AbstractChatClient,
        file_path: str,
        sample_line_limit: int = 100,
    ) -> str:
        path = Path(file_path).expanduser().resolve()
        sample_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:sample_line_limit]
        if not sample_lines:
            return json.dumps(
                {
                    "status": "unavailable",
                    "file_path": str(path),
                    "sample_line_limit": sample_line_limit,
                    "sample_preview": [],
                    "header_pattern": "",
                    "timestamp_start": -1,
                    "timestamp_end": -1,
                    "timestamp_format": "",
                    "confidence": 0.0,
                    "reason": "ログファイルが空です。",
                },
                ensure_ascii=False,
            )

        prompt = (
            "You analyze log headers. Return JSON only without markdown. "
            "Keys: header_pattern, timestamp_start, timestamp_end, timestamp_format, confidence, reason. "
            "header_pattern must match the beginning of each record header line. "
            "timestamp_start and timestamp_end are zero-based slice offsets for the timestamp text on the header line. "
            "timestamp_format must be datetime.strptime-compatible when possible.\n\n"
            f"file_path: {path}\n"
            f"sample_line_limit: {sample_line_limit}\n"
            "sample_lines:\n"
            + "\n".join(f"{index + 1:03d}: {line}" for index, line in enumerate(sample_lines))
        )
        raw_response = await llm_client.simple_chat(prompt)
        parsed = cls._coerce_json_object(raw_response)
        timestamp_start = int(parsed.get("timestamp_start", -1) or -1)
        timestamp_end = int(parsed.get("timestamp_end", -1) or -1)
        payload = {
            "status": "matched"
            if str(parsed.get("header_pattern") or "").strip() and timestamp_start >= 0 and timestamp_end > timestamp_start
            else "unavailable",
            "file_path": str(path),
            "sample_line_limit": sample_line_limit,
            "sample_preview": sample_lines[:10],
            "header_pattern": str(parsed.get("header_pattern") or ""),
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "timestamp_format": str(parsed.get("timestamp_format") or ""),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "reason": str(parsed.get("reason") or ""),
        }
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def detect_log_format_and_search_from_file(
        cls,
        file_path: str,
        *,
        search_terms: list[str] | None = None,
        sample_line_limit: int = 100,
        match_limit: int = 50,
    ) -> str:
        path = Path(file_path).expanduser().resolve()
        text = path.read_text(encoding="utf-8", errors="ignore")
        sample_lines = text.splitlines()[:sample_line_limit]
        detection = cls.detect_log_format_from_lines(sample_lines)
        search_results = cls.search_log_with_patterns(
            text,
            generated_patterns=detection["generated_patterns"],
            search_terms=search_terms,
            match_limit=match_limit,
        )
        return json.dumps(
            {
                "file_path": str(path),
                "sample_line_limit": sample_line_limit,
                "sample_preview": sample_lines[:10],
                "detected_format": detection["primary_format"],
                "format_scores": detection["format_scores"],
                "has_java_stacktrace": detection["has_java_stacktrace"],
                "generated_patterns": detection["generated_patterns"],
                "search_terms": search_terms or [],
                "search_results": search_results,
            },
            ensure_ascii=False,
        )

    @classmethod
    def extract_log_time_range_to_file(
        cls,
        *,
        file_path: str,
        workspace_path: str,
        artifacts_subdir: str,
        header_pattern: str,
        timestamp_start: int,
        timestamp_end: int,
        range_start: str,
        range_end: str,
        time_format: str | None = None,
        output_subdir: str = "log_extracts",
        output_filename: str | None = None,
    ) -> str:
        target_path = Path(file_path).expanduser().resolve()
        source_path = Path(target_path)
        workspace_root = Path(workspace_path).expanduser().resolve()
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        extraction = cls.extract_log_records_in_time_range(
            text,
            header_pattern=header_pattern,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            range_start=range_start,
            range_end=range_end,
            time_format=time_format,
        )

        artifacts_dir = workspace_root / artifacts_subdir / output_subdir.strip("/")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        derived_filename = output_filename or (
            f"{source_path.stem}_{cls.sanitize_output_component(range_start)}_{cls.sanitize_output_component(range_end)}{source_path.suffix or '.log'}"
        )
        output_path = artifacts_dir / derived_filename
        rendered = "\n\n".join(str(record["text"]) for record in extraction["records"])
        output_path.write_text(rendered + ("\n" if rendered else ""), encoding="utf-8")

        return json.dumps(
            {
                "status": "matched" if extraction["matched_record_count"] else "unavailable",
                "file_path": str(source_path),
                "workspace_path": str(workspace_root),
                "output_path": str(output_path),
                "header_pattern": header_pattern,
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "time_format": time_format,
                "range_start": range_start,
                "range_end": range_end,
                **extraction,
            },
            ensure_ascii=False,
        )
