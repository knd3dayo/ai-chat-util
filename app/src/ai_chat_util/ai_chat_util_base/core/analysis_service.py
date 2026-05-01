from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import fitz

from ai_chat_util.ai_chat_util_base.chat.core import AbstractChatClient
from ai_chat_util.common.config.runtime import get_runtime_config
from ai_chat_util.ai_chat_util_base.chat.model import ChatResponse
from ai_chat_util.ai_chat_util_base.analyze_file_util.model import FileUtilDocument
from ai_chat_util.ai_chat_util_base.analyze_file_util.util.file_path_resolver import resolve_existing_path
from ai_chat_util.ai_chat_util_base.analyze_file_util.util.office2pdf import Office2PDFUtil

import ai_chat_util.log.log_settings as log_settings

from ..analyze_pdf_util.util import AnalyzePDFUtil


logger = log_settings.getLogger(__name__)


class AnalysisService:
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
    def summarize_path_basenames(paths: list[str], *, limit: int = 5) -> str:
        names: list[str] = []
        for path in paths[:limit]:
            try:
                names.append(path.rsplit("/", 1)[-1])
            except Exception:
                names.append(str(path))
        more = "" if len(paths) <= limit else f" (+{len(paths) - limit} more)"
        return f"[{', '.join(names)}]{more}"

    @staticmethod
    def prompt_len(prompt: str) -> int:
        return len((prompt or "").strip())

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
    def _is_supported_analysis_file(cls, path: Path) -> bool:
        suffix = path.suffix.lower()
        return (
            suffix in cls._SUPPORTED_TEXT_SUFFIXES
            or suffix in cls._SUPPORTED_IMAGE_SUFFIXES
            or suffix in cls._SUPPORTED_DOCUMENT_SUFFIXES
        )

    @staticmethod
    def _is_hidden_path(path: Path) -> bool:
        return any(part.startswith(".") for part in path.parts)

    @classmethod
    def _expand_directory_analysis_targets(cls, directory_path: str) -> list[str]:
        directory = Path(directory_path)
        resolved_files: list[str] = []

        for candidate in sorted(directory.rglob("*")):
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue

            if cls._is_hidden_path(candidate.relative_to(directory)):
                continue

            if not cls._is_supported_analysis_file(candidate):
                continue

            try:
                if candidate.stat().st_size > cls._MAX_DIRECTORY_ANALYSIS_FILE_BYTES:
                    continue
            except OSError:
                continue

            resolved_files.append(str(candidate.resolve()))
            if len(resolved_files) >= cls._MAX_DIRECTORY_ANALYSIS_FILES:
                break

        if resolved_files:
            return resolved_files

        raise FileNotFoundError(f"No supported analysis files found in directory: {directory.resolve()}")

    @classmethod
    def resolve_existing_file_paths(cls, file_path_list: list[str]) -> list[str]:
        runtime_config = get_runtime_config()
        resolved: list[str] = []
        seen: set[str] = set()
        for file_path in file_path_list:
            result = resolve_existing_path(
                file_path,
                working_directory=runtime_config.mcp.working_directory,
                allow_directory=True,
            )

            candidate_paths = [result.resolved_path]
            if result.path_kind == "directory":
                candidate_paths = cls._expand_directory_analysis_targets(result.resolved_path)

            for candidate_path in candidate_paths:
                if candidate_path in seen:
                    continue
                seen.add(candidate_path)
                resolved.append(candidate_path)
        return resolved

    @staticmethod
    def tool_timeout_seconds() -> float:
        runtime_config = get_runtime_config()
        tool_timeout_cfg = getattr(runtime_config.features, "mcp_tool_timeout_seconds", None)
        try:
            timeout = (
                float(tool_timeout_cfg)
                if tool_timeout_cfg is not None
                else float(runtime_config.llm.timeout_seconds)
            )
        except (TypeError, ValueError):
            timeout = float(runtime_config.llm.timeout_seconds)
        if timeout <= 0:
            timeout = float(runtime_config.llm.timeout_seconds)
        return timeout

    @staticmethod
    def tool_timeout_retries() -> int:
        runtime_config = get_runtime_config()
        try:
            retries_raw = int(getattr(runtime_config.features, "mcp_tool_timeout_retries", 1) or 0)
        except (TypeError, ValueError):
            retries_raw = 1
        return max(0, min(5, retries_raw))

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
        *,
        resolve_paths: bool = True,
    ) -> str:
        target_path = cls.resolve_existing_file_paths([file_path])[0] if resolve_paths else str(Path(file_path).expanduser().resolve())
        path = Path(target_path)
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
        resolve_paths: bool = True,
    ) -> str:
        target_path = cls.resolve_existing_file_paths([file_path])[0] if resolve_paths else str(Path(file_path).expanduser().resolve())
        path = Path(target_path)
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
        resolve_paths: bool = True,
    ) -> str:
        target_path = cls.resolve_existing_file_paths([file_path])[0] if resolve_paths else str(Path(file_path).expanduser().resolve())
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

    @classmethod
    def convert_office_files_to_pdf(
        cls,
        file_path_list: list[str],
        *,
        output_dir: Path | None = None,
        dry_run: bool = False,
        libreoffice_path: str | None = None,
        resolve_paths: bool = True,
    ) -> list[dict[str, str]]:
        resolved_paths = cls.resolve_existing_file_paths(file_path_list) if resolve_paths else file_path_list
        planned: list[dict[str, str]] = []
        for office_path in resolved_paths:
            source_path = Path(office_path)
            pdf_path = source_path.with_suffix(".pdf") if output_dir is None else output_dir / source_path.with_suffix(".pdf").name
            planned.append({"source_path": office_path, "pdf_path": str(pdf_path)})
        if dry_run:
            return planned

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, str]] = []
        for office_path, planned_item in zip(resolved_paths, planned):
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
        output_dir: Path | None = None,
        dry_run: bool = False,
        dpi: int = 144,
        resolve_paths: bool = True,
    ) -> list[dict[str, Any]]:
        resolved_paths = cls.resolve_existing_file_paths(file_path_list) if resolve_paths else file_path_list
        results: list[dict[str, Any]] = []
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        for pdf_path in resolved_paths:
            path = Path(pdf_path)
            image_dir = (output_dir / f"{path.stem}_pages") if output_dir is not None else (path.parent / f"{path.stem}_pages")
            with fitz.open(path) as document:
                page_total = len(document)
                image_paths = [str(image_dir / f"{path.stem}_page_{index:04d}.png") for index in range(1, page_total + 1)]
                if not dry_run:
                    image_dir.mkdir(parents=True, exist_ok=True)
                    for index in range(1, page_total + 1):
                        page = document.load_page(index - 1)
                        pixmap = page.get_pixmap(matrix=matrix)
                        pixmap.save(str(image_dir / f"{path.stem}_page_{index:04d}.png"))
            results.append({"source_path": str(path), "image_dir": str(image_dir), "image_paths": image_paths})
        return results

    @classmethod
    async def run_analysis_tool(
        cls,
        *,
        tool_name: str,
        prompt: str,
        detail: str,
        input_count: int,
        input_kind: str,
        operation: Callable[[], Awaitable[ChatResponse]],
        input_summary: str | None = None,
        use_timeout_retry: bool = False,
        stringify_errors: bool = False,
    ) -> str:
        started = time.perf_counter()
        logger.info(
            "MCP_TOOL_START tool=%s %s=%d detail=%s prompt_len=%d",
            tool_name,
            input_kind,
            input_count,
            detail,
            cls.prompt_len(prompt),
        )
        if input_summary is not None:
            logger.info(
                "MCP_TOOL_INPUT tool=%s %s=%d basenames=%s",
                tool_name,
                input_kind,
                input_count,
                input_summary,
            )

        try:
            if not use_timeout_retry:
                response = await operation()
                return response.output

            timeout = cls.tool_timeout_seconds()
            retries = cls.tool_timeout_retries()
            last_err: Exception | None = None
            for attempt in range(1, retries + 2):
                try:
                    response = await asyncio.wait_for(operation(), timeout=timeout)
                    return response.output
                except asyncio.TimeoutError as exc:
                    last_err = exc
                    logger.warning(
                        "MCP_TOOL_TIMEOUT tool=%s attempt=%s/%s timeout=%ss",
                        tool_name,
                        attempt,
                        retries + 1,
                        timeout,
                    )
                    if attempt <= retries:
                        continue
                    break

            if last_err is not None:
                return (
                    f"ERROR: {tool_name} timed out. "
                    f"timeout={timeout}s retries={retries}. "
                    "同一入力での無限再試行を防ぐため中断しました。"
                )
            return f"ERROR: {tool_name} failed (unknown error)"
        except Exception as exc:
            logger.exception("MCP_TOOL_ERR tool=%s", tool_name)
            if stringify_errors:
                return f"ERROR: {tool_name} failed: {type(exc).__name__}: {str(exc).strip()}"
            raise
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "MCP_TOOL_END tool=%s elapsed_ms=%s",
                tool_name,
                elapsed_ms,
            )

    @classmethod
    async def analyze_image_files(
        cls,
        llm_client: AbstractChatClient,
        file_list: list[str],
        prompt: str,
        detail: str,
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        target_paths = cls.resolve_existing_file_paths(file_list) if resolve_paths else file_list
        return await AnalyzePDFUtil.analyze_image_files(llm_client, target_paths, prompt, detail)

    @classmethod
    async def analyze_pdf_files(
        cls,
        llm_client: AbstractChatClient,
        file_list: list[str],
        prompt: str,
        detail: str = "auto",
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        target_paths = cls.resolve_existing_file_paths(file_list) if resolve_paths else file_list
        return await AnalyzePDFUtil.analyze_pdf_files(llm_client, target_paths, prompt, detail)

    @classmethod
    async def analyze_office_files(
        cls,
        llm_client: AbstractChatClient,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        target_paths = cls.resolve_existing_file_paths(file_path_list) if resolve_paths else file_path_list
        return await AnalyzePDFUtil.analyze_office_files(llm_client, target_paths, prompt, detail)

    @classmethod
    async def analyze_files(
        cls,
        llm_client: AbstractChatClient,
        file_path_list: list[str],
        prompt: str,
        detail: str = "auto",
        *,
        resolve_paths: bool = True,
    ) -> ChatResponse:
        target_paths = cls.resolve_existing_file_paths(file_path_list) if resolve_paths else file_path_list
        return await AnalyzePDFUtil.analyze_files(llm_client, target_paths, prompt, detail)

    @classmethod
    async def analyze_documents_data(
        cls,
        llm_client: AbstractChatClient,
        document_type_list: list[FileUtilDocument],
        prompt: str,
        detail: str = "auto",
    ) -> ChatResponse:
        return await AnalyzePDFUtil.analyze_documents_data(
            llm_client,
            document_type_list,
            prompt,
            detail,
        )