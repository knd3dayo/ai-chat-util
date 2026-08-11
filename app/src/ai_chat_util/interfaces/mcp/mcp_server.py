import argparse
import asyncio
import inspect
import time
from functools import wraps
from typing import Callable

from fastmcp import Context, FastMCP

from ai_chat_util.core.analysis.analyze_file import (
    analyze_documents_data,
    analyze_files,
)
from ai_chat_util.core.analysis.analyze_image import (
    analyze_image_files,
    analyze_image_urls,
)
from ai_chat_util.core.analysis.analyze_log import (
    extract_time_range_from_logfile,
    infer_log_header_pattern,
)
from ai_chat_util.core.analysis.analyze_office import (
    analyze_office_files,
    analyze_office_urls,
)
from ai_chat_util.core.analysis.analyze_office_xml import (
    analyze_office_xml_files,
)
from ai_chat_util.core.analysis.analyze_pdf import (
    analyze_pdf_files,
    analyze_pdf_urls,
    convert_office_files_to_pdf,
    convert_pdf_files_to_images,
)
from ai_chat_util.core.browser.browser_task import (
    run_browser_task,
    run_browser_task_with_output,
)
from ai_chat_util.core.chat import create_llm_client
from ai_chat_util.core.chat.batch_client import BatchClient
from ai_chat_util.core.chat.model import ChatRequest, ChatResponse
from ai_chat_util.core.common.config.runtime import (
    apply_logging_overrides,
    init_runtime,
)
from ai_chat_util.core.request_headers import (
    RequestHeaders,
    bind_current_request_headers,
)
from ai_chat_util.core.resource_app import get_loaded_config_info

from ...core.log import log_settings

logger = log_settings.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP server with specified mode")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "設定ファイル(ai-chat-util-config.yml)のパス。指定時は環境変数 AI_CHAT_UTIL_CONFIG にも反映し、"
            "後続処理に伝播します。未指定の場合は AI_CHAT_UTIL_CONFIG / カレント / プロジェクトルートの順で探索します。"
        ),
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["sse", "http", "stdio"],
        default="stdio",
        help=(
            "Transport mode: 'stdio' (default), 'sse', or 'http' (streamable-http)."
        ),
    )
    parser.add_argument(
        "-t",
        "--tools",
        type=str,
        default="",
        help=(
            "Comma-separated list of tool function names to load (e.g., 'run_chat,analyze_pdf_files'). "
            "If not specified, the default tools are loaded."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind host for sse/http",
    )
    parser.add_argument("-p", "--port", type=int, default=5001, help="Port number to run the server on. Default is 5001.")
    parser.add_argument("-v", "--log_level", type=str, default="", help="Log level to set for the server. Default is empty, which uses the default log level.")
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help=(
            "Log file path for the MCP server process. "
            "Use this when running in stdio mode to avoid mixing logs into stdout."
        ),
    )
    return parser.parse_args()


async def run_chat(chat_request: ChatRequest) -> ChatResponse:
    llm_client = create_llm_client()
    return await llm_client.chat(chat_request)


async def run_simple_chat(prompt: str) -> str:
    llm_client = create_llm_client()
    return await llm_client.simple_chat(prompt)


async def run_batch_chat(chat_requests: list[ChatRequest], concurrency: int = 5) -> list[ChatResponse]:
    llm_batch_client = BatchClient()
    rows = await llm_batch_client.run_batch_chat(chat_requests=chat_requests, concurrency=concurrency)
    return [response for _, response in rows]


async def run_batch_chat_from_excel(
    prompt: str,
    input_excel_path: str,
    output_excel_path: str = "output.xlsx",
    content_column: str = "content",
    file_path_column: str = "file_path",
    output_column: str = "output",
    detail: str = "auto",
    concurrency: int = 16,
) -> dict[str, str]:
    llm_batch_client = BatchClient()
    await llm_batch_client.run_batch_chat_from_excel(
        prompt=prompt,
        input_excel_path=input_excel_path,
        output_excel_path=output_excel_path,
        content_column=content_column,
        file_path_column=file_path_column,
        output_column=output_column,
        detail=detail,
        concurrency=concurrency,
    )
    return {"output_excel_path": output_excel_path}


def prepare_mcp(mcp: FastMCP, tools_option: str):
    def _summarize_mcp_args(tool_name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> dict[str, object]:
        return {
            "tool": tool_name,
            "arg_count": len(args),
            "kw_keys": sorted(str(key) for key in kwargs.keys()),
        }

    def header_aware_tool(mcp_instance: FastMCP, *, tool_name: str):
        def decorator(func: Callable[..., object]):
            is_async = inspect.iscoroutinefunction(func)

            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                context = kwargs.pop("context", None)
                headers_obj: RequestHeaders | None = None
                if isinstance(context, Context):
                    request_context = getattr(context, "request_context", None)
                    request = getattr(request_context, "request", None) if request_context else None
                    if request is not None:
                        headers = {str(k).lower(): str(v) for k, v in request.headers.items()}
                        headers_obj = RequestHeaders.from_mapping(headers)

                try:
                    logger.info(
                        "mcp.request %s",
                        {
                            **_summarize_mcp_args(tool_name, args, kwargs),
                            "trace_id": headers_obj.trace_id if headers_obj else None,
                        },
                    )
                except Exception:
                    pass

                with bind_current_request_headers(headers_obj):
                    try:
                        if is_async:
                            result = await func(*args, **kwargs)  # type: ignore
                        else:
                            result = func(*args, **kwargs)
                    except Exception:
                        dt_ms = int((time.perf_counter() - start) * 1000)
                        try:
                            logger.exception(
                                "mcp.error tool=%s dt_ms=%s trace_id=%s",
                                tool_name,
                                dt_ms,
                                headers_obj.trace_id if headers_obj else None,
                            )
                        except Exception:
                            pass
                        raise

                dt_ms = int((time.perf_counter() - start) * 1000)
                try:
                    logger.info(
                        "mcp.response tool=%s dt_ms=%s trace_id=%s result_type=%s",
                        tool_name,
                        dt_ms,
                        headers_obj.trace_id if headers_obj else None,
                        type(result).__name__,
                    )
                except Exception:
                    pass
                return result

            wrapper.__name__ = tool_name
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if "context" not in [param.name for param in params]:
                params.append(
                    inspect.Parameter(
                        "context",
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=Context,
                        default=None,
                    )
                )
            setattr(wrapper, "__signature__", sig.replace(parameters=params))
            annotations = dict(getattr(wrapper, "__annotations__", {}) or {})
            annotations.setdefault("context", Context)
            wrapper.__annotations__ = annotations
            return mcp_instance.tool()(wrapper)

        return decorator

    tool_registry: dict[str, Callable[..., object]] = {
        "analyze_image_files": analyze_image_files,
        "analyze_pdf_files": analyze_pdf_files,
        "analyze_office_files": analyze_office_files,
        "analyze_office_xml_files": analyze_office_xml_files,
        "analyze_files": analyze_files,
        "analyze_documents_data": analyze_documents_data,
        "analyze_image_urls": analyze_image_urls,
        "analyze_pdf_urls": analyze_pdf_urls,
        "analyze_office_urls": analyze_office_urls,
        "convert_office_files_to_pdf": convert_office_files_to_pdf,
        "convert_pdf_files_to_images": convert_pdf_files_to_images,
        "extract_time_range_from_logfile": extract_time_range_from_logfile,
        "infer_log_header_pattern": infer_log_header_pattern,
        "run_chat": run_chat,
        "run_simple_chat": run_simple_chat,
        "run_batch_chat": run_batch_chat,
        "run_batch_chat_from_excel": run_batch_chat_from_excel,
        "run_browser_task": run_browser_task,
        "run_browser_task_with_output": run_browser_task_with_output,
        "get_loaded_config_info": get_loaded_config_info,
    }
    allowed_registry = dict(tool_registry)

    if tools_option:
        tools = [tool.strip() for tool in tools_option.split(",") if tool.strip()]
        missing = [t for t in tools if t not in allowed_registry]
        if missing:
            raise ValueError(
                f"Unknown tool(s): {missing}. Supported: {sorted(allowed_registry.keys())}"
            )
        for tool in tools:
            header_aware_tool(mcp, tool_name=tool)(allowed_registry[tool])
        return

    for name in allowed_registry.keys():
        header_aware_tool(mcp, tool_name=name)(allowed_registry[name])


async def main():
    args = parse_args()
    init_runtime(args.config or None)
    apply_logging_overrides(
        level=(args.log_level or None),
        file=(args.log_file or None),
    )

    mode = args.mode
    mcp = FastMCP("ai_chat_util")
    prepare_mcp(mcp, args.tools)

    if mode == "stdio":
        await mcp.run_async()
        return

    host = args.host
    port = args.port
    if mode == "sse":
        await mcp.run_async(transport="sse", host=host, port=port)
        return

    await mcp.run_async(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    asyncio.run(main())