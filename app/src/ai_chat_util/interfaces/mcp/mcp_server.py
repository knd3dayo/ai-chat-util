import asyncio
import argparse
import inspect
import time
from functools import wraps
from typing import Callable, Mapping
from fastmcp import FastMCP, Context

from ai_chat_util.ai_chat_util_base.core.common.config.runtime import init_runtime
from ai_chat_util.ai_chat_util_base.core.common.config.runtime import apply_logging_overrides
from ai_chat_util.ai_chat_util_base.request_headers import RequestHeaders, bind_current_request_headers

from ai_chat_util.ai_chat_util_base.core.resource_app import get_loaded_config_info

from ai_chat_util.ai_chat_util_agent.core.app import (
    run_chat,
    run_deepagent_chat,
    run_durable_workflow_from_file,
    run_mermaid_workflow_from_file,
    resume_durable_workflow,
    run_simple_chat,
    run_batch_chat,
    run_deepagent_batch_chat,
    run_simple_batch_chat,
    run_batch_chat_from_excel,
    run_deepagent_batch_chat_from_excel,
)

from ai_chat_util.ai_chat_util_base.analyze_pdf_util.core import (
    analyze_image_files,
    analyze_pdf_files,
    analyze_office_files,
    analyze_files,
    convert_office_files_to_pdf,
    convert_pdf_files_to_images,
    detect_log_format_and_search,
    infer_log_header_pattern,
    extract_log_time_range,
    analyze_documents_data,
    analyze_image_urls,
    analyze_pdf_urls,
    analyze_office_urls,
    analyze_urls
)


def _build_tool_metadata_registry() -> dict[str, dict[str, str]]:
    return {
        "convert_office_files_to_pdf": {
            "requires_approval": "true",
            "action_kind": "write",
            "usage_guidance": (
                "For write-capable usage, call this tool with dry_run=true first to preview the target pdf_path values. "
                "Only after approval should you call it again with dry_run=false to create files."
            ),
        },
        "convert_pdf_files_to_images": {
            "requires_approval": "true",
            "action_kind": "write",
            "usage_guidance": (
                "For write-capable usage, call this tool with dry_run=true first to preview the target image_dir and file pattern. "
                "Only after approval should you call it again with dry_run=false to create files."
            ),
        },
        "extract_log_time_range": {
            "requires_approval": "true",
            "action_kind": "write",
            "usage_guidance": (
                "This tool writes extracted logs under the workspace artifacts directory. "
                "Use infer_log_header_pattern first when the header pattern is unknown."
            ),
        },
    }


def _compose_tool_doc(base_doc: str, metadata: Mapping[str, str] | None) -> str:
    doc = (base_doc or "").rstrip()
    if not metadata:
        return doc

    metadata_lines = ["[MCP_META]"]
    for key, value in metadata.items():
        metadata_lines.append(f"{key}={value}")
    if doc:
        return doc + "\n\n" + "\n".join(metadata_lines)
    return "\n".join(metadata_lines)


# 引数解析用の関数
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
    # -m オプションを追加
    parser.add_argument(
        "-m",
        "--mode",
        choices=["sse", "http", "stdio"],
        default="stdio",
        help=(
            "Transport mode: 'stdio' (default), 'sse', or 'http' (streamable-http)."
        ),
    )
    # -t tools オプションを追加 toolsはカンマ区切りの文字列. search_wikipedia_ja_mcp, vector_search, etc. 指定されていない場合は空文字を設定
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
    # -p オプションを追加　ポート番号を指定する modeがsseの場合に使用.defaultは5001
    parser.add_argument("-p", "--port", type=int, default=5001, help="Port number to run the server on. Default is 5001.")
    # -v LOG_LEVEL オプションを追加 ログレベルを指定する. デフォルトは空白文字
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

def prepare_mcp(mcp: FastMCP, tools_option: str):
    tool_metadata = _build_tool_metadata_registry()

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
                            result = await func(*args, **kwargs)
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
            metadata = tool_metadata.get(tool_name, {})
            base_doc = str(getattr(func, "__doc__", "") or "").rstrip()
            wrapper.__doc__ = _compose_tool_doc(base_doc, metadata)
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
        # analysis tools
        "analyze_image_files": analyze_image_files,
        "analyze_pdf_files": analyze_pdf_files,
        "analyze_office_files": analyze_office_files,
        "analyze_files": analyze_files,
        "analyze_documents_data": analyze_documents_data,
        "analyze_image_urls": analyze_image_urls,
        "analyze_pdf_urls": analyze_pdf_urls,
        "analyze_office_urls": analyze_office_urls,
        "analyze_urls": analyze_urls,
        "convert_office_files_to_pdf": convert_office_files_to_pdf,
        "convert_pdf_files_to_images": convert_pdf_files_to_images,
        "detect_log_format_and_search": detect_log_format_and_search,
        "infer_log_header_pattern": infer_log_header_pattern,
        "extract_log_time_range": extract_log_time_range,
        # chat/batch
        "run_chat": run_chat,
        "run_deepagent_chat": run_deepagent_chat,
        "run_simple_chat": run_simple_chat,
        "run_batch_chat": run_batch_chat,
        "run_deepagent_batch_chat": run_deepagent_batch_chat,
        "deepagent_batch_chat": run_deepagent_batch_chat,
        "run_simple_batch_chat": run_simple_batch_chat,
        "run_batch_chat_from_excel": run_batch_chat_from_excel,
        "run_deepagent_batch_chat_from_excel": run_deepagent_batch_chat_from_excel,
        "deepagent_batch_chat_from_excel": run_deepagent_batch_chat_from_excel,
        "run_mermaid_workflow_from_file": run_mermaid_workflow_from_file,
        "run_durable_workflow_from_file": run_durable_workflow_from_file,
        "resume_durable_workflow": resume_durable_workflow,
        # debug helper
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

    # デフォルトのツールを登録（後方互換: 以前の default と同等 + analyze_documents_data）
    for name in allowed_registry.keys():
        header_aware_tool(mcp, tool_name=name)(allowed_registry[name])
    

async def main():
    # 引数を解析
    args = parse_args()

    # Initialize runtime config first (ai-chat-util-config.yml required)
    init_runtime(args.config or None)

    # Apply process-local logging overrides (especially useful for stdio MCP server).
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

    # mode == "http"
    await mcp.run_async(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    asyncio.run(main())
