import asyncio
import argparse
from typing import Callable
from fastmcp import FastMCP

from ai_chat_util_base.config.runtime import init_runtime
from ai_chat_util_base.config.runtime import apply_logging_overrides

from ai_chat_util.core.resource_app import (
    use_custom_pdf_analyzer,
    get_completion_model,
    create_user_message,
    create_system_message,
    create_assistant_message,
    create_text_content,
    create_pdf_content_from_file,
    create_image_content,
    create_image_content_from_file,
    create_office_content_from_file,
    create_multi_format_contents_from_file,
)

from ai_chat_util.core.app import (
    run_chat,
    run_simple_chat,
    run_batch_chat,
    run_simple_batch_chat,
    run_batch_chat_from_excel,
)

from ai_chat_util.core.tool_app import (
    analyze_image_files,
    analyze_pdf_files,
    analyze_office_files,
    analyze_files,
    analyze_documents_data,
    analyze_image_urls,
    analyze_pdf_urls,
    analyze_office_urls,
    analyze_urls
)


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
        # chat/batch
        "run_chat": run_chat,
        "run_simple_chat": run_simple_chat,
        "run_batch_chat": run_batch_chat,
        "run_simple_batch_chat": run_simple_batch_chat,
        "run_batch_chat_from_excel": run_batch_chat_from_excel,
        # message/content helpers
        "use_custom_pdf_analyzer": use_custom_pdf_analyzer,
        "get_completion_model": get_completion_model,
        "create_user_message": create_user_message,
        "create_system_message": create_system_message,
        "create_assistant_message": create_assistant_message,
        "create_text_content": create_text_content,
        "create_pdf_content_from_file": create_pdf_content_from_file,
        "create_image_content": create_image_content,
        "create_image_content_from_file": create_image_content_from_file,
        "create_office_content_from_file": create_office_content_from_file,
        "create_multi_format_contents_from_file": create_multi_format_contents_from_file,
    }

    if tools_option:
        tools = [tool.strip() for tool in tools_option.split(",") if tool.strip()]
        missing = [t for t in tools if t not in tool_registry]
        if missing:
            raise ValueError(
                f"Unknown tool(s): {missing}. Supported: {sorted(tool_registry.keys())}"
            )
        for tool in tools:
            mcp.tool()(tool_registry[tool])
        return

    # デフォルトのツールを登録（後方互換: 以前の default と同等 + analyze_documents_data）
    for name in (
        "analyze_image_files",
        "analyze_pdf_files",
        "analyze_office_files",
        "analyze_files",
        "analyze_documents_data",
        "analyze_image_urls",
        "analyze_pdf_urls",
        "analyze_office_urls",
        "analyze_urls",
    ):
        mcp.tool()(tool_registry[name])
    

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
