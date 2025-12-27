import asyncio, os
import argparse
from fastmcp import FastMCP

from dotenv import load_dotenv
from ai_chat_util.core.app import (
    run_chat,
    analyze_image_files,
    analyze_pdf_files,
    analyze_office_files,
    analyze_files,
    analyze_image_urls,
    analyze_pdf_urls,
    analyze_office_urls,
    analyze_urls
)


# 引数解析用の関数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP server with specified mode")
    # -m オプションを追加
    parser.add_argument("-m", "--mode", choices=["sse", "http", "stdio"], default="stdio", help="Mode to run the server in: 'http' for Streamable HTTP , 'stdio' for standard input/output.")
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
    # -p オプションを追加　ポート番号を指定する modeがsseの場合に使用.defaultは5001
    parser.add_argument("-p", "--port", type=int, default=5001, help="Port number to run the server on. Default is 5001.")
    # -v LOG_LEVEL オプションを追加 ログレベルを指定する. デフォルトは空白文字
    parser.add_argument("-v", "--log_level", type=str, default="", help="Log level to set for the server. Default is empty, which uses the default log level.")

    return parser.parse_args()

def prepare_mcp(mcp: FastMCP, tools_option: str):
    # tools オプションが指定されている場合は、ツールを登録
    if tools_option:
        tools = [tool.strip() for tool in tools_option.split(",")]
        for tool in tools:
            global_namespace = globals()
            if tool in global_namespace:
                mcp.tool()(global_namespace[tool])

    else:
        # デフォルトのツールを登録
        mcp.tool()(run_chat)
        mcp.tool()(analyze_image_files)
        mcp.tool()(analyze_pdf_files)
        mcp.tool()(analyze_office_files)
        mcp.tool()(analyze_files)
        mcp.tool()(analyze_image_urls)
        mcp.tool()(analyze_pdf_urls)
        mcp.tool()(analyze_office_urls)
        mcp.tool()(analyze_urls)
    

async def main():
    # load_dotenv() を使用して環境変数を読み込む
    load_dotenv()
    # 引数を解析
    args = parse_args()
    mode = args.mode

    mcp = FastMCP() 

    prepare_mcp(mcp, args.tools)


    if mode == "stdio":
        await mcp.run_async()

    elif mode == "sse":
        # port番号を取得
        port = args.port
        await mcp.run_async(transport="sse", host="0.0.0.0", port=port)

    elif mode == "sse":
        # port番号を取得
        port = args.port
        await mcp.run_async(transport="sse", host="0.0.0.0", port=port)

    elif mode == "http":
        # port番号を取得
        port = args.port
        await mcp.run_async(transport="streamable-http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    asyncio.run(main())
