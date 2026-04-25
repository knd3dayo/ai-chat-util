import asyncio
from dotenv import load_dotenv
import argparse
from fastmcp import FastMCP
from ai_chat_util.common.config.runtime import init_runtime
from ai_chat_util.ai_chat_util_base.file_util.core.app import (
    list_file_server_roots,
    list_file_server_entries,
    get_document_type,
    get_mime_type,
    get_sheet_names,
    extract_excel_sheet,
    extract_text_from_file,
    extract_base64_to_text,
    list_zip_contents,
    extract_zip,
    create_zip,
    export_data_to_excel,
    import_data_from_excel,
)
mcp = FastMCP("file_util") #type :ignore

# 引数解析用の関数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP server with specified mode and APP_DATA_PATH.")
    parser.add_argument("--config", type=str, default="", help="Path to ai-chat-util-config.yml")
    # -m オプションを追加
    parser.add_argument("-m", "--mode", choices=["sse", "stdio", "http"], default="stdio", help="Mode to run the server in: 'sse' for Server-Sent Events, 'stdio' for standard input/output.")
    # -t tools オプションを追加 toolsはカンマ区切りの文字列. search_wikipedia_ja_mcp, vector_search, etc. 指定されていない場合は空文字を設定
    parser.add_argument("-t", "--tools", type=str, default="", help="Comma-separated list of tools to use, e.g., 'search_wikipedia_ja_mcp,vector_search_mcp'. If not specified, no tools are loaded.")
    # -p オプションを追加　ポート番号を指定する modeがsseの場合に使用.defaultは5001
    parser.add_argument("-p", "--port", type=int, default=5001, help="Port number to run the server on. Default is 5001.")
    # -v LOG_LEVEL オプションを追加 ログレベルを指定する. デフォルトは空白文字
    parser.add_argument("-v", "--log_level", type=str, default="", help="Log level to set for the server. Default is empty, which uses the default log level.")

    return parser.parse_args()

async def main():
    # load_dotenv() を使用して環境変数を読み込む
    load_dotenv()
    # 引数を解析
    args = parse_args()
    init_runtime(args.config or None)
    mode = args.mode

    # tools オプションが指定されている場合は、ツールを登録
    if args.tools:
        tools = [tool.strip() for tool in args.tools.split(",")]
        for tool_name in tools:
            # tool_nameという名前の関数が存在する場合は登録
            tool = globals().get(tool_name)
            if tool and callable(tool):
                mcp.tool()(tool)
            else:
                print(f"Warning: Tool '{tool_name}' not found or not callable. Skipping registration.")
    else:
        # デフォルトのツールを登録
        mcp.tool()(get_document_type)
        mcp.tool()(get_mime_type)
        mcp.tool()(get_sheet_names)
        mcp.tool()(extract_excel_sheet)
        mcp.tool()(extract_text_from_file)
        mcp.tool()(list_zip_contents)
        mcp.tool()(extract_zip)
        mcp.tool()(create_zip)
        mcp.tool()(extract_base64_to_text)
        mcp.tool()(export_data_to_excel)
        mcp.tool()(import_data_from_excel)
        mcp.tool()(list_file_server_roots)
        mcp.tool()(list_file_server_entries)

    if mode == "stdio":
        await mcp.run_async()

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
