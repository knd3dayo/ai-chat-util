import asyncio, os
import argparse
from fastmcp import FastMCP

from dotenv import load_dotenv
import ai_chat_util.mcp.app as app_functions


# 引数解析用の関数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP server with specified mode and APP_DATA_PATH.")
    # -m オプションを追加
    parser.add_argument("-m", "--mode", choices=["http", "stdio"], default="stdio", help="Mode to run the server in: 'http' for Streamable HTTP , 'stdio' for standard input/output.")
    # -d オプションを追加　APP_DATA_PATH を指定する
    parser.add_argument("-d", "--app_data_path", type=str, help="Path to the application data directory.")
    # 引数を解析して返す
    # -t tools オプションを追加 toolsはカンマ区切りの文字列. search_wikipedia_ja_mcp, vector_search, etc. 指定されていない場合は空文字を設定
    parser.add_argument("-t", "--tools", type=str, default="", help="Comma-separated list of tools to use, e.g., 'search_wikipedia_ja_mcp,vector_search_mcp'. If not specified, no tools are loaded.")
    # -p オプションを追加　ポート番号を指定する modeがsseの場合に使用.defaultは5001
    parser.add_argument("-p", "--port", type=int, default=5001, help="Port number to run the server on. Default is 5001.")
    # -v LOG_LEVEL オプションを追加 ログレベルを指定する. デフォルトは空白文字
    parser.add_argument("-v", "--log_level", type=str, default="", help="Log level to set for the server. Default is empty, which uses the default log level.")

    return parser.parse_args()

def prepare_mcp(mcp: FastMCP, tools_option: str):
    # tools オプションが指定されている場合は、ツールを登録
    if tools_option:
        tools = [tool.strip() for tool in tools_option.split(",")]
        for tool_name in tools:
            # app_functionsにtool_nameという名前の関数が存在する場合は登録
            tool = getattr(app_functions, tool_name, None)
            if tool and callable(tool):
                mcp.tool()(tool)
            else:
                print(f"Warning: Tool '{tool_name}' not found or not callable. Skipping registration.")
    else:
        # デフォルトのツールを登録
        mcp.tool()(app_functions.run_chat)
        mcp.tool()(app_functions.analyze_image_files)
        mcp.tool()(app_functions.analyze_pdf_files)
        mcp.tool()(app_functions.analyze_office_files)
    

async def main():
    # load_dotenv() を使用して環境変数を読み込む
    load_dotenv()
    # 引数を解析
    args = parse_args()
    mode = args.mode

    mcp = FastMCP("ai_chat_util") 

    prepare_mcp(mcp, args.tools)


    if mode == "stdio":
        await mcp.run_async()

    elif mode == "http":
        # port番号を取得
        port = args.port
        await mcp.run_async(transport="streamable-http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    asyncio.run(main())
