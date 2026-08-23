import asyncio
from dotenv import load_dotenv
import argparse
from typing import Annotated, Optional

from fastmcp import FastMCP
from pydantic import Field

from ai_chat_util.core.common.config.runtime import init_runtime
from ai_chat_util.core.analysis.model import (
    MCPBooleanResult,
    MCPDetectExcelTablesResult,
    MCPDocumentTypeResult,
    MCPImportExcelDataResult,
    MCPMimeTypeResult,
    MCPSheetNamesResult,
    MCPTextResult,
    MCPZipContentsResult,
)
from ai_chat_util.core.analysis.base import (
    get_document_type as base_get_document_type,
    get_mime_type as base_get_mime_type,
    get_sheet_names as base_get_sheet_names,
    extract_excel_sheet as base_extract_excel_sheet,
    detect_excel_tables_in_sheet as base_detect_excel_tables_in_sheet,
    extract_text_from_file as base_extract_text_from_file,
    extract_base64_to_text as base_extract_base64_to_text,
    list_zip_contents as base_list_zip_contents,
    extract_zip as base_extract_zip,
    create_zip as base_create_zip,
    export_data_to_excel as base_export_data_to_excel,
    import_data_from_excel as base_import_data_from_excel,
)
mcp = FastMCP("file_util") #type :ignore


async def get_document_type(
    file_path: Annotated[str, Field(description="Path to the file to get types for")]
) -> MCPDocumentTypeResult:
    document_type = await base_get_document_type(file_path)
    return MCPDocumentTypeResult(document_type=document_type)


async def get_mime_type(
    file_path: Annotated[str, Field(description="Path to the file to get MIME type for")]
) -> MCPMimeTypeResult:
    mime_type = await base_get_mime_type(file_path)
    return MCPMimeTypeResult(mime_type=mime_type)


async def get_sheet_names(
    file_path: Annotated[str, Field(description="Path to the Excel file to get sheet names for")]
) -> MCPSheetNamesResult:
    sheet_names = await base_get_sheet_names(file_path)
    return MCPSheetNamesResult(sheet_names=sheet_names)


async def extract_excel_sheet(
    file_path: Annotated[str, Field(description="Path to the Excel file to extract text from")],
    sheet_name: Annotated[str, Field(description="Name of the sheet to extract text from")],
) -> MCPTextResult:
    text = await base_extract_excel_sheet(file_path, sheet_name)
    return MCPTextResult(text=text)


async def detect_excel_tables_in_sheet(
    file_path: Annotated[str, Field(description="Path to the Excel file to detect tables from")],
    sheet_name: Annotated[Optional[str], Field(description="Name of the sheet to scan. If omitted, active sheet is used.")] = None,
    empty_row_tolerance: Annotated[int, Field(description="Number of consecutive empty rows to tolerate before table end.")] = 2,
) -> MCPDetectExcelTablesResult:
    tables = await base_detect_excel_tables_in_sheet(file_path, sheet_name, empty_row_tolerance)
    return MCPDetectExcelTablesResult(tables=tables)


async def extract_text_from_file(
    file_path: Annotated[str, Field(description="Path to the file to extract text from")]
) -> MCPTextResult:
    text = await base_extract_text_from_file(file_path)
    return MCPTextResult(text=text)


async def extract_base64_to_text(
    extension: Annotated[str, Field(description="File extension of the base64 data")],
    base64_data: Annotated[str, Field(description="Base64 encoded data to extract text from")],
) -> MCPTextResult:
    text = await base_extract_base64_to_text(extension, base64_data)
    return MCPTextResult(text=text)


async def list_zip_contents(
    file_path: Annotated[str, Field(description="Path to the ZIP file to list contents from. **Absolute path required**")]
) -> MCPZipContentsResult:
    contents = await base_list_zip_contents(file_path)
    return MCPZipContentsResult(contents=contents)


async def extract_zip(
    file_path: Annotated[str, Field(description="Path to the ZIP file to extract. **Absolute path required**")],
    extract_to: Annotated[str, Field(description="Directory to extract the ZIP contents to. **Absolute path required**")],
    password: Annotated[Optional[str], Field(description="Password for the ZIP file, if any")] = None,
) -> MCPBooleanResult:
    ok = await base_extract_zip(file_path, extract_to, password)
    return MCPBooleanResult(ok=ok)


async def create_zip(
    file_paths: Annotated[list[str], Field(description="List of file or directory paths to include in the ZIP. **Absolute paths required**")],
    output_zip: Annotated[str, Field(description="Path to the output ZIP file. **Absolute path required**")],
    password: Annotated[Optional[str], Field(description="Password for the ZIP file, if any")] = None,
) -> MCPBooleanResult:
    ok = await base_create_zip(file_paths, output_zip, password)
    return MCPBooleanResult(ok=ok)


async def export_data_to_excel(
    data: Annotated[dict[str, list], Field(description="Data to export to Excel, with keys as column headers and values as lists of column data")],
    output_file: Annotated[str, Field(description="Path to the output Excel file")],
    sheet_name: Annotated[Optional[str], Field(description="Name of the sheet to create in the Excel file")] = "Sheet1",
) -> MCPBooleanResult:
    ok = await base_export_data_to_excel(data, output_file, sheet_name)
    return MCPBooleanResult(ok=ok)


async def import_data_from_excel(
    input_file: Annotated[str, Field(description="Path to the Excel file to import data from")],
    sheet_name: Annotated[Optional[str], Field(description="Name of the sheet to import data from")] = "Sheet1",
) -> MCPImportExcelDataResult:
    data = await base_import_data_from_excel(input_file, sheet_name)
    return MCPImportExcelDataResult(data=data)

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
        mcp.tool()(detect_excel_tables_in_sheet)
        mcp.tool()(extract_text_from_file)
        mcp.tool()(list_zip_contents)
        mcp.tool()(extract_zip)
        mcp.tool()(create_zip)
        mcp.tool()(extract_base64_to_text)
        mcp.tool()(export_data_to_excel)
        mcp.tool()(import_data_from_excel)

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
