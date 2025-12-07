import asyncio
from typing import Annotated, Any
from dotenv import load_dotenv
import argparse
from fastmcp import FastMCP
from pydantic import Field
from ai_chat_util.llm.model import ChatRequestContext, ChatHistory, ChatResponse
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.image.image_chat_util import ImageChatClient
from ai_chat_util.pdf.pdf_chat_util import PDFChatClient

mcp = FastMCP("ai_chat_mcp") #type :ignore

# toolは実行時にmcp.tool()で登録する。@mcp.toolは使用しない。
# chat_utilのrun_chat_asyncを呼び出すラッパー関数を定義
async def run_chat_async_mcp(
    completion_request: Annotated[ChatHistory, Field(description="Completion request object")],
    request_context: Annotated[ChatRequestContext, Field(description="Chat request context")]    
) -> Annotated[ChatResponse, Field(description="List of related articles from Wikipedia")]:
    """
    This function searches Wikipedia with the specified keywords and returns related articles.
    """
    client = LLMClient.create_llm_client(LLMConfig(), completion_request, request_context)
    return await client.run_chat()

        
# 画像を分析
async def analyze_image_mcp(
    image_path: Annotated[str, Field(description="Absolute path to the image file to analyze. e.g., /path/to/image.jpg")],
    prompt: Annotated[str, Field(description="Prompt to analyze the image")]
    ) -> Annotated[str, Field(description="Analysis result of the image")]:
    """
    This function analyzes an image using the specified prompt and returns the analysis result.
    """
    client = ImageChatClient(LLMClient.create_llm_client(llm_config=LLMConfig()))
    response = await client.analyze_image_async(image_path, prompt)
    return response

# 複数の画像の分析を行う
async def analyze_images_mcp(
    image_path_list: Annotated[list[str], Field(description="List of absolute paths to the image files to analyze. e.g., [/path/to/image1.jpg, /path/to/image2.jpg]")],
    prompt: Annotated[str, Field(description="Prompt to analyze the images")]
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes two images using the specified prompt and returns the analysis result.
    """
    client = ImageChatClient(LLMClient.create_llm_client(llm_config=LLMConfig()))
    response = await client.analyze_images_async(image_path_list, prompt)
    return response

# 画像グループ1と画像グループ2の分析を行う
async def analyze_image_groups_mcp(
    image_group1: Annotated[list[str], Field(description="List of absolute paths to the first group of image files to analyze.")],
    image_group2: Annotated[list[str], Field(description="List of absolute paths to the second group of image files to analyze.")],
    prompt: Annotated[str, Field(description="Prompt to analyze the image groups")]
    ) -> Annotated[str, Field(description="Analysis result of the image groups")]:
    """
    This function analyzes two groups of images using the specified prompt and returns the analysis result.
    """
    client = ImageChatClient(LLMClient.create_llm_client(llm_config=LLMConfig()))
    # ここでは、各グループの最初の画像のみを使用して分析を行う例を示す
    response = await client.analyze_image_groups_async(image_group1, image_group2, prompt)
    return response

# PDFの分析を行う
async def analyze_pdf_mcp(
    pdf_path: Annotated[str, Field(description="Absolute path to the PDF file to analyze. e.g., /path/to/document.pdf")],
    prompt: Annotated[str, Field(description="Prompt to analyze the PDF")]
    ) -> Annotated[str, Field(description="Analysis result of the PDF")]:
    """
    This function analyzes a PDF using the specified prompt and returns the analysis result.
    """
    client = PDFChatClient(LLMClient.create_llm_client(llm_config=LLMConfig()))
    response = await client.analyze_pdf_async(pdf_path, prompt)
    return response

# 複数のPDFの分析を行う
async def analyze_pdfs_mcp(
    pdf_path_list: Annotated[list[str], Field(description="List of absolute paths to the PDF files to analyze. e.g., [/path/to/document1.pdf, /path/to/document2.pdf]")],
    prompt: Annotated[str, Field(description="Prompt to analyze the PDFs")]
    ) -> Annotated[str, Field(description="Analysis result of the PDFs")]:
    """
    This function analyzes multiple PDFs using the specified prompt and returns the analysis result.
    """
    client = PDFChatClient(LLMClient.create_llm_client(llm_config=LLMConfig()))
    response = await client.analyze_pdfs_async(pdf_path_list, prompt)
    return response

# 引数解析用の関数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP server with specified mode and APP_DATA_PATH.")
    # -m オプションを追加
    parser.add_argument("-m", "--mode", choices=["sse", "stdio"], default="stdio", help="Mode to run the server in: 'sse' for Server-Sent Events, 'stdio' for standard input/output.")
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

async def main():
    # load_dotenv() を使用して環境変数を読み込む
    load_dotenv()
    # 引数を解析
    args = parse_args()
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
        mcp.tool()(run_chat_async_mcp)
        # デフォルトのツールを登録
        mcp.tool()(analyze_image_mcp)
        mcp.tool()(analyze_images_mcp)
        mcp.tool()(analyze_image_groups_mcp)
        mcp.tool()(analyze_pdf_mcp)
        mcp.tool()(analyze_pdfs_mcp)


    if mode == "stdio":
        await mcp.run_async()
    elif mode == "sse":
        # port番号を取得
        port = args.port
        await mcp.run_async(transport="sse", port=port)


if __name__ == "__main__":
    asyncio.run(main())
