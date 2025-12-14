from typing import Annotated
from pydantic import Field
from ai_chat_util.llm.model import ChatRequestContext, ChatHistory, ChatResponse
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig

def use_custom_pdf_analyzer() -> bool:
    """
    Check if the custom PDF analyzer should be used based on the environment variable.
    Returns True if USE_CUSTOM_PDF_ANALYZER is set to "true" (case insensitive), otherwise False.
    """
    import os
    return os.getenv("USE_CUSTOM_PDF_ANALYZER", "false").lower() == "true"

# toolは実行時にmcp.tool()で登録する。@mcp.toolは使用しない。
# chat_utilのrun_chat_asyncを呼び出すラッパー関数を定義
async def run_chat(
    completion_request: Annotated[ChatHistory, Field(description="Completion request object")],
    request_context: Annotated[ChatRequestContext, Field(description="Chat request context")]    
) -> Annotated[ChatResponse, Field(description="List of related articles from Wikipedia")]:
    """
    This function searches Wikipedia with the specified keywords and returns related articles.
    """
    client = LLMClient.create_llm_client(LLMConfig(), completion_request, request_context)
    return await client.chat()


# 複数の画像の分析を行う
async def analyze_image_files(
    image_path_list: Annotated[list[str], Field(description="List of absolute paths to the image files to analyze. e.g., [/path/to/image1.jpg, /path/to/image2.jpg]")],
    prompt: Annotated[str, Field(description="Prompt to analyze the images")],
    detail: Annotated[str, Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes multiple images using the specified prompt and returns the analysis result.
    """
    client = LLMClient.create_llm_client(llm_config=LLMConfig())
    response = await client.analyze_image_files(image_path_list, prompt, detail)
    return response

# 複数のPDFの分析を行う
async def analyze_pdf_files(
    pdf_path_list: Annotated[list[str], Field(description="List of absolute paths to the PDF files to analyze. e.g., [/path/to/document1.pdf, /path/to/document2.pdf]")],
    prompt: Annotated[str, Field(description="Prompt to analyze the PDFs")]
    ) -> Annotated[str, Field(description="Analysis result of the PDFs")]:
    """
    This function analyzes multiple PDFs using the specified prompt and returns the analysis result.
    """
    client = LLMClient.create_llm_client(llm_config=LLMConfig())
    if use_custom_pdf_analyzer():
        response = await client.analyze_pdf_files_custom(pdf_path_list, prompt, detail="auto")
    else:
        response = await client.analyze_pdf_files(pdf_path_list, prompt)
    return response

async def analyze_office_files(
    office_path_list: Annotated[list[str], Field(description="List of absolute paths to the Office files to analyze. e.g., [/path/to/document1.docx, /path/to/spreadsheet1.xlsx]")],
    prompt: Annotated[str, Field(description="Prompt to analyze the Office documents")]
    ) -> Annotated[str, Field(description="Analysis result of the Office documents")]:
    """
    This function analyzes multiple Office documents using the specified prompt and returns the analysis result.
    """ 
    client = LLMClient.create_llm_client(llm_config=LLMConfig())
    if use_custom_pdf_analyzer():
        response = await client.analyze_office_document_files_custom(office_path_list, prompt, detail="auto")
    else:
        response = await client.analyze_office_document_files(office_path_list, prompt)
    return response
