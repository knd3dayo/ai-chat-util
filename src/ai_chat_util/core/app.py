from typing import Annotated, Any
import os, tempfile
import requests
from pydantic import Field, BaseModel
from ai_chat_util.llm.model import ChatRequestContext, ChatHistory, ChatResponse
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig


class RequestModel(BaseModel):
    url: str
    headers: dict[str, Any] = {}

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

# 複数の画像の分析を行う URLから画像をダウンロードして分析する 
async def analyze_image_urls(
    image_path_urls: Annotated[list[RequestModel], Field(description="List of urls to the image files to analyze. e.g., http://path/to/image1.jpg")],
    prompt: Annotated[str, Field(description="Prompt to analyze the images")],
    detail: Annotated[str, Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes multiple images using the specified prompt and returns the analysis result.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path_list = []
        for item in image_path_urls:
            res = requests.get(url=item.url, headers=item.headers)
            with open(os.path.join(tmpdir, os.path.basename(item.url)), "wb") as f:
                f.write(res.content)
            image_path_list.append(os.path.join(tmpdir, os.path.basename(item.url)))

        client = LLMClient.create_llm_client(llm_config=LLMConfig())
        response = await client.analyze_image_files(image_path_list, prompt, detail)
    return response


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

# 複数のPDFの分析を行う URLからPDFをダウンロードして分析する
async def analyze_pdf_urls(
    pdf_path_urls: Annotated[list[RequestModel], Field(description="List of urls to the PDF files to analyze. e.g., http://path/to/document2.pdf")],
    prompt: Annotated[str, Field(description="Prompt to analyze the PDFs")]
    ) -> Annotated[str, Field(description="Analysis result of the PDFs")]:
    """ 
    This function analyzes multiple PDFs using the specified prompt and returns the analysis result.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path_list = []
        for item in pdf_path_urls:
            res = requests.get(url=item.url, headers=item.headers)
            with open(os.path.join(tmpdir, os.path.basename(item.url)), "wb") as f:
                f.write(res.content)
            pdf_path_list.append(os.path.join(tmpdir, os.path.basename(item.url)))

        client = LLMClient.create_llm_client(llm_config=LLMConfig())
        if use_custom_pdf_analyzer():
            response = await client.analyze_pdf_files_custom(pdf_path_list, prompt, detail="auto")
        else:
            response = await client.analyze_pdf_files(pdf_path_list, prompt)
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

# 複数のOfficeドキュメントの分析を行う URLからOfficeドキュメントをダウンロードして分析する
async def analyze_office_urls(
    office_path_urls: Annotated[list[RequestModel], Field(description="List of urls to the Office files to analyze. e.g., http://path/to/document1.docx")],
    prompt: Annotated[str, Field(description="Prompt to analyze the Office documents")]
    ) -> Annotated[str, Field(description="Analysis result of the Office documents")]:
    """ 
    This function analyzes multiple Office documents using the specified prompt and returns the analysis result.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        office_path_list = []
        for item in office_path_urls:
            res = requests.get(url=item.url, headers=item.headers)
            with open(os.path.join(tmpdir, os.path.basename(item.url)), "wb") as f:
                f.write(res.content)
            office_path_list.append(os.path.join(tmpdir, os.path.basename(item.url)))

        client = LLMClient.create_llm_client(llm_config=LLMConfig())
        if use_custom_pdf_analyzer():
            response = await client.analyze_office_document_files_custom(office_path_list, prompt, detail="auto")
        else:
            response = await client.analyze_office_document_files(office_path_list, prompt)
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
