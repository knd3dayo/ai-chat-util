from typing import Annotated
import os, tempfile
import atexit
from pydantic import Field
from ai_chat_util.llm.model import ChatRequestContext, ChatHistory, ChatResponse, RequestModel
from ai_chat_util.llm.llm_client import LLMClient
from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.batch.batch_client import LLMBatchClient
import os

def use_custom_pdf_analyzer() -> Annotated[bool, Field(description="Whether to use the custom PDF analyzer or not")]:
    """
    This function checks whether to use the custom PDF analyzer based on the environment variable.
    """
    use_custom = os.getenv("USE_CUSTOM_PDF_ANALYZER", "false").lower() == "true"
    return use_custom

# toolは実行時にmcp.tool()で登録する。@mcp.toolは使用しない。
# chat_utilのrun_chat_asyncを呼び出すラッパー関数を定義
async def run_chat(
        completion_request: Annotated[ChatHistory, Field(description="Completion request object")],
        request_context: Annotated[ChatRequestContext, Field(description="Chat request context")] = ChatRequestContext()
) -> Annotated[ChatResponse, Field(description="List of related articles from Wikipedia")]:
    """
    This function searches Wikipedia with the specified keywords and returns related articles.
    """
    if request_context is None:
        request_context = ChatRequestContext()
    client = LLMClient.create_llm_client(LLMConfig(), completion_request, request_context)
    return await client.chat()

async def run_simple_batch_chat(
        prompt: Annotated[str, Field(description="Prompt for the batch chat")],
        messages: Annotated[list[str], Field(description="List of messages for the batch chat")],
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[str], Field(description="List of chat responses from batch processing")]:
    """
    This function processes a simple batch chat with the specified prompt and messages, and returns the list of chat responses.
    """
    batch_client = LLMBatchClient()
    results = await batch_client.run_simple_batch_chat(prompt, messages, concurrency)
    return results

async def run_batch_chat(
        chat_histories: Annotated[list[ChatHistory], Field(description="List of chat histories for batch processing")],
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[ChatResponse], Field(description="List of chat responses from batch processing")]:
    """
    This function processes a batch of chat histories concurrently and returns the list of chat responses.
    """
    batch_client = LLMBatchClient()
    results = await batch_client.run_batch_chat(chat_histories, concurrency)
    return [response for _, response, _ in results]

async def run_batch_chat_from_excel(
        prompt: Annotated[str, Field(description="Prompt for the batch chat")],
        input_excel_path: Annotated[str, Field(description="Path to the input Excel file")],
        output_excel_path: Annotated[str, Field(description="Path to the output Excel file")]="output.xlsx",
        content_column: Annotated[str, Field(description="Name of the column containing input messages")]="content",
        file_path_column: Annotated[str, Field(description="Name of the column containing file paths")]="file_path",
        output_column: Annotated[str, Field(description="Name of the column to store output responses")]="output",
        detail: Annotated[str, Field(description="Detail level for file analysis. e.g., 'low', 'high', 'auto'")]= "auto",
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=16
) -> None:
    """
    This function reads chat histories from an Excel file, processes them in batch, and writes the responses to a new Excel file.
    """
    batch_client = LLMBatchClient()
    await batch_client.run_batch_chat_from_excel(
        prompt,
        input_excel_path,
        output_excel_path,
        content_column,
        file_path_column,
        output_column,
        detail,
        concurrency
    )

# 複数の画像の分析を行う URLから画像をダウンロードして分析する 
async def analyze_image_urls(
        image_path_urls: Annotated[list[RequestModel], Field(description="List of urls to the image files to analyze. e.g., http://path/to/image1.jpg")],
        prompt: Annotated[str, Field(description="Prompt to analyze the images")],
        detail: Annotated[str, Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes multiple images using the specified prompt and returns the analysis result.
    """
    llm_client = LLMClient.create_llm_client(LLMConfig())
    response = await llm_client.analyze_image_urls(image_path_urls, prompt, detail)

    return response.output

# 複数の画像の分析を行う
async def analyze_image_files(
        file_list: Annotated[list[str], Field(description="List of absolute paths to the image files to analyze. e.g., [/path/to/image1.jpg, /path/to/image2.jpg]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the images")],
        detail: Annotated[str, Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes multiple images using the specified prompt and returns the analysis result.
    """
    llm_client = LLMClient.create_llm_client(LLMConfig())
    response = await llm_client.analyze_image_files(file_list, prompt, detail)
    return response.output


# 複数のPDFの分析を行う URLからPDFをダウンロードして分析する
async def analyze_pdf_urls(
        pdf_path_urls: Annotated[
            list[RequestModel],
            Field(
                description="List of URLs to the PDF files to analyze. e.g., http://path/to/document2.pdf"
            ),
        ],
        prompt: Annotated[str, Field(description="Prompt to analyze the PDFs")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when USE_CUSTOM_PDF_ANALYZER is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
) -> Annotated[str, Field(description="Analysis result of the PDFs")]:
    """
    This function analyzes multiple PDFs using the specified prompt and returns the analysis result.
    """
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = LLMClient.create_llm_client(LLMConfig())
    path_list = llm_client.download_files(pdf_path_urls, tmpdir.name)
    response = await llm_client.analyze_pdf_files(path_list, prompt, detail)
    return response.output

# 複数のPDFの分析を行う
async def analyze_pdf_files(
        pdf_path_list: Annotated[list[str], Field(description="List of absolute paths to the PDF files to analyze. e.g., [/path/to/document1.pdf, /path/to/document2.pdf]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the PDFs")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when USE_CUSTOM_PDF_ANALYZER is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the PDFs")]:
    """
    This function analyzes multiple PDFs using the specified prompt and returns the analysis result.
    """
    llm_client = LLMClient.create_llm_client(LLMConfig())
    response = await llm_client.analyze_pdf_files(pdf_path_list, prompt, detail)
    return response.output

# 複数のOfficeドキュメントの分析を行う URLからOfficeドキュメントをダウンロードして分析する
async def analyze_office_urls(
        office_path_urls: Annotated[list[RequestModel], Field(description="List of urls to the Office files to analyze. e.g., http://path/to/document1.docx")],
        prompt: Annotated[str, Field(description="Prompt to analyze the Office documents")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when USE_CUSTOM_PDF_ANALYZER is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the Office documents")]:
    """ 
    This function analyzes multiple Office documents using the specified prompt and returns the analysis result.
    """
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = LLMClient.create_llm_client(LLMConfig())
    path_list = llm_client.download_files(office_path_urls, tmpdir.name)

    response = await llm_client.analyze_office_files(path_list, prompt, detail)
    return response.output

async def analyze_office_files(
        office_path_list: Annotated[list[str], Field(description="List of absolute paths to the Office files to analyze. e.g., [/path/to/document1.docx, /path/to/spreadsheet1.xlsx]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the Office documents")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when USE_CUSTOM_PDF_ANALYZER is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the Office documents")]:
    """
    This function analyzes multiple Office documents using the specified prompt and returns the analysis result.
    """ 
    llm_client = LLMClient.create_llm_client(LLMConfig())
    response = await llm_client.analyze_office_files(office_path_list, prompt, detail=detail)
    return response.output

async def analyze_urls(
        file_path_urls: Annotated[list[RequestModel], Field(description="List of urls to the files to analyze. e.g., http://path/to/document1.pdf, http://path/to/image1.jpg")],
        prompt: Annotated[str, Field(description="Prompt to analyze the files")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when USE_CUSTOM_PDF_ANALYZER is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the files")]:
    """
    This function analyzes multiple files of various formats using the specified prompt and returns the analysis result.
    """
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    llm_client = LLMClient.create_llm_client(LLMConfig())
    path_list = llm_client.download_files(file_path_urls, tmpdir.name)
    response = await llm_client.analyze_files(path_list, prompt, detail)
    return response.output

async def analyze_files(
        file_path_list: Annotated[list[str], Field(description="List of absolute paths to the files to analyze. e.g., [/path/to/document1.pdf, /path/to/image1.jpg]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the files")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when USE_CUSTOM_PDF_ANALYZER is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the files")]:
    """
    This function analyzes multiple files of various formats using the specified prompt and returns the analysis result.
    """
    llm_client = LLMClient.create_llm_client(LLMConfig())
    response = await llm_client.analyze_files(file_path_list, prompt, detail=detail)
    return response.output