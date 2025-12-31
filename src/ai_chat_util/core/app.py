from typing import Annotated, Literal
import os, tempfile
import atexit
from pydantic import Field
from ai_chat_util.llm.model import ChatHistory, ChatResponse, RequestModel, ChatRequest, ChatMessage, ChatContent
from ai_chat_util.llm.llm_factory import LLMFactory
from ai_chat_util.llm.llm_config import LLMConfig
from ai_chat_util.batch.batch_client import LLMBatchClient
import os

def use_custom_pdf_analyzer() -> Annotated[bool, Field(description="Whether to use the custom PDF analyzer or not")]:
    """
    This function checks whether to use the custom PDF analyzer based on the environment variable.
    """
    use_custom = os.getenv("USE_CUSTOM_PDF_ANALYZER", "false").lower() == "true"
    return use_custom

def create_chat_history(
        chat_message_list: Annotated[list[ChatMessage], Field(description="List of chat messages to create chat history from")]
) -> Annotated[ChatHistory, Field(description="Chat history created from the list of chat messages")]:
    """
    This function creates a ChatHistory object from a list of chat messages.
    """
    llm_config = LLMConfig()
    return ChatHistory(messages=chat_message_list, model=llm_config.completion_model)

def create_user_message(
        chat_content_list: Annotated[list[ChatContent], Field(description="List of chat contents from the user messages")]
) -> Annotated[ChatMessage, Field(description="Chat history created from user messages")]:
    """
    This function creates a ChatHistory object from a list of user messages.
    """
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    return llm_client.create_user_message(chat_content_list)

def create_assistant_message(
        chat_content_list: Annotated[list[ChatContent], Field(description="List of chat contents from the assistant messages")]
) -> Annotated[ChatMessage, Field(description="Chat history created from assistant messages")]:
    """
    This function creates a ChatHistory object from a list of assistant messages.
    """
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    return llm_client.create_assistant_message(chat_content_list)

def create_system_message(
        chat_content_list: Annotated[list[ChatContent], Field(description="List of chat contents from the system messages")]
) -> Annotated[ChatMessage, Field(description="Chat history created from system messages")]:
    """
    This function creates a ChatHistory object from a list of system messages.
    """
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    return llm_client.create_system_message(chat_content_list)

def create_text_content(
        text: Annotated[str, Field(description="Text content for the chat message")]
) -> Annotated[ChatContent, Field(description="Chat content created from text")]:
    """
    This function creates a ChatContent object from text.
    """
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    return llm_client.create_text_content(text)

def create_pdf_content_from_file(
    file_path: Annotated[str, Field(description="File path for the chat message content")],
    detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for PDF analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from file")]:
    """
    This function creates a ChatContent object from a file.
    """
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    return llm_client.create_pdf_content_from_file(file_path, detail)

def create_image_content_from_file(
        file_path: Annotated[str, Field(description="File path for the chat message content")],
        detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from image file")]:
    """
    This function creates a ChatContent object from an image file.
    """
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    return llm_client.create_image_content_from_file(file_path, detail)

def create_office_content_from_file(
        file_path: Annotated[str, Field(description="File path for the chat message content")],
        detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for Office document analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from Office document file")]:
    """
    This function creates a ChatContent object from an Office document file.
    """
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    return llm_client.create_office_content_from_file(file_path, detail)

def create_multi_format_contents_from_file(
        file_path: Annotated[str, Field(description="File path for the chat message content")],
        detail: Annotated[Literal["low", "high", "auto"], Field(description="Detail level for file analysis. e.g., 'low', 'high', 'auto'")]= "auto"
) -> Annotated[list[ChatContent], Field(description="Chat content created from multi-format file")]:
    """
    This function creates a ChatContent object from a multi-format file.
    """
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    return llm_client.create_multi_format_contents_from_file(file_path, detail)

# toolは実行時にmcp.tool()で登録する。@mcp.toolは使用しない。
# chat_utilのrun_chat_asyncを呼び出すラッパー関数を定義
async def run_chat(
        chat_request: Annotated[ChatRequest, Field(description="Chat request object")]
) -> Annotated[ChatResponse, Field(description="List of related articles from Wikipedia")]:
    """
    This function searches Wikipedia with the specified keywords and returns related articles.
    """
    client = LLMFactory.create_llm_client(LLMConfig(), chat_request)
    return await client.chat()


async def run_simple_chat(
        prompt: Annotated[str, Field(description="Prompt for the chat")],
) -> Annotated[str, Field(description="Chat response from the LLM")]:
    """
    This function processes a simple chat with the specified prompt and returns the chat response.
    """
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    response = await llm_client.simple_chat(prompt)
    return response

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
    llm_client = LLMFactory.create_llm_client(LLMConfig())
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
    llm_client = LLMFactory.create_llm_client(LLMConfig())
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
    llm_client = LLMFactory.create_llm_client(LLMConfig())
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
    llm_client = LLMFactory.create_llm_client(LLMConfig())
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
    llm_client = LLMFactory.create_llm_client(LLMConfig())
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
    llm_client = LLMFactory.create_llm_client(LLMConfig())
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
    llm_client = LLMFactory.create_llm_client(LLMConfig())
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
    llm_client = LLMFactory.create_llm_client(LLMConfig())
    response = await llm_client.analyze_files(file_path_list, prompt, detail=detail)
    return response.output