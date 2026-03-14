from typing import Annotated, Literal
import tempfile
import atexit
from pydantic import Field
from ai_chat_util.model.models import ChatHistory, ChatResponse, WebRequestModel, ChatRequest, ChatMessage, ChatContent
from ai_chat_util.llm.llm_factory import LLMFactory
from ai_chat_util.llm.llm_client import LLMClientUtil
from ai_chat_util.config.runtime import get_runtime_config
from ai_chat_util.llm.batch_client import LLMBatchClient
from file_util.model import FileUtilDocument
from ai_chat_util.util.file_path_resolver import resolve_existing_file_path
from ai_chat_util.config.runtime import get_runtime_config


def _resolve_existing_file_paths(file_path_list: list[str]) -> list[str]:
    """ユーザー入力のパスを、実在するパスへ解決して返す。"""
    llm_config = get_runtime_config()
    resolved: list[str] = []
    for p in file_path_list:
        r = resolve_existing_file_path(p, working_directory=llm_config.paths.working_directory)
        resolved.append(r.resolved_path)
    return resolved

# 複数の画像の分析を行う URLから画像をダウンロードして分析する 
async def analyze_image_urls(
        image_path_urls: Annotated[list[WebRequestModel], Field(description="List of urls to the image files to analyze. e.g., http://path/to/image1.jpg")],
        prompt: Annotated[str, Field(description="Prompt to analyze the images")],
        detail: Annotated[str, Field(description="Detail level for image analysis. e.g., 'low', 'high', 'auto'")]= "auto"
    ) -> Annotated[str, Field(description="Analysis result of the images")]:
    """
    This function analyzes multiple images using the specified prompt and returns the analysis result.
    """
    llm_client = LLMFactory.create_llm_client()
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
    llm_client = LLMFactory.create_llm_client()
    resolved_paths = _resolve_existing_file_paths(file_list)
    response = await llm_client.analyze_image_files(resolved_paths, prompt, detail)
    return response.output


# 複数のPDFの分析を行う URLからPDFをダウンロードして分析する
async def analyze_pdf_urls(
        pdf_path_urls: Annotated[
            list[WebRequestModel],
            Field(
                description="List of URLs to the PDF files to analyze. e.g., http://path/to/document2.pdf"
            ),
        ],
        prompt: Annotated[str, Field(description="Prompt to analyze the PDFs")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
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
    llm_client = LLMFactory.create_llm_client()
    path_list = LLMClientUtil.download_files(pdf_path_urls, tmpdir.name)
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
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the PDFs")]:
    """
    This function analyzes multiple PDFs using the specified prompt and returns the analysis result.
    """
    llm_client = LLMFactory.create_llm_client()
    resolved_paths = _resolve_existing_file_paths(pdf_path_list)
    response = await llm_client.analyze_pdf_files(resolved_paths, prompt, detail)
    return response.output

# 複数のOfficeドキュメントの分析を行う URLからOfficeドキュメントをダウンロードして分析する
async def analyze_office_urls(
        office_path_urls: Annotated[list[WebRequestModel], Field(description="List of urls to the Office files to analyze. e.g., http://path/to/document1.docx")],
        prompt: Annotated[str, Field(description="Prompt to analyze the Office documents")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
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
    llm_client = LLMFactory.create_llm_client()
    path_list = LLMClientUtil.download_files(office_path_urls, tmpdir.name)

    response = await llm_client.analyze_office_files(path_list, prompt, detail)
    return response.output

async def analyze_office_files(
        office_path_list: Annotated[list[str], Field(description="List of absolute paths to the Office files to analyze. e.g., [/path/to/document1.docx, /path/to/spreadsheet1.xlsx]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the Office documents")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the Office documents")]:
    """
    This function analyzes multiple Office documents using the specified prompt and returns the analysis result.
    """ 
    llm_client = LLMFactory.create_llm_client()
    resolved_paths = _resolve_existing_file_paths(office_path_list)
    response = await llm_client.analyze_office_files(resolved_paths, prompt, detail=detail)
    return response.output

async def analyze_urls(
        file_path_urls: Annotated[list[WebRequestModel], Field(description="List of urls to the files to analyze. e.g., http://path/to/document1.pdf, http://path/to/image1.jpg")],
        prompt: Annotated[str, Field(description="Prompt to analyze the files")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
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
    llm_client = LLMFactory.create_llm_client()
    path_list = LLMClientUtil.download_files(file_path_urls, tmpdir.name)
    response = await llm_client.analyze_files(path_list, prompt, detail)
    return response.output

async def analyze_files(
        file_path_list: Annotated[list[str], Field(description="List of absolute paths to the files to analyze. e.g., [/path/to/document1.pdf, /path/to/image1.jpg]")],
        prompt: Annotated[str, Field(description="Prompt to analyze the files")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the files")]:
    """
    This function analyzes multiple files of various formats using the specified prompt and returns the analysis result.
    """
    llm_client = LLMFactory.create_llm_client()
    resolved_paths = _resolve_existing_file_paths(file_path_list)
    response = await llm_client.analyze_files(resolved_paths, prompt, detail=detail)
    return response.output

async def analyze_documents_data(
        document_type_list: Annotated[list[FileUtilDocument], Field(description="List of FileUtilDocument objects to analyze.")],
        prompt: Annotated[str, Field(description="Prompt to analyze the documents")],
        detail: Annotated[
            str,
            Field(
                description=(
                    "Parameter used when features.use_custom_pdf_analyzer is enabled. "
                    "Detail level for analysis. e.g., 'low', 'high', 'auto'"
                )
            ),
        ] = "auto",
    ) -> Annotated[str, Field(description="Analysis result of the files")]:
    """
    This function analyzes multiple files of various formats using the specified prompt and returns the analysis result.
    """
    llm_client = LLMFactory.create_llm_client()
    response = await llm_client.analyze_documents_data(document_type_list, prompt, detail=detail)
    return response.output
