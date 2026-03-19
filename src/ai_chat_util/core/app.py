from typing import Annotated, Literal
import tempfile
import atexit
from pydantic import Field
from ai_chat_util_base.model.ai_chatl_util_models import ChatHistory, ChatResponse, WebRequestModel, ChatRequest, ChatMessage, ChatContent
from ai_chat_util.llm.llm_factory import LLMFactory
from ai_chat_util.llm.llm_client import LLMClientUtil
from ai_chat_util_base.config.runtime import get_runtime_config
from ai_chat_util.llm.llm_batch_client import LLMBatchClient
from file_util.model import FileUtilDocument
from ai_chat_util.util.file_path_resolver import resolve_existing_file_path
from ai_chat_util_base.config.runtime import get_runtime_config

# toolは実行時にmcp.tool()で登録する。@mcp.toolは使用しない。
# chat_utilのrun_chat_asyncを呼び出すラッパー関数を定義
async def run_chat(
        chat_request: Annotated[ChatRequest, Field(description="Chat request object")],
        use_mcp: Annotated[bool, Field(description="Whether to use MCP for the chat or not")]=False
) -> Annotated[ChatResponse, Field(description="List of related articles from Wikipedia")]:
    """
    This function searches Wikipedia with the specified keywords and returns related articles.
    """
    client = LLMFactory.create_llm_client(use_mcp=use_mcp)
    return await client.chat(chat_request)


async def run_simple_chat(
        prompt: Annotated[str, Field(description="Prompt for the chat")],
) -> Annotated[str, Field(description="Chat response from the LLM")]:
    """
    This function processes a simple chat with the specified prompt and returns the chat response.
    """
    llm_client = LLMFactory.create_llm_client(use_mcp=False)
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
    batch_client = LLMBatchClient(use_mcp=False)
    results = await batch_client.run_simple_batch_chat(prompt, messages, concurrency)
    return results

async def run_batch_chat(
        chat_requests: Annotated[list[ChatRequest], Field(description="List of chat histories for batch processing")],
        use_mcp: Annotated[bool, Field(description="Whether to use MCP for the chat or not")]=False,
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[ChatResponse], Field(description="List of chat responses from batch processing")]:
    """
    This function processes a batch of chat histories concurrently and returns the list of chat responses.
    """
    batch_client = LLMBatchClient(use_mcp=use_mcp)
    results = await batch_client.run_batch_chat(chat_requests, concurrency)
    return [response for _, response in results]

async def run_batch_chat_from_excel(
        prompt: Annotated[str, Field(description="Prompt for the batch chat")],
        input_excel_path: Annotated[str, Field(description="Path to the input Excel file")],
        output_excel_path: Annotated[str, Field(description="Path to the output Excel file")]="output.xlsx",
        content_column: Annotated[str, Field(description="Name of the column containing input messages")]="content",
        file_path_column: Annotated[str, Field(description="Name of the column containing file paths")]="file_path",
        output_column: Annotated[str, Field(description="Name of the column to store output responses")]="output",
        detail: Annotated[str, Field(description="Detail level for file analysis. e.g., 'low', 'high', 'auto'")]= "auto",
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=16,
        use_mcp: Annotated[bool, Field(description="Whether to use MCP for the chat or not")]=False
) -> None:
    """
    This function reads chat histories from an Excel file, processes them in batch, and writes the responses to a new Excel file.
    """
    batch_client = LLMBatchClient(use_mcp=use_mcp)
    await batch_client.run_batch_chat_from_excel(
        prompt,
        input_excel_path,
        output_excel_path,
        content_column,
        file_path_column,
        output_column,
        detail,
        concurrency,
    )

