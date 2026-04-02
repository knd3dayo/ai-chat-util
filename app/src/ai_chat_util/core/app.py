from typing import Annotated, Literal

from pydantic import Field

from ai_chat_util.base.agent import DeepAgentBatchClient, MCPBatchClient
from ai_chat_util.base.agent.agent_client_factory import AgentFactory
from ai_chat_util.base.batch import BatchClient
from ai_chat_util.base.chat import create_llm_client
from ai_chat_util.common.model.ai_chatl_util_models import ChatContent, ChatHistory, ChatMessage, ChatRequest, ChatResponse, WebRequestModel


async def run_chat(
        chat_request: Annotated[ChatRequest, Field(description="Chat request object")],
) -> Annotated[ChatResponse, Field(description="List of related articles from Wikipedia")]:
    """
    This function processes a chat request with the standard LLM client.
    """
    client = create_llm_client()
    return await client.chat(chat_request)


async def run_agent_chat(
        chat_request: Annotated[ChatRequest, Field(description="Chat request object")],
) -> Annotated[ChatResponse, Field(description="Agent chat response")]:
    """
    This function processes a chat request with the MCP-backed agent client.
    """
    client = AgentFactory.create_mcp_client()
    return await client.chat(chat_request)


async def run_deepagent_chat(
        chat_request: Annotated[ChatRequest, Field(description="Chat request object")],
) -> Annotated[ChatResponse, Field(description="DeepAgent chat response")]:
    """
    This function processes a chat request with the MCP-backed DeepAgent client.
    """
    client = AgentFactory.create_deepagent_client()
    return await client.chat(chat_request)


async def run_simple_chat(
        prompt: Annotated[str, Field(description="Prompt for the chat")],
) -> Annotated[str, Field(description="Chat response from the LLM")]:
    """
    This function processes a simple chat with the specified prompt and returns the chat response.
    """
    llm_client = create_llm_client()
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
    batch_client = BatchClient()
    results = await batch_client.run_simple_batch_chat(prompt, messages, concurrency)
    return results


async def run_batch_chat(
        chat_requests: Annotated[list[ChatRequest], Field(description="List of chat histories for batch processing")],
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[ChatResponse], Field(description="List of chat responses from batch processing")]:
    """
    This function processes a batch of chat histories with the standard LLM client.
    """
    batch_client = BatchClient()
    results = await batch_client.run_batch_chat(chat_requests, concurrency)
    return [response for _, response in results]


async def run_agent_batch_chat(
        chat_requests: Annotated[list[ChatRequest], Field(description="List of chat histories for agent batch processing")],
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[ChatResponse], Field(description="List of agent chat responses from batch processing")]:
    """
    This function processes a batch of chat histories with the MCP-backed agent client.
    """
    batch_client = MCPBatchClient()
    results = await batch_client.run_batch_chat(chat_requests, concurrency)
    return [response for _, response in results]


async def run_deepagent_batch_chat(
        chat_requests: Annotated[list[ChatRequest], Field(description="List of chat histories for DeepAgent batch processing")],
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=5
) -> Annotated[list[ChatResponse], Field(description="List of DeepAgent chat responses from batch processing")]:
    """
    This function processes a batch of chat histories with the MCP-backed DeepAgent client.
    """
    batch_client = DeepAgentBatchClient()
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
) -> None:
    """
    This function reads chat histories from an Excel file, processes them in batch with the standard LLM client, and writes the responses to a new Excel file.
    """
    batch_client = BatchClient()
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


async def run_agent_batch_chat_from_excel(
        prompt: Annotated[str, Field(description="Prompt for the agent batch chat")],
        input_excel_path: Annotated[str, Field(description="Path to the input Excel file")],
        output_excel_path: Annotated[str, Field(description="Path to the output Excel file")]="output.xlsx",
        content_column: Annotated[str, Field(description="Name of the column containing input messages")]="content",
        file_path_column: Annotated[str, Field(description="Name of the column containing file paths")]="file_path",
        output_column: Annotated[str, Field(description="Name of the column to store output responses")]="output",
        detail: Annotated[str, Field(description="Detail level for file analysis. e.g., 'low', 'high', 'auto'")]= "auto",
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=16,
) -> None:
    """
    This function reads chat histories from an Excel file, processes them in batch with the MCP-backed agent client, and writes the responses to a new Excel file.
    """
    batch_client = MCPBatchClient()
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


async def run_deepagent_batch_chat_from_excel(
        prompt: Annotated[str, Field(description="Prompt for the DeepAgent batch chat")],
        input_excel_path: Annotated[str, Field(description="Path to the input Excel file")],
        output_excel_path: Annotated[str, Field(description="Path to the output Excel file")]="output.xlsx",
        content_column: Annotated[str, Field(description="Name of the column containing input messages")]="content",
        file_path_column: Annotated[str, Field(description="Name of the column containing file paths")]="file_path",
        output_column: Annotated[str, Field(description="Name of the column to store output responses")]="output",
        detail: Annotated[str, Field(description="Detail level for file analysis. e.g., 'low', 'high', 'auto'")]= "auto",
        concurrency: Annotated[int, Field(description="Number of concurrent requests to process")]=16,
) -> None:
    """
    This function reads chat histories from an Excel file, processes them in batch with the MCP-backed DeepAgent client, and writes the responses to a new Excel file.
    """
    batch_client = DeepAgentBatchClient()
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