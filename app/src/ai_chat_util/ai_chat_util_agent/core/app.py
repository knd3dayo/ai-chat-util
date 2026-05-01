from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from ai_chat_util.ai_chat_util_agent.core import DeepAgentBatchClient, MCPBatchClient
from ai_chat_util.ai_chat_util_agent.core.agent_client_factory import AgentFactory
from ai_chat_util.ai_chat_util_base.batch import BatchClient
from ai_chat_util.ai_chat_util_base.core.chat.core import create_llm_client
from ai_chat_util.ai_chat_util_base.core.common.config.runtime import AiChatUtilConfig, get_runtime_config
from ai_chat_util.ai_chat_util_base.core.chat.model import ChatContent, ChatHistory, ChatMessage, ChatRequest, ChatResponse, WebRequestModel
from ai_chat_util.ai_chat_util_base.request_headers import get_current_request_headers
from ai_chat_util.ai_chat_util_workflow import WorkflowExecutionResponse, WorkflowSessionStore, execute_workflow_markdown
from ai_chat_util.ai_chat_util_workflow.chat_client import WorkflowChatClient


def _resolve_workflow_trace_id(trace_id: str = "") -> str:
    normalized = str(trace_id or "").strip()
    if normalized:
        return normalized
    headers = get_current_request_headers()
    header_trace_id = str(getattr(headers, "trace_id", "") or "").strip()
    return header_trace_id


def _is_workflow_approval_text(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    return normalized in {"approve", "approved", "yes", "y"} or normalized.startswith("approve ")


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


async def run_mermaid_workflow_from_file(
        workflow_file_path: Annotated[str, Field(description="Path to a Markdown file containing exactly one mermaid block")],
        message: Annotated[str, Field(description="Initial input text passed into the workflow")]= "",
        max_node_visits: Annotated[int, Field(description="Loop safety limit for a single node")]=8,
        plan_mode: Annotated[bool, Field(description="If true, prepare updated markdown and pause for approval instead of executing")]=False,
        approved_markdown: Annotated[str, Field(description="Updated markdown approved by the client. When set, execution uses this markdown directly")]= "",
    durable: Annotated[bool, Field(description="If true, persist workflow execution state so HITL pauses can be resumed")]=False,
    resume_value: Annotated[str, Field(description="Resume text for a paused durable workflow")]= "",
    enable_tool_approval_nodes: Annotated[bool, Field(description="If true, inject dry-run and approval nodes for update tools")]=False,
    trace_id: Annotated[str, Field(description="Optional trace_id used as the workflow thread id")]= "",
) -> Annotated[WorkflowExecutionResponse, Field(description="Workflow execution response including plan/HITL status")]:
    """
    Execute a Markdown-defined WF workflow using LangGraph.
    """
    workflow_path = Path(workflow_file_path).expanduser().resolve()
    markdown = workflow_path.read_text(encoding="utf-8")
    return await execute_workflow_markdown(
        markdown,
        message=message,
        max_node_visits=max_node_visits,
        plan_mode=plan_mode,
        approved_markdown=approved_markdown,
        durable=durable,
        resume_value=resume_value,
        enable_tool_approval_nodes=enable_tool_approval_nodes,
        thread_id=_resolve_workflow_trace_id(trace_id),
    )


async def run_durable_workflow_from_file(
        workflow_file_path: Annotated[str, Field(description="Path to a Markdown file containing exactly one mermaid block")],
        message: Annotated[str, Field(description="Initial input text passed into the workflow")]= "",
        max_node_visits: Annotated[int, Field(description="Loop safety limit for a single node")]=8,
        plan_mode: Annotated[bool, Field(description="If true, prepare updated markdown and pause for approval instead of executing")]=False,
        trace_id: Annotated[str, Field(description="Optional trace_id used as the durable workflow thread id")]= "",
) -> Annotated[WorkflowExecutionResponse, Field(description="Durable workflow execution response")]:
    """
    Execute a Markdown-defined WF workflow using durable pause/resume semantics.
    """
    return await run_mermaid_workflow_from_file(
        workflow_file_path=workflow_file_path,
        message=message,
        max_node_visits=max_node_visits,
        plan_mode=plan_mode,
        durable=True,
        enable_tool_approval_nodes=True,
        trace_id=trace_id,
    )


async def resume_durable_workflow(
        resume_value: Annotated[str, Field(description="Approval or answer text used to resume a paused workflow")],
        trace_id: Annotated[str, Field(description="trace_id of the paused durable workflow")]= "",
) -> Annotated[WorkflowExecutionResponse, Field(description="Resumed durable workflow response")]:
    """
    Resume a paused durable workflow using its trace_id.
    """
    effective_trace_id = _resolve_workflow_trace_id(trace_id)
    if not effective_trace_id:
        raise ValueError("trace_id is required to resume a durable workflow")

    session_store = WorkflowSessionStore.from_runtime_config()
    session = session_store.load(effective_trace_id)
    if session is None:
        raise ValueError(f"Paused workflow session was not found for trace_id={effective_trace_id}")

    if session.phase == "plan":
        if not _is_workflow_approval_text(resume_value):
            session_store.delete(effective_trace_id)
            return WorkflowExecutionResponse(
                status="completed",
                final_output="Workflow execution was not approved.",
                prepared_markdown=session.prepared_markdown,
                thread_id=effective_trace_id,
            )
        response = await execute_workflow_markdown(
            session.original_markdown,
            message=session.message,
            approved_markdown=session.prepared_markdown,
            max_node_visits=session.max_node_visits,
            durable=True,
            enable_tool_approval_nodes=True,
            thread_id=effective_trace_id,
        )
    else:
        response = await execute_workflow_markdown(
            session.original_markdown,
            message=session.message,
            approved_markdown=session.prepared_markdown,
            max_node_visits=session.max_node_visits,
            resume_value=resume_value,
            durable=True,
            enable_tool_approval_nodes=True,
            thread_id=effective_trace_id,
        )

    if response.status == "paused":
        session_store.save(session.model_copy(update={"phase": "plan" if getattr(response.hitl, "source", "") == "workflow:plan" else "graph"}))
    else:
        session_store.delete(effective_trace_id)
    return response