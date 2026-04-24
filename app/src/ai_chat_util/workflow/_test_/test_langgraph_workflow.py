from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ai_chat_util.workflow import MermaidFlowChart, WorkflowChatClient, WorkflowRunner, WorkflowSessionStore, execute_workflow_markdown
from ai_chat_util.workflow.workflow.flowchat import Flowchart, GraphEdge, GraphNode
from ai_chat_util.workflow.workflow.langgraph_builder import NodeExecutionResult, WorkflowState
from ai_chat_util.workflow.workflow.markdown_workflow import WorkflowMarkdownDocument, WorkflowToolReference
from ai_chat_util.workflow.workflow.runner import WorkflowPauseResult, apply_markdown_context_to_flowchart
from ai_chat_util.ai_chat_util_agent.core.agent_client_util import AgentClientUtil


class FakeExecutor:
    async def __call__(self, node: GraphNode, state: WorkflowState, flowchart: MermaidFlowChart) -> NodeExecutionResult:
        if node.kind == "start":
            return {"output_text": "started"}
        if node.kind == "decision":
            visits = dict(state.get("visit_counts") or {})
            if visits.get(node.id, 0) <= 1:
                return {"output_text": "continue", "selected_edge": "yes"}
            return {"output_text": "complete", "selected_edge": "no"}
        if node.kind == "summary":
            outputs = state.get("node_outputs") or {}
            text = f"summary::{outputs.get('Work', '')}"
            return {"output_text": text, "summary_text": text}
        if node.kind == "end":
            summary = state.get("summary") or state.get("last_output") or ""
            return {"output_text": f"done::{summary}", "summary_text": f"done::{summary}"}
        return {"output_text": f"task::{node.id}"}


class FakeMarkdownPreprocessor:
    def __init__(self, updated_markdown: str):
        self.updated_markdown = updated_markdown

    async def load_tool_references(self) -> list[WorkflowToolReference]:
        return [
            WorkflowToolReference(
                route_name="general_tool_agent",
                name="analyze_files",
                description="Analyze files in the workspace",
                primary_args=["file_path_list", "prompt"],
            )
        ]

    async def prepare_document(self, markdown: str, *, message: str = "") -> WorkflowMarkdownDocument:
        tools = await self.load_tool_references()
        document = WorkflowMarkdownDocument.from_markdown(
            self.updated_markdown,
            available_tools=tools,
            tool_catalog_text="- analyze_files: Analyze files in the workspace",
        )
        document.updated_markdown = self.updated_markdown
        return document


def test_mermaid_parser_infers_branch_loop_and_summary() -> None:
    flowchart = MermaidFlowChart(
        code="""
        flowchart TD
            Start([Start]) --> Decide{Need more work?}
            Decide -->|yes| Work[Handle task]
            Work --> Decide
            Decide -->|no| Summary[summary: Gather the work result]
            Summary --> End([End])
        """
    )

    assert flowchart.get_start_node().id == "Start"
    assert flowchart.get_node("Decide").kind == "decision"
    assert flowchart.get_node("Summary").kind == "summary"
    assert flowchart.has_cycles() is True
    assert [edge.label for edge in flowchart.get_edges_from("Decide")] == ["yes", "no"]


def test_markdown_with_multiple_mermaid_blocks_is_rejected() -> None:
    markdown = """
    # Workflow

    ```mermaid
    flowchart TD
        A[Start] --> B[End]
    ```

    ```mermaid
    flowchart TD
        C[Start] --> D[End]
    ```
    """

    with pytest.raises(ValueError, match="exactly one mermaid block"):
        MermaidFlowChart.from_markdown(markdown)


@pytest.mark.anyio
async def test_workflow_runner_executes_branch_loop_and_summary() -> None:
    flowchart = MermaidFlowChart(
        code="""
        flowchart TD
            Start([Start]) --> Decide{Need more work?}
            Decide -->|yes| Work[Handle task]
            Work --> Decide
            Decide -->|no| Summary[summary: Gather the work result]
            Summary --> End([End])
        """
    )

    runner = WorkflowRunner(flowchart=flowchart, max_node_visits=4)
    runner.set_executor(FakeExecutor())

    result = await runner.run("run the workflow", recursion_limit=25)

    assert result.execution_order == ["Start", "Decide", "Work", "Decide", "Summary", "End"]
    assert result.branch_history == [
        {"node_id": "Decide", "selected_edge": "yes", "next_node_id": "Work"},
        {"node_id": "Decide", "selected_edge": "no", "next_node_id": "Summary"},
    ]
    assert result.visit_counts["Decide"] == 2
    assert result.node_outputs["Summary"] == "summary::task::Work"
    assert result.final_output == "done::summary::task::Work"


@pytest.mark.anyio
async def test_plan_mode_returns_paused_response_with_updated_markdown() -> None:
    original_markdown = """
    # Workflow

    This workflow should inspect available files.

    ```mermaid
    flowchart TD
        Start([Start]) --> End([End])
    ```
    """
    updated_markdown = """
    # Workflow

    This workflow should inspect available files.

    ```mermaid
    flowchart TD
        Start([Start]) --> Work[Use analyze_files]
        Work --> End([End])
    ```
    """

    response = await execute_workflow_markdown(
        original_markdown,
        message="inspect repository files",
        plan_mode=True,
        markdown_preprocessor=FakeMarkdownPreprocessor(updated_markdown),
        node_executor=FakeExecutor(),
    )

    assert response.status == "paused"
    assert response.hitl is not None
    assert response.hitl.kind == "approval"
    assert "Use analyze_files" in response.prepared_markdown


@pytest.mark.anyio
async def test_execute_workflow_markdown_uses_approved_markdown() -> None:
    approved_markdown = """
    # Workflow

    ```mermaid
    flowchart TD
        Start([Start]) --> Decide{Need more work?}
        Decide -->|yes| Work[Handle task]
        Work --> Decide
        Decide -->|no| Summary[summary: Gather the work result]
        Summary --> End([End])
    ```
    """

    response = await execute_workflow_markdown(
        approved_markdown,
        message="run the workflow",
        approved_markdown=approved_markdown,
        markdown_preprocessor=FakeMarkdownPreprocessor(approved_markdown),
        node_executor=FakeExecutor(),
        max_node_visits=4,
        recursion_limit=25,
    )

    assert response.status == "completed"
    assert response.execution_order == ["Start", "Decide", "Work", "Decide", "Summary", "End"]
    assert response.final_output == "done::summary::task::Work"


@pytest.mark.anyio
async def test_workflow_chat_client_resumes_after_approval(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    markdown_path = tmp_path / "workflow.md"
    markdown_path.write_text(
        """
        # Workflow

        ```mermaid
        flowchart TD
            Start([Start]) --> End([End])
        ```
        """,
        encoding="utf-8",
    )
    updated_markdown = """
    # Workflow

    ```mermaid
    flowchart TD
        Start([Start]) --> Work[Handle task]
        Work --> End([End])
    ```
    """

    session_store = WorkflowSessionStore(tmp_path / ".workflow_sessions")
    client = WorkflowChatClient(str(markdown_path), plan_mode=True, session_store=session_store)

    async def _fake_execute(markdown: str, **kwargs):
        approved_markdown = str(kwargs.get("approved_markdown") or "")
        thread_id = str(kwargs.get("thread_id") or "trace-1")
        if approved_markdown:
            from ai_chat_util.workflow.workflow.markdown_workflow import WorkflowExecutionResponse

            return WorkflowExecutionResponse(
                status="completed",
                final_output="done",
                prepared_markdown=approved_markdown,
                flowchart_code="flowchart TD\n Start([Start]) --> Work[Handle task]\n Work --> End([End])",
                tool_catalog_text="",
                thread_id=thread_id,
            )
        from ai_chat_util.common.model.ai_chatl_util_models import HitlRequest
        from ai_chat_util.workflow.workflow.markdown_workflow import WorkflowExecutionResponse

        return WorkflowExecutionResponse(
            status="paused",
            final_output=updated_markdown,
            prepared_markdown=updated_markdown,
            flowchart_code="flowchart TD\n Start([Start]) --> Work[Handle task]\n Work --> End([End])",
            tool_catalog_text="",
            hitl=HitlRequest(kind="approval", prompt=updated_markdown, action_id="1", source="workflow:plan"),
            thread_id=thread_id,
        )

    monkeypatch.setattr("ai_chat_util.workflow.chat_client.execute_workflow_markdown", _fake_execute)

    from ai_chat_util.common.model.ai_chatl_util_models import ChatContent, ChatHistory, ChatMessage, ChatRequest

    first = await client.chat(
        ChatRequest(
            trace_id="1234567890abcdef1234567890abcdef",
            chat_history=ChatHistory(messages=[ChatMessage(role="user", content=[ChatContent(params={"type": "text", "text": "plan it"})])]),
        )
    )
    assert first.status == "paused"
    assert first.trace_id == "1234567890abcdef1234567890abcdef"

    second_client = WorkflowChatClient(str(markdown_path), plan_mode=True, session_store=session_store)
    second = await second_client.chat(
        ChatRequest(
            trace_id="1234567890abcdef1234567890abcdef",
            chat_history=ChatHistory(messages=[ChatMessage(role="user", content=[ChatContent(params={"type": "text", "text": "APPROVE"})])]),
        )
    )
    assert second.status == "completed"
    assert second.output == "done"


@pytest.mark.anyio
async def test_runner_pauses_and_resumes_approval_node() -> None:
    flowchart = MermaidFlowChart(
        nodes=[
            GraphNode(id="Start", label="Start", kind="start"),
            GraphNode(id="Approve", label="Approval", kind="approval", metadata={"approval_target_label": "Write files", "approval_tool_names": "convert_pdf_files_to_images"}),
            GraphNode(id="Work", label="Write files", kind="task"),
            GraphNode(id="End", label="End", kind="end"),
        ],
        edges=[
            GraphEdge(source="Start", target="Approve"),
            GraphEdge(source="Approve", target="Work"),
            GraphEdge(source="Work", target="End"),
        ],
    )

    class _Executor:
        async def __call__(self, node, state, _flowchart):
            if node.id == "Start":
                return {"output_text": "start"}
            if node.id == "Work":
                return {"output_text": "write-done"}
            return {"output_text": "done", "summary_text": "done"}

    runner = WorkflowRunner(flowchart=flowchart)
    runner.set_executor(_Executor())
    saver = InMemorySaver()

    paused = await runner.run("go", thread_id="wf-1", checkpointer=saver)
    assert isinstance(paused, WorkflowPauseResult)
    assert paused.interrupt_payload["kind"] == "approval"

    resumed = await runner.run("go", thread_id="wf-1", checkpointer=saver, resume_value="APPROVE")
    assert resumed.final_output == "done"
    assert resumed.execution_order == ["Start", "Approve", "Work", "End"]


def test_apply_markdown_context_injects_review_nodes() -> None:
    flowchart = MermaidFlowChart(
        code="flowchart TD\nStart([Start]) --> Work[Write files]\nWork[Write files] --> End([End])",
        nodes=[
            GraphNode(id="Start", label="Start", kind="start"),
            GraphNode(id="Work", label="Write files", kind="task"),
            GraphNode(id="End", label="End", kind="end"),
        ],
        edges=[
            GraphEdge(source="Start", target="Work"),
            GraphEdge(source="Work", target="End"),
        ],
    )
    document = WorkflowMarkdownDocument(
        original_markdown="# wf\n\n```mermaid\nflowchart TD\nStart([Start]) --> Work[Write files] --> End([End])\n```\n",
        body_markdown="# wf",
        mermaid_block=MermaidFlowChart.extract_single_mermaid_block(
            "# wf\n\n```mermaid\nflowchart TD\nStart([Start]) --> Work[Write files]\nWork[Write files] --> End([End])\n```\n"
        ),
        available_tools=[
            WorkflowToolReference(
                name="convert_pdf_files_to_images",
                description="write tool",
                primary_args=["pdf_path_list", "dry_run"],
                action_kind="write",
                supports_dry_run=True,
                tool_metadata={"action_kind": "write", "requires_approval": "true"},
            ),
        ],
    )

    updated = apply_markdown_context_to_flowchart(flowchart, document, enable_tool_approval_nodes=True)
    node_ids = {node.id for node in updated.nodes}
    assert "Work__dry_run" in node_ids
    assert "Work__approval" in node_ids


def test_extract_tool_metadata_from_mcp_meta_block() -> None:
    description = "Convert PDF pages into images\n\n[MCP_META]\nrequires_approval=true\naction_kind=write\nusage_guidance=Call dry_run first"
    metadata = AgentClientUtil._extract_tool_metadata(description)

    assert metadata["requires_approval"] == "true"
    assert metadata["action_kind"] == "write"
    assert metadata["usage_guidance"] == "Call dry_run first"
    assert AgentClientUtil._normalize_tool_description(description) == "Convert PDF pages into images"