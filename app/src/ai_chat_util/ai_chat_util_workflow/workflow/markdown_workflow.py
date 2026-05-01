from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...ai_chat_util_base.chat.model import HitlRequest
from .mermaid_models import MermaidCodeBlock


class WorkflowToolReference(BaseModel):
    route_name: str = Field(default="")
    name: str = Field(...)
    description: str = Field(default="")
    primary_args: list[str] = Field(default_factory=list)
    requires_approval: bool = Field(default=False)
    action_kind: str = Field(default="")
    supports_dry_run: bool = Field(default=False)
    usage_guidance: str = Field(default="")
    tool_metadata: dict[str, str] = Field(default_factory=dict)


class WorkflowMarkdownDocument(BaseModel):
    original_markdown: str = Field(default="")
    body_markdown: str = Field(default="")
    mermaid_block: MermaidCodeBlock = Field(...)
    updated_markdown: str = Field(default="")
    available_tools: list[WorkflowToolReference] = Field(default_factory=list)
    tool_catalog_text: str = Field(default="")

    @property
    def mermaid_code(self) -> str:
        return self.mermaid_block.code

    def render_markdown(self) -> str:
        return self.updated_markdown or self.original_markdown

    @classmethod
    def from_markdown(
        cls,
        markdown: str,
        *,
        available_tools: list[WorkflowToolReference] | None = None,
        tool_catalog_text: str = "",
    ) -> "WorkflowMarkdownDocument":
        from ai_chat_util.ai_chat_util_workflow.mermaid.mermaid_flowchart import MermaidFlowChart

        block = MermaidFlowChart.extract_single_mermaid_block(markdown)
        body_markdown = (markdown[: block.start_index] + markdown[block.end_index :]).strip()
        return cls(
            original_markdown=markdown,
            body_markdown=body_markdown,
            mermaid_block=block,
            updated_markdown=markdown,
            available_tools=list(available_tools or []),
            tool_catalog_text=tool_catalog_text,
        )


class WorkflowExecutionResponse(BaseModel):
    status: Literal["completed", "paused"] = Field(default="completed")
    final_output: str = Field(default="")
    prepared_markdown: str = Field(default="")
    flowchart_code: str = Field(default="")
    tool_catalog_text: str = Field(default="")
    hitl: HitlRequest | None = Field(default=None)
    thread_id: str = Field(default="")
    execution_order: list[str] = Field(default_factory=list)
    node_outputs: dict[str, str] = Field(default_factory=dict)
    branch_history: list[dict[str, str]] = Field(default_factory=list)
    visit_counts: dict[str, int] = Field(default_factory=dict)