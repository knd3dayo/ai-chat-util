from __future__ import annotations

import contextlib
import re
import uuid
from typing import Any, Protocol, Sequence

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from pydantic import BaseModel, Field, PrivateAttr

import ai_chat_util.ai_chat_util_base.core.log.log_settings as log_settings
from ...ai_chat_util_agent.core.agent_builder import AgentBuilder
from ...ai_chat_util_agent.core.agent_client_util import AgentClientUtil
from ...ai_chat_util_agent.core.prompts import CodingAgentPrompts
from ...ai_chat_util_agent.core.tool_limits import ToolLimits
from ...ai_chat_util_base.core.chat.model import HitlRequest
from ai_chat_util.ai_chat_util_base.core.common.config.runtime import AiChatUtilConfig, get_runtime_config
from ..mermaid.mermaid_flowchart import MermaidFlowChart
from .flowchat import Flowchart, GraphNode
from .langgraph_builder import (
    LangGraphWorkflowBuilder,
    NodeExecutionResult,
    WorkflowState,
)
from .markdown_workflow import (
    WorkflowExecutionResponse,
    WorkflowMarkdownDocument,
    WorkflowToolReference,
)

logger = log_settings.getLogger(__name__)


class WorkflowMarkdownPreprocessorProtocol(Protocol):
    async def prepare_document(self, markdown: str, *, message: str = "") -> WorkflowMarkdownDocument:
        ...

    async def load_tool_references(self) -> list[WorkflowToolReference]:
        ...


class WorkflowToolAgentRuntime:
    def __init__(self, runtime_config: AiChatUtilConfig):
        self.runtime_config = runtime_config
        self.llm = AgentClientUtil.create_llm(runtime_config)
        self.prompts = CodingAgentPrompts()
        self.tool_limits = ToolLimits.from_config(runtime_config)
        self._agent_cache: dict[tuple[str, ...], Any] = {}

    async def invoke(
        self,
        *,
        node: GraphNode,
        flowchart: Flowchart,
        state: WorkflowState,
        allowed_tool_names: Sequence[str],
        allowed_tools_text: str,
        execution_mode: str = "normal",
        auto_approve_tools: bool = False,
    ) -> str:
        normalized_names = tuple(sorted({name.strip() for name in allowed_tool_names if isinstance(name, str) and name.strip()}))
        if not normalized_names:
            return ""

        agent = await self._get_or_create_agent(normalized_names, auto_approve_tools=auto_approve_tools)
        execution_directive = ""
        analysis_path_directive = ""
        if execution_mode == "dry_run":
            execution_directive = (
                "このノードでは preview のみを行ってください。"
                "dry_run 引数を持つツールは必ず dry_run=True を明示し、実際の更新はしてはいけません。"
            )
        elif auto_approve_tools:
            execution_directive = (
                "このノードのツール実行は人間承認済みです。"
                "dry_run 引数を持つツールを使う場合は dry_run=False を明示して実実行してください。"
            )
        if {"analyze_files", "analyze_image_files", "analyze_pdf_files", "analyze_office_files"} & set(normalized_names):
            analysis_path_directive = (
                "ファイル解析ツールへ渡す path は、ユーザー入力または直前のツール結果に現れた実在パスだけを使ってください。"
                "ディレクトリ確認要求では個別ファイル名を推測せず、ディレクトリパス自体を渡してください。"
                "ユーザーが指定していない config ファイルや仮想パスへ置き換えてはいけません。"
            )
        prompt = (
            f"ワークフロー仕様Markdown:\n{flowchart.markdown or flowchart.code}\n\n"
            f"現在のノードID: {node.id}\n"
            f"ノードの役割: {node.label}\n"
            f"利用可能ツール:\n{allowed_tools_text}\n"
            f"ユーザー入力: {state.get('input_text', '')}\n"
            f"これまでのノード出力:\n{self._previous_outputs_text(state)}\n\n"
            f"実行モード指示: {execution_directive or '通常実行'}\n"
            f"追加制約: {analysis_path_directive or 'なし'}\n\n"
            "必要ならツールを使ってノードの目的を達成してください。"
            "最終出力は system prompt の XML 形式に従ってください。"
        )
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={
                "configurable": {"thread_id": state.get("workflow_trace_id") or uuid.uuid4().hex},
                "recursion_limit": self.tool_limits.tool_recursion_limit,
            },
        )
        output_text, _input_tokens, _output_tokens = AgentClientUtil._extract_output_and_usage(result)
        response_type, extracted_text, hitl_kind, hitl_tool = self._parse_tool_agent_xml(output_text)
        if response_type == "question":
            if hitl_kind == "approval":
                tool_label = hitl_tool or ", ".join(normalized_names)
                return (
                    f"承認が必要なツール {tool_label} を含む要求のため、この workflow ノードでは自動実行しません。\n"
                    f"{extracted_text or output_text}"
                )
            return extracted_text or output_text
        return extracted_text or output_text

    async def _get_or_create_agent(self, allowed_tool_names: tuple[str, ...], *, auto_approve_tools: bool) -> Any:
        cache_key = ("auto" if auto_approve_tools else "guarded",) + allowed_tool_names
        cached = self._agent_cache.get(cache_key)
        if cached is not None:
            return cached

        mcp_config = self.runtime_config.get_mcp_server_config()
        agent_client = MultiServerMCPClient(mcp_config.to_langchain_config())
        langchain_tools = await agent_client.get_tools()
        filtered_tools = [tool for tool in langchain_tools if str(getattr(tool, "name", "")).strip() in set(allowed_tool_names)]

        inferred_approval_tools = AgentBuilder.infer_approval_tools_from_langchain_tools(filtered_tools)
        tools_description = self.prompts.create_tools_description(filtered_tools)
        approval_tools_text = ", ".join(inferred_approval_tools) if inferred_approval_tools else "(なし)"
        hitl_policy_text = (
            self.prompts.auto_approve_hitl_policy_text(approval_tools_text)
            if auto_approve_tools
            else self.prompts.normal_hitl_policy_text(approval_tools_text)
        )
        system_prompt = self.prompts.tool_agent_system_prompt(
            hitl_policy_text,
            agent_name=AgentBuilder.build_tool_agent_name(group_label="workflow_node"),
            followup_poll_interval_seconds=float(getattr(self.runtime_config.features, "mcp_followup_poll_interval_seconds", 2.0) or 0.0),
            status_tail_lines=int(getattr(self.runtime_config.features, "mcp_status_tail_lines", 20) or 0),
            result_tail_lines=int(getattr(self.runtime_config.features, "mcp_get_result_tail_lines", 80) or 0),
        )
        agent = create_agent(
            self.llm,
            filtered_tools,
            system_prompt=system_prompt + "\n" + self.prompts.tool_agent_user_prompt(tools_description, hitl_policy_text),
            name=AgentBuilder.build_tool_agent_name(group_label="workflow_node"),
        )
        self._agent_cache[cache_key] = agent
        return agent

    @staticmethod
    def _previous_outputs_text(state: WorkflowState) -> str:
        node_outputs = state.get("node_outputs") or {}
        return "\n".join(f"- {node_id}: {text}" for node_id, text in node_outputs.items()) or "(none)"

    @staticmethod
    def _parse_tool_agent_xml(output_text: str) -> tuple[str | None, str | None, str | None, str | None]:
        text = output_text or ""
        m_text = re.search(r"<TEXT>\s*(.*?)\s*</TEXT>", text, flags=re.IGNORECASE | re.DOTALL)
        m_type = re.search(r"<RESPONSE_TYPE>\s*(.*?)\s*</RESPONSE_TYPE>", text, flags=re.IGNORECASE | re.DOTALL)
        m_kind = re.search(r"<HITL_KIND>\s*(.*?)\s*</HITL_KIND>", text, flags=re.IGNORECASE | re.DOTALL)
        m_tool = re.search(r"<HITL_TOOL>\s*(.*?)\s*</HITL_TOOL>", text, flags=re.IGNORECASE | re.DOTALL)
        return (
            m_type.group(1).strip().lower() if m_type else None,
            m_text.group(1).strip() if m_text else None,
            m_kind.group(1).strip().lower() if m_kind else None,
            m_tool.group(1).strip() if m_tool else None,
        )


class WorkflowRunResult(BaseModel):
    final_output: str = ""
    summary: str = ""
    execution_order: list[str] = Field(default_factory=list)
    node_outputs: dict[str, str] = Field(default_factory=dict)
    branch_history: list[dict[str, str]] = Field(default_factory=list)
    visit_counts: dict[str, int] = Field(default_factory=dict)
    thread_id: str = ""


class WorkflowPauseResult(BaseModel):
    thread_id: str = ""
    interrupt_payload: dict[str, Any] = Field(default_factory=dict)


class DefaultWorkflowNodeExecutor:
    def __init__(self, runtime_config: AiChatUtilConfig | None = None):
        self.runtime_config = runtime_config or get_runtime_config()
        self.llm = AgentClientUtil.create_llm(self.runtime_config)
        self.tool_agent_runtime = WorkflowToolAgentRuntime(self.runtime_config)

    async def __call__(self, node: GraphNode, state: WorkflowState, flowchart: Flowchart) -> NodeExecutionResult:
        outgoing_edges = flowchart.get_edges_from(node.id)
        node_outputs = state.get("node_outputs") or {}
        previous_outputs_text = "\n".join(f"- {node_id}: {text}" for node_id, text in node_outputs.items()) or "(none)"
        workflow_markdown = flowchart.markdown or flowchart.code
        allowed_tools_text = node.metadata.get("allowed_tools_text", "(none)")
        allowed_tool_names = [
            name.strip()
            for name in str(node.metadata.get("allowed_tool_names") or "").split(",")
            if name.strip()
        ]
        dry_run_tool_names = [
            name.strip()
            for name in str(node.metadata.get("dry_run_tool_names") or "").split(",")
            if name.strip()
        ]
        approval_tool_names = [
            name.strip()
            for name in str(node.metadata.get("approval_tool_names") or "").split(",")
            if name.strip()
        ]

        if node.kind == "dry_run":
            if not dry_run_tool_names:
                return {"output_text": "Dry run skipped."}
            tool_output = await self.tool_agent_runtime.invoke(
                node=node,
                flowchart=flowchart,
                state=state,
                allowed_tool_names=dry_run_tool_names,
                allowed_tools_text=node.metadata.get("dry_run_tools_text", allowed_tools_text),
                execution_mode="dry_run",
                auto_approve_tools=True,
            )
            return {"output_text": tool_output or "Dry run completed."}

        if node.kind == "summary":
            summary_prompt = node.metadata.get("summary_prompt") or node.label
            result = await self.llm.ainvoke(
                [
                    SystemMessage(content="あなたはWF型エージェントの要約ノードです。簡潔で実務向けの要約だけを返してください。"),
                    HumanMessage(
                        content=(
                            f"要約指示:\n{summary_prompt}\n\n"
                            f"ワークフロー仕様Markdown:\n{workflow_markdown}\n\n"
                            f"利用可能ツール:\n{allowed_tools_text}\n\n"
                            f"これまでの各ノード出力:\n{previous_outputs_text}"
                        )
                    ),
                ]
            )
            text = self._stringify_message_content(getattr(result, "content", result))
            return {"output_text": text, "summary_text": text}

        if node.kind == "decision":
            labels = [edge.label for edge in outgoing_edges if edge.label.strip()]
            result = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "あなたはWF型エージェントの判定ノードです。"
                            "与えられた候補ラベルのどれか1つだけを選んでください。"
                            "出力形式は必ず <DECISION>選んだラベル</DECISION><OUTPUT>説明</OUTPUT> にしてください。"
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"判定ノード: {node.label}\n"
                            f"ユーザー入力: {state.get('input_text', '')}\n"
                            f"ワークフロー仕様Markdown:\n{workflow_markdown}\n"
                            f"利用可能ツール:\n{allowed_tools_text}\n"
                            f"これまでの出力:\n{previous_outputs_text}\n"
                            f"選択可能な分岐ラベル: {labels}"
                        )
                    ),
                ]
            )
            text = self._stringify_message_content(getattr(result, "content", result))
            selected_edge = self._extract_tag(text, "DECISION")
            output_text = self._extract_tag(text, "OUTPUT") or text
            return {"output_text": output_text, "selected_edge": selected_edge}

        if node.kind in {"task", "end"} and allowed_tool_names:
            tool_output = await self.tool_agent_runtime.invoke(
                node=node,
                flowchart=flowchart,
                state=state,
                allowed_tool_names=allowed_tool_names,
                allowed_tools_text=allowed_tools_text,
                auto_approve_tools=bool(approval_tool_names),
            )
            if tool_output.strip():
                if node.kind == "end":
                    return {"output_text": tool_output, "summary_text": tool_output}
                return {"output_text": tool_output}

        system_prompt = (
            "あなたはWF型エージェントのノード実行器です。"
            "指定されたノードの責務だけを実行し、次ノードが利用できる簡潔な結果を返してください。"
        )
        if node.kind == "end":
            system_prompt = (
                "あなたはWF型エージェントの終了ノードです。"
                "これまでの流れを踏まえて最終結果だけを簡潔に返してください。"
            )

        result = await self.llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"ワークフロー仕様Markdown:\n{workflow_markdown}\n\n"
                        f"現在のノードID: {node.id}\n"
                        f"ノードの役割: {node.label}\n"
                        f"利用可能ツール:\n{allowed_tools_text}\n"
                        f"ユーザー入力: {state.get('input_text', '')}\n"
                        f"これまでのノード出力:\n{previous_outputs_text}"
                    )
                ),
            ]
        )
        text = self._stringify_message_content(getattr(result, "content", result))
        if node.kind == "end":
            return {"output_text": text, "summary_text": text}
        return {"output_text": text}

    @staticmethod
    def _extract_tag(text: str, tag_name: str) -> str:
        match = re.search(
            rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>",
            text or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _stringify_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)


class WorkflowRunner(BaseModel):
    flowchart: Flowchart = Field(..., description="The flowchart representing the workflow")
    runtime_config: AiChatUtilConfig | None = Field(default=None, description="Runtime config for default node execution")
    max_node_visits: int = Field(default=8, ge=1, description="Safety limit for loop execution")

    _executor: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        return None

    def set_executor(self, executor: Any) -> None:
        self._executor = executor

    def build(self, *, checkpointer: Any | None = None):
        if self._executor is None:
            self._executor = DefaultWorkflowNodeExecutor(self.runtime_config)
        builder = LangGraphWorkflowBuilder(self.flowchart, self._executor)
        return builder.build(checkpointer=checkpointer)

    async def run(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        checkpointer: Any | None = None,
        recursion_limit: int | None = None,
        resume_value: str = "",
    ) -> WorkflowRunResult | WorkflowPauseResult:
        graph = self.build(checkpointer=checkpointer)
        effective_thread_id = thread_id or str(uuid.uuid4())
        invocation: Any
        if resume_value.strip():
            invocation = Command(resume=resume_value)
        else:
            invocation = {
                "input_text": message,
                "workflow_trace_id": effective_thread_id,
                "node_outputs": {},
                "execution_order": [],
                "branch_history": [],
                "visit_counts": {},
                "max_node_visits": self.max_node_visits,
            }
        result = await graph.ainvoke(
            invocation,
            config={
                "configurable": {"thread_id": effective_thread_id},
                "recursion_limit": recursion_limit or max(25, self.max_node_visits * max(len(self.flowchart.nodes), 1)),
            },
        )
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts:
            first_interrupt = interrupts[0]
            payload = getattr(first_interrupt, "value", first_interrupt)
            normalized_payload = payload if isinstance(payload, dict) else {"prompt": str(payload), "kind": "approval"}
            return WorkflowPauseResult(thread_id=effective_thread_id, interrupt_payload=dict(normalized_payload))
        logger.info("Workflow completed thread_id=%s node_count=%s", effective_thread_id, len(self.flowchart.nodes))
        return WorkflowRunResult(
            final_output=str(result.get("final_output") or result.get("last_output") or ""),
            summary=str(result.get("summary") or ""),
            execution_order=list(result.get("execution_order") or []),
            node_outputs=dict(result.get("node_outputs") or {}),
            branch_history=list(result.get("branch_history") or []),
            visit_counts=dict(result.get("visit_counts") or {}),
            thread_id=effective_thread_id,
        )


class DefaultWorkflowMarkdownPreprocessor:
    def __init__(self, runtime_config: AiChatUtilConfig | None = None):
        self.runtime_config = runtime_config or get_runtime_config()
        self.llm = AgentClientUtil.create_llm(self.runtime_config)

    async def load_tool_references(self) -> list[WorkflowToolReference]:
        route_tool_inventory = await AgentClientUtil.resolve_route_tool_inventory(runtime_config=self.runtime_config)
        tools: list[WorkflowToolReference] = []
        for route_name, route_tools in route_tool_inventory.items():
            for tool in route_tools:
                tool_name = str(tool.get("name") or "").strip()
                if not tool_name:
                    continue
                tools.append(
                    WorkflowToolReference(
                        route_name=str(route_name).strip(),
                        name=tool_name,
                        description=str(tool.get("description") or "").strip(),
                        primary_args=[
                            str(arg_name).strip()
                            for arg_name in (tool.get("primary_args") or [])
                            if str(arg_name).strip()
                        ],
                        requires_approval=_metadata_bool(tool.get("tool_metadata"), "requires_approval")
                        or bool(AgentBuilder._APPROVAL_METADATA_PATTERN.search(str(tool.get("description") or ""))),
                        action_kind=str(_metadata_value(tool.get("tool_metadata"), "action_kind") or "").strip(),
                        supports_dry_run=_metadata_bool(tool.get("tool_metadata"), "supports_dry_run")
                        or any(str(arg_name).strip().lower() == "dry_run" for arg_name in (tool.get("primary_args") or [])),
                        usage_guidance=str(_metadata_value(tool.get("tool_metadata"), "usage_guidance") or "").strip(),
                        tool_metadata={
                            str(key).strip(): str(value).strip()
                            for key, value in dict(tool.get("tool_metadata") or {}).items()
                            if str(key).strip()
                        },
                    )
                )
        unique_by_name: dict[str, WorkflowToolReference] = {}
        for tool in tools:
            unique_by_name.setdefault(tool.name, tool)
        return list(unique_by_name.values())

    async def prepare_document(self, markdown: str, *, message: str = "") -> WorkflowMarkdownDocument:
        tool_references = await self.load_tool_references()
        route_tool_inventory: dict[str, list[dict[str, Any]]] = {}
        for tool in tool_references:
            route_tool_inventory.setdefault(tool.route_name or "general_tool_agent", []).append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "primary_args": tool.primary_args,
                }
            )
        tool_catalog_text = AgentClientUtil.build_tool_catalog_response_text(
            route_tool_inventory,
            include_details=True,
        )
        original_document = WorkflowMarkdownDocument.from_markdown(
            markdown,
            available_tools=tool_references,
            tool_catalog_text=tool_catalog_text,
        )

        result = await self.llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "あなたは WF 型エージェント用の Markdown ワークフロー設計補正器です。"
                        "与えられた Markdown 全体を更新し、本文と利用可能ツールに整合する単一の mermaid flowchart を含む Markdown を返してください。"
                        "必ず mermaid block は1つだけにしてください。"
                        "mermaid 以外の本文も維持しつつ、必要に応じて補足説明を加えてください。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"ユーザー入力:\n{message or '(none)'}\n\n"
                        f"現在の Markdown:\n{markdown}\n\n"
                        f"利用可能な MCP ツール一覧:\n{tool_catalog_text}\n\n"
                        "要件:\n"
                        "- Markdown 全体を返すこと\n"
                        "- mermaid block は必ず1つだけにすること\n"
                        "- 利用不可能なツール前提の手順は書かないこと\n"
                        "- 要約ノードが必要なら summary: で表現すること"
                    )
                ),
            ]
        )
        candidate_text = DefaultWorkflowNodeExecutor._stringify_message_content(getattr(result, "content", result)).strip()
        updated_markdown = self._coerce_updated_markdown(original_document, candidate_text)
        updated_document = WorkflowMarkdownDocument.from_markdown(
            updated_markdown,
            available_tools=tool_references,
            tool_catalog_text=tool_catalog_text,
        )
        updated_document.updated_markdown = updated_markdown
        return updated_document

    @staticmethod
    def _coerce_updated_markdown(document: WorkflowMarkdownDocument, candidate_text: str) -> str:
        stripped = candidate_text.strip()
        if not stripped:
            return document.render_markdown()
        try:
            MermaidFlowChart.extract_single_mermaid_block(stripped)
            return stripped
        except ValueError:
            pass

        if stripped.lower().startswith(("graph ", "flowchart ")):
            return MermaidFlowChart.replace_single_mermaid_block(document.original_markdown, stripped)

        return stripped


def _tokenize_for_tool_matching(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+|[ぁ-んァ-ン一-龥]+", text or "")
        if len(token.strip()) >= 2
    }


def _metadata_value(metadata: Any, key: str) -> str:
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get(key) or "").strip()


def _metadata_bool(metadata: Any, key: str) -> bool:
    value = _metadata_value(metadata, key).lower()
    return value in {"1", "true", "yes", "y", "on"}


def _select_relevant_tools(
    node: GraphNode,
    *,
    document: WorkflowMarkdownDocument,
    tool_references: Sequence[WorkflowToolReference],
    include_interactive_tools: bool,
) -> list[WorkflowToolReference]:
    if node.kind in {"start", "summary", "approval", "dry_run"}:
        return []

    query_tokens = _tokenize_for_tool_matching(node.label + "\n" + document.body_markdown)
    scored: list[tuple[int, WorkflowToolReference]] = []
    for tool in tool_references:
        if not include_interactive_tools and _requires_human_review(tool):
            continue
        tool_tokens = _tokenize_for_tool_matching(
            tool.name + "\n" + tool.description + "\n" + " ".join(tool.primary_args)
        )
        score = len(query_tokens & tool_tokens)
        if score > 0:
            scored.append((score, tool))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [tool for _, tool in scored[:5]]


def _requires_human_review(tool: WorkflowToolReference) -> bool:
    return bool(
        tool.requires_approval
        or tool.action_kind.lower() == "write"
        or tool.supports_dry_run
        or any(arg.lower() == "dry_run" for arg in tool.primary_args)
    )


def _inject_review_nodes(flowchart: Flowchart) -> Flowchart:
    rewritten_targets: dict[str, str] = {}
    added_nodes: list[GraphNode] = []
    added_edges: list[tuple[str, str]] = []

    for node in flowchart.nodes:
        approval_tool_names = [name.strip() for name in str(node.metadata.get("approval_tool_names") or "").split(",") if name.strip()]
        dry_run_tool_names = [name.strip() for name in str(node.metadata.get("dry_run_tool_names") or "").split(",") if name.strip()]
        if node.kind not in {"task", "end"} or not approval_tool_names:
            continue

        approval_node_id = f"{node.id}__approval"
        first_target = approval_node_id
        if dry_run_tool_names:
            dry_run_node_id = f"{node.id}__dry_run"
            added_nodes.append(
                GraphNode(
                    id=dry_run_node_id,
                    label=f"Dry Run: {node.label}",
                    kind="dry_run",
                    metadata={
                        "allowed_tool_names": node.metadata.get("dry_run_tool_names", ""),
                        "allowed_tools_text": node.metadata.get("dry_run_tools_text", "(none)"),
                        "dry_run_tool_names": node.metadata.get("dry_run_tool_names", ""),
                        "dry_run_tools_text": node.metadata.get("dry_run_tools_text", "(none)"),
                    },
                )
            )
            added_edges.append((dry_run_node_id, approval_node_id))
            first_target = dry_run_node_id

        added_nodes.append(
            GraphNode(
                id=approval_node_id,
                label=f"Approval: {node.label}",
                kind="approval",
                metadata={
                    "approval_tool_names": node.metadata.get("approval_tool_names", ""),
                    "approval_target_node_id": node.id,
                    "approval_target_label": node.label,
                    "hitl_source": "workflow:tool_approval",
                },
            )
        )
        added_edges.append((approval_node_id, node.id))
        rewritten_targets[node.id] = first_target

    if not rewritten_targets:
        return flowchart

    flowchart.edges = [
        edge.model_copy(update={"target": rewritten_targets.get(edge.target, edge.target)})
        for edge in flowchart.edges
    ]
    for source, target in added_edges:
        from ai_chat_util.ai_chat_util_workflow.workflow.flowchat import GraphEdge

        flowchart.edges.append(GraphEdge(source=source, target=target))
    flowchart.nodes.extend(added_nodes)
    return flowchart


def apply_markdown_context_to_flowchart(
    flowchart: Flowchart,
    document: WorkflowMarkdownDocument,
    *,
    enable_tool_approval_nodes: bool = False,
) -> Flowchart:
    flowchart.markdown = document.render_markdown()
    flowchart.tool_catalog_text = document.tool_catalog_text
    for node in flowchart.nodes:
        relevant_tools = _select_relevant_tools(
            node,
            document=document,
            tool_references=document.available_tools,
            include_interactive_tools=enable_tool_approval_nodes,
        )
        auto_tools = [tool for tool in relevant_tools if not _requires_human_review(tool)]
        approval_tools = [tool for tool in relevant_tools if _requires_human_review(tool)]
        dry_run_tools = [tool for tool in approval_tools if any(arg.lower() == "dry_run" for arg in tool.primary_args)]
        effective_tools = relevant_tools if enable_tool_approval_nodes else auto_tools
        node.metadata["allowed_tool_names"] = ", ".join(tool.name for tool in effective_tools) if effective_tools else ""
        node.metadata["allowed_tools_text"] = (
            "\n".join(f"- {tool.name}: {tool.description}" for tool in effective_tools) if effective_tools else "(none)"
        )
        node.metadata["approval_tool_names"] = ", ".join(tool.name for tool in approval_tools) if approval_tools else ""
        node.metadata["dry_run_tool_names"] = ", ".join(tool.name for tool in dry_run_tools) if dry_run_tools else ""
        node.metadata["dry_run_tools_text"] = (
            "\n".join(
                f"- {tool.name}: {tool.description}" + (f" ({tool.usage_guidance})" if tool.usage_guidance else "")
                for tool in dry_run_tools
            ) if dry_run_tools else "(none)"
        )
    return _inject_review_nodes(flowchart) if enable_tool_approval_nodes else flowchart


@contextlib.asynccontextmanager
async def _create_workflow_checkpointer(runtime_config: AiChatUtilConfig):
    async with contextlib.AsyncExitStack() as exit_stack:
        db_path = AgentClientUtil._default_checkpoint_db_path(runtime_config)
        checkpointer = await AgentClientUtil._create_sqlite_checkpointer(db_path, exit_stack=exit_stack)
        yield checkpointer


async def execute_workflow_markdown(
    markdown: str,
    *,
    message: str = "",
    runtime_config: AiChatUtilConfig | None = None,
    markdown_preprocessor: WorkflowMarkdownPreprocessorProtocol | None = None,
    node_executor: Any | None = None,
    max_node_visits: int = 8,
    plan_mode: bool = False,
    approved_markdown: str = "",
    thread_id: str | None = None,
    recursion_limit: int | None = None,
    resume_value: str = "",
    durable: bool = False,
    enable_tool_approval_nodes: bool = False,
) -> WorkflowExecutionResponse:
    effective_runtime_config = runtime_config or get_runtime_config()
    preprocessor = markdown_preprocessor or DefaultWorkflowMarkdownPreprocessor(effective_runtime_config)

    if approved_markdown.strip():
        tool_references = await preprocessor.load_tool_references()
        route_tool_inventory: dict[str, list[dict[str, Any]]] = {}
        for tool in tool_references:
            route_tool_inventory.setdefault(tool.route_name or "general_tool_agent", []).append(
                {"name": tool.name, "description": tool.description, "primary_args": tool.primary_args}
            )
        tool_catalog_text = AgentClientUtil.build_tool_catalog_response_text(route_tool_inventory, include_details=True)
        document = WorkflowMarkdownDocument.from_markdown(
            approved_markdown,
            available_tools=tool_references,
            tool_catalog_text=tool_catalog_text,
        )
        document.updated_markdown = approved_markdown
    else:
        document = await preprocessor.prepare_document(markdown, message=message)

    if plan_mode and not approved_markdown.strip():
        prepared_markdown = document.render_markdown()
        return WorkflowExecutionResponse(
            status="paused",
            final_output=prepared_markdown,
            prepared_markdown=prepared_markdown,
            flowchart_code=document.mermaid_code,
            tool_catalog_text=document.tool_catalog_text,
            hitl=HitlRequest(
                kind="approval",
                prompt=(
                    "更新済みの Markdown ワークフロー案です。内容を確認して、実行してよければ APPROVE を返してください。\n\n"
                    f"{prepared_markdown}"
                ),
                action_id=str(uuid.uuid4()),
                source="workflow:plan",
            ),
        )

    flowchart = MermaidFlowChart.from_markdown(document.render_markdown())
    flowchart = apply_markdown_context_to_flowchart(
        flowchart,
        document,
        enable_tool_approval_nodes=enable_tool_approval_nodes,
    )
    runner = WorkflowRunner(flowchart=flowchart, runtime_config=effective_runtime_config, max_node_visits=max_node_visits)
    if node_executor is not None:
        runner.set_executor(node_executor)
    if durable:
        async with _create_workflow_checkpointer(effective_runtime_config) as checkpointer:
            result = await runner.run(
                message,
                thread_id=thread_id,
                recursion_limit=recursion_limit,
                checkpointer=checkpointer,
                resume_value=resume_value,
            )
    else:
        result = await runner.run(
            message,
            thread_id=thread_id,
            recursion_limit=recursion_limit,
            resume_value=resume_value,
        )
    if isinstance(result, WorkflowPauseResult):
        payload = dict(result.interrupt_payload)
        prompt = str(payload.get("prompt") or "承認が必要です。")
        kind = str(payload.get("kind") or "approval")
        source = str(payload.get("source") or "workflow:tool_approval")
        return WorkflowExecutionResponse(
            status="paused",
            final_output=prompt,
            prepared_markdown=document.render_markdown(),
            flowchart_code=document.mermaid_code,
            tool_catalog_text=document.tool_catalog_text,
            hitl=HitlRequest(
                kind=kind,
                prompt=prompt,
                action_id=str(uuid.uuid4()),
                source=source,
            ),
            thread_id=result.thread_id,
        )
    return WorkflowExecutionResponse(
        status="completed",
        final_output=result.final_output,
        prepared_markdown=document.render_markdown(),
        flowchart_code=document.mermaid_code,
        tool_catalog_text=document.tool_catalog_text,
        hitl=None,
        thread_id=result.thread_id,
        execution_order=result.execution_order,
        node_outputs=result.node_outputs,
        branch_history=result.branch_history,
        visit_counts=result.visit_counts,
    )

