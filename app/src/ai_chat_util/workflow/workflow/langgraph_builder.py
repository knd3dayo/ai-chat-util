from __future__ import annotations

from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ai_chat_util.workflow.workflow.flowchat import Flowchart, GraphEdge, GraphNode


class WorkflowState(TypedDict, total=False):
    input_text: str
    workflow_trace_id: str
    current_node_id: str
    next_node_id: str | None
    last_output: str
    final_output: str
    summary: str
    node_outputs: dict[str, str]
    execution_order: list[str]
    branch_history: list[dict[str, str]]
    visit_counts: dict[str, int]
    max_node_visits: int


class NodeExecutionResult(TypedDict, total=False):
    output_text: str
    selected_edge: str | None
    summary_text: str | None


NodeExecutor = Callable[[GraphNode, WorkflowState, Flowchart], Awaitable[NodeExecutionResult]]


class LangGraphWorkflowBuilder:
    def __init__(self, flowchart: Flowchart, executor: NodeExecutor):
        self.flowchart = flowchart
        self.executor = executor

    def build(self, *, checkpointer: Any | None = None):
        graph = StateGraph(WorkflowState)
        for node in self.flowchart.nodes:
            graph.add_node(node.id, self._build_node_handler(node))

        start_node = self.flowchart.get_start_node()
        graph.add_edge(START, start_node.id)

        for node in self.flowchart.nodes:
            destinations = {edge.target: edge.target for edge in self.flowchart.get_edges_from(node.id)}
            destinations["__end__"] = END
            graph.add_conditional_edges(node.id, self._build_router(node), destinations)

        return graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()

    def _build_node_handler(self, node: GraphNode):
        async def _handler(state: WorkflowState) -> WorkflowState:
            visit_counts = dict(state.get("visit_counts") or {})
            execution_order = list(state.get("execution_order") or [])
            node_outputs = dict(state.get("node_outputs") or {})
            branch_history = list(state.get("branch_history") or [])
            max_visits = int(state.get("max_node_visits") or 1)

            current_visits = int(visit_counts.get(node.id, 0)) + 1
            if current_visits > max_visits:
                raise RuntimeError(
                    f"Workflow node visit limit exceeded: node={node.id} limit={max_visits}"
                )

            visit_counts[node.id] = current_visits
            execution_state: WorkflowState = dict(state)
            execution_state.update(
                {
                    "current_node_id": node.id,
                    "visit_counts": visit_counts,
                    "execution_order": execution_order,
                    "node_outputs": node_outputs,
                    "branch_history": branch_history,
                }
            )

            if node.kind == "approval":
                tool_names = str(node.metadata.get("approval_tool_names") or "").strip()
                preview_text = str(state.get("last_output") or "").strip()
                prompt_lines = [
                    f"ノード '{node.metadata.get('approval_target_label') or node.label}' の実行前承認が必要です。",
                ]
                if tool_names:
                    prompt_lines.append(f"対象ツール: {tool_names}")
                if preview_text:
                    prompt_lines.append("dry_run または事前確認結果:")
                    prompt_lines.append(preview_text)
                prompt_lines.append("実行してよければ APPROVE を返してください。拒否する場合は REJECT を返してください。")
                resume_value = str(
                    interrupt(
                        {
                            "kind": "approval",
                            "prompt": "\n\n".join(prompt_lines),
                            "source": str(node.metadata.get("hitl_source") or "workflow:tool_approval"),
                            "tool_names": tool_names,
                            "node_id": str(node.metadata.get("approval_target_node_id") or node.id),
                        }
                    )
                    or ""
                ).strip()
                approved = self._is_approval_text(resume_value)
                output_text = "Approval accepted." if approved else "Workflow execution was not approved."
                node_outputs[node.id] = output_text
                execution_order.append(node.id)

                update: WorkflowState = {
                    "current_node_id": node.id,
                    "next_node_id": self._default_next_node(node) if approved else None,
                    "last_output": output_text,
                    "node_outputs": node_outputs,
                    "execution_order": execution_order,
                    "branch_history": branch_history,
                    "visit_counts": visit_counts,
                }
                if not approved:
                    update["final_output"] = output_text
                return update

            result = await self.executor(node, execution_state, self.flowchart)
            output_text = str(result.get("output_text") or "")
            selected_edge = result.get("selected_edge")
            summary_text = str(result.get("summary_text") or "")

            node_outputs[node.id] = output_text
            execution_order.append(node.id)

            next_node_id = self._select_next_node(node, output_text=output_text, selected_edge=selected_edge)
            if selected_edge:
                branch_history.append({"node_id": node.id, "selected_edge": selected_edge, "next_node_id": next_node_id or ""})

            update: WorkflowState = {
                "current_node_id": node.id,
                "next_node_id": next_node_id,
                "last_output": output_text,
                "node_outputs": node_outputs,
                "execution_order": execution_order,
                "branch_history": branch_history,
                "visit_counts": visit_counts,
            }
            if node.kind in {"summary", "end"}:
                update["summary"] = summary_text or output_text
                update["final_output"] = summary_text or output_text
            elif next_node_id is None:
                update["final_output"] = output_text
            return update

        return _handler

    def _default_next_node(self, node: GraphNode) -> str | None:
        outgoing_edges = self.flowchart.get_edges_from(node.id)
        if not outgoing_edges:
            return None
        return outgoing_edges[0].target

    def _build_router(self, node: GraphNode):
        outgoing_edges = self.flowchart.get_edges_from(node.id)

        def _route(state: WorkflowState) -> str:
            next_node_id = state.get("next_node_id")
            if not outgoing_edges or not next_node_id:
                return "__end__"
            return str(next_node_id)

        return _route

    def _select_next_node(self, node: GraphNode, *, output_text: str, selected_edge: str | None) -> str | None:
        outgoing_edges = self.flowchart.get_edges_from(node.id)
        if not outgoing_edges:
            return None
        if len(outgoing_edges) == 1 and not outgoing_edges[0].is_conditional():
            return outgoing_edges[0].target

        normalized_selected = self._normalize_edge_label(selected_edge or self._extract_edge_label(output_text))
        if not normalized_selected:
            raise ValueError(f"Node {node.id} requires a branch selection but executor did not provide one")

        for edge in outgoing_edges:
            if self._normalize_edge_label(edge.label) == normalized_selected:
                return edge.target

        available_labels = [edge.label for edge in outgoing_edges]
        raise ValueError(
            f"Node {node.id} selected unknown branch '{normalized_selected}'. Available labels: {available_labels}"
        )

    @staticmethod
    def _extract_edge_label(output_text: str) -> str:
        text = output_text or ""
        for pattern in (
            r"<DECISION>\s*(.*?)\s*</DECISION>",
            r"<ROUTE>\s*(.*?)\s*</ROUTE>",
            r"<BRANCH>\s*(.*?)\s*</BRANCH>",
        ):
            import re

            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _normalize_edge_label(label: str | None) -> str:
        return str(label or "").strip().lower()

    @staticmethod
    def _is_approval_text(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        return normalized in {"approve", "approved", "yes", "y"} or normalized.startswith("approve ")