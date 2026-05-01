from __future__ import annotations

from abc import abstractmethod
from collections import deque
from typing import Literal

from pydantic import BaseModel, Field, model_validator

NodeKind = Literal["start", "task", "decision", "summary", "dry_run", "approval", "end"]


class GraphNode(BaseModel):
    id: str
    label: str
    kind: NodeKind = "task"
    shape: str = "rect"
    metadata: dict[str, str] = Field(default_factory=dict)

    def normalized_label(self) -> str:
        return self.label.strip().lower()


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)

    def is_conditional(self) -> bool:
        return bool(self.label.strip())


class Subgraph(BaseModel):
    name: str
    nodes: list[str] = Field(default_factory=list)


class Flowchart(BaseModel):
    direction: str = "TD"
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    subgraphs: list[Subgraph] = Field(default_factory=list)
    code: str = Field(default="", description="Mermaid flowchart code")
    markdown: str = Field(default="", description="Markdown document used to derive the workflow")
    tool_catalog_text: str = Field(default="", description="Resolved MCP tool catalog available to the workflow")

    @model_validator(mode="after")
    def _normalize_graph(self) -> "Flowchart":
        return self._apply_graph_inference()

    def _apply_graph_inference(self) -> "Flowchart":
        known_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in known_ids:
                raise ValueError(f"Unknown edge source node: {edge.source}")
            if edge.target not in known_ids:
                raise ValueError(f"Unknown edge target node: {edge.target}")

        if self.nodes:
            incoming = {node.id: 0 for node in self.nodes}
            outgoing = {node.id: 0 for node in self.nodes}
            for edge in self.edges:
                outgoing[edge.source] += 1
                incoming[edge.target] += 1

            for node in self.nodes:
                if node.kind == "task":
                    normalized = node.normalized_label()
                    if normalized in {"start", "開始"}:
                        node.kind = "start"
                    elif normalized in {"end", "終了", "finish"}:
                        node.kind = "end"
                    elif normalized.startswith("summary:") or normalized.startswith("要約:"):
                        node.kind = "summary"
                        node.metadata.setdefault("summary_prompt", node.label.split(":", 1)[1].strip())

            start_candidates = [node for node in self.nodes if incoming[node.id] == 0]
            end_candidates = [node for node in self.nodes if outgoing[node.id] == 0]

            if len(start_candidates) == 1 and start_candidates[0].kind == "task":
                start_candidates[0].kind = "start"
            for node in end_candidates:
                if node.kind == "task":
                    node.kind = "end"
        return self

    def get_node(self, node_id: str) -> GraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"Unknown node id: {node_id}")

    def get_edges_from(self, src_node: GraphNode | str) -> list[GraphEdge]:
        node_id = src_node.id if isinstance(src_node, GraphNode) else src_node
        return [edge for edge in self.edges if edge.source == node_id]

    def get_edges_to(self, target_node: GraphNode | str) -> list[GraphEdge]:
        node_id = target_node.id if isinstance(target_node, GraphNode) else target_node
        return [edge for edge in self.edges if edge.target == node_id]

    def get_target_nodes_from(self, edge: GraphEdge) -> list[GraphNode]:
        return [self.get_node(target_edge.target) for target_edge in self.get_edges_from(edge.source)]

    def get_start_node(self) -> GraphNode:
        start_nodes = [node for node in self.nodes if node.kind == "start"]
        if len(start_nodes) == 1:
            return start_nodes[0]
        if len(start_nodes) > 1:
            raise ValueError("Multiple start nodes found")

        candidates = [node for node in self.nodes if not self.get_edges_to(node.id)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError("No start node found")
        raise ValueError("Multiple start nodes found")

    def get_end_nodes(self) -> list[GraphNode]:
        explicit = [node for node in self.nodes if node.kind == "end"]
        if explicit:
            return explicit
        return [node for node in self.nodes if not self.get_edges_from(node.id)]

    def get_end_node(self) -> GraphNode:
        end_nodes = self.get_end_nodes()
        if not end_nodes:
            raise ValueError("No end node found")
        if len(end_nodes) > 1:
            raise ValueError("Multiple end nodes found")
        return end_nodes[0]

    def has_cycles(self) -> bool:
        indegree = {node.id: 0 for node in self.nodes}
        adjacency = {node.id: [] for node in self.nodes}
        for edge in self.edges:
            indegree[edge.target] += 1
            adjacency[edge.source].append(edge.target)

        queue = deque([node_id for node_id, degree in indegree.items() if degree == 0])
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for target in adjacency[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        return visited != len(self.nodes)

    @abstractmethod
    def parse(self, code: str) -> None:
        raise NotImplementedError