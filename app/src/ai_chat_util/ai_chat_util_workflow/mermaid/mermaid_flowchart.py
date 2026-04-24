from __future__ import annotations

import re

from ai_chat_util.ai_chat_util_workflow.workflow.flowchat import Flowchart, GraphEdge, GraphNode, NodeKind, Subgraph
from ai_chat_util.ai_chat_util_workflow.workflow.mermaid_models import MermaidCodeBlock


class MermaidFlowChart(Flowchart):
    _NODE_ID_PATTERN = r"[A-Za-z0-9_\-]+"
    _SHAPE_PATTERNS: tuple[tuple[str, str], ...] = (
        (r"^(?P<id>[A-Za-z0-9_\-]+)\{(?P<label>.*)\}$", "decision"),
        (r"^(?P<id>[A-Za-z0-9_\-]+)\(\[(?P<label>.*)\]\)$", "stadium"),
        (r"^(?P<id>[A-Za-z0-9_\-]+)\(\((?P<label>.*)\)\)$", "terminal"),
        (r"^(?P<id>[A-Za-z0-9_\-]+)\((?P<label>.*)\)$", "round"),
        (r"^(?P<id>[A-Za-z0-9_\-]+)\[\[(?P<label>.*)\]\]$", "subroutine"),
        (r"^(?P<id>[A-Za-z0-9_\-]+)\[(?P<label>.*)\]$", "rect"),
        (r"^(?P<id>[A-Za-z0-9_\-]+)$", "plain"),
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.code:
            self.parse(self.code)

    @staticmethod
    def extract_mermaid_code(markdown: str) -> list[str]:
        return [block.code for block in MermaidFlowChart.extract_mermaid_blocks(markdown)]

    @staticmethod
    def extract_mermaid_blocks(markdown: str) -> list[MermaidCodeBlock]:
        pattern = r"```mermaid\s*(.*?)```"
        blocks: list[MermaidCodeBlock] = []
        for match in re.finditer(pattern, markdown or "", re.DOTALL | re.IGNORECASE):
            code = match.group(1).strip()
            if not code:
                continue
            blocks.append(
                MermaidCodeBlock(
                    code=code,
                    full_text=match.group(0),
                    start_index=match.start(),
                    end_index=match.end(),
                )
            )
        return blocks

    @staticmethod
    def extract_single_mermaid_block(markdown: str) -> MermaidCodeBlock:
        blocks = MermaidFlowChart.extract_mermaid_blocks(markdown)
        if not blocks:
            raise ValueError("Markdown must contain exactly one mermaid block, but none were found")
        if len(blocks) > 1:
            raise ValueError("Markdown must contain exactly one mermaid block; multiple blocks are not supported yet")
        return blocks[0]

    @staticmethod
    def replace_single_mermaid_block(markdown: str, new_mermaid_code: str) -> str:
        block = MermaidFlowChart.extract_single_mermaid_block(markdown)
        replacement = f"```mermaid\n{new_mermaid_code.strip()}\n```"
        return markdown[: block.start_index] + replacement + markdown[block.end_index :]

    @classmethod
    def from_markdown(cls, markdown: str) -> "MermaidFlowChart":
        mermaid_block = cls.extract_single_mermaid_block(markdown)
        return cls(code=mermaid_block.code)

    def parse(self, code: str) -> None:
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        subgraphs: dict[str, list[str]] = {}
        direction = "TD"
        current_subgraph: str | None = None

        for raw_line in code.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("%%"):
                continue

            if line.startswith(("graph", "flowchart")):
                match = re.match(r"(?:graph|flowchart)\s+([A-Za-z]+)", line)
                if match:
                    direction = match.group(1)
                continue

            if line.startswith("subgraph"):
                current_subgraph = line[len("subgraph"):].strip()
                subgraphs.setdefault(current_subgraph, [])
                continue

            if line == "end":
                current_subgraph = None
                continue

            if "-->" in line:
                source_token, edge_label, target_token = self._parse_edge_line(line)
                source_node = self._get_or_create_node(nodes, source_token, current_subgraph, subgraphs)
                target_node = self._get_or_create_node(nodes, target_token, current_subgraph, subgraphs)
                edges.append(GraphEdge(source=source_node.id, target=target_node.id, label=edge_label or ""))
                continue

            standalone_node = self._parse_node_ref(line)
            if standalone_node is not None:
                self._upsert_node(nodes, standalone_node, current_subgraph, subgraphs)
                continue

            raise ValueError(f"Unsupported mermaid syntax: {line}")

        self.direction = direction
        self.nodes = list(nodes.values())
        self.edges = edges
        self.subgraphs = [Subgraph(name=name, nodes=node_ids) for name, node_ids in subgraphs.items()]
        self.code = code
        self._apply_graph_inference()

    def _parse_edge_line(self, line: str) -> tuple[str, str, str]:
        match = re.match(r"(?P<src>.+?)\s*-->\s*(?:\|(?P<label>.+?)\|\s*)?(?P<dst>.+)", line)
        if not match:
            raise ValueError(f"Unsupported mermaid edge syntax: {line}")
        source_token = match.group("src").strip()
        target_token = match.group("dst").strip()
        edge_label = (match.group("label") or "").strip()
        return source_token, edge_label, target_token

    def _get_or_create_node(
        self,
        nodes: dict[str, GraphNode],
        token: str,
        current_subgraph: str | None,
        subgraphs: dict[str, list[str]],
    ) -> GraphNode:
        parsed_node = self._parse_node_ref(token)
        if parsed_node is None:
            raise ValueError(f"Unsupported mermaid node syntax: {token}")
        return self._upsert_node(nodes, parsed_node, current_subgraph, subgraphs)

    def _upsert_node(
        self,
        nodes: dict[str, GraphNode],
        parsed_node: GraphNode,
        current_subgraph: str | None,
        subgraphs: dict[str, list[str]],
    ) -> GraphNode:
        existing = nodes.get(parsed_node.id)
        if existing is None:
            nodes[parsed_node.id] = parsed_node
            if current_subgraph:
                subgraphs.setdefault(current_subgraph, []).append(parsed_node.id)
            return parsed_node

        if existing.label == existing.id and parsed_node.label != parsed_node.id:
            existing.label = parsed_node.label
        if existing.kind == "task" and parsed_node.kind != "task":
            existing.kind = parsed_node.kind
        if existing.shape == "plain" and parsed_node.shape != "plain":
            existing.shape = parsed_node.shape
        existing.metadata.update(parsed_node.metadata)
        if current_subgraph:
            subgraphs.setdefault(current_subgraph, [])
            if parsed_node.id not in subgraphs[current_subgraph]:
                subgraphs[current_subgraph].append(parsed_node.id)
        return existing

    def _parse_node_ref(self, token: str) -> GraphNode | None:
        text = token.strip()
        for pattern, shape in self._SHAPE_PATTERNS:
            match = re.match(pattern, text)
            if not match:
                continue
            node_id = match.group("id")
            label = (match.groupdict().get("label") or node_id).strip().strip('"')
            kind, metadata = self._infer_node_kind(label, shape)
            return GraphNode(id=node_id, label=label, kind=kind, shape=shape, metadata=metadata)
        return None

    def _infer_node_kind(self, label: str, shape: str) -> tuple[NodeKind, dict[str, str]]:
        normalized = label.strip().lower()
        metadata: dict[str, str] = {}
        if shape == "decision":
            return "decision", metadata
        if normalized in {"start", "開始"}:
            return "start", metadata
        if normalized in {"end", "終了", "finish"}:
            return "end", metadata
        if normalized.startswith("summary:"):
            metadata["summary_prompt"] = label.split(":", 1)[1].strip()
            return "summary", metadata
        if normalized.startswith("要約:"):
            metadata["summary_prompt"] = label.split(":", 1)[1].strip()
            return "summary", metadata
        return "task", metadata
