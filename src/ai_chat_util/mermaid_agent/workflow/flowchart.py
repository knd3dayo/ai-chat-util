from pydantic import BaseModel, Field
from typing import Optional
from abc import  abstractmethod
# node
class GraphNode(BaseModel):
    id: str
    label: str
# edge
class GraphEdge(BaseModel):
    source: GraphNode
    target: GraphNode
    label: Optional[str] = ""
# subgraph
class Subgraph(BaseModel):
    name: str
    nodes: list[str] = Field(default_factory=list)

# full flowchart
class Flowchart(BaseModel):
    direction: str = "TD"
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    subgraphs: list[Subgraph] = Field(default_factory=list)

    code: str = Field(default="", description="Mermaid flowchart code")


    # startのnodeを取得するメソッド。startが複数ある場合はエラーにする
    def get_start_node(self) -> GraphNode:
        target_node_ids = set(edge.source.id for edge in self.edges) - set(edge.target.id for edge in self.edges)
        start_nodes = [node for node in self.nodes if node.id in target_node_ids]
        if not start_nodes:
            raise ValueError("No start node found")
        if len(start_nodes) > 1:
            raise ValueError("Multiple start nodes found")
        return start_nodes[0]
    
    # endのnodeを取得するメソッド。endが複数ある場合はエラーにする
    def get_end_node(self) -> GraphNode:
        target_node_ids = set(edge.target for edge in self.edges) - set(edge.source for edge in self.edges)
        end_nodes = [node for node in self.nodes if node.id in target_node_ids]
        if not end_nodes:
            raise ValueError("No end node found")
        if len(end_nodes) > 1:
            raise ValueError("Multiple end nodes found")
        return end_nodes[0]

    # srcからのエッジを取得するメソッド
    def get_edges_from(self, src_node: GraphNode) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.source == src_node.id]

    # edgesからのターゲットノードを取得するメソッド
    def get_target_nodes_from(self, edge: GraphEdge) -> list[GraphNode]:
        target_edges = self.get_edges_from(edge.source)
        target_node_ids = [edge.target for edge in target_edges]
        return [node for node in self.nodes if node.id in target_node_ids]

    @abstractmethod
    def parse(self, code: str):
        pass