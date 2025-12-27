import re
from ai_chat_util.mermaid_agent.workflow.flowchart import GraphNode, GraphEdge, Subgraph
from ai_chat_util.mermaid_agent.workflow.flowchart import Flowchart

class MermaidFlowChart(Flowchart):

    def __init__(self, **data):
        super().__init__(**data)
        if self.code:
            self.parse(self.code)

    # markdownファイルからmermaidコード部分を取得するメソッド
    @staticmethod
    def extract_mermaid_code( markdown: str) -> list[str]:
        pattern = r"```mermaid(.*?)```"
        matches = re.findall(pattern, markdown, re.DOTALL)
        return [match.strip() for match in matches]

    def parse(self, code: str) :
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        subgraphs: dict[str, list[str]] = {}
        direction = "TD"  # デフォルトは Top Down

        current_subgraph = None

        lines = code.strip().splitlines()

        for line in lines:
            line = line.strip()

            # グラフ方向の検出
            if line.startswith("graph"):
                match = re.match(r"graph\s+(\w+)", line)
                if match:
                    direction = match.group(1)

            # サブグラフの開始
            elif line.startswith("subgraph"):
                match = re.match(r"subgraph\s+(.+)", line)
                if match:
                    current_subgraph = match.group(1).strip()
                    subgraphs[current_subgraph] = []

            # サブグラフの終了
            elif line == "end":
                current_subgraph = None

            # エッジ定義（A[xxx] --> B[yyy]、A --> B、A -->|yes| B すべて対応）
            elif "-->" in line:
                # ノードIDとラベルを同時に抽出するパターン
                # 例: A[ラベルA] -->|yes| B[ラベルB]
                pattern = r'(\w+)(?:\[(.*?)\])?\s*-->\s*(?:\|(.+?)\|\s*)?(\w+)(?:\[(.*?)\])?'
                matches = re.findall(pattern, line)
                for match in matches:
                    source_id, source_label, edge_label, target_id, target_label = match
                    # ノードが未定義の場合は仮定義する（ラベルがあればラベルを使う）
                    if source_id not in nodes:
                        nodes[source_id] = GraphNode(id=source_id, label=source_label if source_label else source_id)
                        if current_subgraph:
                            subgraphs[current_subgraph].append(source_id)
                    else:
                        # 既存ノードでもラベルが空ならラベルを更新
                        if source_label and nodes[source_id].label == source_id:
                            nodes[source_id].label = source_label
                    if target_id not in nodes:
                        nodes[target_id] = GraphNode(id=target_id, label=target_label if target_label else target_id)
                        if current_subgraph:
                            subgraphs[current_subgraph].append(target_id)
                    else:
                        if target_label and nodes[target_id].label == target_id:
                            nodes[target_id].label = target_label
                    edges.append(GraphEdge(source=nodes[source_id], target=nodes[target_id], label=edge_label or ""))
                
            # --> を含まない場合、ノード定義（例: A[Start]）
            # A[Start] --> B[End]と、同じ行にノード定義が複数ある場合も対応
            elif re.match(r'\w+\s*\[.*?\]', line):
                pattern = r'(\w+)\s*\[(.*?)\]'
                matches = re.findall(pattern, line)
                for node_id, label in matches:
                    # すでにノードが存在する場合はスキップ
                    if node_id in nodes:
                        continue
                    nodes[node_id] = GraphNode(id=node_id, label=label)
                    if current_subgraph:
                        subgraphs[current_subgraph].append(node_id)

        node_list = list(nodes.values())
        subgraph_list = [Subgraph(name=name, nodes=nodes) for name, nodes in subgraphs.items()]

        self.direction=direction
        self.nodes=node_list
        self.edges=edges
        self.subgraphs=subgraph_list


if __name__ == "__main__":
    # ✅ 使用例
    mermaid_code = """
    graph LR
        A[ユーザーが指定したディレクトリのファイル一覧を取得] --> B[test.txtが存在するか確認]
        B -->|<yes>| C[ファイルの内容を読み込み]
        B -->|<no>| D[処理を終了]
        C --> D
    """
    flowchart = MermaidFlowChart(code=mermaid_code)
    json_str = flowchart.model_dump_json(indent=2)
    print(json_str)
