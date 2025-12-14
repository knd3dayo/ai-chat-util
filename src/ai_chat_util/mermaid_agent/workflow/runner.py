from typing import Optional, Any
from pydantic import BaseModel, Field
from agent_framework import WorkflowBuilder,AgentRunUpdateEvent
from agent_framework import ChatAgent, AgentExecutorResponse
from ai_chat_util.mermaid_agent.util.agent_util import MSAIAgentUtil,  AppConfig
from ai_chat_util.mermaid_agent.workflow.flowchat import GraphNode, Flowchart
import ai_chat_util.mermaid_agent.log.log_settings as log_settings
from contextlib import AsyncExitStack
logger = log_settings.getLogger(__name__)

class WorkflowRunner(BaseModel):

    app_config: AppConfig = Field(default_factory=lambda: AppConfig(), description="Application configuration for the workflow runner")
    flowchart: Flowchart = Field(..., description="The flowchart representing the workflow")

    def _create_agent_from_node(self, workflow: Flowchart, node: GraphNode) -> ChatAgent:
        # Create an agent using OpenAI ChatCompletion
        agent_util = MSAIAgentUtil(app_config=self.app_config)
        params = agent_util.create_default_agent_params()

        # instructionsを更新
        node_instrunctions = f"""
        あなたはワークフロー内で動作するエージェント: {node.id} です。
        ワークフローの内容は以下の通りです。
        ---
        {workflow.code}
        ---
        あなたは以下の処理を行う役割を持っています。
        ---
        {node.label}
        ---
        * あなたの役割が**開始**や**Start**という場合は、ワークフローからユーザーの意図を整理してください。
        * あなたの役割が**終了**や**End**という場合は、ワークフローの処理結果をまとめて、を終了する旨をユーザーに伝えてください。
        * あなたの役割が、何かをチェックしてyes/noを判断する場合は、必ず出力に<yes>または<no>を含めてください。
        * これまでの文脈を踏まえて、あなたの役割を適切に対応してください。
        """

        params.instructions = node_instrunctions + "\n" + params.instructions
        params.name = f"{node.id}"

        return agent_util.create_agent(params=params)

    async def run(self, message: str):
        
        # start nodeを取得
        start_node = self.flowchart.get_start_node()

        # AsyncExitStackで複数エージェントを安全に管理
        clients = {}
        async with AsyncExitStack() as stack:
            for node in self.flowchart.nodes:
                agent = self._create_agent_from_node(self.flowchart, node)
                agent = await stack.enter_async_context(agent)
                clients[node.id] = agent
                logger.info(f"Created agent for node: {node.id} with label: {node.label}")

            # nodes間のエッジを取得して、WorkflowBuilderに追加
            builder = WorkflowBuilder()
            builder.set_start_executor(clients[start_node.id])    
            for edge in self.flowchart.edges:
                source_agent = clients[edge.source.id]
                target_agent = clients[edge.target.id]
                if edge.label:
                    # labelがyesの場合はtrue、noの場合はfalseとして条件分岐を追加

                    builder.add_edge(source_agent, target_agent, condition=self._check_condition(edge.label))
                    logger.info(f"Added conditional edge from {edge.source} to {edge.target} with label {edge.label}")
                else:
                    builder.add_edge(source_agent, target_agent)
                    logger.info(f"Added edge from {edge.source} to {edge.target}")

            workflow = builder.build()
            last_executor_id = None
            async for event in workflow.run_stream(message):
                if isinstance(event, AgentRunUpdateEvent):
                    if event.executor_id != last_executor_id:
                        if last_executor_id is not None:
                            print()
                        print(f"{event.executor_id}:", end=" ", flush=True)
                        last_executor_id = event.executor_id
                    print(event.data, end="", flush=True)

    def _check_condition(self, label: str) -> Any:
        
        yes_labels = ["<yes>", "yes", "はい", "true", "True"]
        no_labels = ["<no>", "no", "いいえ", "false", "False"]
        if label.lower() in yes_labels:
            def check_yes_condition(output: Any) -> bool:
                if isinstance(output, AgentExecutorResponse):
                    text = output.agent_run_response.text
                    for yes_label in yes_labels:
                        if yes_label in text:
                            return True
                return False
            return check_yes_condition

        elif label.lower() in no_labels:
            def check_no_condition(output: Any) -> bool:
                if isinstance(output, AgentExecutorResponse):
                    text = output.agent_run_response.text
                    for no_label in no_labels:
                        if no_label in text:
                            return True
                return False
            return check_no_condition
        else:
            return lambda output: True  # デフォルトはTrue

