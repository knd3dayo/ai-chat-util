import asyncio
import argparse
import os
from ai_chat_util.mermaid_agent.workflow.runner import WorkflowRunner
from ai_chat_util.mermaid_agent.mermaid.mermaid_flowchart import MermaidFlowChart 

from ai_chat_util.log.log_settings import getLogger
logger = getLogger(__name__)

async def async_main():
    parser = argparse.ArgumentParser(description="MS AI Agent Sample")
    # 引数解析 -f マーメイド図を含むマークダウンファイルパス
    parser.add_argument("-f", "--file", type=str, required=True, help="マーメイド図を含むマークダウンファイルのパス")
    # 引数解析 -m メッセージ
    parser.add_argument("-m", "--message", type=str, default="", help="ワークフローに渡すメッセージ")
    args = parser.parse_args()

    # workflowのmermaidファイルを読み込む
    with open(args.file, "r", encoding="utf-8") as f:
        markdown = f.read()

    mermaid_code_list = MermaidFlowChart.extract_mermaid_code(markdown)  
    for mermaid_code in mermaid_code_list:
        flowchart = MermaidFlowChart(code=mermaid_code)
        workflow_runner = WorkflowRunner(flowchart=flowchart)
        await workflow_runner.run(message=args.message)

if __name__ == "__main__":
        import sys
        asyncio.run(async_main())
