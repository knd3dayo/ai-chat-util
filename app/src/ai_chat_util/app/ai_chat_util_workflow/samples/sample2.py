from __future__ import annotations

import asyncio
from pathlib import Path

from ai_chat_util.app.ai_chat_util_workflow import MermaidFlowChart, WorkflowRunner


async def async_main() -> None:
    markdown_path = Path(__file__).with_name("data").joinpath("sample2.md")
    markdown = markdown_path.read_text(encoding="utf-8")

    flowchart = MermaidFlowChart.from_markdown(markdown)
    workflow_runner = WorkflowRunner(flowchart=flowchart)
    result = await workflow_runner.run(message=str(markdown_path.parent))
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(async_main())
