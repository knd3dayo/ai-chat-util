from __future__ import annotations

import asyncio

from ai_chat_util.app.ai_chat_util_workflow import MermaidFlowChart, WorkflowRunner


async def async_main() -> None:
    mermaid_code = """
    flowchart TD
        Start([Start]) --> Decide{Need more work?}
        Decide -->|yes| Work[Handle task]
        Work --> Decide
        Decide -->|no| Summary[summary: Gather the work result]
        Summary --> End([End])
    """
    flowchart = MermaidFlowChart(code=mermaid_code)
    workflow_runner = WorkflowRunner(flowchart=flowchart)
    result = await workflow_runner.run(message="workflow sample input")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(async_main())
