from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ai_chat_util.base.hitl import create_stdio_hitl_client
from ai_chat_util.common.config.runtime import init_runtime
from ai_chat_util.core.app import run_mermaid_workflow_from_file
from ai_chat_util.workflow import WorkflowChatClient


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run Mermaid workflow with LangGraph")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to a Markdown file containing exactly one mermaid block")
    parser.add_argument("-m", "--message", type=str, default="", help="Initial input for the workflow")
    parser.add_argument("--config", type=str, default="", help="Optional ai-chat-util-config.yml path")
    parser.add_argument("--max-node-visits", type=int, default=8, help="Loop safety limit per node")
    parser.add_argument("--durable", action="store_true", help="Enable durable pause/resume and approval handling")
    parser.add_argument("--plan-mode", action="store_true", help="Prepare updated markdown and pause for approval")
    args = parser.parse_args()

    init_runtime(args.config or None)
    if not args.durable:
        response = await run_mermaid_workflow_from_file(
            workflow_file_path=str(Path(args.file).expanduser().resolve()),
            message=args.message,
            max_node_visits=args.max_node_visits,
            durable=False,
            enable_tool_approval_nodes=False,
        )
        print(response.final_output)
        return

    workflow_client = WorkflowChatClient(
        str(Path(args.file).expanduser().resolve()),
        max_node_visits=args.max_node_visits,
        plan_mode=args.plan_mode,
        durable=True,
    )
    trace_id: str | None = None
    await create_stdio_hitl_client(workflow_client, trace_id=trace_id).run(args.message)


if __name__ == "__main__":
    asyncio.run(async_main())
