#!/bin/sh
export LLM_API_KEY=sk-poc-master-key-12345
uv run -m ai_chat_util.agent.autonomous.mcp.mcp_server -m http --config /home/user/source/repos/ai-chat-util/autonomous-agent-util-config.yml
