#!/bin/sh
export LLM_API_KEY=sk-poc-master-key-12345
uv run -m ai_chat_util.agent.autonomous.mcp.mcp_server -m http --config ./ai-chat-util-config.yml
