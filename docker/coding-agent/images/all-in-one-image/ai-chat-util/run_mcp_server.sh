#!/bin/sh
export LLM_API_KEY=sk-poc-master-key-12345
uv run coding-agent-mcp -m http --config ./ai-chat-util-config.yml
