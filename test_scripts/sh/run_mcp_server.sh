#!/bin/sh
export LLM_API_KEY=sk-poc-master-key-12345
uv --directory /home/user/source/repos/autonomous-agent-util run -m autonomous_agent_util.mcp.mcp_server -m http --config /home/user/source/repos/autonomous-agent-util/autonomous-agent-util-config.yml
