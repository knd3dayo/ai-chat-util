#!/bin/sh
set -eu

basedir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
project_dir="$(CDPATH= cd -- "$basedir/../../../app" && pwd)"

export LLM_API_KEY=sk-poc-master-key-12345
export AI_CHAT_UTIL_CONFIG="$basedir/ai-chat-util-config.yml"

uv --directory "$project_dir" run -m coding_agent_util.mcp.mcp_server -m http --config "$AI_CHAT_UTIL_CONFIG"
