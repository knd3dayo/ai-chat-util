#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/srv/ai_platform/workspaces}"
AI_CHAT_UTIL_CONFIG="${AI_CHAT_UTIL_CONFIG:-/opt/ai-chat-util/dood/ai-chat-util-config.yml}"

mkdir -p "$WORKSPACE_ROOT"

# If the workspace root is bind-mounted from host, it may be owned by root.
# For PoC convenience, allow mapping ownership to the host user.
if [[ -n "${HOST_UID:-}" && -n "${HOST_GID:-}" ]]; then
  # Best-effort: don't fail the container if chown is not permitted
  chown -R "${HOST_UID}:${HOST_GID}" "$WORKSPACE_ROOT" 2>/dev/null || true
fi

# optional: if caller provides host uid/gid, ensure runner propagates it
# (runner reads AI_PLATFORM_HOST_UID/GID)
if [[ -n "${HOST_UID:-}" ]]; then export AI_PLATFORM_HOST_UID="$HOST_UID"; fi
if [[ -n "${HOST_GID:-}" ]]; then export AI_PLATFORM_HOST_GID="$HOST_GID"; fi

cd /opt/ai-chat-util/dood

exec coding-agent-mcp \
  -m http \
  --host "${API_HOST:-0.0.0.0}" \
  -p "${API_PORT:-7101}" \
  --config "$AI_CHAT_UTIL_CONFIG" \
  -v "${MCP_LOG_LEVEL:-INFO}"
