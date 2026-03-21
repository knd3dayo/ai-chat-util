#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/srv/ai_platform/workspaces}"

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

exec python -m uvicorn ai_platform_samplelib.application.autonomous.api.api_server:app \
  --host "${API_HOST:-0.0.0.0}" \
  --port "${API_PORT:-7101}"
