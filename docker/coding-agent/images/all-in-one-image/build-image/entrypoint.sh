#!/bin/bash
set -e

# Prefer a container-reachable LLM base URL when provided.
# This allows using LLM_BASE_URL for host-side tools (e.g. localhost)
# while keeping the container pointing at the Docker network hostname.
if [ -n "${LLM_BASE_URL_IN_CONTAINER:-}" ]; then
    export LLM_BASE_URL="${LLM_BASE_URL_IN_CONTAINER}"
fi

# 1. Firewallの初期化 (sudoが必要)
if [ -f "/usr/local/bin/init-firewall.sh" ]; then
    echo "🛡️ Initializing Firewall..."
    # Preserve env vars such as ALLOW_OPENCODE_APP_EGRESS/OPENCODE_APP_HOST.
    sudo -E /usr/local/bin/init-firewall.sh
fi

# 2. 環境変数をopencode.jsonに反映させるためのPythonスクリプトを実行
mkdir -p /home/codeuser/.config/opencode
python3 /home/codeuser/create_opencode_json.py  /home/codeuser/.config/opencode/opencode.json

# 3. USER_IDとGROUP_IDを環境変数から取得して、codeuserのUIDとGIDを変更
if [ -n "$USER_ID" ] && [ -n "$GROUP_ID" ]; then
    echo "🔧 Setting codeuser UID to $USER_ID and GID to $GROUP_ID"
    sudo usermod -u "$USER_ID" codeuser
    sudo groupmod -g "$GROUP_ID" codeuser
    # 変更後のUIDとGIDで作業ディレクトリの所有権を再設定
    sudo chown -R codeuser:codeuser /workspace
fi

# claude CLI が参照する環境変数にマッピング（compose の ${VAR} 展開に依存しない）
if [ -z "${ANTHROPIC_AUTH_TOKEN}" ] && [ -n "${LLM_API_KEY}" ]; then
    export ANTHROPIC_AUTH_TOKEN="${LLM_API_KEY}"
fi
if [ -z "${ANTHROPIC_BASE_URL}" ] && [ -n "${LLM_BASE_URL}" ]; then
    export ANTHROPIC_BASE_URL="${LLM_BASE_URL}"
fi
if [ -z "${ANTHROPIC_MODEL}" ] && [ -n "${LLM_MODEL}" ]; then
    export ANTHROPIC_MODEL="${LLM_MODEL}"
fi

# cline を実行する場合のみ認証を実行
requested_cmd="${1:-}"
if [ "$requested_cmd" = "cline" ]; then
    if [ -z "${LLM_API_KEY:-}" ] || [ -z "${LLM_PROVIDER:-}" ] || [ -z "${LLM_MODEL:-}" ]; then
        echo "LLM_API_KEY/LLM_PROVIDER/LLM_MODEL are required for cline" >&2
        exit 1
    fi

    if [ -n "${LLM_BASE_URL:-}" ]; then
        runuser -u codeuser -- cline auth -p "$LLM_PROVIDER" -k "$LLM_API_KEY" -m "$LLM_MODEL" -b "$LLM_BASE_URL"
    else
        runuser -u codeuser -- cline auth -p "$LLM_PROVIDER" -k "$LLM_API_KEY" -m "$LLM_MODEL"
    fi
fi

if [ -t 0 ]; then
    exec runuser -u codeuser -- "$@"
else
    exec runuser -u codeuser -- "$@" < /dev/null
fi