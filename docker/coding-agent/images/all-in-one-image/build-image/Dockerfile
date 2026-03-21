FROM node:22-slim

# 1. 必要なツールのインストール (Firewall制御用ツールを追加)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    sudo \
    iptables \
    ipset \
    iproute2

# 2. ユーザー作成 (既存のnodeユーザーを再利用して名前を変更)
RUN usermod -l codeuser node && \
    groupmod -n codeuser node && \
    usermod -d /home/codeuser -m codeuser

# 3. 作業ディレクトリの作成と権限変更
WORKDIR /workspace
RUN chown codeuser:codeuser /workspace

# 4. npm グローバルパスの設定
ENV NPM_CONFIG_PREFIX=/home/codeuser/.npm-global
ENV PATH=$PATH:/home/codeuser/.npm-global/bin:/home/codeuser/.venv-tools/bin

# 6.  LibreOfficeのインストール
RUN apt-get install -y --no-install-recommends libreoffice

#################################
# codeuserの作業
#################################
USER codeuser

# 7. uvのインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/codeuser/.local/bin/:$PATH"

# 8. アプリケーション用ディレクトリと権限設定

RUN mkdir -p /home/codeuser/app
COPY --chown=codeuser:codeuser app /home/codeuser/app
RUN chown codeuser:codeuser /home/codeuser/app

# /home/codeuser/app/*/にMCP用のプロジェクトが配置されているので、
# entrypoint.sh内でこれらをループしてuv syncすることで、MCPプロジェクトの依存関係も解決されるようにする。
RUN for mcp_dir in /home/codeuser/app/*/; do \
    if [ -f "$mcp_dir/pyproject.toml" ]; then \
        cd "$mcp_dir" && uv sync --no-dev; \
    fi; \
done

# 9. コーディングエージェントのインストール
RUN mkdir -p /home/codeuser/.npm-global
## OpenCode AIのインストール
RUN npm install -g opencode-ai

##  opencode.json生成用のPythonスクリプトを配置。(entrypoint.shで実行される)
COPY create_opencode_json.py /home/codeuser/

## Claude Codeのインストール
RUN npm install -g @anthropic-ai/claude-code@latest

## Clineのインストール
RUN npm install -g cline

USER root

# 5. FirewallスクリプトとEntrypointの配置
COPY init-firewall.sh /usr/local/bin/
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/init-firewall.sh /usr/local/bin/entrypoint.sh && \
    # codeuserがsudoをパスワードなしで実行できるように設定 (chown等も含む)
    echo "codeuser ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh, /usr/sbin/chown, /usr/bin/chown" > /etc/sudoers.d/codeuser-admin && \
    chmod 0440 /etc/sudoers.d/codeuser-admin

ENTRYPOINT ["entrypoint.sh"]
CMD ["/bin/bash"]