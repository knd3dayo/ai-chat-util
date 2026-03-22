FROM python:3.12-slim-bookworm

ARG DOCKER_COMPOSE_VERSION=v2.27.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
  git \
    gnupg \
  && rm -rf /var/lib/apt/lists/*

# Install a modern Docker CLI (docker-ce-cli). Debian's docker.io can be too old
# for the host daemon API version in DoOD scenarios.
RUN set -eu; \
  install -m 0755 -d /etc/apt/keyrings; \
  curl -fsSL https://download.docker.com/linux/debian/gpg -o /tmp/docker.gpg; \
  gpg --dearmor -o /etc/apt/keyrings/docker.gpg /tmp/docker.gpg; \
  rm -f /tmp/docker.gpg; \
  chmod a+r /etc/apt/keyrings/docker.gpg; \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" \
    > /etc/apt/sources.list.d/docker.list; \
  apt-get update; \
  apt-get install -y --no-install-recommends docker-ce-cli; \
  rm -rf /var/lib/apt/lists/*

# Install Docker Compose v2 plugin
RUN mkdir -p /usr/local/lib/docker/cli-plugins \
  && arch="$(dpkg --print-architecture)" \
  && case "$arch" in \
      amd64) compose_arch="x86_64" ;; \
      arm64) compose_arch="aarch64" ;; \
      *) echo "Unsupported arch: $arch" >&2; exit 1 ;; \
     esac \
  && curl -fsSL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-${compose_arch}" \
      -o /usr/local/lib/docker/cli-plugins/docker-compose \
  && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

WORKDIR /opt/ai-chat-util

# Install ai-chat-util itself so the bundle can serve the coding MCP server.
COPY app /opt/ai-chat-util/app
RUN python -m pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -e /opt/ai-chat-util/app

# Bundle the executor image assets and a DooD-safe compose template.
# Under DoOD, bind-mount sources are interpreted by the host daemon, so the
# backend compose file must avoid mounts that only exist inside this bundle.
COPY docker/coding-agent/images/all-in-one-image \
  /opt/ai-chat-util/docker/coding-agent/images/all-in-one-image
COPY docker/coding-agent/images/dood/ai-chat-util-config.yml /opt/ai-chat-util/dood/ai-chat-util-config.yml
COPY docker/coding-agent/images/dood/backend-compose.yml /opt/ai-chat-util/dood/backend-compose.yml
COPY docker/coding-agent/images/dood/env_compose /opt/ai-chat-util/dood/env_compose
COPY docker/coding-agent/images/dood/mcp.json /opt/ai-chat-util/dood/mcp.json
COPY docker/coding-agent/images/dood/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV WORKSPACE_ROOT=/srv/ai_platform/workspaces \
    EXECUTOR_ALLOWED_WORKSPACE_ROOT=/srv/ai_platform/workspaces \
    AI_CHAT_UTIL_CONFIG=/opt/ai-chat-util/dood/ai-chat-util-config.yml \
    COMPOSE_DIRECTORY=/opt/ai-chat-util/dood \
    COMPOSE_FILE=backend-compose.yml \
    COMPOSE_SERVICE_NAME=all-in-one-code-executor

EXPOSE 7101

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
