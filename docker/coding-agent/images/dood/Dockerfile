FROM python:3.12-slim-bookworm

ARG DOCKER_COMPOSE_VERSION=v2.27.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
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

WORKDIR /opt/ai-platform

# Install samplelib (SV + autonomous API/runner)
COPY app/ai-platform-samplelib /opt/ai-platform/ai-platform-samplelib
RUN python -m pip install --no-cache-dir -U pip \
  && pip install --no-cache-dir -e /opt/ai-platform/ai-platform-samplelib

# Bundle the executor compose template so the client-side docker compose can read it.
# IMPORTANT: Under DoOD, bind-mount sources are interpreted on the host daemon.
# This template must NOT reference client-local paths (we removed script bind mounts).
COPY app/docker/autonomous-agent-executor/images/all-in-on-image \
  /opt/ai-platform/autonomous-agent-executor/all-in-on-image

COPY app/docker/autonomous-agent-executor/images/one-set-dood/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV WORKSPACE_ROOT=/srv/ai_platform/workspaces \
    EXECUTOR_ALLOWED_WORKSPACE_ROOT=/srv/ai_platform/workspaces \
    COMPOSE_DIRECTORY=/opt/ai-platform/autonomous-agent-executor/all-in-on-image \
    COMPOSE_FILE=docker-compose.yml \
    COMPOSE_SERVICE_NAME=all-in-one-code-executor

EXPOSE 7101

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
