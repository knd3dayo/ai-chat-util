#!/bin/sh
set -eu

basedir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$basedir"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./run-mcp-server.sh [options] <command> [args...]

Commands:
  prepare   Populate ${HOME}/data/mcps with real directories for default mounts
  up        Start MCP server (docker compose up -d)
  down      Stop and remove containers (docker compose down)
  stop      Stop container (docker compose stop)
  restart   Restart container (docker compose restart)
  status    Show compose status and recent logs
  clean     Stop containers and remove orphans
  logs      Show logs (docker compose logs)
  bash      Open bash in the running container (docker compose exec)

Options:
  --mcp-root <dir>            Host MCP root to bind mount (default: ${HOME}/data/mcps)
  --container-mcp-root <dir>  Container mount point for MCP root (default: same as --mcp-root)
  --project <path>            Project path relative to MCP root (default: ai-chat-util/app)
  --config <path>             Config path relative to MCP root (default: <project>/ai-chat-util-config.yml)
  --log-dir <dir>             Host log directory (default: ${HOME}/data/mcp-logs/ai-chat-util)
  --log-level <level>         MCP log level (default: DEBUG)
  -h, --help                  Show this help

Notes:
  - Host-side .venv is expected under the mounted project.
  - Start commands use uv run --no-sync with console scripts such as coding-agent-mcp.
  - If ${HOME}/data/mcps is empty, run `./run-mcp-server.sh prepare` first.
  - Additional arguments after `prepare` are forwarded to `prepare-mcps.sh`.
EOF
}

host_mcp_root="${HOME}/data/mcps"
container_mcp_root=""
project_rel_path="ai-chat-util/app"
config_rel_path=""
host_log_dir="${HOME}/data/mcp-logs/ai-chat-util"
mcp_log_level="DEBUG"

while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --mcp-root)
      shift
      if [ $# -eq 0 ]; then
        echo "--mcp-root requires an argument" >&2
        usage
        exit 1
      fi
      host_mcp_root="$1"
      shift
      ;;
    --mcp-root=*)
      host_mcp_root="${1#--mcp-root=}"
      shift
      ;;
    --container-mcp-root)
      shift
      if [ $# -eq 0 ]; then
        echo "--container-mcp-root requires an argument" >&2
        usage
        exit 1
      fi
      container_mcp_root="$1"
      shift
      ;;
    --container-mcp-root=*)
      container_mcp_root="${1#--container-mcp-root=}"
      shift
      ;;
    --project)
      shift
      if [ $# -eq 0 ]; then
        echo "--project requires an argument" >&2
        usage
        exit 1
      fi
      project_rel_path="$1"
      shift
      ;;
    --project=*)
      project_rel_path="${1#--project=}"
      shift
      ;;
    --config)
      shift
      if [ $# -eq 0 ]; then
        echo "--config requires an argument" >&2
        usage
        exit 1
      fi
      config_rel_path="$1"
      shift
      ;;
    --config=*)
      config_rel_path="${1#--config=}"
      shift
      ;;
    --log-dir)
      shift
      if [ $# -eq 0 ]; then
        echo "--log-dir requires an argument" >&2
        usage
        exit 1
      fi
      host_log_dir="$1"
      shift
      ;;
    --log-dir=*)
      host_log_dir="${1#--log-dir=}"
      shift
      ;;
    --log-level)
      shift
      if [ $# -eq 0 ]; then
        echo "--log-level requires an argument" >&2
        usage
        exit 1
      fi
      mcp_log_level="$1"
      shift
      ;;
    --log-level=*)
      mcp_log_level="${1#--log-level=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "Command is required" >&2
  usage
  exit 1
fi

compose_file="$basedir/docker-compose-mcp-server.yml"
service_name="ai-chat-util-mcp-server"
prepare_script="$basedir/prepare-mcps.sh"

if [ ! -f "$compose_file" ]; then
  echo "Compose file not found: $compose_file" >&2
  exit 1
fi

if [ ! -f "$prepare_script" ]; then
  echo "Prepare script not found: $prepare_script" >&2
  exit 1
fi

if [ -z "$container_mcp_root" ]; then
  container_mcp_root="$host_mcp_root"
fi

if [ -z "$config_rel_path" ]; then
  config_rel_path="$project_rel_path/ai-chat-util-config.yml"
fi

host_project_dir="${host_mcp_root%/}/$project_rel_path"
host_config_path="${host_mcp_root%/}/$config_rel_path"
container_project_dir="${container_mcp_root%/}/$project_rel_path"
container_config_path="${container_mcp_root%/}/$config_rel_path"

USER_ID=$(id -u)
GROUP_ID=$(id -g)
workspace_dir="${WORKSPACE:-${HOME}/data/workspace}"
subcommand="$1"
shift

compose_cmd() {
  env \
    "WORKSPACE=${WORKSPACE:-${HOME}/data/workspace}" \
    "USER_ID=$USER_ID" \
    "GROUP_ID=$GROUP_ID" \
    "HOST_MCP_ROOT=$host_mcp_root" \
    "CONTAINER_MCP_ROOT=$container_mcp_root" \
    "HOST_MCP_LOG_DIR=$host_log_dir" \
    "CONTAINER_MCP_PROJECT_DIR=$container_project_dir" \
    "CONTAINER_MCP_CONFIG_PATH=$container_config_path" \
    "MCP_LOG_LEVEL=$mcp_log_level" \
    docker compose -f "$compose_file" "$@"
}

case "$subcommand" in
  prepare)
    exec "$prepare_script" --root "$host_mcp_root" "$@"
    ;;
  up|restart)
    if [ ! -d "$host_mcp_root" ]; then
      echo "Host MCP root not found: $host_mcp_root" >&2
      exit 1
    fi
    if [ ! -d "$host_project_dir" ]; then
      echo "Project directory not found: $host_project_dir" >&2
      echo "Run ./run-mcp-server.sh prepare to populate ${HOME}/data/mcps with real directories." >&2
      exit 1
    fi
    if [ -L "$host_project_dir" ]; then
      echo "Project directory must be a real directory, not a symlink: $host_project_dir" >&2
      echo "Run ./run-mcp-server.sh prepare to replace symlinks with real directories." >&2
      exit 1
    fi
    if [ ! -f "$host_config_path" ]; then
      echo "Config file not found: $host_config_path" >&2
      exit 1
    fi
    mkdir -p "$workspace_dir" "$host_log_dir"
    ;;
esac

case "$subcommand" in
  up)
    compose_cmd up -d "$service_name"
    ;;
  down)
    compose_cmd down
    ;;
  stop)
    compose_cmd stop "$service_name"
    ;;
  restart)
    compose_cmd restart "$service_name"
    ;;
  status)
    compose_cmd ps
    printf '\n--- recent logs ---\n'
    compose_cmd logs --tail=40 "$service_name"
    ;;
  clean)
    compose_cmd down --remove-orphans
    ;;
  logs)
    compose_cmd logs "$@" "$service_name"
    ;;
  bash)
    compose_cmd exec "$service_name" bash
    ;;
  *)
    echo "Unknown command: $subcommand" >&2
    usage
    exit 1
    ;;
esac
