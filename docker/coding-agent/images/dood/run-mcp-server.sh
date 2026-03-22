#!/bin/sh
set -eu

basedir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
compose_file="$basedir/docker-compose.yml"
service_name="coding-agent-dood-bundle"
env_file="${DOOD_ENV_FILE:-$basedir/.env}"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./run-mcp-server.sh [--env-file <path>] <command> [args...]

Commands:
  build     Build the DooD bundle image
  up        Start the DooD bundle in the background
  down      Stop and remove the DooD bundle
  restart   Recreate the DooD bundle container
  status    Show compose status and recent logs
  clean     Stop the DooD bundle and remove exited task containers
  logs      Show bundle logs (for example: logs -f)
  ps        Show compose status
  sh        Open /bin/sh in the bundle container
  config    Render compose config

Environment:
  LLM_API_KEY   Required for build/up/restart/config unless provided by --env-file
  HOST_UID      Optional, defaults to current uid
  HOST_GID      Optional, defaults to current gid

Notes:
  - If ./.env exists, it is loaded automatically.
  - For non-start commands (down/status/clean/logs/ps/sh), LLM_API_KEY is not required.
EOF
}

load_env_file() {
  candidate="$1"
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then
    set -a
    . "$candidate"
    set +a
  fi
}

compose_cmd() {
  docker compose -f "$compose_file" "$@"
}

clean_exited_task_containers() {
  ids="$(docker ps -aq \
    --filter status=exited \
    --filter ancestor=all-in-one-code-executor-image \
    --filter name=all-in-one-code-executor-run || true)"
  if [ -n "$ids" ]; then
    docker rm -f $ids >/dev/null
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --env-file)
      shift
      if [ $# -eq 0 ]; then
        echo "--env-file requires a path" >&2
        usage
        exit 1
      fi
      env_file="$1"
      shift
      ;;
    --env-file=*)
      env_file="${1#--env-file=}"
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
  usage
  exit 1
fi

subcommand="$1"
shift

load_env_file "$env_file"

if [ -z "${HOST_UID:-}" ]; then
  HOST_UID="$(id -u)"
  export HOST_UID
fi
if [ -z "${HOST_GID:-}" ]; then
  HOST_GID="$(id -g)"
  export HOST_GID
fi

case "$subcommand" in
  build|up|restart|config)
    if [ -z "${LLM_API_KEY:-}" ]; then
      echo "LLM_API_KEY is required. Export it or place it in ${env_file}." >&2
      exit 1
    fi
    export LLM_API_KEY
    ;;
  down|status|clean|logs|ps|sh)
    if [ -z "${LLM_API_KEY:-}" ]; then
      LLM_API_KEY="__unused_for_non_start_commands__"
      export LLM_API_KEY
    fi
    ;;
  *)
    echo "Unknown command: $subcommand" >&2
    usage
    exit 1
    ;;
esac

case "$subcommand" in
  build)
    compose_cmd build "$@"
    ;;
  up)
    compose_cmd up -d "$@"
    ;;
  down)
    compose_cmd down "$@"
    ;;
  restart)
    compose_cmd up -d --force-recreate "$@"
    ;;
  status)
    compose_cmd ps
    printf '\n--- recent logs ---\n'
    compose_cmd logs --tail=40 "$@"
    ;;
  clean)
    compose_cmd down "$@"
    clean_exited_task_containers
    ;;
  logs)
    compose_cmd logs "$@"
    ;;
  ps)
    compose_cmd ps "$@"
    ;;
  sh)
    compose_cmd exec "$service_name" sh "$@"
    ;;
  config)
    compose_cmd config "$@"
    ;;
esac