#!/bin/sh
set -eu

basedir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$basedir"

env_file="$basedir/env_run"
. "$env_file"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./run-opencode-web.sh [--config <dir>] [--host <host>] <command> [args...]

Commands:
  up        Start server (docker compose up -d)
  down      Stop and remove containers (docker compose down)
  stop      Stop container (docker compose stop)
  restart   Restart container (docker compose restart)
  logs      Show logs (docker compose logs)

Options:
  --config <dir>   Copy <dir> contents into workspace before running (default: ./config/common)
  --host <host>    Bind opencode web to <host> (default: 0.0.0.0)
  -h, --help       Show this help

Notes:
  - This script manages opencode web via docker compose.
EOF
}

config_dir="$basedir/config/common"
opencode_web_host="0.0.0.0"

while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --config)
      shift
      if [ $# -eq 0 ]; then
        echo "--config requires an argument" >&2
        usage
        exit 1
      fi
      config_dir="$1"
      shift
      ;;
    --config=*)
      config_dir="${1#--config=}"
      shift
      ;;
    --host)
      shift
      if [ $# -eq 0 ]; then
        echo "--host requires an argument" >&2
        usage
        exit 1
      fi
      opencode_web_host="$1"
      shift
      ;;
    --host=*)
      opencode_web_host="${1#--host=}"
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

compose_file="$basedir/docker-compose-opnecode-web.yml"
service_name="opencode-web-executor"

if [ ! -f "$compose_file" ]; then
  echo "Compose file not found: $compose_file" >&2
  exit 1
fi

USER_ID=$(id -u)
GROUP_ID=$(id -g)

workspace_dir="${WORKSPACE:-${HOME}/data/workspace}"

if [ ! -d "$config_dir" ]; then
  echo "Config directory not found: $config_dir" >&2
  exit 1
fi
if [ $# -eq 0 ]; then
  echo "Command is required" >&2
  usage
  exit 1
fi

subcommand="$1"
shift

# Only copy config into workspace when we (re)start the container.
case "$subcommand" in
  up|restart)
    mkdir -p "$workspace_dir"
    cp -pr "$config_dir"/. "$workspace_dir"/
    ;;
esac

compose_env="WORKSPACE=${WORKSPACE:-${HOME}/data/workspace} USER_ID=$USER_ID GROUP_ID=$GROUP_ID OPENCODE_WEB_HOSTNAME=$opencode_web_host OPENCODE_WEB_PORT=4096"

case "$subcommand" in
  up)
    env $compose_env docker compose -f "$compose_file" up -d "$service_name"
    ;;
  down)
    env $compose_env docker compose -f "$compose_file" down
    ;;
  stop)
    env $compose_env docker compose -f "$compose_file" stop "$service_name"
    ;;
  restart)
    env $compose_env docker compose -f "$compose_file" restart "$service_name"
    ;;
  logs)
    # Pass-through args to docker compose logs (e.g. -f, --tail)
    env $compose_env docker compose -f "$compose_file" logs "$@" "$service_name"
    ;;
  *)
    echo "Unknown command: $subcommand" >&2
    usage
    exit 1
    ;;
esac
