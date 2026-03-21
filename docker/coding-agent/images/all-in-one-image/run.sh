#!/bin/sh
set -eu
basedir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$basedir"


usage() {
  cat >&2 <<'EOF'
Usage:
  ./run.sh [--config <dir>] <command: opencode|claude|cline> [args...]

Options:
  --config <dir>   Copy <dir> contents into workspace before running (default: ./config/common)
  -h, --help       Show this help
EOF
}

config_dir="$basedir/config/common"

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
  # commandはopencode、claude、clineのいずれかを想定しているが、指定されない場合はエラーとする
  usage
  exit 1
elif [ "$1" = "open-code" ]; then
  echo "Invalid command: open-code (use: opencode)" >&2
  usage
  exit 1
elif [ "$1" != "opencode" ] && [ "$1" != "claude" ] && [ "$1" != "cline" ]; then
  echo "Invalid command: $1" >&2
  usage
  exit 1
else
  COMPOSE_COMMAND=$1
  shift
fi

service_name=all-in-one-code-executor
command=${COMPOSE_COMMAND}

if [ -z "$command" ]; then
  echo "COMPOSE_COMMAND is not set in env_run" >&2
  exit 1
fi

# USER_IDとGROUP_IDを実行ユーザーから取得して、codeuserのUIDとGIDを変更する処理を追加
USER_ID=$(id -u)
GROUP_ID=$(id -g)

workspace_dir="${WORKSPACE:-${HOME}/workspace}"

if [ ! -d "$config_dir" ]; then
  echo "Config directory not found: $config_dir" >&2
  exit 1
fi
mkdir -p "$workspace_dir"
# Copy config directory contents into workspace (include dotfiles).
cp -pr "$config_dir"/. "$workspace_dir"/

compose_run_extra_args=""
# docker compose run does not publish ports unless --service-ports is set.
if [ "$command" = "opencode" ] && [ "${1:-}" = "web" ]; then
  compose_run_extra_args="--service-ports"
fi

env WORKSPACE=${WORKSPACE:-${HOME}/data/workspace} USER_ID=$USER_ID GROUP_ID=$GROUP_ID docker compose run $compose_run_extra_args --rm $service_name $command $@
