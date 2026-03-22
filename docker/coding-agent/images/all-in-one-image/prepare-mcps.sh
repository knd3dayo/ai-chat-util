#!/bin/sh
set -eu

basedir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$basedir/../../../.." && pwd)"
source_repos_root="$(CDPATH= cd -- "$repo_root/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./prepare-mcps.sh [options]

Options:
  --root <dir>              Destination MCP root (default: ${HOME}/data/mcps)
  --source-repos <dir>      Source repos root (default: parent of current repo)
  --project <name>          Sync only the named project (repeatable)
                           Supported: ai-chat-util, deonodo-log-util, denodo-vql-client
  --dry-run                 Show rsync changes without applying them
  --skip-uv-sync            Skip `uv sync --no-dev` after copying projects
  -h, --help                Show this help

Notes:
  - Copies real directories into the MCP root; symlinks at destinations are replaced.
  - Preserves host-managed .venv by syncing it as part of each project.
EOF
}

target_root="${HOME}/data/mcps"
selected_projects=""
dry_run=0
skip_uv_sync=0

while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --root)
      shift
      if [ $# -eq 0 ]; then
        echo "--root requires an argument" >&2
        usage
        exit 1
      fi
      target_root="$1"
      shift
      ;;
    --root=*)
      target_root="${1#--root=}"
      shift
      ;;
    --source-repos)
      shift
      if [ $# -eq 0 ]; then
        echo "--source-repos requires an argument" >&2
        usage
        exit 1
      fi
      source_repos_root="$1"
      shift
      ;;
    --source-repos=*)
      source_repos_root="${1#--source-repos=}"
      shift
      ;;
    --project)
      shift
      if [ $# -eq 0 ]; then
        echo "--project requires an argument" >&2
        usage
        exit 1
      fi
      selected_projects="$selected_projects${selected_projects:+ }$1"
      shift
      ;;
    --project=*)
      selected_projects="$selected_projects${selected_projects:+ }${1#--project=}"
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --skip-uv-sync)
      skip_uv_sync=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

should_sync_project() {
  project_name="$1"

  if [ -z "$selected_projects" ]; then
    return 0
  fi

  for selected in $selected_projects; do
    if [ "$selected" = "$project_name" ]; then
      return 0
    fi
  done

  return 1
}

validate_project_name() {
  project_name="$1"

  case "$project_name" in
    ai-chat-util|deonodo-log-util|denodo-vql-client)
      ;;
    *)
      echo "Unsupported project name: $project_name" >&2
      usage
      exit 1
      ;;
  esac
}

if [ -n "$selected_projects" ]; then
  for selected in $selected_projects; do
    validate_project_name "$selected"
  done
fi

sync_dir() {
  project_name="$1"
  src="$2"
  dst="$3"

  if [ ! -d "$src" ]; then
    echo "Source directory not found: $src" >&2
    exit 1
  fi

  parent_dir="$(dirname "$dst")"
  mkdir -p "$parent_dir"

  if [ -L "$dst" ]; then
    rm -f "$dst"
  fi
  mkdir -p "$dst"

  echo "Syncing $project_name"

  rsync_args="-a --delete"
  if [ "$dry_run" -eq 1 ]; then
    rsync_args="$rsync_args --dry-run --itemize-changes"
  fi

  # shellcheck disable=SC2086
  rsync $rsync_args "$src/" "$dst/"

  if [ "$dry_run" -eq 0 ] && [ "$skip_uv_sync" -eq 0 ] && [ -f "$dst/pyproject.toml" ]; then
    (
      cd "$dst"
      uv sync --no-dev
    )
  fi
}

mkdir -p "$target_root"

if should_sync_project "ai-chat-util"; then
  sync_dir "ai-chat-util" "$repo_root/app" "$target_root/ai-chat-util/app"
fi

if should_sync_project "deonodo-log-util" && [ -d "$source_repos_root/mcp/deonodo-log-util" ]; then
  sync_dir "deonodo-log-util" "$source_repos_root/mcp/deonodo-log-util" "$target_root/deonodo-log-util"
fi

if should_sync_project "denodo-vql-client" && [ -d "$source_repos_root/mcp/denodo-vql-client" ]; then
  sync_dir "denodo-vql-client" "$source_repos_root/mcp/denodo-vql-client" "$target_root/denodo-vql-client"
fi

if [ "$dry_run" -eq 1 ]; then
  echo "Dry-run complete for MCP root: $target_root"
else
  echo "Prepared MCP root at: $target_root"
fi
