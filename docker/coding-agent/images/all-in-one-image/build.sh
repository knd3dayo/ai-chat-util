#!/bin/sh
set -eu

basedir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$basedir"

env_file="$basedir/env_build"
. "$env_file"

image_name="${IMAGE_NAME}"

if [ -z "$image_name" ]; then
	echo "IMAGE_NAME is not set in env_build" >&2
	exit 1
fi

cleanup() {
	# Best-effort cleanup: ensure this runs even when the script is interrupted.
	# (e.g. Ctrl+C) and don't let cleanup errors mask the original exit.
	set +e
	rm -rf "$basedir/app"
}
trap cleanup EXIT INT TERM HUP

# apps.txt で指定されたライブラリを staging して Docker build context にコピー
mkdir -p "$basedir/app"
for lib in $(cat "$basedir/apps.txt"); do
# libを$basedir/app配下にコピー
# 例えば、libが"ai-platform-samplelib"の場合、$basedir/app/ai-platform-samplelibにコピーされる
	cp -r "$lib" "$basedir/app/"
done


docker build \
	-t "$image_name" \
	-f "$basedir/Dockerfile" \
	.
