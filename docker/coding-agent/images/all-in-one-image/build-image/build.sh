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

docker build -t "$image_name" -f "$basedir/Dockerfile" .
