#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="$project_root/containers/usb-evidence/Dockerfile"
image_ref="localllm/usb-evidence:ubuntu24.04-20260808"
base_ref="docker.io/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea"

die() {
  echo "setup-usb-evidence-tools: $*" >&2
  exit 1
}

command -v docker >/dev/null || die "Docker is required"
docker info >/dev/null 2>&1 ||
  die "the current user cannot reach the Docker daemon; no sudo fallback is attempted"
[[ -r "$dockerfile" ]] || die "missing Dockerfile: $dockerfile"
grep -Fxq "FROM $base_ref" "$dockerfile" ||
  die "Dockerfile base is not the reviewed digest-pinned Ubuntu image"

docker build \
  --platform linux/amd64 \
  --pull \
  --provenance=false \
  --tag "$image_ref" \
  --file "$dockerfile" \
  "$project_root/containers/usb-evidence"

actual_base="$(
  docker image inspect "$image_ref" \
    --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}'
)"
[[ "$actual_base" == "$base_ref" ]] ||
  die "built image base label mismatch: expected $base_ref, found ${actual_base:-none}"

"$project_root/scripts/verify-usb-evidence-tools.sh"

echo "USB evidence image: $image_ref"
docker image inspect "$image_ref" --format 'Image ID: {{.Id}}'
