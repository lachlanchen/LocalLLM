#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="$project_root/containers/python-sandbox/Dockerfile"
image_ref="localllm/python-sandbox:3.12.11-20260809"
base_ref="docker.io/library/python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49"

die() {
  echo "setup-agent-sandbox: $*" >&2
  exit 1
}

command -v docker >/dev/null || die "Docker is required"
[[ "$(command -v docker)" == "/usr/bin/docker" ]] || die "Docker must be installed at /usr/bin/docker"
docker info >/dev/null 2>&1 ||
  die "the current user cannot reach the Docker daemon; no sudo fallback is attempted"
[[ -r "$dockerfile" ]] || die "missing Dockerfile: $dockerfile"
grep -Fxq "FROM $base_ref" "$dockerfile" ||
  die "Dockerfile base is not the reviewed digest-pinned Python image"

docker build \
  --platform linux/amd64 \
  --pull \
  --provenance=false \
  --tag "$image_ref" \
  --file "$dockerfile" \
  "$project_root/containers/python-sandbox"

actual_base="$(
  docker image inspect "$image_ref" \
    --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}'
)"
[[ "$actual_base" == "$base_ref" ]] ||
  die "built image base label mismatch: expected $base_ref, found ${actual_base:-none}"

actual_profile="$(
  docker image inspect "$image_ref" \
    --format '{{index .Config.Labels "io.localllm.sandbox.profile"}}'
)"
[[ "$actual_profile" == "python-v1" ]] ||
  die "built image profile mismatch: ${actual_profile:-none}"

"$project_root/scripts/verify-agent-sandbox.sh"

echo "Agent sandbox image: $image_ref"
docker image inspect "$image_ref" --format 'Image ID: {{.Id}}'
echo "Code execution remains disabled until LOCALLLM_AGENT_CODE_EXECUTION_ENABLED=true is set."
