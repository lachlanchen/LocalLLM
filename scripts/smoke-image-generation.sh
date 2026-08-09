#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
api_python="$project_root/apps/api/.venv/bin/python"
gpu="${LOCALLLM_IMAGE_GENERATION_GPU:-0}"

[[ "$#" -eq 0 ]] || {
  echo "smoke-image-generation: this fixed smoke accepts no command-line arguments" >&2
  exit 2
}

[[ -x "$api_python" ]] || {
  echo "smoke-image-generation: bootstrap the LocalLLM API environment first" >&2
  exit 1
}
[[ "$gpu" =~ ^([0-9]|1[0-5])$ ]] || {
  echo "smoke-image-generation: GPU index must be between 0 and 15" >&2
  exit 1
}

exec "$api_python" "$project_root/scripts/smoke-image-generation.py" --gpu "$gpu"
