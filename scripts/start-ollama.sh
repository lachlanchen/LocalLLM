#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ollama_bin="$project_root/.local/ollama/bin/ollama"

if [[ ! -x "$ollama_bin" ]]; then
  "$project_root/scripts/install-ollama-local.sh"
fi

mkdir -p "$project_root/.local/models/ollama"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$project_root/.local/models/ollama}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-15m}"
if [[ -n "${LOCALLLM_OLLAMA_CUDA_VISIBLE_DEVICES:-}" ]]; then
  [[ "$LOCALLLM_OLLAMA_CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "start-ollama: LOCALLLM_OLLAMA_CUDA_VISIBLE_DEVICES must be a comma-separated list of GPU indexes" >&2
    exit 1
  }
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$LOCALLLM_OLLAMA_CUDA_VISIBLE_DEVICES"
  export OLLAMA_VULKAN=0
fi
exec "$ollama_bin" serve
