#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
platform="$(uname -s)"
ollama_bin="${LOCALLLM_OLLAMA_BIN:-}"
if [[ -z "$ollama_bin" ]]; then
  if [[ "$platform" == Darwin ]]; then
    for candidate in "$project_root/.local/ollama/ollama" \
      "$project_root/.local/ollama/bin/ollama" /Applications/Ollama.app/Contents/Resources/ollama; do
      if [[ -x "$candidate" ]]; then
        ollama_bin="$candidate"
        break
      fi
    done
    ollama_bin="${ollama_bin:-$(command -v ollama || true)}"
  else
    ollama_bin="$project_root/.local/ollama/bin/ollama"
  fi
fi

if [[ ! -x "$ollama_bin" ]]; then
  if [[ "$platform" == Linux && "$(uname -m)" == x86_64 && -z "${LOCALLLM_OLLAMA_BIN:-}" ]]; then
    "$project_root/scripts/install-ollama-local.sh"
  else
    echo "start-ollama: install an Ollama build for this platform and set LOCALLLM_OLLAMA_BIN to its executable" >&2
    exit 1
  fi
fi

default_models="$project_root/.local/models/ollama"
if [[ "$platform" == Darwin && -d "$project_root/models" ]]; then
  default_models="$project_root/models"
fi
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$default_models}"
mkdir -p "$OLLAMA_MODELS"
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
