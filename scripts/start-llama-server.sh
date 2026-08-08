#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="$project_root/.local/opt/llama.cpp-b10327/bin/llama-server"

die() {
  echo "start-llama-server: $*" >&2
  exit 1
}

[[ -x "$server_bin" ]] ||
  die "llama-server is not installed; run scripts/setup-llama-cpp.sh first"

model_path="${1:-}"
[[ -n "$model_path" ]] || die "usage: scripts/start-llama-server.sh /absolute/path/model.gguf"
[[ -f "$model_path" && -r "$model_path" ]] || die "model is not a readable file: $model_path"

port="${LOCALLLM_LLAMA_CPP_PORT:-8010}"
context_size="${LOCALLLM_LLAMA_CPP_CONTEXT:-16384}"
gpu_layers="${LOCALLLM_LLAMA_CPP_GPU_LAYERS:-all}"
split_mode="${LOCALLLM_LLAMA_CPP_SPLIT_MODE:-layer}"
flash_attention="${LOCALLLM_LLAMA_CPP_FLASH_ATTN:-auto}"
model_alias="${LOCALLLM_LLAMA_CPP_ALIAS:-$(basename "$model_path" .gguf)}"

[[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) ||
  die "LOCALLLM_LLAMA_CPP_PORT must be between 1 and 65535"
[[ "$context_size" =~ ^[1-9][0-9]*$ ]] ||
  die "LOCALLLM_LLAMA_CPP_CONTEXT must be a positive integer"
[[ "$gpu_layers" =~ ^(all|auto|[0-9]+)$ ]] ||
  die "LOCALLLM_LLAMA_CPP_GPU_LAYERS must be all, auto, or a non-negative integer"
case "$split_mode" in
  none | layer | row | tensor) ;;
  *) die "LOCALLLM_LLAMA_CPP_SPLIT_MODE must be none, layer, row, or tensor" ;;
esac
case "$flash_attention" in
  on | off | auto) ;;
  *) die "LOCALLLM_LLAMA_CPP_FLASH_ATTN must be on, off, or auto" ;;
esac

server_args=(
  --model "$model_path"
  --alias "$model_alias"
  --host 127.0.0.1
  --port "$port"
  --ctx-size "$context_size"
  --n-gpu-layers "$gpu_layers"
  --split-mode "$split_mode"
  --flash-attn "$flash_attention"
  --jinja
  --cors-origins localhost
  --no-cors-credentials
  --no-slots
  --no-ui
)

if [[ -n "${LOCALLLM_LLAMA_CPP_TENSOR_SPLIT:-}" ]]; then
  server_args+=(--tensor-split "$LOCALLLM_LLAMA_CPP_TENSOR_SPLIT")
fi
if [[ -n "${LOCALLLM_LLAMA_CPP_MMPROJ:-}" ]]; then
  [[ -f "$LOCALLLM_LLAMA_CPP_MMPROJ" && -r "$LOCALLLM_LLAMA_CPP_MMPROJ" ]] ||
    die "LOCALLLM_LLAMA_CPP_MMPROJ is not a readable file"
  server_args+=(--mmproj "$LOCALLLM_LLAMA_CPP_MMPROJ")
fi
if [[ -n "${LOCALLLM_LLAMA_CPP_API_KEY:-}" ]]; then
  export LLAMA_API_KEY="$LOCALLLM_LLAMA_CPP_API_KEY"
fi

echo "llama.cpp OpenAI-compatible API: http://127.0.0.1:${port}/v1"
exec "$server_bin" "${server_args[@]}"
