#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ollama_bin="$project_root/.local/ollama/bin/ollama"
mode="${1:-core}"

text_models=(
  qwen3:4b-q4_K_M
  qwen3:4b-q8_0
  qwen3:8b-q4_K_M
  qwen3:8b-q8_0
  qwen3:30b-a3b-instruct-2507-q4_K_M
  qwen3:30b-a3b-instruct-2507-q8_0
)
vision_models=(
  qwen3-vl:8b-instruct-q4_K_M
  qwen3-vl:8b-instruct-q8_0
  qwen3-vl:30b-a3b-instruct-q4_K_M
)
embedding_models=(
  bge-m3:latest
)
code_models=(
  qwen3-coder:30b-a3b-q4_K_M
)
core_models=(
  qwen3:8b-q4_K_M
  qwen3-vl:8b-instruct-q4_K_M
  qwen3:30b-a3b-instruct-2507-q4_K_M
  bge-m3:latest
)
all_models=(
  qwen3:4b-q4_K_M
  qwen3:4b-q8_0
  qwen3:8b-q4_K_M
  qwen3-vl:8b-instruct-q4_K_M
  qwen3:30b-a3b-instruct-2507-q4_K_M
  qwen3:8b-q8_0
  qwen3-vl:8b-instruct-q8_0
  qwen3-vl:30b-a3b-instruct-q4_K_M
  qwen3:30b-a3b-instruct-2507-q8_0
  qwen3-coder:30b-a3b-q4_K_M
  bge-m3:latest
)

if [[ ! -x "$ollama_bin" ]]; then
  "$project_root/scripts/install-ollama-local.sh"
fi
if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null; then
  echo "Ollama is not running. Start it with scripts/start-ollama.sh" >&2
  exit 1
fi

case "$mode" in
  core) selected_models=("${core_models[@]}") ;;
  text) selected_models=("${text_models[@]}") ;;
  vision) selected_models=("${vision_models[@]}") ;;
  embedding) selected_models=("${embedding_models[@]}") ;;
  code) selected_models=("${code_models[@]}") ;;
  all) selected_models=("${all_models[@]}") ;;
  status) exec "$ollama_bin" list ;;
  verify)
    installed="$($ollama_bin list | awk 'NR > 1 {print $1}')"
    missing=0
    for localllm_model in "${all_models[@]}"; do
      if ! grep -Fxq "$localllm_model" <<<"$installed"; then
        echo "MISSING $localllm_model"
        missing=1
      else
        echo "OK      $localllm_model"
      fi
    done
    exit "$missing"
    ;;
  *) echo "Usage: $0 {core|text|vision|embedding|code|all|status|verify}" >&2; exit 2 ;;
esac

for localllm_model in "${selected_models[@]}"; do
  "$ollama_bin" pull "$localllm_model"
done
