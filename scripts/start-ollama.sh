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
exec "$ollama_bin" serve

