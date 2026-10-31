#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

api_python=""
if [[ "$(uname -s)" == Darwin ]]; then
  export PATH="$project_root/.local/bin:$project_root/.local/node/bin:$PATH"
  api_python="${LOCALLLM_API_PYTHON:-$project_root/.venv/bin/python}"
  if [[ ! -x "$api_python" ]]; then
    echo "run: set LOCALLLM_API_PYTHON to the Python executable with the LocalLLM API dependencies installed" >&2
    exit 1
  fi
fi

scripts/ensure-web-dist.sh --ensure

ollama_pid=""
api_pid=""
cleanup() {
  if [[ -n "$api_pid" ]] && kill -0 "$api_pid" 2>/dev/null; then
    kill "$api_pid"
  fi
  if [[ -n "$ollama_pid" ]] && kill -0 "$ollama_pid" 2>/dev/null; then
    kill "$ollama_pid"
  fi
}
trap cleanup EXIT INT TERM

if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  scripts/start-ollama.sh >.local/ollama.log 2>&1 &
  ollama_pid="$!"
  for _attempt in $(seq 1 80); do
    curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 0.25
  done
fi

if [[ -n "$api_python" ]]; then
  "$api_python" -m uvicorn localllm.main:app \
    --app-dir apps/api --host 127.0.0.1 --port 8008 &
else
  uv run --project apps/api uvicorn localllm.main:app \
    --app-dir apps/api --host 127.0.0.1 --port 8008 &
fi
api_pid="$!"
wait "$api_pid"
