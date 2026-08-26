#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

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

uv run --project apps/api uvicorn localllm.main:app \
  --app-dir apps/api --host 127.0.0.1 --port 8008 &
api_pid="$!"
wait "$api_pid"
