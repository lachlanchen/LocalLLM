#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ollama_bin="$project_root/.local/ollama/bin/ollama"

printf 'LocalLLM root: %s\n' "$project_root"
printf 'Disk:\n'
df -h "$project_root"
printf '\nNVIDIA:\n'
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader 2>&1 || true
if [[ -f /proc/driver/nvidia/version ]]; then head -n 1 /proc/driver/nvidia/version; fi
modinfo -F version nvidia 2>/dev/null || true
printf '\nOllama:\n'
curl -fsS http://127.0.0.1:11434/api/version 2>&1 || true
if [[ -x "$ollama_bin" ]]; then "$ollama_bin" list 2>&1 || true; fi
printf '\nApplication:\n'
curl -fsS http://127.0.0.1:8008/healthz 2>&1 || true
printf '\nReverse engineering:\n'
[[ -x "$project_root/.local/opt/ghidra_12.0.3_PUBLIC/ghidraRun" ]] && echo "Ghidra ready"
if [[ -x "$project_root/.venv-tools/bin/python" ]]; then
  "$project_root/.venv-tools/bin/python" -c 'from importlib.metadata import version; print("pyghidra-mcp", version("pyghidra-mcp"))'
fi
