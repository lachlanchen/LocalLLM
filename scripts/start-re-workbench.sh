#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ghidra_home="$project_root/.local/opt/ghidra_12.0.3_PUBLIC"
mcp_bin="$project_root/.venv-tools/bin/pyghidra-mcp"
mode="${1:-help}"

case "$mode" in
  gui)
    [[ -x "$ghidra_home/ghidraRun" ]] || { echo "Run scripts/setup-re-toolchain.sh first" >&2; exit 1; }
    exec "$ghidra_home/ghidraRun"
    ;;
  mcp)
    shift
    [[ -x "$mcp_bin" ]] || { echo "Run scripts/setup-re-toolchain.sh first" >&2; exit 1; }
    [[ "$#" -gt 0 ]] || { echo "Usage: scripts/start-re-workbench.sh mcp BINARY [BINARY ...]" >&2; exit 2; }
    re_project_name="${LOCALLLM_RE_PROJECT_NAME:-investigation}"
    if [[ ! "$re_project_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
      echo "LOCALLLM_RE_PROJECT_NAME must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$" >&2
      exit 2
    fi
    re_port="${LOCALLLM_RE_MCP_PORT:-18765}"
    if [[ ! "$re_port" =~ ^[0-9]+$ ]] || ((re_port < 1 || re_port > 65535)); then
      echo "LOCALLLM_RE_MCP_PORT must be an integer between 1 and 65535" >&2
      exit 2
    fi
    re_project_dir="$project_root/.local/re-projects/$re_project_name"
    mkdir -p "$re_project_dir"
    export GHIDRA_INSTALL_DIR="$ghidra_home"
    exec "$mcp_bin" \
      --transport streamable-http \
      --host 127.0.0.1 \
      --port "$re_port" \
      --project-path "$re_project_dir" \
      --project-name "$re_project_name" \
      --wait-for-analysis \
      -- "$@"
    ;;
  *)
    echo "Usage: scripts/start-re-workbench.sh {gui|mcp BINARY [BINARY ...]}"
    ;;
esac
