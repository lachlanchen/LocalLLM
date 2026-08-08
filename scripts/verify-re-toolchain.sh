#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ghidra_home="$project_root/.local/opt/ghidra_12.0.3_PUBLIC"
mcp_bin="$project_root/.venv-tools/bin/pyghidra-mcp"
python_bin="$project_root/.venv-tools/bin/python"
runtime_root="$project_root/.local/runtime"
target="${1:-/bin/true}"
server_pid=""
verify_dir=""

die() {
  echo "verify-re-toolchain: $*" >&2
  exit 1
}

if [[ "$#" -gt 1 ]]; then
  die "usage: scripts/verify-re-toolchain.sh [BENIGN_BINARY]"
fi
for required_command in curl find grep mktemp realpath sleep tail timeout; do
  command -v "$required_command" >/dev/null || die "missing prerequisite: $required_command"
done
[[ -x "$ghidra_home/ghidraRun" ]] || die "Ghidra is not installed; run scripts/setup-re-toolchain.sh first"
[[ -x "$mcp_bin" && -x "$python_bin" ]] || die "PyGhidra-MCP is not installed; run scripts/setup-re-toolchain.sh first"
[[ -f "$target" && -r "$target" ]] || die "verification target must be a readable regular file: $target"
target="$(realpath -- "$target")"

mkdir -p "$runtime_root"
verify_dir="$(mktemp -d "$runtime_root/re-verify.XXXXXX")"
server_log="$verify_dir/server.log"
project_dir="$verify_dir/project"
project_name="localllm_verify_${BASHPID}"

stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for _ in {1..50}; do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
  fi
  if [[ -n "$server_pid" ]]; then
    wait "$server_pid" 2>/dev/null || true
  fi
}

cleanup() {
  local status="$?"
  trap - EXIT
  stop_server
  if [[ "$status" -ne 0 && -f "$server_log" ]]; then
    echo "--- PyGhidra-MCP verification log ---" >&2
    tail -n 80 "$server_log" >&2 || true
  fi
  case "$verify_dir" in
    "$runtime_root"/re-verify.*)
      if [[ -d "$verify_dir" ]]; then
        find "$verify_dir" -xdev -depth -delete || status=1
      fi
      ;;
    "")
      ;;
    *)
      echo "verify-re-toolchain: refusing to clean unexpected path: $verify_dir" >&2
      status=1
      ;;
  esac
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

if LOCALLLM_RE_PROJECT_NAME='../escape' \
  "$project_root/scripts/start-re-workbench.sh" mcp "$target" \
  >"$verify_dir/confinement.log" 2>&1; then
  die "start-re-workbench accepted an escaping project name"
fi
grep -F 'LOCALLLM_RE_PROJECT_NAME must match' "$verify_dir/confinement.log" >/dev/null ||
  die "start-re-workbench did not fail at the project-name confinement check"

port="$($python_bin - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"

GHIDRA_INSTALL_DIR="$ghidra_home" "$mcp_bin" \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port "$port" \
  --project-path "$project_dir" \
  --project-name "$project_name" \
  --wait-for-analysis \
  "$target" \
  >"$server_log" 2>&1 &
server_pid="$!"

ready=0
for _ in {1..240}; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid" || true
    die "PyGhidra-MCP exited before becoming ready"
  fi
  if curl --silent --max-time 1 --output /dev/null "http://127.0.0.1:$port/mcp"; then
    ready=1
    break
  fi
  sleep 0.5
done
[[ "$ready" == "1" ]] || die "PyGhidra-MCP did not become ready within 120 seconds"

LOCALLLM_RE_VERIFY_URL="http://127.0.0.1:$port/mcp" \
LOCALLLM_RE_VERIFY_TARGET="$target" \
timeout 90 "$python_bin" - <<'PY'
import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

EXPECTED_TOOLS = {
    "decompile_function",
    "delete_project_binary",
    "disassemble",
    "gen_callgraph",
    "import_binary",
    "list_exports",
    "list_imports",
    "list_project_binaries",
    "list_project_binary_metadata",
    "list_xrefs",
    "read_bytes",
    "rename_function",
    "rename_variable",
    "save",
    "search_code",
    "search_strings",
    "search_symbols_by_name",
    "set_comment",
    "set_function_prototype",
    "set_variable_type",
}


def response_json(response, tool_name):
    if response.isError:
        raise RuntimeError(f"{tool_name} returned an MCP error: {response}")
    texts = [getattr(block, "text", "") for block in response.content]
    text = next((item for item in texts if item.strip()), "")
    if not text:
        raise RuntimeError(f"{tool_name} returned no text content")
    return json.loads(text)


async def verify():
    url = os.environ["LOCALLLM_RE_VERIFY_URL"]
    target = Path(os.environ["LOCALLLM_RE_VERIFY_TARGET"]).resolve()
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            initialized = await session.initialize()

            listed_tools = await session.list_tools()
            actual_tools = {tool.name for tool in listed_tools.tools}
            if actual_tools != EXPECTED_TOOLS:
                missing = sorted(EXPECTED_TOOLS - actual_tools)
                extra = sorted(actual_tools - EXPECTED_TOOLS)
                raise RuntimeError(
                    f"unexpected MCP tool surface: missing={missing}, extra={extra}"
                )

            binaries_response = await session.call_tool("list_project_binaries", {})
            binaries = response_json(binaries_response, "list_project_binaries").get("programs", [])
            matches = [
                binary
                for binary in binaries
                if Path(binary.get("file_path", "")).resolve() == target
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one imported verification target, found {len(matches)} in {binaries!r}"
                )
            imported = matches[0]
            if not imported.get("analysis_complete"):
                raise RuntimeError("verification target analysis did not complete")

            search_response = await session.call_tool(
                "search_symbols_by_name",
                {
                    "binary_name": imported["name"],
                    "query": ".*",
                    "functions_only": True,
                    "limit": 10,
                },
            )
            symbols = response_json(search_response, "search_symbols_by_name").get("symbols", [])
            if not symbols or not any(symbol.get("name") for symbol in symbols):
                raise RuntimeError("read-only symbol search returned no functions")

            print(
                "RE toolchain verification PASS: "
                f"server={initialized.serverInfo.name} {initialized.serverInfo.version}, "
                f"tools={len(actual_tools)}, binary={imported['name']}, "
                f"search_hits={len(symbols)}"
            )


async def main():
    async with asyncio.timeout(80):
        await verify()


asyncio.run(main())
PY
