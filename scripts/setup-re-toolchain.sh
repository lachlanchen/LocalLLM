#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
download_dir="$project_root/.local/downloads"
install_dir="$project_root/.local/opt"
tools_dir="$project_root/.local/tools"
ghidra_archive="$download_dir/ghidra_12.0.3_PUBLIC_20260210.zip"
ghidra_home="$install_dir/ghidra_12.0.3_PUBLIC"
ghidra_url="https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_12.0.3_build/ghidra_12.0.3_PUBLIC_20260210.zip"
ghidra_sha256="90d3fffb20b00030dcef8d2a24dd0f422d3a61e432b3ad43f77233ac6d667981"

command -v java >/dev/null || { echo "Java 21 is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
mkdir -p "$download_dir" "$install_dir" "$tools_dir"

if [[ ! -f "$ghidra_archive" ]]; then
  curl -fL --progress-bar "$ghidra_url" -o "$ghidra_archive"
fi
printf '%s  %s\n' "$ghidra_sha256" "$ghidra_archive" | sha256sum -c -
if [[ ! -d "$ghidra_home" ]]; then
  unzip -q "$ghidra_archive" -d "$install_dir"
fi

if [[ ! -d "$tools_dir/OGhidra/.git" ]]; then
  git clone --depth 1 https://github.com/LLNL/OGhidra.git "$tools_dir/OGhidra"
else
  git -C "$tools_dir/OGhidra" pull --ff-only
fi
if [[ ! -d "$tools_dir/pyghidra-mcp/.git" ]]; then
  git clone --depth 1 https://github.com/clearbluejar/pyghidra-mcp.git "$tools_dir/pyghidra-mcp"
else
  git -C "$tools_dir/pyghidra-mcp" pull --ff-only
fi

# OGhidra currently ships legacy KEY=value metadata that Ghidra 12 rejects.
if grep -q '^GHIDRA_MODULE_' "$tools_dir/OGhidra/OGhidraMCP/Module.manifest"; then
  : > "$tools_dir/OGhidra/OGhidraMCP/Module.manifest"
fi

uv python install 3.12
uv venv "$project_root/.venv-tools" --python 3.12 --clear
uv pip install --python "$project_root/.venv-tools/bin/python" \
  -e "$tools_dir/pyghidra-mcp" -e "$tools_dir/OGhidra"

(
  cd "$tools_dir/OGhidra"
  env GHIDRA_INSTALL_DIR="$ghidra_home" ./build_ghidra_plugin.sh
)
plugin_zip="$(find "$tools_dir/OGhidra/OGhidraMCP/dist" -maxdepth 1 -name '*.zip' -print -quit)"
unzip -qo "$plugin_zip" -d "$ghidra_home/Ghidra/Extensions"

echo "Ghidra: $ghidra_home/ghidraRun"
echo "OGhidra: $tools_dir/OGhidra"
echo "PyGhidra-MCP: $project_root/.venv-tools/bin/pyghidra-mcp"

