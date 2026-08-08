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
oghidra_url="https://github.com/LLNL/OGhidra.git"
oghidra_commit="93a4380fc748a393690be9bfd2c2156fade82757"
pyghidra_mcp_url="https://github.com/clearbluejar/pyghidra-mcp.git"
pyghidra_mcp_commit="f29063b8636100b71e9c3aec61fe056827c556e4"
oghidra_patch="$project_root/patches/oghidra-local-security.patch"

die() {
  echo "setup-re-toolchain: $*" >&2
  exit 1
}

for required_command in cmp curl git java javac mktemp sed sha256sum strings unzip uv; do
  command -v "$required_command" >/dev/null || die "missing prerequisite: $required_command"
done

java_specification_version="$(
  java -XshowSettings:properties -version 2>&1 |
    awk -F= '/java.specification.version/ {gsub(/[[:space:]]/, "", $2); print $2; exit}'
)"
[[ "$java_specification_version" == "21" ]] || die "Java 21 is required (found ${java_specification_version:-unknown})"
javac_version="$(javac -version 2>&1 | awk '{print $2}')"
[[ "$javac_version" == "21" || "$javac_version" == 21.* ]] ||
  die "Java 21 compiler is required (found ${javac_version:-unknown})"
[[ -r "$oghidra_patch" ]] || die "missing required patch: $oghidra_patch"

ensure_pinned_repo() {
  local name="$1"
  local url="$2"
  local repo_dir="$3"
  local commit="$4"

  if [[ -e "$repo_dir" && ! -d "$repo_dir/.git" ]]; then
    die "$repo_dir exists but is not a Git checkout"
  fi
  if [[ ! -d "$repo_dir/.git" ]]; then
    mkdir -p "$repo_dir"
    git -C "$repo_dir" init -q
    git -C "$repo_dir" remote add origin "$url"
  fi

  local origin_url
  origin_url="$(git -C "$repo_dir" remote get-url origin 2>/dev/null || true)"
  [[ "$origin_url" == "$url" ]] || die "$name origin mismatch: expected $url, found ${origin_url:-none}"

  if ! git -C "$repo_dir" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$repo_dir" fetch --depth 1 origin "$commit"
  fi

  local current_commit
  current_commit="$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$current_commit" != "$commit" ]]; then
    if [[ -n "$(git -C "$repo_dir" status --porcelain --untracked-files=all)" ]]; then
      die "$name has local changes; refusing to replace them while selecting pinned commit $commit"
    fi
    git -C "$repo_dir" checkout --detach "$commit"
  fi

  [[ "$(git -C "$repo_dir" rev-parse HEAD)" == "$commit" ]] || die "$name did not resolve to pinned commit $commit"
  echo "$name pinned at $commit"
}

apply_repo_patch() {
  local name="$1"
  local repo_dir="$2"
  local patch_file="$3"

  if git -C "$repo_dir" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "$name local security patch already applied"
  elif git -C "$repo_dir" apply --check "$patch_file"; then
    git -C "$repo_dir" apply "$patch_file"
    echo "$name local security patch applied"
  else
    die "$name does not match the expected clean or patched pinned source"
  fi
}

require_clean_repo() {
  local name="$1"
  local repo_dir="$2"
  git -C "$repo_dir" diff --quiet HEAD -- ||
    die "$name has tracked changes at the pinned revision"
  [[ -z "$(git -C "$repo_dir" ls-files --others --exclude-standard)" ]] ||
    die "$name has untracked files at the pinned revision"
}

require_exact_patch() {
  local name="$1"
  local repo_dir="$2"
  local patch_file="$3"
  local actual_patch
  actual_patch="$(mktemp)"
  git -C "$repo_dir" diff --binary HEAD | sed 's/^ $//' > "$actual_patch"
  if ! cmp -s "$actual_patch" "$patch_file"; then
    unlink "$actual_patch"
    die "$name contains changes beyond the repository-tracked patch"
  fi
  unlink "$actual_patch"
  [[ -z "$(git -C "$repo_dir" ls-files --others --exclude-standard)" ]] ||
    die "$name has unexpected untracked files"
}

mkdir -p "$download_dir" "$install_dir" "$tools_dir"

if [[ ! -f "$ghidra_archive" ]]; then
  curl -fL --progress-bar "$ghidra_url" -o "$ghidra_archive"
fi
printf '%s  %s\n' "$ghidra_sha256" "$ghidra_archive" | sha256sum -c -
if [[ ! -d "$ghidra_home" ]]; then
  unzip -q "$ghidra_archive" -d "$install_dir"
fi

ensure_pinned_repo "OGhidra" "$oghidra_url" "$tools_dir/OGhidra" "$oghidra_commit"
ensure_pinned_repo "PyGhidra-MCP" "$pyghidra_mcp_url" "$tools_dir/pyghidra-mcp" "$pyghidra_mcp_commit"
apply_repo_patch "OGhidra" "$tools_dir/OGhidra" "$oghidra_patch"
require_exact_patch "OGhidra" "$tools_dir/OGhidra" "$oghidra_patch"
require_clean_repo "PyGhidra-MCP" "$tools_dir/pyghidra-mcp"

oghidra_plugin_source="$tools_dir/OGhidra/OGhidraMCP/src/main/java/com/lauriewired/GhidraMCPPlugin.java"
grep -Fq 'new InetSocketAddress(LOOPBACK_ADDRESS, this.currentPort)' "$oghidra_plugin_source" ||
  die "OGhidra loopback binding is missing after patching"
[[ "$(grep -Fc 'server.createContext(' "$oghidra_plugin_source")" == "1" ]] ||
  die "OGhidra contains an endpoint outside the centralized request guard"

uv python install 3.12
uv venv "$project_root/.venv-tools" --python 3.12 --clear
uv pip install --python "$project_root/.venv-tools/bin/python" \
  -e "$tools_dir/pyghidra-mcp" -e "$tools_dir/OGhidra"

(
  cd "$tools_dir/OGhidra"
  env GHIDRA_INSTALL_DIR="$ghidra_home" ./build_ghidra_plugin.sh
)
compiled_plugin_class="$tools_dir/OGhidra/OGhidraMCP/build/classes/java/main/com/lauriewired/GhidraMCPPlugin.class"
compiled_filter_class="$tools_dir/OGhidra/OGhidraMCP/build/classes/java/main/com/lauriewired/GhidraMCPPlugin\$LoopbackBrowserFilter.class"
[[ -f "$compiled_plugin_class" ]] || die "OGhidra extension build produced no plugin class"
[[ -f "$compiled_filter_class" ]] || die "OGhidra extension build produced no browser filter class"
strings "$compiled_plugin_class" | grep -F '127.0.0.1' >/dev/null ||
  die "compiled OGhidra extension is missing the loopback binding constant"
strings "$compiled_filter_class" |
  grep -F 'Forbidden: OGhidra accepts browser requests only from loopback origins.' >/dev/null ||
  die "compiled OGhidra extension is missing the browser-origin guard"
plugin_zip="$(find "$tools_dir/OGhidra/OGhidraMCP/dist" -maxdepth 1 -name '*.zip' -print -quit)"
[[ -n "$plugin_zip" && -f "$plugin_zip" ]] || die "OGhidra extension build produced no ZIP archive"
unzip -qo "$plugin_zip" -d "$ghidra_home/Ghidra/Extensions"

"$project_root/scripts/verify-re-toolchain.sh"

echo "Ghidra: $ghidra_home/ghidraRun"
echo "OGhidra: $tools_dir/OGhidra"
echo "PyGhidra-MCP: $project_root/.venv-tools/bin/pyghidra-mcp"
