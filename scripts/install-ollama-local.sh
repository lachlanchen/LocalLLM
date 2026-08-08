#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_dir="$project_root/.local/ollama"

if [[ -x "$install_dir/bin/ollama" ]]; then
  "$install_dir/bin/ollama" --version || true
  exit 0
fi

mkdir -p "$install_dir"
curl -fL --progress-bar https://ollama.com/download/ollama-linux-amd64.tar.zst \
  | tar --zstd -xf - -C "$install_dir"
"$install_dir/bin/ollama" --version || true

