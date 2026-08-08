#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_dir="$project_root/.local/ollama"
download_dir="$project_root/.local/downloads"
ollama_version="0.32.6"
ollama_archive="$download_dir/ollama-linux-amd64-v${ollama_version}.tar.zst"
ollama_url="https://github.com/ollama/ollama/releases/download/v${ollama_version}/ollama-linux-amd64.tar.zst"
ollama_sha256="dec2fa50d24e6868ca3c4c977d69d059399372105f951a9acc320a5a79aadcfc"

for prerequisite in curl tar sha256sum zstd; do
  command -v "$prerequisite" >/dev/null || {
    echo "$prerequisite is required" >&2
    exit 1
  }
done

if [[ -x "$install_dir/bin/ollama" ]] \
  && "$install_dir/bin/ollama" --version 2>&1 | grep -q "$ollama_version"; then
  "$install_dir/bin/ollama" --version
  exit 0
fi

mkdir -p "$install_dir" "$download_dir"
if [[ ! -f "$ollama_archive" ]]; then
  curl -fL --progress-bar "$ollama_url" -o "$ollama_archive.part"
  mv "$ollama_archive.part" "$ollama_archive"
fi
printf '%s  %s\n' "$ollama_sha256" "$ollama_archive" | sha256sum -c -
tar --zstd -xf "$ollama_archive" -C "$install_dir"
"$install_dir/bin/ollama" --version
