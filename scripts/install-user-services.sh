#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_base="${XDG_CONFIG_HOME:-$HOME/.config}"
service_dir="$config_base/systemd/user"
mkdir -p "$service_dir"

for unit in localllm-ollama localllm-api; do
  sed "s|@PROJECT_ROOT@|$project_root|g" \
    "$project_root/deploy/systemd/$unit.service.in" > "$service_dir/$unit.service"
done
systemctl --user daemon-reload
systemctl --user enable --now localllm-ollama.service localllm-api.service
systemctl --user --no-pager --full status localllm-ollama.service localllm-api.service || true

