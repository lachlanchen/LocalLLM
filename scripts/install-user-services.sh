#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_base="${XDG_CONFIG_HOME:-$HOME/.config}"
service_dir="$config_base/systemd/user"
runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
readiness_attempts="${LOCALLLM_SERVICE_READY_ATTEMPTS:-120}"
readiness_interval_seconds="${LOCALLLM_SERVICE_READY_INTERVAL_SECONDS:-0.25}"

die() {
  echo "install-user-services: $*" >&2
  exit 1
}

for required_command in chmod curl id mkdir mktemp mv sed sleep systemctl; do
  command -v "$required_command" >/dev/null || die "missing prerequisite: $required_command"
done
[[ "$readiness_attempts" =~ ^[1-9][0-9]*$ ]] ||
  die "LOCALLLM_SERVICE_READY_ATTEMPTS must be a positive integer"
[[ "$readiness_interval_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "LOCALLLM_SERVICE_READY_INTERVAL_SECONDS must be a non-negative number"

show_unit_diagnostics() {
  local unit="$1"
  echo "--- $unit status ---" >&2
  systemctl --user --no-pager --full status "$unit" >&2 || true
  if command -v journalctl >/dev/null; then
    echo "--- $unit recent journal ---" >&2
    journalctl --user --unit "$unit" --no-pager --lines 50 >&2 || true
  fi
}

wait_for_http_ready() {
  local label="$1"
  local url="$2"
  local unit="$3"
  local attempt
  local last_error="HTTP probe was not attempted"

  for ((attempt = 1; attempt <= readiness_attempts; attempt++)); do
    if last_error="$(
      curl --fail --silent --show-error \
        --connect-timeout 1 --max-time 2 --output /dev/null "$url" 2>&1
    )"; then
      echo "$label ready: $url (attempt $attempt/$readiness_attempts)"
      return 0
    fi

    if ! systemctl --user is-active --quiet "$unit"; then
      echo "$label stopped before becoming ready: $unit" >&2
      [[ -z "$last_error" ]] || echo "Last HTTP error: $last_error" >&2
      show_unit_diagnostics "$unit"
      return 1
    fi
    sleep "$readiness_interval_seconds"
  done

  echo "$label readiness timed out after $readiness_attempts attempts: $url" >&2
  [[ -z "$last_error" ]] || echo "Last HTTP error: $last_error" >&2
  show_unit_diagnostics "$unit"
  return 1
}

for required_path in \
  "$project_root/.local/ollama/bin/ollama" \
  "$project_root/apps/api/.venv/bin/uvicorn" \
  "$project_root/apps/web/dist/index.html"; do
  [[ -e "$required_path" ]] || {
    echo "Missing runtime prerequisite: $required_path" >&2
    exit 1
  }
done

mkdir -p "$service_dir" "$project_root/data"
chmod 700 "$project_root/data"
[[ ! -f "$project_root/.env" ]] || chmod 600 "$project_root/.env"

if [[ -S "$runtime_dir/bus" ]]; then
  export XDG_RUNTIME_DIR="$runtime_dir"
  export DBUS_SESSION_BUS_ADDRESS="unix:path=$runtime_dir/bus"
fi

for unit in localllm-ollama localllm-api; do
  rendered_unit="$(mktemp "$service_dir/$unit.service.XXXXXX")"
  sed "s|@PROJECT_ROOT@|$project_root|g" \
    "$project_root/deploy/systemd/$unit.service.in" > "$rendered_unit"
  chmod 600 "$rendered_unit"
  mv "$rendered_unit" "$service_dir/$unit.service"
done
systemctl --user daemon-reload
systemctl --user enable --now localllm-ollama.service localllm-api.service
systemctl --user is-enabled --quiet localllm-ollama.service localllm-api.service
if ! systemctl --user is-active --quiet \
  localllm-ollama.service localllm-api.service; then
  echo "One or more LocalLLM services failed during activation" >&2
  show_unit_diagnostics "localllm-ollama.service"
  show_unit_diagnostics "localllm-api.service"
  exit 1
fi
wait_for_http_ready \
  "Ollama" "http://127.0.0.1:11434/api/version" "localllm-ollama.service"
wait_for_http_ready \
  "LocalLLM API" "http://127.0.0.1:8008/healthz" "localllm-api.service"
systemctl --user --no-pager --full status localllm-ollama.service localllm-api.service
