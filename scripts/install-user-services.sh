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

for required_command in awk chmod curl id mkdir mktemp mv sed sleep systemctl; do
  command -v "$required_command" >/dev/null || die "missing prerequisite: $required_command"
done

read_numeric_setting() {
  local name="$1"
  local fallback="$2"
  local value="${!name:-}"

  if [[ -z "$value" && -f "$project_root/.env" ]]; then
    value="$(awk -F= -v key="$name" '$1 == key { value = substr($0, index($0, "=") + 1) } END { print value }' "$project_root/.env")"
  fi
  printf '%s' "${value:-$fallback}"
}

expected_gpu_count="$(read_numeric_setting LOCALLLM_EXPECTED_GPU_COUNT 0)"
gpu_ready_attempts="$(read_numeric_setting LOCALLLM_GPU_READY_ATTEMPTS 120)"
gpu_ready_interval_seconds="$(read_numeric_setting LOCALLLM_GPU_READY_INTERVAL_SECONDS 1)"
ollama_context_length="$(read_numeric_setting LOCALLLM_OLLAMA_CONTEXT_LENGTH 65536)"
ollama_sched_spread="$(read_numeric_setting LOCALLLM_OLLAMA_SCHED_SPREAD 0)"
[[ "$expected_gpu_count" =~ ^[0-9]+$ ]] ||
  die "LOCALLLM_EXPECTED_GPU_COUNT must be a non-negative integer"
[[ "$gpu_ready_attempts" =~ ^[1-9][0-9]*$ ]] ||
  die "LOCALLLM_GPU_READY_ATTEMPTS must be a positive integer"
[[ "$gpu_ready_interval_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "LOCALLLM_GPU_READY_INTERVAL_SECONDS must be a non-negative number"
[[ "$ollama_context_length" =~ ^[1-9][0-9]*$ ]] ||
  die "LOCALLLM_OLLAMA_CONTEXT_LENGTH must be a positive integer"
((ollama_context_length >= 2048 && ollama_context_length <= 262144)) ||
  die "LOCALLLM_OLLAMA_CONTEXT_LENGTH must be between 2048 and 262144"
case "${ollama_sched_spread,,}" in
  0|false|no) ollama_sched_spread=0 ;;
  1|true|yes) ollama_sched_spread=1 ;;
  *) die "LOCALLLM_OLLAMA_SCHED_SPREAD must be 0/1, true/false, or yes/no" ;;
esac
gpu_start_timeout_seconds="$(awk -v attempts="$gpu_ready_attempts" -v interval="$gpu_ready_interval_seconds" 'BEGIN { value = attempts * interval + 30; rounded = int(value); if (value > rounded) rounded++; print rounded }')"
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
        --connect-timeout 1 --max-time 3 --output /dev/null "$url" 2>&1
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

verify_ollama_gpu_inventory() {
  if ((expected_gpu_count == 0)); then
    return 0
  fi
  command -v journalctl >/dev/null ||
    die "journalctl is required to verify Ollama GPU discovery"

  local attempt
  local discovered_count
  local inventory_summary
  local invocation_id
  invocation_id="$(
    systemctl --user show --property InvocationID --value localllm-ollama.service
  )"
  [[ "$invocation_id" =~ ^[0-9a-fA-F]{32}$ ]] ||
    die "could not determine the current Ollama service invocation ID"

  # Ollama can accept HTTP before asynchronous accelerator discovery is logged.
  # Poll only this boot and this systemd invocation. The parser counts distinct
  # PCI IDs and accepts a legacy compute ID only with a known accelerator
  # library, so CPU fallback rows cannot produce a false pass.
  for ((attempt = 1; attempt <= gpu_ready_attempts; attempt++)); do
    inventory_summary="$(
      journalctl --user --boot --unit localllm-ollama.service \
        _SYSTEMD_INVOCATION_ID="$invocation_id" --no-pager --output=cat \
        2>/dev/null | awk -f "$project_root/scripts/parse-ollama-gpu-inventory.awk"
    )"
    discovered_count="${inventory_summary%% *}"
    [[ "$discovered_count" =~ ^[0-9]+$ ]] || discovered_count=0
    if ((discovered_count >= expected_gpu_count)); then
      echo "Ollama GPU inventory PASS: $inventory_summary (expected at least $expected_gpu_count; attempt $attempt/$gpu_ready_attempts)"
      return 0
    fi
    if ! systemctl --user is-active --quiet localllm-ollama.service; then
      echo "Ollama stopped during GPU discovery verification" >&2
      show_unit_diagnostics "localllm-ollama.service"
      return 1
    fi
    sleep "$gpu_ready_interval_seconds"
  done

  echo "Ollama discovered $discovered_count distinct inference GPU(s), expected at least $expected_gpu_count; current invocation: $invocation_id" >&2
  show_unit_diagnostics "localllm-ollama.service"
  return 1
}

for required_path in \
  "$project_root/.local/ollama/bin/ollama" \
  "$project_root/apps/api/.venv/bin/uvicorn" \
  "$project_root/apps/web/dist/index.html" \
  "$project_root/scripts/parse-ollama-gpu-inventory.awk" \
  "$project_root/scripts/wait-for-gpus.sh"; do
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
  sed \
    -e "s|@PROJECT_ROOT@|$project_root|g" \
    -e "s|@EXPECTED_GPU_COUNT@|$expected_gpu_count|g" \
    -e "s|@GPU_READY_ATTEMPTS@|$gpu_ready_attempts|g" \
    -e "s|@GPU_READY_INTERVAL_SECONDS@|$gpu_ready_interval_seconds|g" \
    -e "s|@GPU_START_TIMEOUT_SECONDS@|$gpu_start_timeout_seconds|g" \
    -e "s|@OLLAMA_CONTEXT_LENGTH@|$ollama_context_length|g" \
    -e "s|@OLLAMA_SCHED_SPREAD@|$ollama_sched_spread|g" \
    "$project_root/deploy/systemd/$unit.service.in" > "$rendered_unit"
  chmod 600 "$rendered_unit"
  mv "$rendered_unit" "$service_dir/$unit.service"
done
systemctl --user daemon-reload
systemctl --user enable localllm-ollama.service localllm-api.service
systemctl --user is-enabled --quiet localllm-ollama.service localllm-api.service
if ! systemctl --user restart localllm-ollama.service; then
  echo "Ollama failed during activation" >&2
  show_unit_diagnostics "localllm-ollama.service"
  exit 1
fi
wait_for_http_ready \
  "Ollama" "http://127.0.0.1:11434/api/version" "localllm-ollama.service"
verify_ollama_gpu_inventory
if ! systemctl --user restart localllm-api.service; then
  echo "LocalLLM API failed during activation" >&2
  show_unit_diagnostics "localllm-api.service"
  exit 1
fi
wait_for_http_ready \
  "LocalLLM API" "http://127.0.0.1:8008/readyz" "localllm-api.service"
systemctl --user --no-pager --full status localllm-ollama.service localllm-api.service
