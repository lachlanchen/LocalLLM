#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${LOCALLLM_NOVNC_RUNTIME_DIR:-$project_root/.local/runtime/browser}"
profile_dir="$runtime_dir/profile"
log_dir="$runtime_dir/logs"
display_number="${LOCALLLM_NOVNC_DISPLAY_NUMBER:-100}"
vnc_port="${LOCALLLM_NOVNC_VNC_PORT:-5930}"
novnc_port="${LOCALLLM_NOVNC_PORT:-6130}"
cdp_port="${LOCALLLM_NOVNC_CDP_PORT:-9470}"
readiness_attempts="${LOCALLLM_NOVNC_READY_ATTEMPTS:-80}"
readiness_interval_seconds="${LOCALLLM_NOVNC_READY_INTERVAL_SECONDS:-0.25}"
novnc_web_root="/usr/share/novnc"
app_url="http://127.0.0.1:8008/"

started_names=()
started_pids=()
started_expectations=()
start_committed="false"

die() {
  echo "launch-novnc: $*" >&2
  exit 1
}

require_commands() {
  local command_name
  local missing=()
  for command_name in "$@"; do
    command -v "$command_name" >/dev/null || missing+=("$command_name")
  done
  if [[ "${#missing[@]}" -ne 0 ]]; then
    die "missing prerequisite commands: ${missing[*]}"
  fi
}

validate_configuration() {
  local port
  [[ "$display_number" =~ ^[0-9]+$ ]] ||
    die "LOCALLLM_NOVNC_DISPLAY_NUMBER must be a non-negative integer"
  for port in "$vnc_port" "$novnc_port" "$cdp_port"; do
    [[ "$port" =~ ^[1-9][0-9]*$ ]] && ((port <= 65535)) ||
      die "noVNC ports must be integers from 1 through 65535"
  done
  [[ "$readiness_attempts" =~ ^[1-9][0-9]*$ ]] ||
    die "LOCALLLM_NOVNC_READY_ATTEMPTS must be a positive integer"
  [[ "$readiness_interval_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    die "LOCALLLM_NOVNC_READY_INTERVAL_SECONDS must be a non-negative number"
}

pid_matches_expected() {
  local pid="$1"
  local expected="$2"
  local process_args
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  process_args="$(ps -p "$pid" -o args= 2>/dev/null)" || return 1
  grep -Fq -- "$expected" <<<"$process_args"
}

pid_is_owned() {
  local pid_file="$1"
  local expected="$2"
  local owned_pid
  [[ -f "$pid_file" ]] || return 1
  owned_pid="$(<"$pid_file")"
  pid_matches_expected "$owned_pid" "$expected"
}

pid_is_current_child() {
  local pid="$1"
  local expected="$2"
  local parent_pid
  pid_matches_expected "$pid" "$expected" || return 1
  parent_pid="$(ps -p "$pid" -o ppid= 2>/dev/null)" || return 1
  parent_pid="${parent_pid//[[:space:]]/}"
  [[ "$parent_pid" == "$$" ]]
}

pid_is_running() {
  local pid="$1"
  local process_state
  process_state="$(ps -p "$pid" -o stat= 2>/dev/null)" || return 1
  process_state="${process_state//[[:space:]]/}"
  [[ -n "$process_state" && "$process_state" != Z* ]]
}

display_process_is_running() {
  ps -eo comm=,args= 2>/dev/null | awk -v display=":$display_number" '
    $1 == "Xvfb" {
      for (field = 2; field <= NF; field++) {
        if ($field == display) {
          found = 1
          exit
        }
      }
    }
    END { exit(found ? 0 : 1) }
  '
}

cleanup_stale_display_artifacts() {
  local lock_path="/tmp/.X${display_number}-lock"
  local socket_path="/tmp/.X11-unix/X${display_number}"
  local lock_pid=""

  [[ -e "$lock_path" || -S "$socket_path" ]] || return 0
  # An unreadable lock cannot be inspected well enough to distinguish a stale
  # artifact from another user's live display, so never remove it automatically.
  [[ ! -e "$lock_path" || -r "$lock_path" ]] || return 1
  if [[ -r "$lock_path" ]]; then
    lock_pid="$(tr -d '[:space:]' <"$lock_path")"
    # A live PID is treated as foreign even when its command is unrelated: PID
    # reuse makes deleting another session's lock unsafe.
    if [[ "$lock_pid" =~ ^[1-9][0-9]*$ ]] && pid_is_running "$lock_pid"; then
      return 1
    fi
    # A malformed lock cannot be proved stale, so fail closed.
    [[ "$lock_pid" =~ ^[1-9][0-9]*$ ]] || return 1
  fi
  display_process_is_running && return 1

  rm -f -- "$lock_path" "$socket_path"
  echo "Removed stale X display artifacts for :$display_number" >&2
}

port_is_listening() {
  local port="$1"
  local listeners
  listeners="$(ss -H -ltn "sport = :$port" 2>/dev/null || true)"
  [[ -n "$listeners" ]]
}

show_log_tail() {
  local label="$1"
  local log_file="$2"
  if [[ -r "$log_file" ]]; then
    echo "--- $label log: $log_file ---" >&2
    tail -n 60 "$log_file" >&2 || true
  else
    echo "$label produced no readable log: $log_file" >&2
  fi
}

wait_for_x_socket() {
  local pid="$1"
  local log_file="$2"
  local attempt
  for ((attempt = 1; attempt <= readiness_attempts; attempt++)); do
    if ! pid_is_running "$pid"; then
      echo "Xvfb exited before display :$display_number became ready" >&2
      show_log_tail "Xvfb" "$log_file"
      return 1
    fi
    [[ -S "/tmp/.X11-unix/X$display_number" ]] && return 0
    sleep "$readiness_interval_seconds"
  done
  echo "Xvfb readiness timed out after $readiness_attempts attempts on :$display_number" >&2
  show_log_tail "Xvfb" "$log_file"
  return 1
}

wait_for_listener() {
  local label="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local attempt
  for ((attempt = 1; attempt <= readiness_attempts; attempt++)); do
    if ! pid_is_running "$pid"; then
      echo "$label exited before 127.0.0.1:$port became ready" >&2
      show_log_tail "$label" "$log_file"
      return 1
    fi
    port_is_listening "$port" && return 0
    sleep "$readiness_interval_seconds"
  done
  echo "$label readiness timed out after $readiness_attempts attempts on 127.0.0.1:$port" >&2
  show_log_tail "$label" "$log_file"
  return 1
}

wait_for_http() {
  local label="$1"
  local url="$2"
  local pid="$3"
  local log_file="$4"
  local attempt
  local last_error="HTTP probe was not attempted"
  for ((attempt = 1; attempt <= readiness_attempts; attempt++)); do
    if ! pid_is_running "$pid"; then
      echo "$label exited before $url became ready" >&2
      show_log_tail "$label" "$log_file"
      return 1
    fi
    if last_error="$(
      curl --fail --silent --show-error \
        --connect-timeout 1 --max-time 2 --output /dev/null "$url" 2>&1
    )"; then
      return 0
    fi
    sleep "$readiness_interval_seconds"
  done
  echo "$label readiness timed out after $readiness_attempts attempts: $url" >&2
  [[ -z "$last_error" ]] || echo "Last HTTP error: $last_error" >&2
  show_log_tail "$label" "$log_file"
  return 1
}

register_started_child() {
  local name="$1"
  local pid="$2"
  local expected="$3"
  started_names+=("$name")
  started_pids+=("$pid")
  started_expectations+=("$expected")
  printf '%s\n' "$pid" > "$runtime_dir/$name.pid"
}

wait_for_process_exit() {
  local pid="$1"
  local attempt
  for ((attempt = 1; attempt <= 40; attempt++)); do
    pid_is_running "$pid" || return 0
    sleep 0.1
  done
  return 1
}

cleanup_started_children() {
  local index name pid expected pid_file recorded_pid should_wait
  for ((index = ${#started_pids[@]} - 1; index >= 0; index--)); do
    name="${started_names[$index]}"
    pid="${started_pids[$index]}"
    expected="${started_expectations[$index]}"
    pid_file="$runtime_dir/$name.pid"
    should_wait="false"

    if pid_is_running "$pid"; then
      if pid_is_current_child "$pid" "$expected"; then
        should_wait="true"
        kill -TERM "$pid" 2>/dev/null || true
        if ! wait_for_process_exit "$pid"; then
          kill -KILL "$pid" 2>/dev/null || true
        fi
      else
        echo "Refusing to terminate non-child or unexpected process $pid during cleanup" >&2
      fi
    else
      should_wait="true"
    fi
    if [[ "$should_wait" == "true" ]]; then
      wait "$pid" 2>/dev/null || true
    fi

    if [[ -f "$pid_file" ]]; then
      recorded_pid="$(<"$pid_file")"
      [[ "$recorded_pid" != "$pid" ]] || rm -f -- "$pid_file"
    fi
  done
}

start_exit_handler() {
  local status="$1"
  trap - EXIT INT TERM HUP
  if [[ "$start_committed" != "true" && "${#started_pids[@]}" -ne 0 ]]; then
    echo "noVNC startup failed; cleaning only children launched by this attempt" >&2
    cleanup_started_children
  fi
  exit "$status"
}

start() {
  require_commands awk curl google-chrome grep mkdir nohup ps rm sleep ss tail timeout tr \
    websockify Xvfb x11vnc xdotool
  [[ -r "$novnc_web_root/vnc.html" ]] ||
    die "missing noVNC client: $novnc_web_root/vnc.html"

  mkdir -p "$profile_dir" "$log_dir" "$runtime_dir/evidence"
  if [[ -e "/tmp/.X${display_number}-lock" ]] &&
     ! pid_is_owned "$runtime_dir/xvfb.pid" "Xvfb :$display_number"; then
    cleanup_stale_display_artifacts ||
      die "display :$display_number is occupied by another process"
  elif [[ -S "/tmp/.X11-unix/X${display_number}" ]] &&
       ! pid_is_owned "$runtime_dir/xvfb.pid" "Xvfb :$display_number"; then
    cleanup_stale_display_artifacts ||
      die "display :$display_number is occupied by another process"
  fi
  local candidate_port
  for candidate_port in "$vnc_port" "$novnc_port" "$cdp_port"; do
    if port_is_listening "$candidate_port"; then
      die "port $candidate_port is occupied by another process"
    fi
  done

  trap 'start_exit_handler $?' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP

  local child_pid
  nohup Xvfb ":$display_number" -screen 0 1440x1000x24 -nolisten tcp -ac \
    >"$log_dir/xvfb.log" 2>&1 &
  child_pid="$!"
  register_started_child "xvfb" "$child_pid" "Xvfb :$display_number"
  wait_for_x_socket "$child_pid" "$log_dir/xvfb.log"

  nohup x11vnc -display ":$display_number" -localhost -nopw -forever -shared \
    -rfbport "$vnc_port" >"$log_dir/x11vnc.log" 2>&1 &
  child_pid="$!"
  register_started_child "x11vnc" "$child_pid" "-rfbport $vnc_port"
  wait_for_listener "x11vnc" "$vnc_port" "$child_pid" "$log_dir/x11vnc.log"

  nohup websockify --web="$novnc_web_root" \
    "127.0.0.1:$novnc_port" "127.0.0.1:$vnc_port" \
    >"$log_dir/websockify.log" 2>&1 &
  child_pid="$!"
  register_started_child "websockify" "$child_pid" "127.0.0.1:$novnc_port"
  wait_for_http "websockify" "http://127.0.0.1:$novnc_port/vnc.html" \
    "$child_pid" "$log_dir/websockify.log"

  DISPLAY=":$display_number" nohup google-chrome \
    --user-data-dir="$profile_dir" \
    --remote-debugging-address=127.0.0.1 \
    --remote-debugging-port="$cdp_port" \
    --window-position=0,0 \
    --window-size=1440,1000 \
    --no-first-run \
    --no-default-browser-check \
    --disable-dev-shm-usage \
    --disable-gpu \
    "$app_url" >"$log_dir/chrome.log" 2>&1 &
  child_pid="$!"
  register_started_child "chrome" "$child_pid" "--user-data-dir=$profile_dir"
  wait_for_http "Chrome CDP" "http://127.0.0.1:$cdp_port/json/version" \
    "$child_pid" "$log_dir/chrome.log"

  DISPLAY=":$display_number" timeout 10 xdotool \
    search --sync --onlyvisible --class 'google-chrome' \
    windowmove --sync 0 0 windowsize --sync 1440 1000 >/dev/null 2>&1 || true

  start_committed="true"
  trap - EXIT INT TERM HUP
  status
}

status() {
  local cdp_ok="false"
  local novnc_ok="false"
  curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
    "http://127.0.0.1:$cdp_port/json/version" >/dev/null 2>&1 && cdp_ok="true"
  curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
    "http://127.0.0.1:$novnc_port/vnc.html" >/dev/null 2>&1 && novnc_ok="true"
  printf '{"display":":%s","vnc":"127.0.0.1:%s","novnc":"http://127.0.0.1:%s/vnc.html?host=127.0.0.1&port=%s&autoconnect=1&resize=scale","cdp":"http://127.0.0.1:%s","cdpReady":%s,"novncReady":%s}\n' \
    "$display_number" "$vnc_port" "$novnc_port" "$novnc_port" "$cdp_port" "$cdp_ok" "$novnc_ok"
}

stop() {
  local process_name pid_file expected process_pid
  for process_name in chrome websockify x11vnc xvfb; do
    pid_file="$runtime_dir/$process_name.pid"
    case "$process_name" in
      chrome) expected="--user-data-dir=$profile_dir" ;;
      websockify) expected="127.0.0.1:$novnc_port" ;;
      x11vnc) expected="-rfbport $vnc_port" ;;
      xvfb) expected="Xvfb :$display_number" ;;
    esac
    if pid_is_owned "$pid_file" "$expected"; then
      process_pid="$(<"$pid_file")"
      kill -TERM "$process_pid" 2>/dev/null || true
      if ! wait_for_process_exit "$process_pid" &&
         pid_matches_expected "$process_pid" "$expected"; then
        kill -KILL "$process_pid" 2>/dev/null || true
        wait_for_process_exit "$process_pid" || true
      fi
    elif [[ -f "$pid_file" ]]; then
      echo "Ignoring stale or foreign PID file: $pid_file" >&2
    fi
    rm -f -- "$pid_file"
  done
  # Xorg normally removes these paths itself. Recover only when the lock PID is
  # dead and no matching Xvfb process remains; live or ambiguous displays stay.
  cleanup_stale_display_artifacts || true
}

action="${1:-start}"
validate_configuration
case "$action" in
  start) start ;;
  status)
    require_commands curl grep ps
    status
    ;;
  stop)
    require_commands awk grep ps rm sleep tr
    stop
    ;;
  *) echo "Usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
