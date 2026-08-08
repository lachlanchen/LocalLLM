#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$project_root/.local/runtime/browser"
profile_dir="$runtime_dir/profile"
log_dir="$runtime_dir/logs"
display_number="100"
vnc_port="5930"
novnc_port="6130"
cdp_port="9470"
app_url="http://127.0.0.1:8008/"

pid_is_owned() {
  local pid_file="$1"
  local expected="$2"
  [[ -f "$pid_file" ]] || return 1
  local owned_pid
  owned_pid="$(<"$pid_file")"
  [[ -n "$owned_pid" ]] || return 1
  ps -p "$owned_pid" -o args= 2>/dev/null | grep -Fq -- "$expected"
}

wait_for_port() {
  local port="$1"
  for _attempt in $(seq 1 80); do
    if curl -fsS "http://127.0.0.1:$port/json/version" >/dev/null 2>&1 ||
       ss -ltn | grep -q "127.0.0.1:$port"; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

start() {
  mkdir -p "$profile_dir" "$log_dir" "$runtime_dir/evidence"
  if [[ -e "/tmp/.X${display_number}-lock" ]] && ! pid_is_owned "$runtime_dir/xvfb.pid" "Xvfb :$display_number"; then
    echo "Display :$display_number is occupied by another process" >&2
    exit 1
  fi
  for candidate_port in "$vnc_port" "$novnc_port" "$cdp_port"; do
    if ss -ltn | grep -q ":${candidate_port} "; then
      echo "Port $candidate_port is occupied by another process" >&2
      exit 1
    fi
  done

  nohup Xvfb ":$display_number" -screen 0 1440x1000x24 -nolisten tcp -ac \
    >"$log_dir/xvfb.log" 2>&1 &
  echo "$!" > "$runtime_dir/xvfb.pid"
  for _attempt in $(seq 1 40); do
    [[ -S "/tmp/.X11-unix/X$display_number" ]] && break
    sleep 0.25
  done

  nohup x11vnc -display ":$display_number" -localhost -nopw -forever -shared \
    -rfbport "$vnc_port" >"$log_dir/x11vnc.log" 2>&1 &
  echo "$!" > "$runtime_dir/x11vnc.pid"
  wait_for_port "$vnc_port"

  nohup websockify --web=/usr/share/novnc "127.0.0.1:$novnc_port" "127.0.0.1:$vnc_port" \
    >"$log_dir/websockify.log" 2>&1 &
  echo "$!" > "$runtime_dir/websockify.pid"
  wait_for_port "$novnc_port"

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
  echo "$!" > "$runtime_dir/chrome.pid"
  wait_for_port "$cdp_port"

  DISPLAY=":$display_number" xdotool search --sync --onlyvisible --class 'google-chrome' \
    windowmove --sync 0 0 windowsize --sync 1440 1000 >/dev/null 2>&1 || true
  status
}

status() {
  local cdp_ok="false"
  local novnc_ok="false"
  curl -fsS "http://127.0.0.1:$cdp_port/json/version" >/dev/null 2>&1 && cdp_ok="true"
  curl -fsS "http://127.0.0.1:$novnc_port/vnc.html" >/dev/null 2>&1 && novnc_ok="true"
  printf '{"display":":%s","vnc":"127.0.0.1:%s","novnc":"http://127.0.0.1:%s/vnc.html?host=127.0.0.1&port=%s&autoconnect=1&resize=scale","cdp":"http://127.0.0.1:%s","cdpReady":%s,"novncReady":%s}\n' \
    "$display_number" "$vnc_port" "$novnc_port" "$novnc_port" "$cdp_port" "$cdp_ok" "$novnc_ok"
}

stop() {
  for process_name in chrome websockify x11vnc xvfb; do
    pid_file="$runtime_dir/$process_name.pid"
    if [[ -f "$pid_file" ]]; then
      process_pid="$(<"$pid_file")"
      if kill -0 "$process_pid" 2>/dev/null; then kill "$process_pid"; fi
    fi
  done
}

case "${1:-start}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  *) echo "Usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac

