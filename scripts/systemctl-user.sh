#!/usr/bin/env bash
set -euo pipefail

# GUI and tmux shells can inherit a session-private D-Bus address that does not
# own org.freedesktop.systemd1.  User services belong to the canonical per-user
# manager, so always address its runtime bus explicitly.
runtime_dir="${LOCALLLM_USER_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}}"
user_bus="$runtime_dir/bus"

if [[ ! -S "$user_bus" ]]; then
  echo "systemctl-user: canonical user bus is unavailable: $user_bus" >&2
  exit 1
fi

export XDG_RUNTIME_DIR="$runtime_dir"
export DBUS_SESSION_BUS_ADDRESS="unix:path=$user_bus"
exec systemctl --user "$@"
