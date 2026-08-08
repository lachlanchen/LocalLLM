#!/usr/bin/env bash
set -euo pipefail

image_ref="localllm/usb-evidence:ubuntu24.04-20260808"

die() {
  echo "analyze-usb-pcap: $*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
Usage: scripts/analyze-usb-pcap.sh CAPTURE.pcap[ng] [TSHARK_ARGUMENT ...]

The evidence file is mounted read-only into an offline, capability-free
container. With no extra arguments, the first 500 frames are summarized.

Examples:
  scripts/analyze-usb-pcap.sh capture.pcapng
  scripts/analyze-usb-pcap.sh capture.pcapng -Y 'usb.transfer_type == 0x03' -V
  scripts/analyze-usb-pcap.sh capture.pcapng -Y usb -T json
EOF
  exit 2
}

[[ "$#" -ge 1 ]] || usage
capture="$1"
shift

for required_command in docker id realpath; do
  command -v "$required_command" >/dev/null || die "$required_command is required"
done
docker info >/dev/null 2>&1 ||
  die "the current user cannot reach the Docker daemon; no sudo fallback is attempted"
docker image inspect "$image_ref" >/dev/null 2>&1 ||
  die "image is not installed; run scripts/setup-usb-evidence-tools.sh"
[[ -f "$capture" && -r "$capture" ]] ||
  die "capture must be a readable regular file: $capture"

capture="$(realpath -- "$capture")"
[[ "$capture" != *','* ]] ||
  die "capture paths containing commas are unsupported by Docker bind-mount syntax"

for argument in "$@"; do
  case "$argument" in
    -i*|--interface*|-D|--list-interfaces*|-L|--list-data-link-types*)
      die "live-capture option '$argument' is not allowed in the offline wrapper"
      ;;
  esac
done

if [[ "$#" -eq 0 ]]; then
  set -- \
    -c 500 \
    -T fields \
    -E header=y \
    -E quote=d \
    -E separator=, \
    -e frame.number \
    -e frame.time_relative \
    -e _ws.col.Protocol \
    -e usb.bus_id \
    -e usb.device_address \
    -e usb.endpoint_address \
    -e usb.transfer_type \
    -e usb.urb_type \
    -e usb.capdata \
    -e _ws.col.Info
fi

exec docker run \
  --rm \
  --init \
  --network none \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --read-only \
  --pids-limit 128 \
  --memory 1g \
  --cpus 2 \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  --mount "type=bind,source=$capture,target=/evidence/capture.pcap,readonly" \
  "$image_ref" \
  -n -r /evidence/capture.pcap "$@"
