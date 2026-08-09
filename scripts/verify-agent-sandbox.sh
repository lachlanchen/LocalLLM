#!/usr/bin/env bash
set -euo pipefail

image_ref="localllm/python-sandbox:3.12.11-20260809"
expected_base="docker.io/library/python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49"

die() {
  echo "verify-agent-sandbox: $*" >&2
  exit 1
}

command -v docker >/dev/null || die "Docker is required"
[[ "$(command -v docker)" == "/usr/bin/docker" ]] || die "Docker must be installed at /usr/bin/docker"
docker image inspect "$image_ref" >/dev/null 2>&1 ||
  die "image is not installed; run scripts/setup-agent-sandbox.sh"

actual_base="$(
  docker image inspect "$image_ref" \
    --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}'
)"
[[ "$actual_base" == "$expected_base" ]] ||
  die "unexpected base label: ${actual_base:-none}"

actual_profile="$(
  docker image inspect "$image_ref" \
    --format '{{index .Config.Labels "io.localllm.sandbox.profile"}}'
)"
[[ "$actual_profile" == "python-v1" ]] ||
  die "unexpected sandbox profile: ${actual_profile:-none}"

runtime=(
  /usr/bin/docker run
  --rm
  --interactive
  --network none
  --cap-drop ALL
  --security-opt no-new-privileges:true
  --read-only
  --pids-limit 64
  --memory 512m
  --memory-swap 512m
  --cpus 1
  --ulimit nofile=128:128
  --ulimit nproc=64:64
  --user 65532:65532
  --workdir /work
  --env HOME=/work
  --env TMPDIR=/work
  --tmpfs /work:rw,noexec,nosuid,nodev,size=64m,uid=65532,gid=65532,mode=700
  --log-driver none
  --entrypoint /usr/local/bin/python3
  "$image_ref"
  -I -S -B -u -
)

verification="$("${runtime[@]}" <<'PY'
import os
import pathlib
import socket

assert os.getuid() == 65532
status = pathlib.Path("/proc/self/status").read_text()
assert "CapEff:\t0000000000000000" in status
assert "NoNewPrivs:\t1" in status

try:
    pathlib.Path("/etc/localllm-sandbox-write-test").write_text("forbidden")
except OSError:
    pass
else:
    raise AssertionError("read-only root filesystem was writable")

probe = pathlib.Path("/work/probe.txt")
probe.write_text("ephemeral")
assert probe.read_text() == "ephemeral"

sock = socket.socket()
sock.settimeout(0.25)
try:
    sock.connect(("1.1.1.1", 53))
except OSError:
    pass
else:
    raise AssertionError("external network unexpectedly reachable")
finally:
    sock.close()

print("uid=65532 caps=none no-new-privileges=1 rootfs=readonly work=tmpfs network=none")
PY
)"

[[ "$verification" == "uid=65532 caps=none no-new-privileges=1 rootfs=readonly work=tmpfs network=none" ]] ||
  die "unexpected sandbox verification output: $verification"

printf '%s\n' "$verification"
echo "Agent sandbox verification PASS"
