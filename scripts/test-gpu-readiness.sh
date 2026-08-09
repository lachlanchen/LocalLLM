#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parser="$project_root/scripts/parse-ollama-gpu-inventory.awk"
fixtures="$project_root/scripts/tests/fixtures"

fail() {
  echo "test-gpu-readiness: $*" >&2
  exit 1
}

assert_inventory() {
  local fixture="$1"
  local expected="$2"
  local actual
  actual="$(awk -f "$parser" "$fixtures/$fixture")"
  [[ "$actual" == "$expected" ]] ||
    fail "$fixture: expected '$expected', got '$actual'"
}

assert_inventory \
  ollama-inventory-two-gpus.log \
  "2 pci_id=0000:09:00.0 pci_id=0000:01:00.0"
assert_inventory ollama-inventory-cpu-only.log "0"
assert_inventory ollama-inventory-mixed.log "1 pci_id=0000:01:00.0"
assert_inventory \
  ollama-inventory-legacy.log \
  "2 library=cuda,id=0 library=rocm,id=1"

function nvidia-smi {
  [[ "$*" == "--query-gpu=index --format=csv,noheader,nounits" ]] || return 2
  case "${FAKE_NVIDIA_SMI_MODE:?}" in
    success-two)
      printf '0\n1\n'
      ;;
    duplicate-one)
      printf '0\n0\n'
      ;;
    partial-failure)
      printf '0\n1\n'
      return 1
      ;;
    *)
      return 2
      ;;
  esac
}
export -f nvidia-smi

FAKE_NVIDIA_SMI_MODE=success-two \
  LOCALLLM_EXPECTED_GPU_COUNT=2 \
  LOCALLLM_GPU_READY_ATTEMPTS=1 \
  LOCALLLM_GPU_READY_INTERVAL_SECONDS=0 \
  bash "$project_root/scripts/wait-for-gpus.sh" >/dev/null

if FAKE_NVIDIA_SMI_MODE=partial-failure \
  LOCALLLM_EXPECTED_GPU_COUNT=2 \
  LOCALLLM_GPU_READY_ATTEMPTS=1 \
  LOCALLLM_GPU_READY_INTERVAL_SECONDS=0 \
  bash "$project_root/scripts/wait-for-gpus.sh" >/dev/null 2>&1; then
  fail "partial nvidia-smi output with a failing exit status was accepted"
fi

if FAKE_NVIDIA_SMI_MODE=duplicate-one \
  LOCALLLM_EXPECTED_GPU_COUNT=2 \
  LOCALLLM_GPU_READY_ATTEMPTS=1 \
  LOCALLLM_GPU_READY_INTERVAL_SECONDS=0 \
  bash "$project_root/scripts/wait-for-gpus.sh" >/dev/null 2>&1; then
  fail "duplicate GPU indexes were counted as distinct devices"
fi

echo "GPU readiness fixtures PASS"
