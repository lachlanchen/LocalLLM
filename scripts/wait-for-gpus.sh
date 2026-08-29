#!/usr/bin/env bash
set -euo pipefail

expected_count="${LOCALLLM_EXPECTED_GPU_COUNT:-0}"
attempts="${LOCALLLM_GPU_READY_ATTEMPTS:-120}"
interval_seconds="${LOCALLLM_GPU_READY_INTERVAL_SECONDS:-1}"
selected_indexes="${LOCALLLM_OLLAMA_CUDA_VISIBLE_DEVICES:-}"

die() {
  echo "wait-for-gpus: $*" >&2
  exit 1
}

[[ "$expected_count" =~ ^[0-9]+$ ]] ||
  die "LOCALLLM_EXPECTED_GPU_COUNT must be a non-negative integer"
[[ "$attempts" =~ ^[1-9][0-9]*$ ]] ||
  die "LOCALLLM_GPU_READY_ATTEMPTS must be a positive integer"
[[ "$interval_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "LOCALLLM_GPU_READY_INTERVAL_SECONDS must be a non-negative number"
if [[ -n "$selected_indexes" ]]; then
  [[ "$selected_indexes" =~ ^[0-9]+(,[0-9]+)*$ ]] ||
    die "LOCALLLM_OLLAMA_CUDA_VISIBLE_DEVICES must be a comma-separated list of GPU indexes"
  [[ "${CUDA_DEVICE_ORDER:-}" == PCI_BUS_ID ]] ||
    die "CUDA_DEVICE_ORDER must be PCI_BUS_ID for deterministic GPU-index selection"
  [[ "${CUDA_VISIBLE_DEVICES:-}" == "$selected_indexes" ]] ||
    die "CUDA_VISIBLE_DEVICES does not match the reviewed Ollama GPU selection"
  IFS=',' read -r -a selected_gpu_indexes <<<"$selected_indexes"
  ((${#selected_gpu_indexes[@]} == expected_count)) ||
    die "LOCALLLM_EXPECTED_GPU_COUNT must equal the selected Ollama GPU count"
fi

if ((expected_count == 0)); then
  echo "GPU readiness check disabled (LOCALLLM_EXPECTED_GPU_COUNT=0)"
  exit 0
fi

command -v nvidia-smi >/dev/null ||
  die "nvidia-smi is required when LOCALLLM_EXPECTED_GPU_COUNT is nonzero"

for ((attempt = 1; attempt <= attempts; attempt++)); do
  detected_count=0
  if gpu_indexes="$(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null
  )"; then
    detected_count="$(
      awk '
        /^[[:space:]]*[0-9]+[[:space:]]*$/ {
          index_value = $0
          gsub(/[[:space:]]/, "", index_value)
          if (!seen[index_value]++) count++
        }
        END { print count + 0 }
      ' <<<"$gpu_indexes"
    )"
  fi

  if [[ "$detected_count" =~ ^[0-9]+$ ]] && ((detected_count >= expected_count)); then
    if [[ -n "$selected_indexes" ]]; then
      missing_index=''
      for selected_gpu_index in "${selected_gpu_indexes[@]}"; do
        if ! awk -v wanted="$selected_gpu_index" '
          /^[[:space:]]*[0-9]+[[:space:]]*$/ {
            value = $0
            gsub(/[[:space:]]/, "", value)
            if (value == wanted) found = 1
          }
          END { exit(found ? 0 : 1) }
        ' <<<"$gpu_indexes"; then
          missing_index="$selected_gpu_index"
          break
        fi
      done
      if [[ -n "$missing_index" ]]; then
        if ((attempt < attempts)); then sleep "$interval_seconds"; fi
        continue
      fi
      echo "GPU readiness PASS: selected indexes $selected_indexes are present"
      exit 0
    else
      echo "GPU readiness PASS: detected $detected_count, expected at least $expected_count"
      exit 0
    fi
  fi

  if ((attempt < attempts)); then
    sleep "$interval_seconds"
  fi
done

die "detected ${detected_count:-0} GPU(s), expected at least $expected_count after $attempts attempts"
