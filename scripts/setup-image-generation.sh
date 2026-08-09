#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="$project_root/.local/image-generation"
venv_dir="$runtime_root/venv"
requirements="$project_root/tools/image-generation/requirements.txt"
requirements_lock="$project_root/tools/image-generation/requirements.lock.txt"
bootstrap_python="${LOCALLLM_IMAGE_GENERATION_BOOTSTRAP_PYTHON:-python3}"

die() {
  echo "setup-image-generation: $*" >&2
  exit 1
}

command -v "$bootstrap_python" >/dev/null || die "missing Python interpreter: $bootstrap_python"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required to verify local CUDA hardware"
command -v sha256sum >/dev/null || die "sha256sum is required for runtime lock verification"
[[ -x /usr/bin/bwrap ]] || die "bubblewrap is required at /usr/bin/bwrap for worker network isolation"
[[ -r "$requirements" ]] || die "missing pinned requirements: $requirements"
[[ -r "$requirements_lock" ]] || die "missing hash-locked requirements: $requirements_lock"

"$bootstrap_python" - <<'PY'
import sys

if sys.version_info[:3] != (3, 10, 13):
    raise SystemExit(
        "setup-image-generation: Python 3.10.13 is required by the attested runtime layout"
    )
print(f"Bootstrap Python: {sys.version.split()[0]}")
PY

gpu_indexes="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)" || \
  die "nvidia-smi could not enumerate CUDA devices"
[[ -n "$gpu_indexes" ]] || die "nvidia-smi found no CUDA devices"
echo "Visible NVIDIA devices: $(wc -l <<<"$gpu_indexes" | tr -d '[:space:]')"

mkdir -p "$runtime_root" "$runtime_root/cache" "$runtime_root/tmp"
chmod 0700 "$runtime_root" "$runtime_root/cache" "$runtime_root/tmp"
rebuild_venv=false
if [[ ! -x "$venv_dir/bin/python" ]]; then
  rebuild_venv=true
elif grep -Fq 'include-system-site-packages = true' "$venv_dir/pyvenv.cfg"; then
  rebuild_venv=true
else
  existing_python="$($venv_dir/bin/python -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
  existing_site="$($venv_dir/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null || true)"
  if [[ "$existing_python" != "3.10.13" || -z "$existing_site" ]]; then
    rebuild_venv=true
  elif [[ -e "$existing_site/localllm-host-torch.pth" ]]; then
    rebuild_venv=true
  fi
fi
if [[ "$rebuild_venv" == true ]]; then
  "$bootstrap_python" -m venv --clear "$venv_dir"
fi

lock_sha256="$(sha256sum "$requirements_lock" | awk '{print $1}')"
force_reinstall=false
if [[ "$rebuild_venv" == false ]]; then
  installed_lock_sha256="$($venv_dir/bin/python - "$venv_dir/.localllm-runtime.json" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    marker = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit
value = marker.get("lock_sha256")
if isinstance(value, str):
    print(value)
PY
)"
  [[ "$installed_lock_sha256" == "$lock_sha256" ]] || force_reinstall=true
fi

pip_arguments=(
  --disable-pip-version-check
  --upgrade
  --require-hashes
  --requirement "$requirements_lock"
)
if [[ "$force_reinstall" == true ]]; then
  pip_arguments+=(--force-reinstall)
fi
PYTHONNOUSERSITE=1 "$venv_dir/bin/python" -m pip install "${pip_arguments[@]}"

PYTHONNOUSERSITE=1 "$venv_dir/bin/python" - "$requirements" "$requirements_lock" <<'PY'
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import sys

import torch
from diffusers import ZImagePipeline

expected = dict(
    line.split("==", 1)
    for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if (line := raw.strip()) and not line.startswith("#")
)
for package, wanted in expected.items():
    actual = version(package)
    if actual != wanted:
        raise SystemExit(f"dependency mismatch: {package} {actual}, expected {wanted}")
if not torch.cuda.is_available():
    raise SystemExit("project-local image runtime cannot access CUDA")
if ZImagePipeline.__name__ != "ZImagePipeline":
    raise SystemExit("installed Diffusers does not expose ZImagePipeline")

marker_packages = expected
requirements_bytes = Path(sys.argv[1]).read_bytes()
requirements_lock_bytes = Path(sys.argv[2]).read_bytes()
marker = {
    "python": platform.python_version(),
    "requirements_sha256": hashlib.sha256(requirements_bytes).hexdigest(),
    "lock_sha256": hashlib.sha256(requirements_lock_bytes).hexdigest(),
    "packages": marker_packages,
}
marker_path = Path(sys.prefix) / ".localllm-runtime.json"
temporary_path = marker_path.with_name(f".{marker_path.name}.{os.getpid()}.tmp")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(temporary_path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(marker, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, marker_path)
    marker_path.chmod(0o600)
finally:
    temporary_path.unlink(missing_ok=True)
print(f"Image runtime ready: Diffusers {version('diffusers')}, PyTorch {torch.__version__}")
print(f"Visible CUDA devices: {torch.cuda.device_count()}")
print(f"Runtime marker: {marker_path}")
PY

echo "Runtime: $venv_dir"
echo "Image generation remains disabled until LOCALLLM_IMAGE_GENERATION_ENABLED=true is set."
