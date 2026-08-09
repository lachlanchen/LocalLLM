#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_python="$project_root/.local/image-generation/venv/bin/python"
model_dir="$project_root/.local/models/image-generation/z-image-turbo-f332072a"
model_revision="f332072aa78be7aecdf3ee76d5c247082da564a6"
requirements="$project_root/tools/image-generation/requirements.txt"
requirements_lock="$project_root/tools/image-generation/requirements.lock.txt"
selected_gpu="${LOCALLLM_IMAGE_GENERATION_GPU:-0}"

die() {
  echo "verify-image-generation: $*" >&2
  exit 1
}

[[ "$selected_gpu" =~ ^([0-9]|1[0-5])$ ]] || die "GPU index must be between 0 and 15"
[[ -x /usr/bin/bwrap ]] || die "bubblewrap is required at /usr/bin/bwrap"
[[ -x "$runtime_python" ]] || die "project-local image runtime is not installed"
[[ -r "$project_root/scripts/image-generation-worker.py" ]] || die "worker script is missing"
[[ -r "$requirements" ]] || die "pinned runtime requirements are missing"
[[ -r "$requirements_lock" ]] || die "hash-locked runtime requirements are missing"

"$runtime_python" - "$model_dir" "$model_revision" "$requirements" "$requirements_lock" <<'PY'
from __future__ import annotations

import hashlib
import json
import platform
import stat
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from diffusers import ZImagePipeline

model_dir = Path(sys.argv[1])
revision = sys.argv[2]
requirements_path = Path(sys.argv[3])
requirements_lock_path = Path(sys.argv[4])
expected_versions = dict(
    line.split("==", 1)
    for raw in requirements_path.read_text(encoding="utf-8").splitlines()
    if (line := raw.strip()) and not line.startswith("#")
)
for package, wanted in expected_versions.items():
    actual = version(package)
    if actual != wanted:
        raise SystemExit(f"dependency mismatch: {package} {actual}, expected {wanted}")
if ZImagePipeline.__name__ != "ZImagePipeline":
    raise SystemExit("Diffusers does not expose ZImagePipeline")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to the image runtime")

marker_packages = expected_versions
runtime_marker_path = Path(sys.prefix) / ".localllm-runtime.json"
runtime_marker_stat = runtime_marker_path.lstat()
if (
    not stat.S_ISREG(runtime_marker_stat.st_mode)
    or runtime_marker_stat.st_nlink != 1
    or runtime_marker_stat.st_size > 4096
):
    raise SystemExit("image runtime marker is unsafe")
runtime_marker = json.loads(runtime_marker_path.read_text(encoding="utf-8"))
expected_runtime_marker = {
    "python": platform.python_version(),
    "requirements_sha256": hashlib.sha256(requirements_path.read_bytes()).hexdigest(),
    "lock_sha256": hashlib.sha256(requirements_lock_path.read_bytes()).hexdigest(),
    "packages": marker_packages,
}
if runtime_marker != expected_runtime_marker:
    raise SystemExit("image runtime marker does not match the current requirements/runtime")

marker_path = model_dir / ".localllm-model.json"
if marker_path.is_symlink() or not marker_path.is_file():
    raise SystemExit("pinned model marker is missing")
marker = json.loads(marker_path.read_text(encoding="utf-8"))
expected_marker = {
    "model_id": "Tongyi-MAI/Z-Image-Turbo",
    "revision": revision,
    "license": "Apache-2.0",
    "weights_sha256": {
        "text_encoder/model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
        "text_encoder/model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
        "text_encoder/model-00003-of-00003.safetensors": "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804",
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": "a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015",
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": "aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa",
        "vae/diffusion_pytorch_model.safetensors": "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",
    },
}
if marker != expected_marker:
    raise SystemExit("pinned model marker mismatch")

weights = list(model_dir.glob("**/*.safetensors"))
payload_bytes = 0
for path in weights:
    file_stat = path.lstat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise SystemExit(f"unsafe model weight: {path}")
    payload_bytes += file_stat.st_size
if not 32_000_000_000 <= payload_bytes <= 34_000_000_000:
    raise SystemExit(f"unexpected model payload size: {payload_bytes}")
if {str(path.relative_to(model_dir)) for path in weights} != set(marker["weights_sha256"]):
    raise SystemExit("model weight file set does not match the pinned manifest")
for relative_path, expected_hash in marker["weights_sha256"].items():
    digest = hashlib.sha256()
    with (model_dir / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise SystemExit(f"weight checksum mismatch: {relative_path}")
print(f"Runtime verified: Diffusers {version('diffusers')}, PyTorch {torch.__version__}")
print(f"Pinned model verified: {marker['model_id']} @ {marker['revision']}")
print(f"Safetensors payload: {payload_bytes} bytes")
PY

if command -v nvidia-smi >/dev/null; then
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  ((selected_gpu < gpu_count)) || die "selected GPU $selected_gpu does not exist"
  free_mib="$(nvidia-smi --id="$selected_gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  [[ "$free_mib" =~ ^[0-9]+$ ]] || die "could not read free memory for GPU $selected_gpu"
  echo "Selected physical GPU $selected_gpu: ${free_mib} MiB currently free"
fi

echo "Static verification only; no weights were loaded and no image was generated."
