#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_python="$project_root/.local/image-generation/venv/bin/python"
model_dir="$project_root/.local/models/image-generation/z-image-turbo-f332072a"
model_id="Tongyi-MAI/Z-Image-Turbo"
model_revision="f332072aa78be7aecdf3ee76d5c247082da564a6"
expected_payload_bytes=32800000000
required_free_after_bytes=$((100 * 1024 * 1024 * 1024))

die() {
  echo "download-image-generation-model: $*" >&2
  exit 1
}

[[ -x "$runtime_python" ]] ||
  die "run scripts/setup-image-generation.sh before downloading the model"
if [[ -L "$model_dir" ]]; then
  die "model directory must not be a symlink: $model_dir"
fi

available_bytes="$(df --output=avail -B1 "$project_root" | awk 'NR == 2 {print $1}')"
[[ "$available_bytes" =~ ^[0-9]+$ ]] || die "could not determine available storage"
required_before=$((expected_payload_bytes + required_free_after_bytes))
if ((available_bytes < required_before)); then
  die "need at least $required_before free bytes before download; found $available_bytes"
fi

mkdir -p "$model_dir"
chmod 0700 "$model_dir"

HF_HOME="$project_root/.local/image-generation/cache/huggingface" \
HF_XET_HIGH_PERFORMANCE=1 \
"$runtime_python" - "$model_dir" "$model_id" "$model_revision" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

destination = Path(sys.argv[1]).resolve()
model_id = sys.argv[2]
revision = sys.argv[3]

snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=destination,
    allow_patterns=[
        "README.md",
        "model_index.json",
        "scheduler/*.json",
        "text_encoder/*.json",
        "text_encoder/*.safetensors",
        "tokenizer/*.json",
        "tokenizer/*.model",
        "tokenizer/*.txt",
        "transformer/*.json",
        "transformer/*.safetensors",
        "vae/*.json",
        "vae/*.safetensors",
    ],
)

required = [
    destination / "model_index.json",
    destination / "scheduler/scheduler_config.json",
    destination / "text_encoder/config.json",
    destination / "tokenizer/tokenizer_config.json",
    destination / "transformer/config.json",
    destination / "vae/config.json",
]
if not all(path.is_file() and not path.is_symlink() for path in required):
    raise SystemExit("download did not produce the complete regular-file Diffusers layout")

safetensors = list(destination.glob("**/*.safetensors"))
if not safetensors:
    raise SystemExit("download produced no safetensors weights")
payload_bytes = 0
for path in safetensors:
    file_stat = path.lstat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise SystemExit(f"unsafe weight entry: {path}")
    payload_bytes += file_stat.st_size
if not 32_000_000_000 <= payload_bytes <= 34_000_000_000:
    raise SystemExit(f"unexpected safetensors payload size: {payload_bytes}")

expected_hashes = {
    "text_encoder/model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "text_encoder/model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "text_encoder/model-00003-of-00003.safetensors": "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": "a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": "aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa",
    "vae/diffusion_pytorch_model.safetensors": "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",
}
if {str(path.relative_to(destination)) for path in safetensors} != set(expected_hashes):
    raise SystemExit("downloaded weight file set does not match the pinned manifest")
for relative_path, expected_hash in expected_hashes.items():
    digest = hashlib.sha256()
    with (destination / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise SystemExit(f"weight checksum mismatch: {relative_path}")

marker = destination / ".localllm-model.json"
temporary = destination / ".localllm-model.json.part"
if temporary.is_symlink():
    raise SystemExit("unsafe model marker temporary path")
if temporary.exists():
    if not stat.S_ISREG(temporary.lstat().st_mode):
        raise SystemExit("unsafe model marker temporary entry")
    temporary.unlink()
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(temporary, flags, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "model_id": model_id,
            "revision": revision,
            "license": "Apache-2.0",
            "weights_sha256": expected_hashes,
        },
        handle,
        sort_keys=True,
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, marker)
print(f"Pinned model payload: {payload_bytes} bytes across {len(safetensors)} files")
PY

"$project_root/scripts/verify-image-generation.sh"
echo "Image generation remains disabled until LOCALLLM_IMAGE_GENERATION_ENABLED=true is set."
