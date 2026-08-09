#!/usr/bin/env python3
"""Offline, single-GPU Z-Image-Turbo worker for the optional LocalLLM lane."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
MODEL_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
MODEL_LICENSE = "Apache-2.0"
MODEL_WEIGHT_SHA256 = {
    "text_encoder/model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "text_encoder/model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "text_encoder/model-00003-of-00003.safetensors": "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": "a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": "aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa",
    "vae/diffusion_pytorch_model.safetensors": "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",
}
MAX_REQUEST_BYTES = 8 * 1024
MAX_RUNTIME_MARKER_BYTES = 16 * 1024
MAX_IMAGE_PIXELS = 1_572_864
JOB_ID_PATTERN = re.compile(r"img_[0-9a-f]{32}")


def emit(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def fail(message: str) -> None:
    emit({"ok": False, "error": message})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-marker", type=Path, required=True)
    return parser.parse_args()


def verify_model(model_dir: Path) -> Path:
    resolved = model_dir.resolve(strict=True)
    marker_path = resolved / ".localllm-model.json"
    if (
        marker_path.is_symlink()
        or not marker_path.is_file()
        or marker_path.stat().st_size > 4096
    ):
        raise RuntimeError("pinned model marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker != {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "weights_sha256": MODEL_WEIGHT_SHA256,
    }:
        raise RuntimeError("pinned model marker does not match this worker")
    return resolved


def verify_output_directory(output_dir: Path) -> Path:
    if output_dir.is_symlink():
        raise RuntimeError("output directory must not be a symlink")
    resolved = output_dir.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError("output directory is unavailable")
    return resolved


def verify_runtime(runtime_marker: Path) -> None:
    if (
        runtime_marker.is_symlink()
        or not runtime_marker.is_file()
        or not 0 < runtime_marker.stat().st_size <= MAX_RUNTIME_MARKER_BYTES
    ):
        raise RuntimeError("runtime attestation is missing")
    marker = json.loads(runtime_marker.read_text(encoding="utf-8"))
    if not isinstance(marker, dict) or set(marker) != {
        "python",
        "requirements_sha256",
        "lock_sha256",
        "packages",
    }:
        raise RuntimeError("runtime attestation is malformed")
    packages = marker.get("packages")
    if (
        marker.get("python") != platform.python_version()
        or not isinstance(marker.get("requirements_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", marker["requirements_sha256"])
        or not isinstance(marker.get("lock_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", marker["lock_sha256"])
        or not isinstance(packages, dict)
        or not 1 <= len(packages) <= 128
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", name)
            or not isinstance(wanted, str)
            or not 1 <= len(wanted) <= 100
            or version(name) != wanted
            for name, wanted in packages.items()
        )
    ):
        raise RuntimeError("runtime dependency versions do not match")


def validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("request must be an object")
    allowed = {
        "job_id",
        "prompt",
        "width",
        "height",
        "steps",
        "seed",
        "output_format",
        "jpeg_quality",
    }
    if set(payload) != allowed:
        raise ValueError("request fields did not match the worker protocol")
    job_id = payload.get("job_id")
    prompt = payload.get("prompt")
    width = payload.get("width")
    height = payload.get("height")
    steps = payload.get("steps")
    seed = payload.get("seed")
    output_format = payload.get("output_format")
    jpeg_quality = payload.get("jpeg_quality")
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("invalid job id")
    if not isinstance(prompt, str) or not 1 <= len(prompt) <= 2000:
        raise ValueError("invalid prompt")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in prompt):
        raise ValueError("invalid prompt")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not 512 <= width <= 1536
        or width % 64
        or not isinstance(height, int)
        or isinstance(height, bool)
        or not 512 <= height <= 1536
        or height % 64
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise ValueError("invalid dimensions")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 4 <= steps <= 12:
        raise ValueError("invalid step count")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 4_294_967_295
    ):
        raise ValueError("invalid seed")
    if output_format not in {"png", "jpeg"}:
        raise ValueError("invalid output format")
    if (
        not isinstance(jpeg_quality, int)
        or isinstance(jpeg_quality, bool)
        or not 70 <= jpeg_quality <= 95
    ):
        raise ValueError("invalid JPEG quality")
    return payload


def output_path(output_dir: Path, payload: dict[str, Any]) -> Path:
    suffix = "jpg" if payload["output_format"] == "jpeg" else "png"
    candidate = output_dir / f".{payload['job_id']}.{suffix}.part"
    if candidate.parent.resolve() != output_dir:
        raise ValueError("output path escaped its directory")
    return candidate


def save_private_image(
    image: Any, path: Path, output_format: str, jpeg_quality: int
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            rgb = image.convert("RGB")
            if output_format == "png":
                rgb.save(handle, format="PNG", compress_level=6)
            else:
                rgb.save(handle, format="JPEG", quality=jpeg_quality, optimize=True)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def main() -> int:
    args = parse_arguments()
    try:
        model_dir = verify_model(args.model_dir)
        destination = verify_output_directory(args.output_dir)
        verify_runtime(args.runtime_marker)

        import torch
        from diffusers import ZImagePipeline

        if (
            os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        ):
            raise RuntimeError("offline mode is required")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
            raise RuntimeError("an explicit GPU selection is required")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("the worker must see exactly one CUDA GPU")

        pipeline = ZImagePipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=True)
    except Exception:  # noqa: BLE001 - never disclose dependency or model details
        fail("worker initialization failed")
        return 1

    emit({"ok": True, "ready": True, "revision": MODEL_REVISION})
    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            return 0
        if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
            fail("request exceeded the worker protocol limit")
            return 2
        path: Path | None = None
        try:
            payload = validate_request(json.loads(line))
            path = output_path(destination, payload)
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                result = pipeline(
                    prompt=payload["prompt"],
                    height=payload["height"],
                    width=payload["width"],
                    num_inference_steps=payload["steps"],
                    guidance_scale=0.0,
                    generator=torch.Generator("cuda").manual_seed(payload["seed"]),
                )
            if not result.images:
                raise RuntimeError("pipeline produced no image")
            save_private_image(
                result.images[0],
                path,
                payload["output_format"],
                payload["jpeg_quality"],
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            peak_memory = int(torch.cuda.max_memory_allocated())
            del result
            emit(
                {
                    "ok": True,
                    "duration_ms": duration_ms,
                    "peak_gpu_memory_bytes": peak_memory,
                }
            )
        except Exception:  # noqa: BLE001 - keep the persistent protocol fail-closed
            if path is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            fail("generation failed")


if __name__ == "__main__":
    raise SystemExit(main())
