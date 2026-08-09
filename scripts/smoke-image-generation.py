#!/usr/bin/env python3
"""Run one fixed, local-only image-generation smoke and release its GPU."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps/api"))

from localllm.config import Settings
from localllm.image_generation import (
    ImageGenerationManager,
    ImageGenerationRequest,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one 512px Z-Image-Turbo smoke image and then unload the worker."
    )
    parser.add_argument("--gpu", type=int, default=0, choices=range(16))
    return parser.parse_args()


def read_gpu_memory(gpu: int) -> int | None:
    nvidia_smi = Path("/usr/bin/nvidia-smi")
    if not nvidia_smi.is_file():
        return None
    try:
        result = subprocess.run(
            (
                str(nvidia_smi),
                f"--id={gpu}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    if result.returncode == 0 and value.isdigit():
        return int(value)
    return None


async def monitor_gpu_memory(gpu: int, stop: asyncio.Event, samples: list[int]) -> None:
    while not stop.is_set():
        try:
            value = await asyncio.to_thread(read_gpu_memory, gpu)
            if value is not None:
                samples.append(value)
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            continue


async def run(gpu: int) -> int:
    settings = Settings(
        data_dir=PROJECT_ROOT / ".local/image-generation/smoke-data",
        image_generation_enabled=True,
        image_generation_gpu=gpu,
        image_generation_timeout_seconds=900,
        _env_file=None,
    )
    manager = ImageGenerationManager(settings)
    stop_monitor = asyncio.Event()
    gpu_memory_samples: list[int] = []
    monitor = asyncio.create_task(
        monitor_gpu_memory(gpu, stop_monitor, gpu_memory_samples),
        name="image-smoke-gpu-monitor",
    )
    report: dict[str, object] | None = None
    exit_code = 1
    try:
        wall_start = time.monotonic()
        submitted = await manager.submit(
            ImageGenerationRequest(
                prompt=(
                    "A friendly small robot painting a bright Hong Kong harbor postcard, "
                    "clean modern illustration, vivid daylight, fine detail"
                ),
                width=512,
                height=512,
                steps=9,
                seed=42,
                output_format="png",
            )
        )
        while True:
            job = await manager.get(submitted.id)
            if job.status not in {"queued", "running"}:
                break
            await asyncio.sleep(0.25)
        if job.status != "succeeded":
            print(json.dumps(job.model_dump(), indent=2), file=sys.stderr)
        else:
            path, _content_type, size = await manager.image_path(job.id)
            wall_time_ms = int((time.monotonic() - wall_start) * 1000)
            report = {
                "id": job.id,
                "status": job.status,
                "physical_gpu": gpu,
                "output": str(path),
                "output_bytes": size,
                "duration_ms": job.duration_ms,
                "cold_load_plus_generation_ms": wall_time_ms,
                "torch_peak_gpu_memory_bytes": job.peak_gpu_memory_bytes,
            }
            exit_code = 0
    except Exception as exc:  # noqa: BLE001 - operator-facing smoke failure boundary
        print(f"image-generation smoke failed: {str(exc)[:500]}", file=sys.stderr)
    finally:
        stop_monitor.set()
        await monitor
        await manager.shutdown()
    if report is not None:
        report["nvidia_smi_peak_memory_mib"] = max(gpu_memory_samples, default=None)
        print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(arguments().gpu)))
