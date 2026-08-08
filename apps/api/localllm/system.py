from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any


async def _command(*args: str, timeout: float = 5.0) -> tuple[int, str]:
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return process.returncode or 0, output.decode(errors="replace").strip()
    except FileNotFoundError:
        return 127, "unavailable"
    except (TimeoutError, asyncio.TimeoutError):
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except (TimeoutError, asyncio.TimeoutError):
                process.kill()
                await process.wait()
        return 124, "timed out"
    except BaseException:
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except (TimeoutError, asyncio.TimeoutError):
                process.kill()
                await process.wait()
        raise


async def gpu_status() -> dict[str, Any]:
    code, output = await _command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )
    if code == 0:
        devices = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 7:
                devices.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_used_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                        "temperature_c": int(parts[5]),
                        "power_w": float(parts[6]),
                    }
                )
        return {"ok": True, "devices": devices}

    kernel_version = "unknown"
    version_file = Path("/proc/driver/nvidia/version")
    if version_file.exists():
        kernel_version = version_file.read_text(errors="replace").splitlines()[0]
    module_code, module_version = await _command("modinfo", "-F", "version", "nvidia")
    return {
        "ok": False,
        "devices": [],
        "error": output,
        "diagnosis": "NVIDIA driver/library mismatch; reboot is normally required"
        if "mismatch" in output.lower()
        else "NVIDIA runtime unavailable",
        "loaded_kernel_module": kernel_version,
        "installed_kernel_module": module_version if module_code == 0 else "unknown",
    }


def storage_status(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path.resolve())
    return {"total": usage.total, "used": usage.used, "free": usage.free}


async def tool_status(path: str | Path, version_args: tuple[str, ...] = ()) -> dict[str, Any]:
    resolved = shutil.which(str(path))
    if not resolved and Path(path).exists():
        resolved = str(Path(path).resolve())
    if not resolved:
        return {"installed": False, "path": str(path)}
    result: dict[str, Any] = {"installed": True, "path": resolved}
    if version_args:
        code, output = await _command(resolved, *version_args)
        result["version"] = output.splitlines()[0] if code == 0 and output else "unknown"
    return result


def find_project_root() -> Path:
    configured = os.environ.get("LOCALLLM_ROOT")
    if configured:
        return Path(configured).resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "package.json").exists() and (candidate / "apps").exists():
            return candidate
    return current
