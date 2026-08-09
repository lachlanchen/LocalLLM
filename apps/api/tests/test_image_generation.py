from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import platform
import struct
import zlib
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from localllm import image_generation as image_module
from localllm.config import Settings, get_settings
from localllm.image_generation import (
    ImageGenerationManager,
    ImageGenerationRequest,
    WorkerProtocolError,
    router,
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _write_png(path: Path, width: int, height: int) -> None:
    row = b"\x00" + b"\x22\x88\xcc" * width
    image_data = zlib.compress(row * height)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", image_data)
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _test_app(settings: Settings, manager: ImageGenerationManager) -> FastAPI:
    app = FastAPI()
    app.state.image_generation = manager
    app.include_router(router)
    app.dependency_overrides[get_settings] = lambda: settings
    return app


async def _ample_gpu_memory(_index: int) -> int:
    return image_module.MIN_IMAGE_GPU_FREE_BYTES + 1024**3


def test_request_schema_is_strict_and_bounded() -> None:
    valid = ImageGenerationRequest(prompt="A bright robot workshop")

    assert valid.steps == 9
    assert valid.width == 1024
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ImageGenerationRequest.model_validate(
            {"prompt": "Fetch an input image", "image_url": "https://example.test/image.png"}
        )
    with pytest.raises(ValidationError, match="multiples of 64"):
        ImageGenerationRequest(prompt="test", width=513)
    with pytest.raises(ValidationError, match="image area"):
        ImageGenerationRequest(prompt="test", width=1536, height=1536)
    with pytest.raises(ValidationError, match="control characters"):
        ImageGenerationRequest(prompt="unsafe\x00prompt")


def test_runtime_readiness_requires_exact_hashed_lock_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root = tmp_path / "runtime"
    python_home = tmp_path / "python-home"
    site_packages = runtime_root / "lib/python3.10/site-packages"
    python = runtime_root / "bin/python"
    worker = tmp_path / "worker.py"
    bwrap = tmp_path / "bwrap"
    requirements = tmp_path / "requirements.txt"
    lock = tmp_path / "requirements.lock.txt"
    marker = runtime_root / ".localllm-runtime.json"
    site_packages.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    (python_home / "bin").mkdir(parents=True)
    (python_home / "lib/python3.10").mkdir(parents=True)
    for executable in (python, python_home / "bin/python3.10", worker, bwrap):
        executable.write_text("test", encoding="utf-8")
        executable.chmod(0o700)
    requirements.write_text("alpha==1.2.3\nbeta==4.5.6\n", encoding="utf-8")
    lock.write_text("alpha==1.2.3 --hash=sha256:abc\n", encoding="utf-8")
    marker.write_text(
        json.dumps(
            {
                "python": image_module.RUNTIME_PYTHON_VERSION,
                "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
                "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "packages": {"alpha": "1.2.3", "beta": "4.5.6"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(image_module, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(image_module, "RUNTIME_PYTHON", python)
    monkeypatch.setattr(image_module, "RUNTIME_PYTHON_HOME", python_home)
    monkeypatch.setattr(image_module, "RUNTIME_SITE_PACKAGES", site_packages)
    monkeypatch.setattr(image_module, "RUNTIME_MARKER", marker)
    monkeypatch.setattr(image_module, "WORKER_SCRIPT", worker)
    monkeypatch.setattr(image_module, "BWRAP_BINARY", bwrap)
    monkeypatch.setattr(image_module, "REQUIREMENTS_FILE", requirements)
    monkeypatch.setattr(image_module, "REQUIREMENTS_LOCK_FILE", lock)

    assert image_module._runtime_is_ready() is True
    lock.write_text("alpha==9 --hash=sha256:def\n", encoding="utf-8")
    assert image_module._runtime_is_ready() is False


def test_worker_command_uses_private_namespaces_and_only_selected_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        image_generation_enabled=True,
        image_generation_gpu=1,
        _env_file=None,
    )
    manager = ImageGenerationManager(settings)
    manager._ensure_storage()
    monkeypatch.setattr(image_module, "_gpu_device_is_ready", lambda _index: True)

    command = manager._worker_command()
    pairs = list(zip(command, command[1:], strict=False))

    assert "--unshare-pid" in command
    assert "--unshare-net" in command
    assert "--unshare-ipc" in command
    assert "--unshare-uts" in command
    assert ("--tmpfs", "/") in pairs
    assert ("--tmpfs", "/run") in pairs
    assert ("--tmpfs", "/runtime/python/lib/python3.10/site-packages") in pairs
    assert ("--ro-bind", "/") not in pairs
    assert "/dev/nvidia1" in command
    assert "/dev/nvidia0" not in command
    assert str(image_module.PROJECT_ROOT / ".env") not in command
    cuda_index = command.index("CUDA_VISIBLE_DEVICES")
    assert command[cuda_index + 1] == "0"
    separator = command.index("--")
    assert command[separator + 1 :] == [
        "/runtime/python/bin/python3.10",
        "/app/image-generation-worker.py",
        "--serve",
        "--model-dir",
        "/model",
        "--output-dir",
        "/output",
        "--runtime-marker",
        "/runtime/attestation.json",
    ]


def test_systemd_api_unit_launches_bwrap_from_a_separate_transient_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        image_generation_enabled=True,
        image_generation_timeout_seconds=240,
        _env_file=None,
    )
    manager = ImageGenerationManager(settings)
    sandbox = [str(image_module.BWRAP_BINARY), "--unshare-pid", "--", "/worker"]
    monkeypatch.setenv("SYSTEMD_EXEC_PID", "123")
    monkeypatch.setattr(image_module, "_transient_worker_launcher_is_ready", lambda: True)

    command, environment, unit = manager._worker_launch_command(sandbox)

    assert command[:6] == [
        str(image_module.SYSTEMD_RUN_BINARY),
        "--user",
        "--pipe",
        "--wait",
        "--collect",
        "--quiet",
    ]
    assert "--property=RuntimeMaxSec=270" in command
    assert command[-len(sandbox) :] == sandbox
    assert unit is not None and unit.startswith("localllm-image-worker-")
    assert environment["DBUS_SESSION_BUS_ADDRESS"] == (f"unix:path={image_module.SYSTEMD_USER_BUS}")


def test_worker_attests_live_package_versions(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "localllm_test_image_worker", image_module.WORKER_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    marker = tmp_path / "runtime.json"
    payload = {
        "python": platform.python_version(),
        "requirements_sha256": "0" * 64,
        "lock_sha256": "1" * 64,
        "packages": {"pydantic": version("pydantic")},
    }
    marker.write_text(json.dumps(payload), encoding="utf-8")

    worker.verify_runtime(marker)
    payload["packages"]["pydantic"] = "0.0.0"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="dependency versions"):
        worker.verify_runtime(marker)


def test_router_status_is_available_while_lane_is_disabled(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=False, _env_file=None)
    manager = ImageGenerationManager(settings)

    with TestClient(_test_app(settings, manager)) as client:
        response = client.get("/api/images/status")

    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert response.json()["limits"]["concurrency"] == 1
    assert response.json()["model"]["revision"] == image_module.MODEL_REVISION


@pytest.mark.asyncio
async def test_read_operations_do_not_create_storage(tmp_path: Path) -> None:
    data_dir = tmp_path / "missing-data"
    manager = ImageGenerationManager(
        Settings(data_dir=data_dir, image_generation_enabled=False, _env_file=None)
    )

    await manager.status()
    assert await manager.list_jobs() == []

    assert not data_dir.exists()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_gpu_capacity_fails_closed_until_the_selected_card_has_room(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=True, _env_file=None)
    manager = ImageGenerationManager(settings)
    monkeypatch.setattr(image_module, "_runtime_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_model_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_gpu_device_is_ready", lambda _index: True)

    async def low_memory(_index: int) -> int:
        return image_module.MIN_IMAGE_GPU_FREE_BYTES - 1

    monkeypatch.setattr(image_module, "_gpu_free_memory_bytes", low_memory)
    status = await manager.status()

    assert status["gpu_ready"] is True
    assert status["gpu_capacity_ready"] is False
    assert status["gpu_free_memory_bytes"] == image_module.MIN_IMAGE_GPU_FREE_BYTES - 1
    assert status["minimum_gpu_free_memory_bytes"] == image_module.MIN_IMAGE_GPU_FREE_BYTES
    assert status["available"] is False
    with pytest.raises(image_module.ImageGenerationUnavailable, match="at least 22 GiB free"):
        await manager.submit(ImageGenerationRequest(prompt="must not start", width=512, height=512))

    # A worker that already owns the model may continue without paying the
    # cold-load allocation again; status remains transparent about current free VRAM.
    manager._worker = SimpleNamespace(returncode=None)  # type: ignore[assignment]
    warm_status = await manager.status()
    assert warm_status["gpu_capacity_ready"] is True
    assert warm_status["available"] is True
    manager._worker = None
    await manager.shutdown()


@pytest.mark.asyncio
async def test_queued_job_rechecks_gpu_capacity_before_worker_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=True, _env_file=None)
    manager = ImageGenerationManager(settings)
    monkeypatch.setattr(image_module, "_runtime_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_model_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_gpu_device_is_ready", lambda _index: True)
    probe_count = 0
    worker_called = False

    async def changing_memory(_index: int) -> int:
        nonlocal probe_count
        probe_count += 1
        if probe_count == 1:
            return image_module.MIN_IMAGE_GPU_FREE_BYTES + 1024**3
        return image_module.MIN_IMAGE_GPU_FREE_BYTES - 1

    async def forbidden_worker(_job, _path: Path):
        nonlocal worker_called
        worker_called = True
        return {"ok": True}

    monkeypatch.setattr(image_module, "_gpu_free_memory_bytes", changing_memory)
    manager._invoke_worker = forbidden_worker  # type: ignore[method-assign]
    submitted = await manager.submit(
        ImageGenerationRequest(prompt="capacity changes while queued", width=512, height=512)
    )
    task = manager._jobs[submitted.id].task
    assert task is not None
    await task

    completed = await manager.get(submitted.id)
    assert completed.status == "failed"
    assert completed.error == "the selected image GPU no longer has 22 GiB free"
    assert worker_called is False
    await manager.shutdown()


def test_create_requires_local_api_key_and_rejects_unbounded_or_extra_input(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=False, _env_file=None)
    manager = ImageGenerationManager(settings)

    with TestClient(_test_app(settings, manager)) as client:
        unauthenticated = client.post("/api/images/jobs", json={"prompt": "test"})
        oversized = client.post(
            "/api/images/jobs",
            content=b"{" + b" " * image_module.MAX_REQUEST_BYTES + b"}",
            headers={"X-LocalLLM-Key": settings.api_key},
        )
        remote_input = client.post(
            "/api/images/jobs",
            json={"prompt": "test", "image_url": "https://example.test/input.png"},
            headers={"X-LocalLLM-Key": settings.api_key},
        )

    assert unauthenticated.status_code == 401
    assert oversized.status_code == 413
    assert remote_input.status_code == 422


def test_router_lists_and_deletes_recovered_output_after_restart(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=False, _env_file=None)
    manager = ImageGenerationManager(settings)
    manager._ensure_storage()
    job_id = "img_11111111111111111111111111111111"
    _write_png(manager.output_dir / f"{job_id}.png", 512, 512)

    with TestClient(_test_app(settings, manager)) as client:
        unauthenticated = client.get("/api/images/jobs")
        listed = client.get("/api/images/jobs", headers={"X-LocalLLM-Key": settings.api_key})
        unauthenticated_image = client.get(f"/api/images/jobs/{job_id}/image")
        image = client.get(
            f"/api/images/jobs/{job_id}/image",
            headers={"X-LocalLLM-Key": settings.api_key},
        )
        removed = client.delete(
            f"/api/images/jobs/{job_id}", headers={"X-LocalLLM-Key": settings.api_key}
        )
        missing = client.get(
            f"/api/images/jobs/{job_id}", headers={"X-LocalLLM-Key": settings.api_key}
        )

    assert unauthenticated.status_code == 401
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == job_id
    assert listed.json()[0]["settings_known"] is False
    assert unauthenticated_image.status_code == 401
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert removed.status_code == 204
    assert missing.status_code == 404
    assert list(manager.output_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_manager_completes_validated_job_and_keeps_prompt_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=True, _env_file=None)
    manager = ImageGenerationManager(settings)
    monkeypatch.setattr(image_module, "_runtime_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_model_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_gpu_device_is_ready", lambda _index: True)
    monkeypatch.setattr(image_module, "_gpu_free_memory_bytes", _ample_gpu_memory)

    async def fake_worker(job, path: Path):
        _write_png(path, job.request.width, job.request.height)
        return {"ok": True, "duration_ms": 123, "peak_gpu_memory_bytes": 456}

    manager._invoke_worker = fake_worker  # type: ignore[method-assign]
    submitted = await manager.submit(
        ImageGenerationRequest(prompt="private prompt", width=512, height=512)
    )
    for _ in range(100):
        completed = await manager.get(submitted.id)
        if completed.status != "queued" and completed.status != "running":
            break
        await asyncio.sleep(0.01)

    assert completed.status == "succeeded"
    assert completed.duration_ms == 123
    assert completed.peak_gpu_memory_bytes == 456
    assert "private prompt" not in completed.model_dump_json()
    path, media_type, _size = await manager.image_path(submitted.id)
    assert path.name == f"{submitted.id}.png"
    assert path.stat().st_mode & 0o777 == 0o600
    assert media_type == "image/png"
    metadata = manager.output_dir / f"{submitted.id}.json"
    assert metadata.stat().st_mode & 0o777 == 0o600
    assert b"private prompt" not in metadata.read_bytes()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_completed_job_survives_restart_and_remains_listable_and_deletable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=True, _env_file=None)
    first_manager = ImageGenerationManager(settings)
    monkeypatch.setattr(image_module, "_runtime_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_model_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_gpu_device_is_ready", lambda _index: True)
    monkeypatch.setattr(image_module, "_gpu_free_memory_bytes", _ample_gpu_memory)

    async def fake_worker(job, path: Path):
        _write_png(path, job.request.width, job.request.height)
        return {"ok": True, "duration_ms": 321}

    first_manager._invoke_worker = fake_worker  # type: ignore[method-assign]
    submitted = await first_manager.submit(
        ImageGenerationRequest(prompt="never persist this", width=512, height=512, seed=77)
    )
    assert first_manager._jobs[submitted.id].task is not None
    await first_manager._jobs[submitted.id].task
    await first_manager.shutdown()

    restarted_manager = ImageGenerationManager(
        Settings(data_dir=tmp_path, image_generation_enabled=False, _env_file=None)
    )
    listed = await restarted_manager.list_jobs()

    assert [job.id for job in listed] == [submitted.id]
    assert listed[0].status == "succeeded"
    assert listed[0].seed == 77
    assert listed[0].duration_ms == 321
    assert listed[0].settings_known is True
    path, media_type, _size = await restarted_manager.image_path(submitted.id)
    assert path.exists()
    assert media_type == "image/png"

    await restarted_manager.delete(submitted.id)
    assert await restarted_manager.list_jobs() == []
    assert list(restarted_manager.output_dir.iterdir()) == []
    await restarted_manager.shutdown()


@pytest.mark.asyncio
async def test_read_reconciliation_is_nonmutating_and_startup_cleanup_is_bounded(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=False, _env_file=None)
    manager = ImageGenerationManager(settings)
    manager._ensure_storage()
    legacy_id = "img_0123456789abcdef0123456789abcdef"
    corrupt_id = "img_abcdef0123456789abcdef0123456789"
    legacy_path = manager.output_dir / f"{legacy_id}.png"
    corrupt_path = manager.output_dir / f"{corrupt_id}.png"
    _write_png(legacy_path, 512, 576)
    corrupt_path.write_bytes(b"not an image")

    listed = await manager.list_jobs()

    assert [job.id for job in listed] == [legacy_id]
    assert listed[0].settings_known is False
    assert listed[0].width == 512
    assert listed[0].height == 576
    assert not manager._metadata_path(legacy_id).exists()
    assert corrupt_path.exists()

    await manager.initialize()

    assert manager._metadata_path(legacy_id).exists()
    assert not corrupt_path.exists()
    assert not manager._metadata_path(corrupt_id).exists()
    status = await manager.status()
    assert status["usage"]["output_files"] == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_generation_concurrency_is_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=True, _env_file=None)
    manager = ImageGenerationManager(settings)
    monkeypatch.setattr(image_module, "_runtime_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_model_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_gpu_device_is_ready", lambda _index: True)
    monkeypatch.setattr(image_module, "_gpu_free_memory_bytes", _ample_gpu_memory)
    active = 0
    maximum = 0

    async def fake_worker(job, path: Path):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.04)
        _write_png(path, job.request.width, job.request.height)
        active -= 1
        return {"ok": True}

    manager._invoke_worker = fake_worker  # type: ignore[method-assign]
    first = await manager.submit(ImageGenerationRequest(prompt="first", width=512, height=512))
    second = await manager.submit(ImageGenerationRequest(prompt="second", width=512, height=512))
    await asyncio.gather(manager._jobs[first.id].task, manager._jobs[second.id].task)

    assert maximum == 1
    assert (await manager.get(first.id)).status == "succeeded"
    assert (await manager.get(second.id)).status == "succeeded"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_delete_cancels_running_job_and_removes_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=True, _env_file=None)
    manager = ImageGenerationManager(settings)
    monkeypatch.setattr(image_module, "_runtime_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_model_is_ready", lambda: True)
    monkeypatch.setattr(image_module, "_gpu_device_is_ready", lambda _index: True)
    monkeypatch.setattr(image_module, "_gpu_free_memory_bytes", _ample_gpu_memory)
    started = asyncio.Event()

    async def blocked_worker(job, path: Path):
        path.write_bytes(b"partial")
        started.set()
        await asyncio.Event().wait()
        return {"ok": True}

    manager._invoke_worker = blocked_worker  # type: ignore[method-assign]
    submitted = await manager.submit(ImageGenerationRequest(prompt="cancel", width=512, height=512))
    await asyncio.wait_for(started.wait(), timeout=1)
    partial = manager._output_path(manager._jobs[submitted.id], temporary=True)
    await manager.delete(submitted.id)

    assert not partial.exists()
    with pytest.raises(KeyError):
        await manager.get(submitted.id)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_idle_worker_unloads_and_a_new_job_cancels_pending_idle_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=True, _env_file=None)
    manager = ImageGenerationManager(settings)
    stopped = asyncio.Event()
    stop_calls = 0

    async def fake_stop() -> None:
        nonlocal stop_calls
        stop_calls += 1
        manager._worker = None
        stopped.set()

    manager._worker = SimpleNamespace(returncode=None)  # type: ignore[assignment]
    manager._stop_worker = fake_stop  # type: ignore[method-assign]
    monkeypatch.setattr(image_module, "IDLE_UNLOAD_SECONDS", 0.02)
    manager._schedule_idle_unload()
    manager._cancel_idle_unload()
    await asyncio.sleep(0.04)
    assert stop_calls == 0

    manager._schedule_idle_unload()
    await asyncio.wait_for(stopped.wait(), timeout=1)
    assert stop_calls == 1
    await manager.shutdown()


@pytest.mark.asyncio
async def test_explicit_unload_releases_worker_without_deleting_output(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=True, _env_file=None)
    manager = ImageGenerationManager(settings)
    manager._ensure_storage()
    output = manager.output_dir / "preserved.png"
    _write_png(output, 512, 512)
    worker = await asyncio.create_subprocess_exec(
        "python3", "-c", "import time; time.sleep(30)", start_new_session=True
    )
    manager._worker = worker

    assert await manager.unload() is True
    assert worker.returncode is not None
    assert output.exists()
    await manager.shutdown()


def test_unload_endpoint_is_authenticated_and_rejects_request_bodies(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, image_generation_enabled=False, _env_file=None)
    manager = ImageGenerationManager(settings)

    with TestClient(_test_app(settings, manager)) as client:
        unauthenticated = client.post("/api/images/unload")
        body = client.post(
            "/api/images/unload",
            content=b"not empty",
            headers={"X-LocalLLM-Key": settings.api_key},
        )
        released = client.post("/api/images/unload", headers={"X-LocalLLM-Key": settings.api_key})

    assert unauthenticated.status_code == 401
    assert body.status_code == 413
    assert released.status_code == 200
    assert released.json() == {"released": False, "worker_running": False}


def test_output_validator_rejects_wrong_dimensions_and_symlinks(tmp_path: Path) -> None:
    request = ImageGenerationRequest(prompt="test", width=512, height=512)
    wrong = tmp_path / "wrong.png"
    _write_png(wrong, 512, 576)
    with pytest.raises(WorkerProtocolError, match="dimensions"):
        image_module._validate_image(wrong, request)

    target = tmp_path / "target.png"
    _write_png(target, 512, 512)
    link = tmp_path / "link.png"
    link.symlink_to(target)
    with pytest.raises(WorkerProtocolError, match="regular file"):
        image_module._validate_image(link, request)
