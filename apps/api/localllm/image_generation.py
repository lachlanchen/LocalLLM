from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import signal
import stat
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .config import Settings, get_settings

LOGGER = logging.getLogger(__name__)

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
MODEL_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
MODEL_LICENSE = "Apache-2.0"
MODEL_PARAMETER_COUNT = 6_154_908_736
MODEL_WEIGHT_MIN_BYTES = 32_000_000_000
MODEL_WEIGHT_MAX_BYTES = 34_000_000_000
MODEL_WEIGHT_SHA256 = {
    "text_encoder/model-00001-of-00003.safetensors": (
        "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"
    ),
    "text_encoder/model-00002-of-00003.safetensors": (
        "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5"
    ),
    "text_encoder/model-00003-of-00003.safetensors": (
        "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52"
    ),
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": (
        "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804"
    ),
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": (
        "a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015"
    ),
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": (
        "aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa"
    ),
    "vae/diffusion_pytorch_model.safetensors": (
        "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3"
    ),
}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIRECTORY = PROJECT_ROOT / ".local/models/image-generation/z-image-turbo-f332072a"
RUNTIME_PYTHON = PROJECT_ROOT / ".local/image-generation/venv/bin/python"
RUNTIME_ROOT = PROJECT_ROOT / ".local/image-generation/venv"
RUNTIME_SITE_PACKAGES = RUNTIME_ROOT / "lib/python3.10/site-packages"
RUNTIME_MARKER = RUNTIME_ROOT / ".localllm-runtime.json"
RUNTIME_PYTHON_HOME = RUNTIME_PYTHON.resolve().parents[1]
RUNTIME_PYTHON_VERSION = "3.10.13"
REQUIREMENTS_FILE = PROJECT_ROOT / "tools/image-generation/requirements.txt"
REQUIREMENTS_LOCK_FILE = PROJECT_ROOT / "tools/image-generation/requirements.lock.txt"
WORKER_SCRIPT = PROJECT_ROOT / "scripts/image-generation-worker.py"
BWRAP_BINARY = Path("/usr/bin/bwrap")
SYSTEMD_RUN_BINARY = Path("/usr/bin/systemd-run")
SYSTEMCTL_BINARY = Path("/usr/bin/systemctl")
SYSTEMD_USER_BUS = Path(f"/run/user/{os.getuid()}/bus")
NVIDIA_SMI_BINARY = Path("/usr/bin/nvidia-smi")
MODEL_MARKER = MODEL_DIRECTORY / ".localllm-model.json"
OUTPUT_DIRECTORY_NAME = "image-generation"
OUTPUT_QUOTA_BYTES = 1024 * 1024 * 1024
MAX_OUTPUT_FILES = 128
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 4096
MAX_RUNTIME_MARKER_BYTES = 16 * 1024
MAX_PENDING_JOBS = 4
MAX_RETAINED_JOBS = MAX_OUTPUT_FILES
MAX_REQUEST_BYTES = 8 * 1024
MAX_WORKER_LINE_BYTES = 64 * 1024
MAX_IMAGE_PIXELS = 1_572_864
IDLE_UNLOAD_SECONDS = 120
GPU_MEMORY_PROBE_TIMEOUT_SECONDS = 3
# The measured 512px BF16 worker peak is 21,352,528,384 bytes. Requiring 22 GiB
# free leaves a bounded allocator/activation margin while still admitting an
# otherwise idle 24 GB 4090 with the desktop compositor resident.
MIN_IMAGE_GPU_FREE_BYTES = 22 * 1024**3
JOB_ID_PATTERN = re.compile(r"img_[0-9a-f]{32}")
IMAGE_FILE_PATTERN = re.compile(r"(img_[0-9a-f]{32})\.(png|jpg)")
METADATA_FILE_PATTERN = re.compile(r"(img_[0-9a-f]{32})\.json")
PART_FILE_PATTERN = re.compile(r"\.(img_[0-9a-f]{32})\.(png|jpg|json)\.part")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ImageGenerationRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=2000)
    width: int = Field(default=1024, ge=512, le=1536)
    height: int = Field(default=1024, ge=512, le=1536)
    # The upstream example uses 9 scheduler steps, which produces 8 DiT forwards.
    steps: int = Field(default=9, ge=4, le=12)
    seed: int = Field(default=0, ge=0, le=4_294_967_295)
    output_format: Literal["png", "jpeg"] = "png"
    jpeg_quality: int = Field(default=90, ge=70, le=95)

    @model_validator(mode="after")
    def validate_dimensions(self) -> ImageGenerationRequest:
        if self.width % 64 or self.height % 64:
            raise ValueError("width and height must be multiples of 64")
        if self.width * self.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"image area must not exceed {MAX_IMAGE_PIXELS} pixels")
        if any(ord(character) < 32 and character not in "\n\r\t" for character in self.prompt):
            raise ValueError("prompt contains unsupported control characters")
        return self


class ImageJobResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at: int
    started_at: int | None = None
    completed_at: int | None = None
    width: int
    height: int
    steps: int
    seed: int
    output_format: Literal["png", "jpeg"]
    image_url: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    peak_gpu_memory_bytes: int | None = None
    settings_known: bool = True


class _PersistedImageJob(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^img_[0-9a-f]{32}$")
    created_at: int = Field(ge=0)
    started_at: int | None = Field(default=None, ge=0)
    completed_at: int = Field(ge=0)
    width: int = Field(ge=512, le=1536)
    height: int = Field(ge=512, le=1536)
    steps: int = Field(ge=4, le=12)
    seed: int = Field(ge=0, le=4_294_967_295)
    output_format: Literal["png", "jpeg"]
    duration_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    peak_gpu_memory_bytes: int | None = Field(default=None, ge=0, le=128 * 1024**3)
    settings_known: bool = True

    @model_validator(mode="after")
    def validate_record(self) -> _PersistedImageJob:
        if self.width % 64 or self.height % 64:
            raise ValueError("stored dimensions must be multiples of 64")
        if self.width * self.height > MAX_IMAGE_PIXELS:
            raise ValueError("stored image area exceeds the limit")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("stored job timestamps are inconsistent")
        if self.completed_at < (self.started_at or self.created_at):
            raise ValueError("stored job timestamps are inconsistent")
        return self


class ImageGenerationUnavailable(RuntimeError):
    pass


class ImageGenerationCapacityError(RuntimeError):
    pass


class ImageGenerationTimeout(RuntimeError):
    pass


class WorkerProtocolError(RuntimeError):
    pass


@dataclass(slots=True)
class _Job:
    id: str
    request: ImageGenerationRequest
    created_at: int = field(default_factory=lambda: int(time.time()))
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] = "queued"
    started_at: int | None = None
    completed_at: int | None = None
    error: str | None = None
    duration_ms: int | None = None
    peak_gpu_memory_bytes: int | None = None
    task: asyncio.Task[None] | None = None
    settings_known: bool = True


def _extension(output_format: str) -> str:
    return "jpg" if output_format == "jpeg" else "png"


def _content_type(output_format: str) -> str:
    return "image/jpeg" if output_format == "jpeg" else "image/png"


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"unsupported JSON numeric value: {value}")


async def _bounded_request(request: Request) -> ImageGenerationRequest:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_length = int(declared)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
        if declared_length < 0:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")
        if declared_length > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Image request exceeds the size limit")

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="Image request exceeds the size limit")
        body.extend(chunk)
    try:
        decoded = json.loads(body, parse_constant=_reject_nonfinite_json)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")
    try:
        return ImageGenerationRequest.model_validate(decoded)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status_code=422, detail=errors[:20]) from exc


def _model_is_ready() -> bool:
    try:
        if MODEL_MARKER.is_symlink() or not MODEL_MARKER.is_file():
            return False
        if MODEL_MARKER.stat().st_size > 4096:
            return False
        marker = json.loads(MODEL_MARKER.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if marker != {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "weights_sha256": MODEL_WEIGHT_SHA256,
    }:
        return False
    required = (
        MODEL_DIRECTORY / "model_index.json",
        MODEL_DIRECTORY / "scheduler/scheduler_config.json",
        MODEL_DIRECTORY / "text_encoder/config.json",
        MODEL_DIRECTORY / "tokenizer/tokenizer_config.json",
        MODEL_DIRECTORY / "transformer/config.json",
        MODEL_DIRECTORY / "vae/config.json",
    )
    if not all(path.is_file() and not path.is_symlink() for path in required):
        return False
    try:
        weights = list(MODEL_DIRECTORY.glob("**/*.safetensors"))
        weight_bytes = sum(path.lstat().st_size for path in weights)
    except OSError:
        return False
    return (
        bool(weights)
        and {str(path.relative_to(MODEL_DIRECTORY)) for path in weights} == set(MODEL_WEIGHT_SHA256)
        and all(stat.S_ISREG(path.lstat().st_mode) for path in weights)
        and MODEL_WEIGHT_MIN_BYTES <= weight_bytes <= MODEL_WEIGHT_MAX_BYTES
    )


def _locked_requirements() -> dict[str, str] | None:
    try:
        lines = REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    locked: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version or name in locked:
            return None
        locked[name] = version
    return locked or None


def _runtime_is_ready() -> bool:
    try:
        marker_stat = RUNTIME_MARKER.lstat()
        requirements_stat = REQUIREMENTS_FILE.lstat()
        lock_stat = REQUIREMENTS_LOCK_FILE.lstat()
        if (
            RUNTIME_MARKER.is_symlink()
            or not stat.S_ISREG(marker_stat.st_mode)
            or marker_stat.st_nlink != 1
            or not 0 < marker_stat.st_size <= MAX_RUNTIME_MARKER_BYTES
            or REQUIREMENTS_FILE.is_symlink()
            or not stat.S_ISREG(requirements_stat.st_mode)
            or REQUIREMENTS_LOCK_FILE.is_symlink()
            or not stat.S_ISREG(lock_stat.st_mode)
        ):
            return False
        requirements_bytes = REQUIREMENTS_FILE.read_bytes()
        lock_bytes = REQUIREMENTS_LOCK_FILE.read_bytes()
        marker = json.loads(RUNTIME_MARKER.read_text(encoding="utf-8"))
        locked = _locked_requirements()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if marker != {
        "python": RUNTIME_PYTHON_VERSION,
        "requirements_sha256": hashlib.sha256(requirements_bytes).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "packages": locked,
    }:
        return False
    required_paths = (
        RUNTIME_PYTHON,
        RUNTIME_PYTHON_HOME / "bin/python3.10",
        RUNTIME_PYTHON_HOME / "lib/python3.10",
        RUNTIME_SITE_PACKAGES,
        WORKER_SCRIPT,
        BWRAP_BINARY,
    )
    return (
        locked is not None
        and all(path.exists() and not path.is_symlink() for path in required_paths[1:])
        and RUNTIME_PYTHON.is_file()
        and os.access(RUNTIME_PYTHON, os.X_OK)
        and os.access(BWRAP_BINARY, os.X_OK)
    )


def _gpu_device_is_ready(index: int) -> bool:
    required = (Path("/dev/nvidiactl"), Path("/dev/nvidia-uvm"), Path(f"/dev/nvidia{index}"))
    try:
        return all(
            not path.is_symlink() and stat.S_ISCHR(path.lstat().st_mode) for path in required
        )
    except OSError:
        return False


async def _gpu_free_memory_bytes(index: int) -> int | None:
    """Return physical-card free VRAM from a bounded trusted local probe."""

    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            str(NVIDIA_SMI_BINARY),
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _stderr = await asyncio.wait_for(
            process.communicate(), timeout=GPU_MEMORY_PROBE_TIMEOUT_SECONDS
        )
    except asyncio.CancelledError:
        if process is not None and process.returncode is None:
            process.kill()
            with contextlib.suppress(Exception):
                await asyncio.shield(process.wait())
        raise
    except (OSError, asyncio.TimeoutError):
        if process is not None and process.returncode is None:
            process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
        return None
    if process.returncode != 0 or len(stdout) > 4096:
        return None
    try:
        for raw_line in stdout.decode("ascii", errors="strict").splitlines():
            raw_index, raw_free = (part.strip() for part in raw_line.split(",", 1))
            if int(raw_index) != index:
                continue
            free_mib = int(raw_free)
            if 0 <= free_mib <= 1024 * 1024:
                return free_mib * 1024**2
    except (UnicodeDecodeError, ValueError):
        return None
    return None


def _transient_worker_launcher_is_ready() -> bool:
    try:
        bus_stat = SYSTEMD_USER_BUS.lstat()
    except OSError:
        return False
    return (
        not SYSTEMD_USER_BUS.is_symlink()
        and stat.S_ISSOCK(bus_stat.st_mode)
        and os.access(SYSTEMD_RUN_BINARY, os.X_OK)
        and os.access(SYSTEMCTL_BINARY, os.X_OK)
    )


def _read_png_dimensions(header: bytes) -> tuple[int, int] | None:
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if header[12:16] != b"IHDR" or int.from_bytes(header[8:12], "big") != 13:
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def _read_jpeg_dimensions(header: bytes) -> tuple[int, int] | None:
    if len(header) < 4 or header[:2] != b"\xff\xd8":
        return None
    position = 2
    while position + 4 <= len(header):
        if header[position] != 0xFF:
            return None
        while position < len(header) and header[position] == 0xFF:
            position += 1
        if position >= len(header):
            return None
        marker = header[position]
        position += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(header):
            return None
        segment_length = int.from_bytes(header[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(header):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if segment_length < 7:
                return None
            height = int.from_bytes(header[position + 3 : position + 5], "big")
            width = int.from_bytes(header[position + 5 : position + 7], "big")
            return width, height
        position += segment_length
    return None


def _inspect_image(path: Path, output_format: Literal["png", "jpeg"]) -> tuple[int, int, int]:
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise WorkerProtocolError("worker produced no image") from exc
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise WorkerProtocolError("worker output is not a private regular file")
    if file_stat.st_size <= 0 or file_stat.st_size > MAX_OUTPUT_BYTES:
        raise WorkerProtocolError("worker output exceeded the image size limit")
    with path.open("rb") as handle:
        header = handle.read(min(file_stat.st_size, 1024 * 1024))
    dimensions = (
        _read_png_dimensions(header) if output_format == "png" else _read_jpeg_dimensions(header)
    )
    if dimensions is None:
        raise WorkerProtocolError("worker output type could not be verified")
    width, height = dimensions
    if (
        width < 512
        or width > 1536
        or height < 512
        or height > 1536
        or width % 64
        or height % 64
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise WorkerProtocolError("worker output dimensions exceeded the image limits")
    return file_stat.st_size, width, height


def _validate_image(path: Path, request: ImageGenerationRequest) -> int:
    size, width, height = _inspect_image(path, request.output_format)
    dimensions = (width, height)
    if dimensions != (request.width, request.height):
        raise WorkerProtocolError("worker output type or dimensions did not match the request")
    return size


class ImageGenerationManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.output_dir = settings.data_dir.resolve() / OUTPUT_DIRECTORY_NAME
        self._jobs: dict[str, _Job] = {}
        self._jobs_lock = asyncio.Lock()
        self._generation_slot = asyncio.Semaphore(1)
        self._worker: asyncio.subprocess.Process | None = None
        self._worker_unit: str | None = None
        self._worker_stderr_task: asyncio.Task[None] | None = None
        self._worker_stderr_tail = bytearray()
        self._idle_unload_task: asyncio.Task[None] | None = None
        self._storage_reconciled = False
        self._storage_maintained = False
        self._closed = False

    def _ensure_storage(self) -> None:
        data_dir = self.settings.data_dir.resolve()
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        data_dir.chmod(0o700)
        candidate = data_dir / OUTPUT_DIRECTORY_NAME
        if candidate.is_symlink():
            raise ImageGenerationUnavailable("image output directory must not be a symlink")
        candidate.mkdir(mode=0o700, exist_ok=True)
        if candidate.resolve().parent != data_dir:
            raise ImageGenerationUnavailable("image output directory escaped the data directory")
        candidate.chmod(0o700)

    def _storage_exists_and_is_safe(self) -> bool:
        data_dir = self.settings.data_dir.resolve()
        candidate = data_dir / OUTPUT_DIRECTORY_NAME
        if not candidate.exists():
            return False
        if candidate.is_symlink() or not candidate.is_dir():
            raise ImageGenerationUnavailable("image output directory is unsafe")
        if candidate.resolve().parent != data_dir:
            raise ImageGenerationUnavailable("image output directory escaped the data directory")
        return True

    def _metadata_path(self, job_id: str, *, temporary: bool = False) -> Path:
        filename = f".{job_id}.json.part" if temporary else f"{job_id}.json"
        return self.output_dir / filename

    @staticmethod
    def _safe_regular_file(path: Path, *, maximum_bytes: int | None = None) -> os.stat_result:
        try:
            entry_stat = path.lstat()
        except OSError as exc:
            raise ImageGenerationUnavailable(
                "image output storage changed during validation"
            ) from exc
        if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
            raise ImageGenerationUnavailable("image output directory contains an unsafe entry")
        if maximum_bytes is not None and entry_stat.st_size > maximum_bytes:
            raise ImageGenerationUnavailable("image output metadata exceeded its size limit")
        return entry_stat

    def _persist_job(self, job: _Job) -> None:
        if job.status != "succeeded" or job.completed_at is None:
            raise ImageGenerationUnavailable("only completed image jobs can be persisted")
        record = _PersistedImageJob(
            id=job.id,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            width=job.request.width,
            height=job.request.height,
            steps=job.request.steps,
            seed=job.request.seed,
            output_format=job.request.output_format,
            duration_ms=job.duration_ms,
            peak_gpu_memory_bytes=job.peak_gpu_memory_bytes,
            settings_known=job.settings_known,
        )
        encoded = (record.model_dump_json() + "\n").encode("utf-8")
        if len(encoded) > MAX_METADATA_BYTES:
            raise ImageGenerationUnavailable("image output metadata exceeded its size limit")
        temporary_path = self._metadata_path(job.id, temporary=True)
        final_path = self._metadata_path(job.id)
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary_path, flags, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = None
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, final_path)
            final_path.chmod(0o600)
            directory_descriptor = os.open(
                self.output_dir,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            with contextlib.suppress(FileNotFoundError):
                temporary_path.unlink()
            raise ImageGenerationUnavailable(
                "image output metadata could not be persisted"
            ) from exc

    def _read_persisted_job(self, path: Path) -> _PersistedImageJob | None:
        try:
            entry_stat = path.lstat()
            if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                raise ImageGenerationUnavailable("image output directory contains unsafe metadata")
            if entry_stat.st_size <= 0 or entry_stat.st_size > MAX_METADATA_BYTES:
                return None
            encoded = path.read_bytes()
            decoded = json.loads(encoded, parse_constant=_reject_nonfinite_json)
            return _PersistedImageJob.model_validate(decoded)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, ValidationError):
            return None

    @staticmethod
    def _request_for_stored_job(
        record: _PersistedImageJob,
    ) -> ImageGenerationRequest:
        return ImageGenerationRequest(
            prompt="restored local image",
            width=record.width,
            height=record.height,
            steps=record.steps,
            seed=record.seed,
            output_format=record.output_format,
        )

    def _restore_job(self, record: _PersistedImageJob) -> _Job:
        return _Job(
            id=record.id,
            request=self._request_for_stored_job(record),
            created_at=record.created_at,
            status="succeeded",
            started_at=record.started_at,
            completed_at=record.completed_at,
            duration_ms=record.duration_ms,
            peak_gpu_memory_bytes=record.peak_gpu_memory_bytes,
            settings_known=record.settings_known,
        )

    def _recover_legacy_job(
        self,
        job_id: str,
        image_path: Path,
        output_format: Literal["png", "jpeg"],
        *,
        persist: bool,
    ) -> _Job:
        _size, width, height = _inspect_image(image_path, output_format)
        timestamp = max(0, int(image_path.stat().st_mtime))
        job = _Job(
            id=job_id,
            request=ImageGenerationRequest(
                prompt="restored legacy local image",
                width=width,
                height=height,
                steps=9,
                seed=0,
                output_format=output_format,
            ),
            created_at=timestamp,
            status="succeeded",
            started_at=timestamp,
            completed_at=timestamp,
            settings_known=False,
        )
        if persist:
            self._persist_job(job)
        return job

    def _reconcile_storage_locked(self, *, mutate: bool) -> None:
        images: dict[str, tuple[Path, Literal["png", "jpeg"]]] = {}
        metadata: dict[str, Path] = {}
        parts: list[Path] = []
        for entry in self.output_dir.iterdir():
            self._safe_regular_file(entry)
            if match := IMAGE_FILE_PATTERN.fullmatch(entry.name):
                job_id, suffix = match.groups()
                if job_id in images:
                    raise ImageGenerationUnavailable(
                        "duplicate image outputs require operator review"
                    )
                images[job_id] = (entry, "jpeg" if suffix == "jpg" else "png")
            elif match := METADATA_FILE_PATTERN.fullmatch(entry.name):
                metadata[match.group(1)] = entry
            elif PART_FILE_PATTERN.fullmatch(entry.name):
                parts.append(entry)
            else:
                raise ImageGenerationUnavailable("image output directory contains an unknown entry")

        if mutate:
            for path in parts:
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

            for job_id, metadata_path in metadata.items():
                if job_id not in images:
                    with contextlib.suppress(FileNotFoundError):
                        metadata_path.unlink()

        for job_id, (image_path, output_format) in images.items():
            metadata_path = metadata.get(job_id)
            record = self._read_persisted_job(metadata_path) if metadata_path is not None else None
            if record is not None:
                expected_format = record.output_format
                try:
                    _validate_image(image_path, self._request_for_stored_job(record))
                except WorkerProtocolError:
                    record = None
                else:
                    if record.id != job_id or expected_format != output_format:
                        record = None
            if record is None:
                try:
                    job = self._recover_legacy_job(
                        job_id, image_path, output_format, persist=mutate
                    )
                except WorkerProtocolError:
                    if mutate:
                        for path in (image_path, metadata_path):
                            if path is not None:
                                with contextlib.suppress(FileNotFoundError):
                                    path.unlink()
                    continue
            else:
                job = self._restore_job(record)
            self._jobs.setdefault(job_id, job)

    async def _ensure_reconciled(self, *, mutate: bool = False) -> None:
        if self._storage_reconciled and (not mutate or self._storage_maintained):
            return
        if mutate:
            self._ensure_storage()
        elif not self._storage_exists_and_is_safe():
            async with self._jobs_lock:
                self._storage_reconciled = True
            return
        async with self._jobs_lock:
            if self._storage_reconciled and (not mutate or self._storage_maintained):
                return
            self._reconcile_storage_locked(mutate=mutate)
            self._storage_reconciled = True
            self._storage_maintained = self._storage_maintained or mutate

    async def initialize(self) -> None:
        """Perform bounded app-startup cleanup before any HTTP request is served."""
        await self._ensure_reconciled(mutate=True)

    def _storage_usage(self) -> tuple[int, int]:
        if not self.output_dir.exists():
            return 0, 0
        if self.output_dir.is_symlink() or not self.output_dir.is_dir():
            raise ImageGenerationUnavailable("image output directory is unsafe")
        count = 0
        used = 0
        for entry in self.output_dir.iterdir():
            entry_stat = self._safe_regular_file(entry)
            if IMAGE_FILE_PATTERN.fullmatch(entry.name):
                count += 1
            elif not (
                METADATA_FILE_PATTERN.fullmatch(entry.name)
                or PART_FILE_PATTERN.fullmatch(entry.name)
            ):
                raise ImageGenerationUnavailable("image output directory contains an unknown entry")
            used += entry_stat.st_size
        return count, used

    def _output_path(self, job: _Job, *, temporary: bool = False) -> Path:
        suffix = _extension(job.request.output_format)
        filename = f".{job.id}.{suffix}.part" if temporary else f"{job.id}.{suffix}"
        return self.output_dir / filename

    def _public_job(self, job: _Job) -> ImageJobResponse:
        return ImageJobResponse(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            width=job.request.width,
            height=job.request.height,
            steps=job.request.steps,
            seed=job.request.seed,
            output_format=job.request.output_format,
            image_url=(f"/api/images/jobs/{job.id}/image" if job.status == "succeeded" else None),
            error=job.error,
            duration_ms=job.duration_ms,
            peak_gpu_memory_bytes=job.peak_gpu_memory_bytes,
            settings_known=job.settings_known,
        )

    async def status(self) -> dict[str, Any]:
        try:
            await self._ensure_reconciled()
            count, used = self._storage_usage()
            storage_ready = True
        except ImageGenerationUnavailable:
            count, used, storage_ready = 0, 0, False
        async with self._jobs_lock:
            queued = sum(job.status == "queued" for job in self._jobs.values())
            running = sum(job.status == "running" for job in self._jobs.values())
        runtime_ready = _runtime_is_ready()
        model_ready = _model_is_ready()
        gpu_ready = _gpu_device_is_ready(self.settings.image_generation_gpu)
        worker_running = self._worker is not None and self._worker.returncode is None
        gpu_free_memory_bytes = (
            await _gpu_free_memory_bytes(self.settings.image_generation_gpu) if gpu_ready else None
        )
        gpu_capacity_ready = worker_running or (
            gpu_free_memory_bytes is not None and gpu_free_memory_bytes >= MIN_IMAGE_GPU_FREE_BYTES
        )
        return {
            "enabled": self.settings.image_generation_enabled,
            "available": (
                runtime_ready and model_ready and gpu_ready and gpu_capacity_ready and storage_ready
            ),
            "runtime_ready": runtime_ready,
            "model_ready": model_ready,
            "gpu_ready": gpu_ready,
            "gpu_capacity_ready": gpu_capacity_ready,
            "gpu_free_memory_bytes": gpu_free_memory_bytes,
            "minimum_gpu_free_memory_bytes": MIN_IMAGE_GPU_FREE_BYTES,
            "worker_running": worker_running,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "license": MODEL_LICENSE,
                "parameters": MODEL_PARAMETER_COUNT,
            },
            "gpu": self.settings.image_generation_gpu,
            "limits": {
                "concurrency": 1,
                "pending_jobs": MAX_PENDING_JOBS,
                "timeout_seconds": self.settings.image_generation_timeout_seconds,
                "output_quota_bytes": OUTPUT_QUOTA_BYTES,
                "max_output_files": MAX_OUTPUT_FILES,
                "max_output_bytes": MAX_OUTPUT_BYTES,
                "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
            },
            "usage": {
                "queued": queued,
                "running": running,
                "output_files": count,
                "output_bytes": used,
            },
        }

    async def submit(self, request: ImageGenerationRequest) -> ImageJobResponse:
        if self._closed:
            raise ImageGenerationUnavailable("image generator is shutting down")
        if not self.settings.image_generation_enabled:
            raise ImageGenerationUnavailable("image generation is disabled")
        if not _runtime_is_ready():
            raise ImageGenerationUnavailable("image generation runtime is not installed")
        if not _model_is_ready():
            raise ImageGenerationUnavailable("the pinned image model is not installed")
        if not _gpu_device_is_ready(self.settings.image_generation_gpu):
            raise ImageGenerationUnavailable("the selected image GPU device is unavailable")
        worker_running = self._worker is not None and self._worker.returncode is None
        if not worker_running:
            free_memory = await _gpu_free_memory_bytes(self.settings.image_generation_gpu)
            if free_memory is None or free_memory < MIN_IMAGE_GPU_FREE_BYTES:
                raise ImageGenerationUnavailable(
                    "the selected image GPU needs at least 22 GiB free; "
                    "close or move other GPU workloads, including resident Ollama models, "
                    "or choose an idle physical card"
                )
        await self._ensure_reconciled(mutate=True)
        async with self._jobs_lock:
            active = sum(job.status in {"queued", "running"} for job in self._jobs.values())
            if active >= MAX_PENDING_JOBS:
                raise ImageGenerationCapacityError("the image generation queue is full")
            if len(self._jobs) >= MAX_RETAINED_JOBS:
                raise ImageGenerationCapacityError(
                    "delete completed image jobs before creating more"
                )
            output_count, output_bytes = self._storage_usage()
            reserved = active * MAX_OUTPUT_BYTES
            if output_count + active >= MAX_OUTPUT_FILES:
                raise ImageGenerationCapacityError("the image output file quota is full")
            if output_bytes + reserved + MAX_OUTPUT_BYTES > OUTPUT_QUOTA_BYTES:
                raise ImageGenerationCapacityError("the image output byte quota is full")
            job = _Job(id=f"img_{secrets.token_hex(16)}", request=request)
            self._jobs[job.id] = job
            job.task = asyncio.create_task(self._run_job(job), name=f"image-generation-{job.id}")
            return self._public_job(job)

    async def get(self, job_id: str) -> ImageJobResponse:
        self._require_valid_job_id(job_id)
        await self._ensure_reconciled()
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return self._public_job(job)

    async def list_jobs(self) -> list[ImageJobResponse]:
        await self._ensure_reconciled()
        async with self._jobs_lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: (job.completed_at or job.created_at, job.created_at, job.id),
                reverse=True,
            )
            return [self._public_job(job) for job in jobs[:MAX_OUTPUT_FILES]]

    async def image_path(self, job_id: str) -> tuple[Path, str, int]:
        self._require_valid_job_id(job_id)
        await self._ensure_reconciled()
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "succeeded":
                raise KeyError(job_id)
            path = self._output_path(job)
            size = _validate_image(path, job.request)
            return path, _content_type(job.request.output_format), size

    async def delete(self, job_id: str) -> None:
        self._require_valid_job_id(job_id)
        await self._ensure_reconciled(mutate=True)
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            task = job.task
            if task is not None and not task.done():
                task.cancel()
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        async with self._jobs_lock:
            current = self._jobs.pop(job_id, None)
        if current is not None:
            for path in (
                self._output_path(current),
                self._output_path(current, temporary=True),
                self._metadata_path(current.id),
                self._metadata_path(current.id, temporary=True),
            ):
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

    async def unload(self) -> bool:
        """Release resident model memory without deleting completed image outputs."""
        self._cancel_idle_unload()
        if self._generation_slot.locked():
            raise ImageGenerationCapacityError("an image generation job is still running")
        async with self._generation_slot:
            async with self._jobs_lock:
                if any(job.status in {"queued", "running"} for job in self._jobs.values()):
                    raise ImageGenerationCapacityError("an image generation job is still active")
            was_running = self._worker is not None and self._worker.returncode is None
            await self._stop_worker()
            return was_running

    @staticmethod
    def _require_valid_job_id(job_id: str) -> None:
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise KeyError(job_id)

    async def _run_job(self, job: _Job) -> None:
        temporary_path = self._output_path(job, temporary=True)
        final_path = self._output_path(job)
        try:
            async with self._generation_slot:
                self._cancel_idle_unload()
                job.status = "running"
                job.started_at = int(time.time())
                worker_running = self._worker is not None and self._worker.returncode is None
                if not worker_running:
                    free_memory = await _gpu_free_memory_bytes(self.settings.image_generation_gpu)
                    if free_memory is None or free_memory < MIN_IMAGE_GPU_FREE_BYTES:
                        raise ImageGenerationCapacityError(
                            "the selected image GPU no longer has 22 GiB free"
                        )
                result = await self._invoke_worker(job, temporary_path)
                _validate_image(temporary_path, job.request)
                temporary_path.chmod(0o600)
                os.replace(temporary_path, final_path)
                final_path.chmod(0o600)
                duration = result.get("duration_ms")
                peak_memory = result.get("peak_gpu_memory_bytes")
                if isinstance(duration, int) and 0 <= duration <= 3_600_000:
                    job.duration_ms = duration
                if isinstance(peak_memory, int) and 0 <= peak_memory <= 128 * 1024**3:
                    job.peak_gpu_memory_bytes = peak_memory
                job.status = "succeeded"
                job.completed_at = int(time.time())
                self._persist_job(job)
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "generation was cancelled"
            raise
        except ImageGenerationTimeout:
            job.status = "failed"
            job.error = "generation exceeded the configured timeout"
        except ImageGenerationCapacityError as exc:
            job.status = "failed"
            job.error = str(exc)
        except Exception as exc:
            stderr_tail = bytes(self._worker_stderr_tail).decode("utf-8", errors="replace")
            stderr_tail = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]+", " ", stderr_tail)[-2000:]
            LOGGER.warning(
                "image generation job %s failed in %s: %s; worker stderr tail=%r",
                job.id,
                type(exc).__name__,
                str(exc)[:500],
                stderr_tail,
            )
            await self._stop_worker()
            job.status = "failed"
            job.error = "the local image worker failed validation"
        finally:
            if job.completed_at is None:
                job.completed_at = int(time.time())
            with contextlib.suppress(FileNotFoundError):
                temporary_path.unlink()
            if job.status != "succeeded":
                for path in (final_path, self._metadata_path(job.id)):
                    with contextlib.suppress(FileNotFoundError):
                        path.unlink()
            self._schedule_idle_unload()

    def _cancel_idle_unload(self) -> None:
        task, self._idle_unload_task = self._idle_unload_task, None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _schedule_idle_unload(self) -> None:
        self._cancel_idle_unload()
        if self._closed or self._worker is None or self._worker.returncode is not None:
            return

        async def unload_after_idle() -> None:
            try:
                await asyncio.sleep(IDLE_UNLOAD_SECONDS)
                async with self._generation_slot:
                    async with self._jobs_lock:
                        active = any(
                            job.status in {"queued", "running"} for job in self._jobs.values()
                        )
                    if not active:
                        await self._stop_worker()
            finally:
                if self._idle_unload_task is asyncio.current_task():
                    self._idle_unload_task = None

        self._idle_unload_task = asyncio.create_task(
            unload_after_idle(), name="image-worker-idle-unload"
        )

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                return
            self._worker_stderr_tail.extend(chunk)
            if len(self._worker_stderr_tail) > MAX_WORKER_LINE_BYTES:
                del self._worker_stderr_tail[:-MAX_WORKER_LINE_BYTES]

    async def _read_worker_message(self, timeout: float) -> dict[str, Any]:
        worker = self._worker
        if worker is None or worker.stdout is None:
            raise WorkerProtocolError("image worker is not running")
        try:
            line = await asyncio.wait_for(worker.stdout.readline(), timeout=timeout)
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise WorkerProtocolError("image worker response exceeded its limit") from exc
        if not line or len(line) > MAX_WORKER_LINE_BYTES:
            raise WorkerProtocolError("image worker returned no bounded response")
        try:
            decoded = json.loads(line, parse_constant=_reject_nonfinite_json)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise WorkerProtocolError("image worker returned malformed data") from exc
        if not isinstance(decoded, dict):
            raise WorkerProtocolError("image worker returned malformed data")
        return decoded

    def _worker_environment(self) -> dict[str, str]:
        cache_root = PROJECT_ROOT / ".local/image-generation/cache"
        temp_root = PROJECT_ROOT / ".local/image-generation/tmp"
        cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        cache_root.chmod(0o700)
        temp_root.chmod(0o700)
        environment = {
            # Only the selected physical device node is mounted. It is therefore
            # logical device zero inside the private device namespace.
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": "/cache/huggingface",
            "TORCH_HOME": "/cache/torch",
            "XDG_CACHE_HOME": "/cache",
            "HOME": "/home/worker",
            "TMPDIR": "/work",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHOME": "/runtime/python",
            "PYTHONPATH": "/runtime/venv/lib/python3.10/site-packages",
            "TOKENIZERS_PARALLELISM": "false",
            "PATH": "/runtime/python/bin:/usr/bin:/bin",
        }
        cuda_home = Path("/usr/local/cuda-13.0")
        if cuda_home.is_dir() and not cuda_home.is_symlink():
            environment["CUDA_HOME"] = str(cuda_home)
            environment["CUDA_PATH"] = str(cuda_home)
        return environment

    def _worker_command(self) -> list[str]:
        if not _gpu_device_is_ready(self.settings.image_generation_gpu):
            raise ImageGenerationUnavailable("the selected image GPU device is unavailable")
        cache_root = PROJECT_ROOT / ".local/image-generation/cache"
        temp_root = PROJECT_ROOT / ".local/image-generation/tmp"
        selected_device = Path(f"/dev/nvidia{self.settings.image_generation_gpu}")
        environment = self._worker_environment()
        command = [
            str(BWRAP_BINARY),
            "--die-with-parent",
            "--new-session",
            "--unshare-user-try",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--clearenv",
            "--tmpfs",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/run",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/usr",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--symlink",
            "usr/sbin",
            "/sbin",
            "--dir",
            "/etc",
            "--ro-bind",
            "/etc/ld.so.cache",
            "/etc/ld.so.cache",
            "--ro-bind",
            "/etc/nsswitch.conf",
            "/etc/nsswitch.conf",
            "--ro-bind",
            "/etc/passwd",
            "/etc/passwd",
            "--ro-bind",
            "/etc/group",
            "/etc/group",
            "--dir",
            "/runtime",
            "--dir",
            "/runtime/python",
            "--ro-bind",
            str(RUNTIME_PYTHON_HOME / "bin"),
            "/runtime/python/bin",
            "--ro-bind",
            str(RUNTIME_PYTHON_HOME / "lib"),
            "/runtime/python/lib",
            "--tmpfs",
            "/runtime/python/lib/python3.10/site-packages",
            "--dir",
            "/runtime/venv",
            "--dir",
            "/runtime/venv/lib",
            "--dir",
            "/runtime/venv/lib/python3.10",
            "--ro-bind",
            str(RUNTIME_SITE_PACKAGES),
            "/runtime/venv/lib/python3.10/site-packages",
            "--ro-bind",
            str(RUNTIME_MARKER),
            "/runtime/attestation.json",
            "--dir",
            "/app",
            "--ro-bind",
            str(WORKER_SCRIPT),
            "/app/image-generation-worker.py",
            "--dir",
            "/model",
            "--ro-bind",
            str(MODEL_DIRECTORY),
            "/model",
            "--dir",
            "/output",
            "--bind",
            str(self.output_dir),
            "/output",
            "--dir",
            "/cache",
            "--bind",
            str(cache_root),
            "/cache",
            "--dir",
            "/work",
            "--bind",
            str(temp_root),
            "/work",
            "--dir",
            "/home",
            "--dir",
            "/home/worker",
            "--dev-bind",
            "/dev/nvidiactl",
            "/dev/nvidiactl",
            "--dev-bind",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm",
            "--dev-bind",
            str(selected_device),
            str(selected_device),
            "--cap-drop",
            "ALL",
            "--chdir",
            "/app",
        ]
        for name, value in sorted(environment.items()):
            command.extend(("--setenv", name, value))
        command.extend(
            (
                "--",
                "/runtime/python/bin/python3.10",
                "/app/image-generation-worker.py",
                "--serve",
                "--model-dir",
                "/model",
                "--output-dir",
                "/output",
                "--runtime-marker",
                "/runtime/attestation.json",
            )
        )
        return command

    def _worker_launch_command(
        self, sandbox_command: list[str]
    ) -> tuple[list[str], dict[str, str], str | None]:
        """Launch bwrap outside a systemd mount namespace when necessary.

        Ubuntu's AppArmor user-namespace policy rejects a nested user namespace
        from units that already use systemd mount namespacing (for example
        ``PrivateTmp``). A transient user unit starts the trusted bwrap binary
        from the user manager's namespace; the untrusted Python worker still
        starts only after bwrap has entered the strict sandbox.
        """

        base_environment = {"PATH": "/usr/bin:/bin"}
        if not os.environ.get("SYSTEMD_EXEC_PID"):
            return sandbox_command, base_environment, None
        if not _transient_worker_launcher_is_ready():
            return sandbox_command, base_environment, None
        unit = f"localllm-image-worker-{secrets.token_hex(8)}"
        launcher = [
            str(SYSTEMD_RUN_BINARY),
            "--user",
            "--pipe",
            "--wait",
            "--collect",
            "--quiet",
            "--service-type=exec",
            f"--unit={unit}",
            "--property=UMask=0077",
            "--property=TimeoutStopSec=5",
            f"--property=RuntimeMaxSec={self.settings.image_generation_timeout_seconds + 30}",
            "--",
            *sandbox_command,
        ]
        return (
            launcher,
            {
                **base_environment,
                "DBUS_SESSION_BUS_ADDRESS": f"unix:path={SYSTEMD_USER_BUS}",
            },
            unit,
        )

    async def _ensure_worker(self, deadline: float) -> None:
        if self._worker is not None and self._worker.returncode is None:
            return
        await self._stop_worker()
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise ImageGenerationTimeout
        command, process_environment, worker_unit = self._worker_launch_command(
            self._worker_command()
        )
        self._worker = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=process_environment,
            limit=MAX_WORKER_LINE_BYTES,
            start_new_session=True,
        )
        self._worker_unit = worker_unit
        self._worker_stderr_tail.clear()
        if self._worker.stderr is not None:
            self._worker_stderr_task = asyncio.create_task(
                self._drain_stderr(self._worker.stderr), name="image-worker-stderr"
            )
        try:
            ready = await self._read_worker_message(remaining)
            if ready != {"ok": True, "ready": True, "revision": MODEL_REVISION}:
                raise WorkerProtocolError("image worker did not attest the pinned model")
        except BaseException:
            await self._stop_worker()
            raise

    async def _invoke_worker(self, job: _Job, output_path: Path) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.image_generation_timeout_seconds
        try:
            await self._ensure_worker(deadline)
            worker = self._worker
            if worker is None or worker.stdin is None:
                raise WorkerProtocolError("image worker is unavailable")
            payload = {
                "job_id": job.id,
                "prompt": job.request.prompt,
                "width": job.request.width,
                "height": job.request.height,
                "steps": job.request.steps,
                "seed": job.request.seed,
                "output_format": job.request.output_format,
                "jpeg_quality": job.request.jpeg_quality,
            }
            encoded = (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
            )
            if len(encoded) > MAX_REQUEST_BYTES:
                raise WorkerProtocolError("worker request exceeded its protocol limit")
            worker.stdin.write(encoded)
            await worker.stdin.drain()
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            result = await self._read_worker_message(remaining)
            if result.get("ok") is not True:
                raise WorkerProtocolError("image worker reported a generation failure")
            return result
        except asyncio.CancelledError:
            await self._stop_worker()
            raise
        except asyncio.TimeoutError as exc:
            await self._stop_worker()
            raise ImageGenerationTimeout from exc
        except BaseException:
            await self._stop_worker()
            raise

    async def _stop_worker(self) -> None:
        worker, self._worker = self._worker, None
        worker_unit, self._worker_unit = self._worker_unit, None
        stderr_task, self._worker_stderr_task = self._worker_stderr_task, None
        if worker is not None:
            if worker.stdin is not None:
                worker.stdin.close()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    await worker.stdin.wait_closed()
            if worker.returncode is None:
                try:
                    await asyncio.wait_for(worker.wait(), timeout=5)
                except asyncio.TimeoutError:
                    if worker_unit is not None:
                        await self._stop_worker_unit(worker_unit)
                    else:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(worker.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(worker.wait(), timeout=2)
                    except asyncio.TimeoutError:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(worker.pid, signal.SIGKILL)
                        await worker.wait()
            if worker.stdout is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(worker.stdout.read(), timeout=1)
            with contextlib.suppress(Exception):
                await worker.wait()
        if stderr_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(stderr_task), timeout=1)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await stderr_task
        # Let asyncio process the pipe-closed callbacks before the caller can
        # close its event loop (notably the standalone smoke script).
        await asyncio.sleep(0)

    async def _stop_worker_unit(self, unit: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                str(SYSTEMCTL_BINARY),
                "--user",
                "stop",
                unit,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env={
                    "PATH": "/usr/bin:/bin",
                    "DBUS_SESSION_BUS_ADDRESS": f"unix:path={SYSTEMD_USER_BUS}",
                },
                start_new_session=True,
            )
        except OSError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=7)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    async def shutdown(self) -> None:
        self._closed = True
        self._cancel_idle_unload()
        async with self._jobs_lock:
            tasks = [job.task for job in self._jobs.values() if job.task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._stop_worker()


def get_image_generation_manager(request: Request) -> ImageGenerationManager:
    manager = getattr(request.app.state, "image_generation", None)
    if not isinstance(manager, ImageGenerationManager):
        raise HTTPException(status_code=503, detail="Image generation manager is unavailable")
    return manager


def require_image_generation_key(
    authorization: str | None = Header(default=None),
    x_localllm_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.api_key:
        return
    bearer = ""
    if authorization:
        scheme, separator, credential = authorization.partition(" ")
        if separator and scheme.lower() == "bearer":
            bearer = credential
    if not (
        hmac.compare_digest(bearer, settings.api_key)
        or (x_localllm_key is not None and hmac.compare_digest(x_localllm_key, settings.api_key))
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid LocalLLM API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def _router_lifespan(app: Any):
    existing = getattr(app.state, "image_generation", None)
    owned = not isinstance(existing, ImageGenerationManager)
    if owned:
        app.state.image_generation = ImageGenerationManager(get_settings())
    try:
        with contextlib.suppress(ImageGenerationUnavailable):
            await app.state.image_generation.initialize()
        yield
    finally:
        if owned:
            await app.state.image_generation.shutdown()


router = APIRouter(prefix="/api/images", tags=["image-generation"], lifespan=_router_lifespan)


@router.get("/status")
async def image_generation_status(
    manager: ImageGenerationManager = Depends(get_image_generation_manager),
) -> dict[str, Any]:
    return await manager.status()


@router.post(
    "/jobs",
    status_code=202,
    response_model=ImageJobResponse,
    dependencies=[Depends(require_image_generation_key)],
)
async def create_image_job(
    request: Request,
    manager: ImageGenerationManager = Depends(get_image_generation_manager),
) -> ImageJobResponse:
    image_request = await _bounded_request(request)
    try:
        return await manager.submit(image_request)
    except ImageGenerationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImageGenerationCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.get(
    "/jobs",
    response_model=list[ImageJobResponse],
    dependencies=[Depends(require_image_generation_key)],
)
async def list_image_jobs(
    manager: ImageGenerationManager = Depends(get_image_generation_manager),
) -> list[ImageJobResponse]:
    try:
        return await manager.list_jobs()
    except ImageGenerationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/unload",
    dependencies=[Depends(require_image_generation_key)],
)
async def unload_image_worker(
    request: Request,
    manager: ImageGenerationManager = Depends(get_image_generation_manager),
) -> dict[str, bool]:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) != 0:
                raise HTTPException(status_code=413, detail="Unload request body must be empty")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    async for chunk in request.stream():
        if chunk:
            raise HTTPException(status_code=413, detail="Unload request body must be empty")
    try:
        released = await manager.unload()
    except ImageGenerationCapacityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"released": released, "worker_running": False}


@router.get(
    "/jobs/{job_id}",
    response_model=ImageJobResponse,
    dependencies=[Depends(require_image_generation_key)],
)
async def get_image_job(
    job_id: str,
    manager: ImageGenerationManager = Depends(get_image_generation_manager),
) -> ImageJobResponse:
    try:
        return await manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Image job was not found") from exc
    except ImageGenerationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/jobs/{job_id}/image",
    dependencies=[Depends(require_image_generation_key)],
)
async def get_image_job_output(
    job_id: str,
    manager: ImageGenerationManager = Depends(get_image_generation_manager),
) -> StreamingResponse:
    try:
        path, media_type, size = await manager.image_path(job_id)
    except (KeyError, WorkerProtocolError) as exc:
        raise HTTPException(status_code=404, detail="Image output was not found") from exc
    except ImageGenerationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Image output was not found") from exc

    async def body():
        try:
            while True:
                chunk = await asyncio.to_thread(os.read, descriptor, 64 * 1024)
                if not chunk:
                    return
                yield chunk
        finally:
            os.close(descriptor)

    return StreamingResponse(
        body(),
        media_type=media_type,
        headers={
            "Content-Length": str(size),
            "Content-Disposition": f'inline; filename="{job_id}.{path.suffix.lstrip(".")}"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete(
    "/jobs/{job_id}",
    status_code=204,
    dependencies=[Depends(require_image_generation_key)],
)
async def delete_image_job(
    job_id: str,
    manager: ImageGenerationManager = Depends(get_image_generation_manager),
) -> Response:
    try:
        await manager.delete(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Image job was not found") from exc
    except ImageGenerationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(status_code=204)


__all__ = [
    "ImageGenerationManager",
    "ImageGenerationRequest",
    "MODEL_ID",
    "MODEL_REVISION",
    "router",
]
