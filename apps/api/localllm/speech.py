from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings, get_settings, prepare_private_data_dir

MAX_SPEECH_AUDIO_BYTES = 12 * 1024 * 1024
MAX_SPEECH_DURATION_SECONDS = 180.0
MAX_SPEECH_TRANSCRIPT_CHARACTERS = 32_000
SPEECH_MULTIPART_REQUEST_BYTES = MAX_SPEECH_AUDIO_BYTES + 64 * 1024
_LANGUAGE = re.compile(r"^(?:auto|[a-z]{2,3})$")
_AUDIO_TYPES = {
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
}


class SpeechRuntimeError(Exception):
    def __init__(self, code: str, status_code: int = 503) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


def _audio_signature_is_valid(media_type: str, payload: bytes) -> bool:
    if media_type in {"audio/mp4", "audio/x-m4a"}:
        return len(payload) >= 12 and payload[4:8] == b"ftyp"
    if media_type == "audio/webm":
        return payload.startswith(b"\x1a\x45\xdf\xa3")
    if media_type == "audio/ogg":
        return payload.startswith(b"OggS")
    if media_type in {"audio/wav", "audio/x-wav"}:
        return len(payload) >= 12 and payload.startswith(b"RIFF") and payload[8:12] == b"WAVE"
    if media_type == "audio/mpeg":
        return payload.startswith(b"ID3") or (
            len(payload) >= 2 and payload[0] == 0xFF and payload[1] & 0xE0 == 0xE0
        )
    return False


def _private_directory(path: Path) -> Path:
    result = prepare_private_data_dir(path)
    entry = result.stat()
    if entry.st_uid != os.getuid() or stat.S_IMODE(entry.st_mode) & 0o077:
        raise SpeechRuntimeError("speech_storage_not_private")
    for child in result.iterdir():
        child_entry = child.lstat()
        if not stat.S_ISREG(child_entry.st_mode) or child_entry.st_nlink != 1:
            raise SpeechRuntimeError("speech_storage_contains_unknown_entry")
        child.unlink()
    return result


class SpeechTranscriptionManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._input_root: Path | None = None
        self._busy = False
        self._fault: str | None = None

    def status(self) -> dict[str, Any]:
        if not self.settings.speech_enabled:
            state = "disabled"
        elif self._fault:
            state = "faulted"
        elif self._busy:
            state = "busy"
        elif self._process is not None and self._process.returncode is None:
            state = "ready"
        else:
            state = "cold"
        return {
            "schema": "localllm/speech-status/v1",
            "enabled": self.settings.speech_enabled,
            "state": state,
            "model_loaded": state in {"ready", "busy"},
            "accepted_media_types": sorted(_AUDIO_TYPES),
            "maximum_audio_bytes": MAX_SPEECH_AUDIO_BYTES,
            "maximum_duration_seconds": int(MAX_SPEECH_DURATION_SECONDS),
            "persistence": "transient-until-transcribed",
            "fault": self._fault,
        }

    def _validate_runtime(self) -> Path:
        if not self.settings.speech_enabled:
            raise SpeechRuntimeError("speech_disabled")
        model_path = self.settings.speech_model_path
        if model_path is None or not model_path.is_absolute() or not model_path.is_dir():
            raise SpeechRuntimeError("speech_model_unavailable")
        for executable in (self.settings.speech_python_path, self.settings.speech_ffprobe_path):
            if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
                raise SpeechRuntimeError("speech_runtime_unavailable")
        if self.settings.speech_device == "cuda" and self.settings.speech_device_index != 1:
            raise SpeechRuntimeError("speech_gpu_policy_rejected")
        if self._input_root is None:
            data_root = Path(os.path.abspath(os.fspath(self.settings.data_dir)))
            self._input_root = _private_directory(data_root / "speech-inflight")
        return model_path

    async def _stop_worker(self) -> None:
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def close(self) -> None:
        async with self._lock:
            await self._stop_worker()

    async def _start_worker(self) -> asyncio.subprocess.Process:
        model_path = self._validate_runtime()
        if self._process is not None and self._process.returncode is None:
            return self._process
        await self._stop_worker()
        worker = Path(__file__).with_name("speech_worker.py")
        if not worker.is_file() or self._input_root is None:
            raise SpeechRuntimeError("speech_runtime_unavailable")
        environment = dict(os.environ)
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        process = await asyncio.create_subprocess_exec(
            os.fspath(self.settings.speech_python_path),
            os.fspath(worker),
            "--model",
            os.fspath(model_path),
            "--input-root",
            os.fspath(self._input_root),
            "--device",
            self.settings.speech_device,
            "--device-index",
            str(0 if self.settings.speech_device == "cpu" else self.settings.speech_device_index),
            "--compute-type",
            self.settings.speech_compute_type,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
        )
        self._process = process
        try:
            line = await asyncio.wait_for(
                process.stdout.readline(), timeout=self.settings.speech_worker_start_timeout_seconds
            )
            ready = json.loads(line)
            if ready != {"schema": "localllm/speech-worker-ready/v1", "ready": True}:
                raise ValueError
        except (TimeoutError, ValueError, json.JSONDecodeError):
            await self._stop_worker()
            raise SpeechRuntimeError("speech_worker_start_failed") from None
        self._fault = None
        return process

    async def _probe_duration(self, path: Path) -> float:
        process = await asyncio.create_subprocess_exec(
            os.fspath(self.settings.speech_ffprobe_path),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            os.fspath(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise SpeechRuntimeError("invalid_audio", 422) from None
        try:
            duration = float(json.loads(stdout)["format"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise SpeechRuntimeError("invalid_audio", 422) from None
        if process.returncode != 0 or duration <= 0:
            raise SpeechRuntimeError("invalid_audio", 422)
        if duration > MAX_SPEECH_DURATION_SECONDS:
            raise SpeechRuntimeError("audio_too_long", 413)
        return duration

    async def _exchange(self, path: Path, language: str) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        for attempt in range(2):
            process = await self._start_worker()
            if process.stdin is None or process.stdout is None:
                raise SpeechRuntimeError("speech_worker_unavailable")
            payload = json.dumps(
                {"id": request_id, "path": os.fspath(path), "language": language},
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            try:
                process.stdin.write(payload)
                await process.stdin.drain()
                line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=self.settings.speech_timeout_seconds
                )
                result = json.loads(line)
                if (
                    result.get("schema") != "localllm/speech-worker-result/v1"
                    or result.get("id") != request_id
                ):
                    raise ValueError
                if result.get("ok") is not True:
                    code = result.get("error")
                    if code == "no_speech_detected":
                        raise SpeechRuntimeError(code, 422)
                    raise SpeechRuntimeError("transcription_failed")
                text = result.get("text")
                if not isinstance(text, str) or not text.strip() or len(text) > MAX_SPEECH_TRANSCRIPT_CHARACTERS:
                    raise ValueError
                return result
            except SpeechRuntimeError:
                raise
            except (BrokenPipeError, ConnectionResetError, ValueError, json.JSONDecodeError):
                await self._stop_worker()
                if attempt == 1:
                    raise SpeechRuntimeError("speech_worker_failed") from None
            except TimeoutError:
                await self._stop_worker()
                raise SpeechRuntimeError("transcription_timeout", 504) from None
        raise SpeechRuntimeError("speech_worker_failed")

    async def transcribe(self, payload: bytes, media_type: str, language: str) -> dict[str, Any]:
        if media_type not in _AUDIO_TYPES:
            raise SpeechRuntimeError("unsupported_audio_type", 415)
        if not payload or len(payload) > MAX_SPEECH_AUDIO_BYTES:
            raise SpeechRuntimeError("audio_too_large", 413)
        if not _audio_signature_is_valid(media_type, payload):
            raise SpeechRuntimeError("audio_signature_mismatch", 422)
        if not _LANGUAGE.fullmatch(language):
            raise SpeechRuntimeError("invalid_language", 422)
        async with self._lock:
            self._busy = True
            path: Path | None = None
            try:
                self._validate_runtime()
                if self._input_root is None:
                    raise SpeechRuntimeError("speech_storage_unavailable")
                path = self._input_root / f"audio-{uuid.uuid4().hex}{_AUDIO_TYPES[media_type]}"
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                finally:
                    os.close(descriptor)
                measured_duration = await self._probe_duration(path)
                result = await self._exchange(path, language)
                self._fault = None
                return {
                    "schema": "localllm/speech-transcription/v1",
                    "text": result["text"].strip(),
                    "language": result.get("language", "und"),
                    "language_probability": result.get("language_probability", 0.0),
                    "duration_seconds": round(measured_duration, 3),
                    "audio_retained": False,
                }
            except SpeechRuntimeError as exc:
                if exc.status_code >= 500:
                    self._fault = exc.code
                raise
            finally:
                self._busy = False
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        self._fault = "speech_audio_cleanup_failed"


def get_speech(request: Request) -> SpeechTranscriptionManager:
    return request.app.state.speech


def require_speech_api_key(
    request: Request, current: Settings = Depends(get_settings)
) -> None:
    expected = current.speech_api_key.get_secret_value()
    if not current.speech_enabled or not expected:
        raise HTTPException(status_code=503, detail="Speech transcription is unavailable")
    values = tuple(
        value for name, value in request.scope.get("headers", ()) if name.lower() == b"authorization"
    )
    if len(values) != 1 or not values[0].startswith(b"Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid speech API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    candidate = values[0][len(b"Bearer ") :]
    if not hmac.compare_digest(candidate, expected.encode("ascii")):
        raise HTTPException(
            status_code=401,
            detail="Invalid speech API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


router = APIRouter(prefix="/api/speech", tags=["speech"])


@router.get("/status")
async def speech_status(manager: SpeechTranscriptionManager = Depends(get_speech)) -> JSONResponse:
    response = JSONResponse(content=manager.status())
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@router.post("/transcriptions", dependencies=[Depends(require_speech_api_key)])
async def speech_transcription(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("auto"),
    manager: SpeechTranscriptionManager = Depends(get_speech),
) -> JSONResponse:
    if request.query_params:
        raise HTTPException(status_code=422, detail="Unsupported query parameter")
    media_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    payload = await file.read(MAX_SPEECH_AUDIO_BYTES + 1)
    await file.close()
    try:
        result = await manager.transcribe(payload, media_type, language.strip().lower())
    except SpeechRuntimeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    response = JSONResponse(content=result)
    response.headers["Cache-Control"] = "no-store, private"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
