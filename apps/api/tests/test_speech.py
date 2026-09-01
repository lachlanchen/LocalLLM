from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localllm.config import Settings, get_settings
from localllm.main import _request_body_limit
from localllm.speech import (
    MAX_SPEECH_AUDIO_BYTES,
    SPEECH_MULTIPART_REQUEST_BYTES,
    SpeechRuntimeError,
    SpeechTranscriptionManager,
    router,
)


class FakeSpeechManager:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str, str]] = []

    def status(self):
        return {
            "schema": "localllm/speech-status/v1",
            "enabled": True,
            "state": "cold",
            "model_loaded": False,
            "accepted_media_types": ["audio/mp4"],
            "maximum_audio_bytes": MAX_SPEECH_AUDIO_BYTES,
            "maximum_duration_seconds": 180,
            "persistence": "transient-until-transcribed",
            "fault": None,
        }

    async def transcribe(self, payload: bytes, media_type: str, language: str):
        self.calls.append((payload, media_type, language))
        return {
            "schema": "localllm/speech-transcription/v1",
            "text": "Voice input works.",
            "language": "en",
            "language_probability": 0.99,
            "duration_seconds": 1.25,
            "audio_retained": False,
        }


def speech_app(tmp_path: Path) -> tuple[TestClient, FakeSpeechManager]:
    model = tmp_path / "model"
    model.mkdir()
    settings = Settings(
        data_dir=tmp_path / "data",
        speech_enabled=True,
        speech_model_path=model,
        speech_api_key="speech-test-key",
        _env_file=None,
    )
    manager = FakeSpeechManager()
    app = FastAPI()
    app.state.speech = manager
    app.include_router(router)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app), manager


def test_speech_status_is_public_bounded_and_no_store(tmp_path: Path) -> None:
    client, _manager = speech_app(tmp_path)

    response = client.get("/api/speech/status")

    assert response.status_code == 200
    assert response.json()["state"] == "cold"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_speech_transcription_requires_dedicated_bearer_key(tmp_path: Path) -> None:
    client, manager = speech_app(tmp_path)
    audio = b"\x00\x00\x00\x18ftypM4A "

    missing = client.post(
        "/api/speech/transcriptions",
        files={"file": ("voice.m4a", audio, "audio/mp4")},
    )
    wrong = client.post(
        "/api/speech/transcriptions",
        headers={"authorization": "Bearer wrong"},
        files={"file": ("voice.m4a", audio, "audio/mp4")},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert manager.calls == []


def test_speech_transcription_passes_only_bytes_media_type_and_language(tmp_path: Path) -> None:
    client, manager = speech_app(tmp_path)
    audio = b"\x00\x00\x00\x18ftypM4A "

    response = client.post(
        "/api/speech/transcriptions?language=en",
        headers={"authorization": "Bearer speech-test-key"},
        files={"file": ("private-original-name.m4a", audio, "audio/mp4")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "schema": "localllm/speech-transcription/v1",
        "text": "Voice input works.",
        "language": "en",
        "language_probability": 0.99,
        "duration_seconds": 1.25,
        "audio_retained": False,
    }
    assert response.headers["cache-control"] == "no-store, private"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert manager.calls == [(audio, "audio/mp4", "en")]


def test_speech_transcription_rejects_unknown_query_fields(tmp_path: Path) -> None:
    client, manager = speech_app(tmp_path)

    response = client.post(
        "/api/speech/transcriptions?language=en&callback=https://attacker.test",
        headers={"authorization": "Bearer speech-test-key"},
        files={"file": ("voice.m4a", b"\x00\x00\x00\x18ftypM4A ", "audio/mp4")},
    )

    assert response.status_code == 422
    assert manager.calls == []


@pytest.mark.asyncio
async def test_manager_deletes_transient_audio_after_success(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    settings = Settings(
        data_dir=tmp_path / "data",
        speech_enabled=True,
        speech_model_path=model,
        speech_python_path=Path("/bin/true"),
        speech_ffprobe_path=Path("/bin/true"),
        speech_api_key="speech-test-key",
        _env_file=None,
    )
    manager = SpeechTranscriptionManager(settings)

    async def probe(_path: Path) -> float:
        return 1.5

    async def exchange(_path: Path, _language: str):
        return {
            "text": "A local transcript.",
            "language": "en",
            "language_probability": 0.9,
        }

    monkeypatch.setattr(manager, "_probe_duration", probe)
    monkeypatch.setattr(manager, "_exchange", exchange)
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt "

    result = await manager.transcribe(wav, "audio/wav", "auto")

    assert result["text"] == "A local transcript."
    assert result["audio_retained"] is False
    assert list((settings.data_dir / "speech-inflight").iterdir()) == []


@pytest.mark.asyncio
async def test_manager_fails_closed_on_media_mismatch(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    manager = SpeechTranscriptionManager(settings)

    with pytest.raises(SpeechRuntimeError, match="audio_signature_mismatch") as error:
        await manager.transcribe(b"not an mp4", "audio/mp4", "auto")

    assert error.value.status_code == 422
    assert not (settings.data_dir / "speech-inflight").exists()


def test_main_request_boundary_covers_speech_multipart() -> None:
    assert _request_body_limit("/api/speech/transcriptions") == SPEECH_MULTIPART_REQUEST_BYTES
