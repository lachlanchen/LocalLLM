from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localllm.config import Settings, get_settings
from localllm.main import app, readiness
from localllm.ollama import OllamaProbe


class ProbeOllama:
    def __init__(self, probe: OllamaProbe) -> None:
        self._probe = probe

    async def probe(self) -> OllamaProbe:
        return self._probe


def _settings(required_models: list[str]) -> Settings:
    return Settings(required_models=required_models, _env_file=None)


def test_liveness_is_dependency_free_and_not_cacheable() -> None:
    class UnusableOllama:
        async def probe(self) -> OllamaProbe:
            raise AssertionError("liveness must not touch Ollama")

        async def health(self) -> dict[str, object]:
            raise AssertionError("liveness must not touch Ollama")

    with TestClient(app) as client:
        client.app.state.ollama = UnusableOllama()
        response = client.get("/livez")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "ok": True,
        "status": "alive",
        "service": {"name": "localllm-api", "version": app.version},
    }


def test_compatibility_health_stays_200_when_ollama_is_down() -> None:
    class UnavailableOllama:
        async def health(self) -> dict[str, object]:
            return {"ok": False, "error": "offline"}

    with TestClient(app) as client:
        client.app.state.ollama = UnavailableOllama()
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "service": "localllm-api",
        "ollama": {"ok": False, "error": "offline"},
    }


def test_readiness_requires_ollama_and_every_configured_model() -> None:
    app.dependency_overrides[get_settings] = lambda: _settings(["localllm-fast", "localllm-vision"])
    try:
        with TestClient(app) as client:
            client.app.state.ollama = ProbeOllama(OllamaProbe(ok=True, models=("qwen3:8b-q4_K_M",)))
            missing = client.get("/readyz")
            client.app.state.ollama = ProbeOllama(
                OllamaProbe(
                    ok=True,
                    models=("qwen3:8b-q4_K_M", "qwen3-vl:8b-instruct-q4_K_M"),
                )
            )
            ready = client.get("/readyz")
    finally:
        app.dependency_overrides.clear()

    assert missing.status_code == 503
    assert missing.headers["cache-control"] == "no-store"
    assert missing.json()["status"] == "not_ready"
    assert missing.json()["checks"]["required_models"]["missing"] == ["localllm-vision"]
    assert ready.status_code == 200
    assert ready.json()["ok"] is True
    assert ready.json()["checks"]["required_models"]["missing"] == []


def test_readiness_returns_stable_sanitized_dependency_failure() -> None:
    app.dependency_overrides[get_settings] = lambda: _settings(["localllm-fast"])
    try:
        with TestClient(app) as client:
            client.app.state.ollama = ProbeOllama(
                OllamaProbe(ok=False, error_code="ollama_catalog_unavailable")
            )
            response = client.get("/readyz")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["checks"]["ollama"] == {
        "ok": False,
        "code": "ollama_catalog_unavailable",
    }
    assert response.json()["checks"]["required_models"]["missing"] == ["localllm-fast"]
    assert "offline" not in response.text


def test_node_capabilities_contract_is_versioned_and_reports_actual_models() -> None:
    app.dependency_overrides[get_settings] = lambda: _settings(["localllm-fast", "localllm-vision"])
    try:
        with TestClient(app) as client:
            client.app.state.ollama = ProbeOllama(
                OllamaProbe(
                    ok=True,
                    models=("custom/model:latest", "qwen3:8b-q4_K_M"),
                )
            )
            response = client.get("/api/node/capabilities")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["ready"] is False
    assert payload["runtime"] == {
        "provider": "ollama",
        "ready": True,
        "error_code": None,
    }
    assert [protocol["id"] for protocol in payload["protocols"]] == [
        "openai.models.list.v1",
        "openai.models.retrieve.v1",
        "openai.chat-completions.v1",
        "openai.responses.v1",
        "openai.embeddings.v1",
    ]
    advertised_routes = {
        (protocol["method"], protocol["path"]) for protocol in payload["protocols"]
    }
    implemented_routes = {
        (method, path.replace("{model:path}", "{model}"))
        for route in app.routes
        if (path := getattr(route, "path", "")).startswith("/v1/")
        for method in getattr(route, "methods", set())
    }
    assert advertised_routes == implemented_routes
    fast = next(model for model in payload["models"] if model["id"] == "qwen3:8b-q4_K_M")
    custom = next(model for model in payload["models"] if model["id"] == "custom/model:latest")
    assert fast["aliases"] == ["localllm-fast"]
    assert fast["modalities"] == ["text", "tools"]
    assert custom == {
        "id": "custom/model:latest",
        "aliases": [],
        "catalogued": False,
        "modalities": [],
        "context_tokens": None,
    }


def test_node_capabilities_remains_discoverable_when_runtime_is_unready() -> None:
    app.dependency_overrides[get_settings] = lambda: _settings([])
    try:
        with TestClient(app) as client:
            client.app.state.ollama = ProbeOllama(
                OllamaProbe(ok=False, error_code="ollama_catalog_unavailable")
            )
            response = client.get("/api/node/capabilities")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["models"] == []


@pytest.mark.asyncio
async def test_readiness_cancellation_propagates() -> None:
    started = asyncio.Event()
    blocker = asyncio.Event()

    class BlockingOllama:
        async def probe(self) -> OllamaProbe:
            started.set()
            await blocker.wait()
            return OllamaProbe(ok=True)

    task = asyncio.create_task(
        readiness(current=_settings([]), ollama=BlockingOllama())  # type: ignore[arg-type]
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_service_installer_uses_truthful_readiness_probe() -> None:
    root = Path(__file__).parents[3]
    installer = (root / "scripts" / "install-user-services.sh").read_text()

    assert '"LocalLLM API" "http://127.0.0.1:8008/readyz"' in installer
    assert "http://127.0.0.1:8008/healthz" not in installer
