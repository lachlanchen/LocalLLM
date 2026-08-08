from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from localllm.catalog import MODEL_CATALOG
from localllm.main import app


class FakeOllama:
    async def health(self) -> dict[str, Any]:
        return {"ok": True, "version": "test"}

    async def tags(self) -> list[dict[str, Any]]:
        return [{"name": "qwen3:8b-q4_K_M", "size": 1234}]

    async def get_model(self, model: str) -> httpx.Response:
        request = httpx.Request("GET", f"http://ollama.test/v1/models/{model}")
        return httpx.Response(200, json={"id": model, "object": "model"}, request=request)


class FakeStream:
    def __init__(self, status_code: int, payload: Any | None = None):
        request = httpx.Request("POST", "http://ollama.test/v1/chat/completions")
        self.response = httpx.Response(
            status_code,
            json=payload,
            headers={"content-type": "application/json"},
            request=request,
        )
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True
        await self.response.aclose()

    async def iter_raw(self):
        try:
            yield b'data: {"id":"chatcmpl_test"}\n\n'
            yield b"data: [DONE]\n\n"
        finally:
            await self.aclose()


class FakeProxyOllama(FakeOllama):
    def __init__(self, stream: FakeStream | None = None):
        self.stream = stream
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def proxy_stream(self, endpoint: str, payload: dict[str, Any]) -> FakeStream:
        self.calls.append((endpoint, payload))
        assert self.stream is not None
        return self.stream

    async def proxy_json(self, endpoint: str, payload: dict[str, Any]) -> httpx.Response:
        self.calls.append((endpoint, payload))
        request = httpx.Request("POST", f"http://ollama.test{endpoint}")
        if endpoint == "/v1/responses":
            body = {"id": "resp_test", "object": "response", "output": []}
        elif endpoint == "/v1/embeddings":
            body = {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": payload["model"],
            }
        else:
            body = {"id": "chatcmpl_test", "object": "chat.completion", "choices": []}
        return httpx.Response(200, json=body, request=request)


def test_catalog_and_openai_model_aliases() -> None:
    with TestClient(app) as client:
        client.app.state.ollama = FakeOllama()
        catalog = client.get("/api/models/catalog")
        assert catalog.status_code == 200
        assert len(catalog.json()["models"]) == len(MODEL_CATALOG)
        installed = [model for model in catalog.json()["models"] if model["installed"]]
        assert [model["id"] for model in installed] == ["qwen3:8b-q4_K_M"]

        unauthorized = client.get("/v1/models")
        assert unauthorized.status_code == 401
        assert unauthorized.json() == {
            "error": {
                "message": "Invalid LocalLLM API key",
                "type": "authentication_error",
                "param": None,
                "code": None,
            }
        }
        assert unauthorized.headers["www-authenticate"] == "Bearer"

        authorized = client.get(
            "/v1/models", headers={"Authorization": "bearer local-dev-key"}
        )
        assert authorized.status_code == 200
        ids = {model["id"] for model in authorized.json()["data"]}
        assert "qwen3:8b-q4_K_M" in ids
        assert "localllm-fast" in ids


def test_model_catalog_degrades_explicitly_but_openai_models_return_503() -> None:
    class UnavailableCatalog(FakeOllama):
        async def tags(self) -> list[dict[str, Any]]:
            raise HTTPException(status_code=503, detail="Ollama model catalog is unavailable")

    headers = {"Authorization": "Bearer local-dev-key"}
    with TestClient(app) as client:
        client.app.state.ollama = UnavailableCatalog()
        catalog = client.get("/api/models/catalog")
        models = client.get("/v1/models", headers=headers)
        retrieval = client.get("/v1/models/localllm-fast", headers=headers)

    assert catalog.status_code == 200
    assert catalog.json()["installed"] == []
    assert catalog.json()["ollama"] == {
        "ok": False,
        "error": "Ollama model catalog is unavailable",
    }
    for response in (models, retrieval):
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "service_unavailable"
        assert "catalog is unavailable" in response.json()["error"]["message"]


def test_health_reports_gateway_and_runtime() -> None:
    with TestClient(app) as client:
        client.app.state.ollama = FakeOllama()
        response = client.get("/healthz")
        assert response.status_code == 200
    assert response.json()["service"] == "localllm-api"
    assert response.json()["ollama"]["version"] == "test"


def test_model_retrieval_accepts_installed_alias_and_rejects_path_traversal() -> None:
    with TestClient(app) as client:
        client.app.state.ollama = FakeOllama()
        headers = {"Authorization": "Bearer local-dev-key"}
        installed = client.get("/v1/models/localllm-fast", headers=headers)
        traversal = client.get(
            "/v1/models/%2e%2e/%2e%2e/api/version",
            headers=headers,
        )

    assert installed.status_code == 200
    assert installed.json()["id"] == "qwen3:8b-q4_K_M"
    assert traversal.status_code == 404
    assert traversal.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    ("upstream_error", "expected_message"),
    [
        ({"error": "model 'missing' not found"}, "model 'missing' not found"),
        (
            {"error": {"message": "invalid model", "type": "invalid_request_error"}},
            "invalid model",
        ),
    ],
)
def test_streaming_preflight_preserves_error_status_and_json_shape(
    upstream_error: dict[str, Any], expected_message: str
) -> None:
    stream = FakeStream(404, upstream_error)
    fake = FakeProxyOllama(stream)

    with TestClient(app) as client:
        client.app.state.ollama = fake
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer local-dev-key"},
            json={"model": "missing", "messages": [], "stream": True},
        )

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["message"] == expected_message
    assert isinstance(response.json()["error"], dict)
    assert stream.closed


def test_successful_stream_is_forwarded_and_closed() -> None:
    stream = FakeStream(200, {"unused": True})
    stream.response.headers["content-type"] = "text/event-stream"
    fake = FakeProxyOllama(stream)

    with TestClient(app) as client:
        client.app.state.ollama = fake
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer local-dev-key"},
            json={"model": "localllm-fast", "messages": [], "stream": True},
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert b'"id":"chatcmpl_test"' in body
    assert b"data: [DONE]" in body
    assert stream.closed


def test_streaming_connection_failure_is_openai_compatible() -> None:
    class UnavailableOllama(FakeProxyOllama):
        async def proxy_stream(self, endpoint: str, payload: dict[str, Any]) -> FakeStream:
            raise HTTPException(status_code=503, detail="Ollama is unavailable: offline")

    with TestClient(app) as client:
        client.app.state.ollama = UnavailableOllama()
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer local-dev-key"},
            json={"model": "localllm-deep", "input": "hello", "stream": True},
        )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "message": "Ollama is unavailable: offline",
            "type": "service_unavailable",
            "param": None,
            "code": None,
        }
    }


def test_responses_and_embeddings_contracts_are_forwarded() -> None:
    fake = FakeProxyOllama()
    headers = {"Authorization": "Bearer local-dev-key"}

    with TestClient(app) as client:
        client.app.state.ollama = fake
        response = client.post(
            "/v1/responses",
            headers=headers,
            json={"model": "localllm-deep", "input": "hello"},
        )
        embeddings = client.post(
            "/v1/embeddings",
            headers=headers,
            json={"model": "localllm-embed", "input": ["one", "two"]},
        )

    assert response.status_code == 200
    assert response.json()["object"] == "response"
    assert embeddings.status_code == 200
    assert embeddings.json()["object"] == "list"
    assert embeddings.json()["data"][0]["object"] == "embedding"
    assert [endpoint for endpoint, _payload in fake.calls] == [
        "/v1/responses",
        "/v1/embeddings",
    ]
