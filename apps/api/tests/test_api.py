import asyncio
import json
import socket
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from localllm.catalog import MODEL_CATALOG
from localllm.main import _proxy_openai, app


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

        authorized = client.get("/v1/models", headers={"Authorization": "bearer local-dev-key"})
        assert authorized.status_code == 200
        model_list = authorized.json()
        assert set(model_list) == {"object", "data"}
        assert all(
            set(model) == {"id", "object", "created", "owned_by"}
            for model in model_list["data"]
        )
        ids = {model["id"] for model in model_list["data"]}
        assert "qwen3:8b-q4_K_M" in ids
        assert "localllm-fast" in ids


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_proxy_cancels_preheader_upstream_after_client_disconnect(stream: bool) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    blocker = asyncio.Event()

    async def receive_disconnect() -> dict[str, str]:
        await started.wait()
        return {"type": "http.disconnect"}

    class BlockingOllama:
        async def _wait(self) -> Any:
            started.set()
            try:
                await blocker.wait()
            finally:
                cancelled.set()

        async def proxy_stream(self, endpoint: str, payload: dict[str, Any]) -> Any:
            return await self._wait()

        async def proxy_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
            return await self._wait()

    response = await _proxy_openai(
        Request({"type": "http"}, receive_disconnect),
        "/v1/chat/completions",
        {"model": "localllm-fast", "messages": [], "stream": stream},
        BlockingOllama(),  # type: ignore[arg-type]
    )

    assert response.status_code == 499
    assert json.loads(response.body) == {
        "error": {
            "message": "Client closed request",
            "type": "request_cancelled",
            "param": None,
            "code": None,
        }
    }
    assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_real_asgi_disconnect_cancels_preheader_upstream(stream: bool) -> None:
    class BlockingOllama:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()

        async def _wait(self) -> None:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

        async def proxy_stream(self, endpoint: str, payload: dict[str, Any]) -> FakeStream:
            await self._wait()
            return FakeStream(200, {"ok": True})

        async def proxy_json(self, endpoint: str, payload: dict[str, Any]) -> httpx.Response:
            await self._wait()
            request = httpx.Request("POST", f"http://ollama.test{endpoint}")
            return httpx.Response(200, json={"ok": True}, request=request)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = int(listener.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="critical",
            access_log=False,
            lifespan="on",
            timeout_graceful_shutdown=1,
        )
    )
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    fake = BlockingOllama()
    writer: asyncio.StreamWriter | None = None
    try:
        async def wait_until_started() -> None:
            while not server.started:
                if server_task.done():
                    await server_task
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_until_started(), timeout=2)
        app.state.ollama = fake
        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        body = json.dumps(
            {"model": "localllm-fast", "messages": [], "stream": stream}
        ).encode()
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            + f"Host: 127.0.0.1:{port}\r\n".encode()
            + b"Authorization: Bearer local-dev-key\r\n"
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        writer.write(request)
        await writer.drain()
        await asyncio.wait_for(fake.started.wait(), timeout=1)
        writer.close()
        await writer.wait_closed()
        writer = None

        await asyncio.wait_for(fake.cancelled.wait(), timeout=2)
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        fake.release.set()
        server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(server_task), timeout=3)
        except TimeoutError:
            server.force_exit = True
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
        listener.close()


def test_installed_coder_is_exposed_by_exact_id_and_stable_alias() -> None:
    class FakeCoderOllama(FakeOllama):
        async def tags(self) -> list[dict[str, Any]]:
            return [{"name": "qwen3-coder:30b-a3b-q4_K_M", "size": 19_000_000_000}]

    with TestClient(app) as client:
        client.app.state.ollama = FakeCoderOllama()
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer local-dev-key"},
        )

    assert response.status_code == 200
    ids = {model["id"] for model in response.json()["data"]}
    assert "qwen3-coder:30b-a3b-q4_K_M" in ids
    assert "localllm-code" in ids


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
    assert response.headers["x-content-type-options"] == "nosniff"
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


def test_openai_body_limit_preflights_with_openai_error_envelope() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            headers={
                "Authorization": "Bearer local-dev-key",
                "Content-Type": "application/json",
                "Content-Length": str(9 * 1024 * 1024),
            },
            content=b"{}",
        )

    assert response.status_code == 413
    assert response.json()["error"] == {
        "message": "Request body exceeds the endpoint size limit",
        "type": "invalid_request_error",
        "param": None,
        "code": "request_too_large",
    }


def test_management_body_limit_stops_chunked_payload_before_parsing() -> None:
    def chunks():
        yield b'{"model":"'
        yield b"x" * (9 * 1024)
        yield b'"}'

    with TestClient(app) as client:
        response = client.post(
            "/api/models/pull",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the endpoint size limit"}


@pytest.mark.parametrize(
    "path",
    [
        "/api/models/pull",
        "/api/re/mcp/investigate",
        "/api/re/triage",
    ],
)
def test_management_json_rejects_nonfinite_numbers_without_echoing_input(path: str) -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            path,
            content=b'{"model":NaN}',
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Request body must be valid JSON"}
    assert "NaN" not in response.text


@pytest.mark.parametrize(
    "path",
    [
        "/api/chat/completions",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/embeddings",
    ],
)
def test_generic_json_routes_reject_overflowing_floats(path: str) -> None:
    headers = {"Content-Type": "application/json"}
    if path.startswith("/v1/"):
        headers["Authorization"] = "Bearer local-dev-key"
    with TestClient(app, raise_server_exceptions=False) as client:
        client.app.state.ollama = FakeProxyOllama()
        response = client.post(path, content=b'{"temperature":1e9999}', headers=headers)

    assert response.status_code == 400
    assert "1e9999" not in response.text
