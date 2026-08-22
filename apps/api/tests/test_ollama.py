from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest
from fastapi import HTTPException

from localllm.config import Settings
from localllm.ollama import OllamaClient


def install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.AsyncClient]:
    real_client = httpx.AsyncClient
    clients: list[httpx.AsyncClient] = []

    def client_factory(*args, **kwargs):
        client = real_client(*args, transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("localllm.ollama.httpx.AsyncClient", client_factory)
    return clients


@pytest.mark.asyncio
async def test_get_model_percent_encodes_the_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_path = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_path
        seen_path = request.url.raw_path.decode()
        return httpx.Response(200, json={"id": "model"})

    install_mock_transport(monkeypatch, handler)
    ollama = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))

    response = await ollama.get_model("team/model:tag")

    assert response.status_code == 200
    assert seen_path == "/v1/models/team%2Fmodel%3Atag"


@pytest.mark.asyncio
async def test_tags_surfaces_ollama_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    clients = install_mock_transport(monkeypatch, handler)
    ollama = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))

    with pytest.raises(HTTPException) as exc_info:
        await ollama.tags()

    assert exc_info.value.status_code == 503
    assert "model catalog is unavailable" in str(exc_info.value.detail)
    assert len(clients) == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
async def test_proxy_json_closes_client_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    clients = install_mock_transport(monkeypatch, handler)
    ollama = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))

    response = await ollama.proxy_json(
        "/v1/chat/completions", {"model": "localllm-fast", "messages": []}
    )

    assert response.json() == {"ok": True}
    assert seen_payload["model"] == "qwen3:8b-q4_K_M"
    assert len(clients) == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
async def test_proxy_json_resolves_stable_coder_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    clients = install_mock_transport(monkeypatch, handler)
    ollama = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))

    await ollama.proxy_json(
        "/v1/chat/completions", {"model": "localllm-code", "messages": []}
    )

    assert seen_payload["model"] == "qwen3-coder:30b-a3b-q4_K_M"
    assert len(clients) == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
async def test_proxy_json_closes_client_after_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    clients = install_mock_transport(monkeypatch, handler)
    ollama = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))

    with pytest.raises(HTTPException) as exc_info:
        await ollama.proxy_json("/v1/responses", {"model": "localllm-deep"})

    assert exc_info.value.status_code == 503
    assert len(clients) == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
async def test_proxy_stream_returns_upstream_headers_before_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    clients = install_mock_transport(monkeypatch, handler)
    ollama = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))

    stream = await ollama.proxy_stream("/v1/chat/completions", {"model": "missing", "stream": True})

    assert stream.response.status_code == 404
    assert not clients[0].is_closed
    assert await stream.response.aread() == b'{"error":"model not found"}'
    await stream.aclose()
    assert clients[0].is_closed


@pytest.mark.asyncio
async def test_proxy_stream_closes_client_after_preflight_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    clients = install_mock_transport(monkeypatch, handler)
    ollama = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))

    with pytest.raises(HTTPException) as exc_info:
        await ollama.proxy_stream("/v1/chat/completions", {"model": "missing", "stream": True})

    assert exc_info.value.status_code == 503
    assert len(clients) == 1
    assert clients[0].is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_proxy_closes_client_when_caller_cancels_preflight(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    started = asyncio.Event()
    blocker = asyncio.Event()
    clients: list[httpx.AsyncClient] = []
    real_client = httpx.AsyncClient

    class BlockingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            started.set()
            await blocker.wait()
            return httpx.Response(200, json={"ok": True})

    def client_factory(*args, **kwargs):
        client = real_client(*args, transport=BlockingTransport(), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("localllm.ollama.httpx.AsyncClient", client_factory)
    ollama = OllamaClient(Settings(ollama_base_url="http://127.0.0.1:11434"))
    payload = {"model": "localllm-fast", "messages": [], "stream": stream}
    if stream:
        operation = ollama.proxy_stream("/v1/chat/completions", payload)
    else:
        operation = ollama.proxy_json("/v1/chat/completions", payload)
    task = asyncio.create_task(operation)

    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(clients) == 1
    assert clients[0].is_closed
