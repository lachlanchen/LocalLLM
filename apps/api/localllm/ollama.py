from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException

from .catalog import resolve_model
from .config import Settings


class OllamaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.timeout = httpx.Timeout(600.0, connect=5.0)

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.base_url}/api/version")
                response.raise_for_status()
                return {"ok": True, **response.json()}
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    async def tags(self) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                return response.json().get("models", [])
        except httpx.HTTPError:
            return []

    async def get_json(self, endpoint: str) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                return await client.get(f"{self.base_url}{endpoint}")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama is unavailable: {exc}") from exc

    async def proxy_json(self, endpoint: str, payload: dict[str, Any]) -> httpx.Response:
        payload = dict(payload)
        if "model" in payload:
            payload["model"] = resolve_model(str(payload["model"]))
        try:
            client = httpx.AsyncClient(timeout=self.timeout)
            request = client.build_request("POST", f"{self.base_url}{endpoint}", json=payload)
            response = await client.send(request, stream=False)
            await client.aclose()
            return response
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama is unavailable: {exc}") from exc

    async def proxy_stream(self, endpoint: str, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        payload = dict(payload)
        if "model" in payload:
            payload["model"] = resolve_model(str(payload["model"]))
        client = httpx.AsyncClient(timeout=self.timeout)
        try:
            async with client.stream(
                "POST", f"{self.base_url}{endpoint}", json=payload
            ) as response:
                if response.is_error:
                    body = await response.aread()
                    message = body.decode(errors="replace")
                    yield f"data: {json.dumps({'error': message})}\n\n".encode()
                    return
                async for chunk in response.aiter_raw():
                    yield chunk
        except httpx.HTTPError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
        finally:
            await client.aclose()

    async def pull(self, model: str) -> AsyncIterator[bytes]:
        payload = {"model": resolve_model(model), "stream": True}
        client = httpx.AsyncClient(timeout=None)
        try:
            async with client.stream("POST", f"{self.base_url}/api/pull", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield f"data: {line}\n\n".encode()
            yield b'data: {"status":"complete"}\n\n'
        except httpx.HTTPError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
        finally:
            await client.aclose()
