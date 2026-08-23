from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import HTTPException

from .catalog import resolve_model
from .config import Settings


@dataclass
class OllamaStream:
    """An upstream response whose connection stays open until iteration finishes."""

    response: httpx.Response
    client: httpx.AsyncClient

    async def aclose(self) -> None:
        try:
            await self.response.aclose()
        finally:
            await self.client.aclose()

    async def iter_raw(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.response.aiter_raw():
                yield chunk
        except httpx.HTTPError as exc:
            error = {
                "error": {
                    "message": f"Ollama stream interrupted: {exc}",
                    "type": "upstream_error",
                    "param": None,
                    "code": None,
                }
            }
            yield f"data: {json.dumps(error)}\n\n".encode()
        finally:
            await self.aclose()


@dataclass(frozen=True)
class OllamaProbe:
    """Sanitized dependency state for readiness and node discovery."""

    ok: bool
    models: tuple[str, ...] = ()
    error_code: str | None = None


class OllamaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.timeout = httpx.Timeout(600.0, connect=5.0)

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
                response = await client.get(f"{self.base_url}/api/version")
                response.raise_for_status()
                return {"ok": True, **response.json()}
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    async def probe(self) -> OllamaProbe:
        """Verify that Ollama's model catalog is reachable and structurally valid.

        A successful ``/api/tags`` response proves both runtime reachability and
        catalog availability in one bounded request. Cancellation deliberately
        propagates; the async context manager still closes its transport.
        """

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(2.0, connect=1.0), trust_env=False
            ) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                    raise ValueError("invalid Ollama model catalog")
                names: set[str] = set()
                for item in payload["models"]:
                    if not isinstance(item, dict):
                        raise ValueError("invalid Ollama model catalog entry")
                    name = item.get("name") or item.get("model")
                    if not isinstance(name, str) or not name:
                        raise ValueError("invalid Ollama model catalog entry")
                    names.add(name)
                return OllamaProbe(ok=True, models=tuple(sorted(names)))
        except (httpx.HTTPError, ValueError, TypeError):
            return OllamaProbe(ok=False, error_code="ollama_catalog_unavailable")

    async def tags(self) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                return response.json().get("models", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=503, detail=f"Ollama model catalog is unavailable: {exc}"
            ) from exc

    async def get_model(self, model: str) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
                encoded_model = quote(model, safe="")
                return await client.get(f"{self.base_url}/v1/models/{encoded_model}")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama is unavailable: {exc}") from exc

    async def proxy_json(self, endpoint: str, payload: dict[str, Any]) -> httpx.Response:
        payload = dict(payload)
        if "model" in payload:
            payload["model"] = resolve_model(str(payload["model"]))
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                return await client.post(f"{self.base_url}{endpoint}", json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama is unavailable: {exc}") from exc

    async def proxy_stream(self, endpoint: str, payload: dict[str, Any]) -> OllamaStream:
        """Open a streaming request and return after upstream response headers arrive.

        Performing this preflight before FastAPI creates its ``StreamingResponse`` lets
        the gateway preserve an upstream 4xx/5xx status and JSON error body.
        """

        payload = dict(payload)
        if "model" in payload:
            payload["model"] = resolve_model(str(payload["model"]))
        client = httpx.AsyncClient(timeout=self.timeout, trust_env=False)
        try:
            request = client.build_request("POST", f"{self.base_url}{endpoint}", json=payload)
            response = await client.send(request, stream=True)
            return OllamaStream(response=response, client=client)
        except httpx.HTTPError as exc:
            await client.aclose()
            raise HTTPException(status_code=503, detail=f"Ollama is unavailable: {exc}") from exc
        except BaseException:
            await client.aclose()
            raise

    async def pull(self, model: str) -> AsyncIterator[bytes]:
        payload = {"model": resolve_model(model), "stream": True}
        client = httpx.AsyncClient(timeout=None, trust_env=False)
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
