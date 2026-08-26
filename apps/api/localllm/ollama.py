from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import HTTPException

from .catalog import resolve_model
from .config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass
class OllamaStream:
    """An upstream response whose connection stays open until iteration finishes."""

    response: httpx.Response
    request_id: str

    async def aclose(self) -> None:
        await self.response.aclose()

    async def iter_raw(self) -> AsyncIterator[bytes]:
        emitted_output = False
        try:
            async for chunk in self.response.aiter_raw():
                if chunk:
                    emitted_output = True
                yield chunk
        except httpx.HTTPError as exc:
            if emitted_output:
                code = "ollama_stream_interrupted_after_output"
                message = (
                    "The local model stream was interrupted after output began; "
                    "the partial response was not retried."
                )
            else:
                code = "ollama_stream_interrupted_before_output"
                message = "The local model stream was interrupted before output began."
            LOGGER.warning(
                "Ollama stream transport failed request_id=%s code=%s error_type=%s",
                self.request_id,
                code,
                type(exc).__name__,
            )
            error = {
                "error": {
                    "message": message,
                    "type": "upstream_error",
                    "param": None,
                    "code": code,
                    "request_id": self.request_id,
                }
            }
            yield f"data: {json.dumps(error)}\n\n".encode()
        finally:
            await self.aclose()


@dataclass(frozen=True)
class OllamaModelMetadata:
    """Sanitized installed-model provenance retained for node discovery."""

    id: str
    digest: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class OllamaProbe:
    """Sanitized dependency state for readiness and node discovery."""

    ok: bool
    models: tuple[str, ...] = ()
    model_metadata: tuple[OllamaModelMetadata, ...] = ()
    error_code: str | None = None


class OllamaTransportError(HTTPException):
    """Sanitized transient transport failure with an operator-correlatable ID."""

    def __init__(self, *, request_id: str, error_code: str = "ollama_upstream_unavailable"):
        super().__init__(
            status_code=503,
            detail="The local model runtime is temporarily unavailable.",
        )
        self.request_id = request_id
        self.error_code = error_code


class OllamaClient:
    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.timeout = httpx.Timeout(600.0, connect=5.0)
        self._client = client or httpx.AsyncClient(timeout=self.timeout, trust_env=False)
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the shared transport when this client owns its lifecycle."""

        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _request_id() -> str:
        return f"ollama_{uuid.uuid4().hex}"

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get(f"{self.base_url}/api/version", timeout=3.0)
            response.raise_for_status()
            return {"ok": True, **response.json()}
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    async def probe(self) -> OllamaProbe:
        """Verify that Ollama's model catalog is reachable and structurally valid.

        A successful ``/api/tags`` response proves both runtime reachability and
        catalog availability in one bounded request. Cancellation deliberately
        propagates without closing the process-wide pooled transport.
        """

        try:
            response = await self._client.get(
                f"{self.base_url}/api/tags",
                timeout=httpx.Timeout(2.0, connect=1.0),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                raise ValueError("invalid Ollama model catalog")
            names: set[str] = set()
            metadata: dict[str, OllamaModelMetadata] = {}
            for item in payload["models"]:
                if not isinstance(item, dict):
                    raise ValueError("invalid Ollama model catalog entry")
                name = item.get("name") or item.get("model")
                if not isinstance(name, str) or not name:
                    raise ValueError("invalid Ollama model catalog entry")
                if name in names:
                    raise ValueError("duplicate Ollama model catalog entry")
                names.add(name)
                digest = item.get("digest")
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    digest = None
                size = item.get("size")
                if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                    size = None
                metadata[name] = OllamaModelMetadata(
                    id=name,
                    digest=digest,
                    size_bytes=size,
                )
            ordered_names = tuple(sorted(names))
            return OllamaProbe(
                ok=True,
                models=ordered_names,
                model_metadata=tuple(metadata[name] for name in ordered_names),
            )
        except (httpx.HTTPError, ValueError, TypeError):
            return OllamaProbe(ok=False, error_code="ollama_catalog_unavailable")

    async def tags(self) -> list[dict[str, Any]]:
        try:
            response = await self._client.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            return response.json().get("models", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=503, detail=f"Ollama model catalog is unavailable: {exc}"
            ) from exc

    async def get_model(self, model: str) -> httpx.Response:
        try:
            encoded_model = quote(model, safe="")
            return await self._client.get(
                f"{self.base_url}/v1/models/{encoded_model}", timeout=10.0
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama is unavailable: {exc}") from exc

    async def proxy_json(self, endpoint: str, payload: dict[str, Any]) -> httpx.Response:
        payload = dict(payload)
        if "model" in payload:
            payload["model"] = resolve_model(str(payload["model"]))
        request_id = self._request_id()
        try:
            # A request is attempted exactly once. In particular, callers must never
            # replay generation after an upstream may have produced hidden output.
            return await self._client.post(f"{self.base_url}{endpoint}", json=payload)
        except httpx.HTTPError as exc:
            LOGGER.warning(
                "Ollama request transport failed request_id=%s stage=before_output "
                "error_type=%s",
                request_id,
                type(exc).__name__,
            )
            raise OllamaTransportError(request_id=request_id) from exc

    async def proxy_stream(self, endpoint: str, payload: dict[str, Any]) -> OllamaStream:
        """Open a streaming request and return after upstream response headers arrive.

        Performing this preflight before FastAPI creates its ``StreamingResponse`` lets
        the gateway preserve an upstream 4xx/5xx status and JSON error body.
        """

        payload = dict(payload)
        if "model" in payload:
            payload["model"] = resolve_model(str(payload["model"]))
        request_id = self._request_id()
        try:
            # ``send`` is deliberately single-shot: a failed or partial generation is
            # classified for the caller and never replayed automatically.
            request = self._client.build_request(
                "POST", f"{self.base_url}{endpoint}", json=payload
            )
            response = await self._client.send(request, stream=True)
            return OllamaStream(response=response, request_id=request_id)
        except httpx.HTTPError as exc:
            LOGGER.warning(
                "Ollama stream preflight failed request_id=%s stage=before_output "
                "error_type=%s",
                request_id,
                type(exc).__name__,
            )
            raise OllamaTransportError(request_id=request_id) from exc

    async def pull(self, model: str) -> AsyncIterator[bytes]:
        payload = {"model": resolve_model(model), "stream": True}
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/api/pull", json=payload, timeout=None
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        yield f"data: {line}\n\n".encode()
            yield b'data: {"status":"complete"}\n\n'
        except httpx.HTTPError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
