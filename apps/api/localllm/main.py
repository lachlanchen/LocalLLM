from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
import time
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from typing import Any, Literal, TypeVar

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.background import BackgroundTask
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .agent_runtime import router as agent_runtime_router
from .catalog import MODEL_ALIASES, MODEL_CATALOG, resolve_model
from .config import Settings, get_settings, prepare_private_data_dir
from .conversations import (
    MAX_SUMMARY_CHARS,
    ConversationCapacityError,
    ConversationCompactRequest,
    ConversationConflictError,
    ConversationCreate,
    ConversationDelete,
    ConversationStore,
    ConversationUpdate,
    deterministic_summary,
    harden_database_permissions,
    summary_prompt,
)
from .grounded_chat import router as grounded_chat_router
from .image_generation import router as image_generation_router
from .mcp_bridge import investigate_with_mcp, mcp_status
from .ollama import OllamaClient, OllamaStream
from .research import ResearchCapacityError, ResearchManager
from .reverse_engineering import (
    MAX_UPLOAD_REQUEST_SIZE,
    ai_triage,
    delete_inspection,
    inspect_upload,
    re_toolchain_status,
)
from .system import find_project_root, gpu_status, storage_status, tool_status


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ModelAction(StrictRequest):
    model: str = Field(min_length=1, max_length=200)


class ResearchRequest(StrictRequest):
    question: str = Field(min_length=8, max_length=4000)
    model: str = "localllm-deep"
    mode: Literal["web", "papers", "both"] = "both"
    depth: Literal["quick", "standard", "deep"] = "standard"


class SearchRequest(StrictRequest):
    query: str = Field(min_length=3, max_length=800)
    mode: Literal["web", "papers", "both"] = "both"
    limit: int = Field(default=12, ge=1, le=30)


class SearchSourceResponse(BaseModel):
    title: str
    url: str
    snippet: str
    provider: str
    providers: list[str]
    kind: str
    authors: list[str]
    year: int | None
    published_date: str | None
    doi: str | None
    citation_count: int | None
    score: float
    query: str
    provenance: list[dict[str, Any]]


class SearchProviderResponse(BaseModel):
    name: str
    kind: str
    ok: bool
    result_count: int
    duration_ms: int
    error: str | None = None
    queries: list[str]


class SearchResponse(BaseModel):
    query: str
    mode: Literal["web", "papers", "both"]
    sources: list[SearchSourceResponse]
    providers: list[SearchProviderResponse]
    warnings: list[str]


class TriageRequest(StrictRequest):
    metadata: dict[str, Any]
    model: str = Field(default="localllm-deep", min_length=1, max_length=200)


class McpInvestigationRequest(StrictRequest):
    binary_name: str = Field(min_length=1, max_length=500)
    question: str = Field(min_length=8, max_length=4000)
    model: str = "localllm-deep"


class OpenAIAuthenticationError(Exception):
    """Authentication failure that must retain an OpenAI-compatible body."""


class _RequestBodyTooLarge(BaseException):
    """Private receive-channel sentinel that inner exception middleware cannot consume."""


class _ClientDisconnected(Exception):
    """The downstream client disconnected while an upstream request was starting."""


RequestModel = TypeVar("RequestModel", bound=BaseModel)
UpstreamResult = TypeVar("UpstreamResult")
MAX_SEARCH_JSON_BYTES = 16 * 1024
MAX_RESEARCH_JSON_BYTES = 32 * 1024
MAX_OPENAI_JSON_BYTES = 25 * 1024 * 1024
REQUEST_BODY_LIMITS = {
    "/api/agent/chat": MAX_OPENAI_JSON_BYTES,
    "/api/agent/code/confirmations": 20 * 1024,
    "/api/agent/code/executions": 40 * 1024,
    "/api/agent/plans/propose": 20 * 1024,
    "/api/agent/plans/validate": 20 * 1024,
    "/api/chat/completions": MAX_OPENAI_JSON_BYTES,
    "/api/images/jobs": 8 * 1024,
    "/api/models/pull": 8 * 1024,
    "/api/re/inspect": MAX_UPLOAD_REQUEST_SIZE,
    "/api/re/mcp/investigate": 32 * 1024,
    "/api/re/triage": 4 * 1024 * 1024,
    "/api/research": MAX_RESEARCH_JSON_BYTES,
    "/api/search": MAX_SEARCH_JSON_BYTES,
    "/v1/chat/completions": MAX_OPENAI_JSON_BYTES,
    "/v1/embeddings": 8 * 1024 * 1024,
    "/v1/responses": MAX_OPENAI_JSON_BYTES,
}
_CONVERSATION_MUTATION_PATH = re.compile(r"^/api/conversations(?:/conv_[0-9a-f]{32})?$")
_CONVERSATION_COMPACT_PATH = re.compile(r"^/api/conversations/conv_[0-9a-f]{32}/compact$")


def _request_body_limit(path: str) -> int | None:
    exact = REQUEST_BODY_LIMITS.get(path)
    if exact is not None:
        return exact
    if _CONVERSATION_MUTATION_PATH.fullmatch(path):
        return MAX_OPENAI_JSON_BYTES
    if _CONVERSATION_COMPACT_PATH.fullmatch(path):
        return 8 * 1024
    return None


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._body_slots = asyncio.Semaphore(4)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, status: int) -> None:
        path = str(scope.get("path", ""))
        if path.startswith("/v1/"):
            response = JSONResponse(
                status_code=status,
                content={
                    "error": {
                        "message": "Request body exceeds the endpoint size limit",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "request_too_large",
                    }
                },
            )
        else:
            response = JSONResponse(
                status_code=status,
                content={"detail": "Request body exceeds the endpoint size limit"},
            )
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return
        limit = _request_body_limit(str(scope.get("path", "")))
        if limit is None:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                declared_length = int(declared)
            except ValueError:
                response = JSONResponse(
                    status_code=400, content={"detail": "Invalid Content-Length"}
                )
                await response(scope, receive, send)
                return
            if declared_length < 0:
                response = JSONResponse(
                    status_code=400, content={"detail": "Invalid Content-Length"}
                )
                await response(scope, receive, send)
                return
            if declared_length > limit:
                await self._reject(scope, receive, send, 413)
                return

        received = 0
        slot_acquired = False
        slot_released = False
        response_started = False

        def release_slot() -> None:
            nonlocal slot_released
            if slot_acquired and not slot_released:
                self._body_slots.release()
                slot_released = True

        async def limited_receive() -> Message:
            nonlocal received, slot_acquired
            if not slot_acquired:
                await self._body_slots.acquire()
                slot_acquired = True
            message = await receive()
            if message.get("type") == "http.disconnect":
                release_slot()
                return message
            if message.get("type") != "http.request":
                return message
            chunk = message.get("body", b"")
            if received + len(chunk) > limit:
                release_slot()
                raise _RequestBodyTooLarge
            received += len(chunk)
            if not message.get("more_body", False):
                release_slot()
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLarge:
            if not response_started:
                await self._reject(scope, receive, send, 413)
        finally:
            release_slot()


def _bounded_json_integer(value: str) -> int:
    if len(value) > 256:
        raise ValueError("JSON integer is too long")
    return int(value)


def _bounded_json_float(value: str) -> float:
    if len(value) > 256:
        raise ValueError("JSON number is too long")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite JSON numbers are not allowed")
    return parsed


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is not allowed")


def _json_structure_is_bounded(value: object, max_depth: int = 100) -> bool:
    """Bound container depth consistently across supported Python parsers."""

    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if depth > max_depth or nodes > 100_000:
            return False
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return True


async def _bounded_json_object(request: Request, max_bytes: int) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(status_code=413, detail="JSON request exceeds the size limit")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail="JSON request exceeds the size limit")
        body.extend(chunk)
    try:
        decoded = json.loads(
            body,
            parse_int=_bounded_json_integer,
            parse_float=_bounded_json_float,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")
    if not _json_structure_is_bounded(decoded):
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")
    return decoded


async def _bounded_json_model(
    request: Request,
    model_type: type[RequestModel],
    max_bytes: int,
) -> RequestModel:
    decoded = await _bounded_json_object(request, max_bytes)
    try:
        return model_type.model_validate(decoded)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status_code=422, detail=errors[:20]) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    prepare_private_data_dir(settings.data_dir)
    app.state.settings = settings
    app.state.ollama = OllamaClient(settings)
    app.state.research = ResearchManager(settings)
    app.state.conversations = ConversationStore(settings.data_dir)
    try:
        yield
    finally:
        await app.state.research.shutdown()


app = FastAPI(
    title="LocalLLM Studio API",
    version="0.1.0",
    description="Private model control plane with OpenAI-compatible endpoints.",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.add_middleware(RequestBodyLimitMiddleware)
settings = get_settings()
loopback_origins = [
    f"http://127.0.0.1:{settings.port}",
    f"http://localhost:{settings.port}",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]
allowed_origins = list(dict.fromkeys([*settings.allowed_origins, *loopback_origins]))
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-LocalLLM-Key"],
)


@app.middleware("http")
async def browser_security_boundary(request: Request, call_next):
    """Reject cross-site browser mutations and add local-app security headers."""
    peer = request.client.host if request.client else ""
    if peer and peer != "testclient":
        try:
            if not ipaddress.ip_address(peer).is_loopback:
                return JSONResponse(status_code=403, content={"detail": "Loopback access only"})
        except ValueError:
            return JSONResponse(status_code=403, content={"detail": "Loopback access only"})
    api_request = request.url.path.startswith(("/api/", "/v1/"))
    if api_request or request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        fetch_site = request.headers.get("sec-fetch-site", "").lower()
        if (origin and origin not in allowed_origins) or fetch_site == "cross-site":
            return JSONResponse(status_code=403, content={"detail": "Cross-site request blocked"})
    if request.method == "POST" and request.url.path == "/api/re/inspect":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_UPLOAD_REQUEST_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Upload request exceeds the binary inspection limit"},
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; connect-src 'self'; font-src 'self' data:; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'"
    )
    return response


app.include_router(grounded_chat_router)
app.include_router(agent_runtime_router)
app.include_router(image_generation_router)


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


def get_research(request: Request) -> ResearchManager:
    return request.app.state.research


def get_conversations(request: Request) -> ConversationStore:
    return request.app.state.conversations


def require_api_key(
    authorization: str | None = Header(default=None),
    x_localllm_key: str | None = Header(default=None),
    current: Settings = Depends(get_settings),
) -> None:
    if not current.api_key:
        return
    bearer = None
    if authorization:
        scheme, separator, credential = authorization.partition(" ")
        if separator and scheme.lower() == "bearer":
            bearer = credential
    if bearer != current.api_key and x_localllm_key != current.api_key:
        raise OpenAIAuthenticationError


def _passthrough(response: httpx.Response) -> Response:
    media_type = response.headers.get("content-type", "application/json").split(";")[0]
    try:
        content = response.json()
        return JSONResponse(content=content, status_code=response.status_code)
    except json.JSONDecodeError:
        return Response(
            content=response.content, status_code=response.status_code, media_type=media_type
        )


def _openai_error(status_code: int, message: str, error_type: str = "upstream_error") -> Response:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(OpenAIAuthenticationError)
async def openai_authentication_error(
    _request: Request, _exc: OpenAIAuthenticationError
) -> Response:
    response = _openai_error(401, "Invalid LocalLLM API key", "authentication_error")
    response.headers["WWW-Authenticate"] = "Bearer"
    return response


@app.exception_handler(RequestValidationError)
async def sanitized_request_validation_error(
    _request: Request, exc: RequestValidationError
) -> Response:
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    return JSONResponse(status_code=422, content={"detail": errors[:20]})


def _openai_upstream_error(response: httpx.Response) -> Response:
    """Normalize an upstream failure without encoding its JSON body as a string."""

    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _openai_error(response.status_code, response.text or "Ollama request failed")

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            normalized = dict(error)
            normalized.setdefault("message", "Ollama request failed")
            normalized.setdefault("type", "upstream_error")
            normalized.setdefault("param", None)
            normalized.setdefault("code", None)
            return JSONResponse(status_code=response.status_code, content={"error": normalized})
        if isinstance(error, str):
            return _openai_error(response.status_code, error)
        detail = payload.get("detail")
        if isinstance(detail, str):
            return _openai_error(response.status_code, detail)
        message = payload.get("message")
        if isinstance(message, str):
            return _openai_error(response.status_code, message)

    return _openai_error(response.status_code, "Ollama request failed")


async def _streaming_passthrough(stream: OllamaStream) -> Response:
    response = stream.response
    if response.is_error:
        try:
            await response.aread()
            return _openai_upstream_error(response)
        except httpx.HTTPError as exc:
            return _openai_error(response.status_code, f"Could not read Ollama error: {exc}")
        finally:
            await stream.aclose()

    media_type = response.headers.get("content-type", "text/event-stream").split(";")[0]
    return StreamingResponse(
        stream.iter_raw(),
        status_code=response.status_code,
        media_type=media_type,
        background=BackgroundTask(stream.aclose),
    )


@app.get("/healthz")
async def health(ollama: OllamaClient = Depends(get_ollama)) -> dict[str, Any]:
    state = await ollama.health()
    return {"ok": True, "service": "localllm-api", "ollama": state}


@app.get("/api/system/status")
async def system_status(
    current: Settings = Depends(get_settings), ollama: OllamaClient = Depends(get_ollama)
) -> dict[str, Any]:
    root = find_project_root()
    gpu, ollama_health, python, node, docker = await __import__("asyncio").gather(
        gpu_status(),
        ollama.health(),
        tool_status("python3", ("--version",)),
        tool_status("node", ("--version",)),
        tool_status("docker", ("--version",)),
    )
    return {
        "service": {"ok": True, "version": app.version, "time": int(time.time())},
        "gpu": gpu,
        "ollama": ollama_health,
        "storage": storage_status(root),
        "runtime": {"python": python, "node": node, "docker": docker},
        "binding": {
            "host": current.host,
            "port": current.port,
            "local_only": current.host in {"127.0.0.1", "localhost"},
        },
    }


@app.get("/api/models/catalog")
async def model_catalog(ollama: OllamaClient = Depends(get_ollama)) -> dict[str, Any]:
    try:
        installed_raw = await ollama.tags()
        ollama_state: dict[str, Any] = {"ok": True}
    except HTTPException as exc:
        installed_raw = []
        ollama_state = {"ok": False, "error": str(exc.detail)}
    installed = {item.get("name") or item.get("model") for item in installed_raw}
    models = [{**model, "installed": model["id"] in installed} for model in MODEL_CATALOG]
    return {
        "models": models,
        "installed": installed_raw,
        "aliases": MODEL_ALIASES,
        "ollama": ollama_state,
        "planned_download_gb": round(sum(model["size_gb"] for model in MODEL_CATALOG), 1),
    }


@app.post("/api/models/pull")
async def pull_model(
    request: Request, ollama: OllamaClient = Depends(get_ollama)
) -> StreamingResponse:
    action = await _bounded_json_model(request, ModelAction, 8 * 1024)
    allowed = {model["id"] for model in MODEL_CATALOG}
    resolved = resolve_model(action.model)
    if resolved not in allowed:
        raise HTTPException(status_code=400, detail="Model is not in the curated catalog")
    return StreamingResponse(ollama.pull(resolved), media_type="text/event-stream")


@app.post("/api/chat/completions")
async def app_chat(request: Request, ollama: OllamaClient = Depends(get_ollama)) -> Response:
    payload = await _bounded_json_object(request, MAX_OPENAI_JSON_BYTES)
    return await _proxy_openai(request, "/v1/chat/completions", payload, ollama)


@app.get("/api/conversations")
async def list_conversations(
    store: ConversationStore = Depends(get_conversations),
) -> dict[str, Any]:
    return await asyncio.to_thread(store.list)


@app.post("/api/conversations", status_code=201)
async def create_conversation(
    request: Request,
    store: ConversationStore = Depends(get_conversations),
) -> dict[str, Any]:
    payload = await _bounded_json_model(request, ConversationCreate, MAX_OPENAI_JSON_BYTES)
    try:
        conversation = await asyncio.to_thread(store.create, payload)
    except ConversationCapacityError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    harden_database_permissions(store)
    return conversation


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversations),
) -> dict[str, Any]:
    conversation = await asyncio.to_thread(store.get, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.patch("/api/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: Request,
    store: ConversationStore = Depends(get_conversations),
) -> dict[str, Any]:
    payload = await _bounded_json_model(request, ConversationUpdate, MAX_OPENAI_JSON_BYTES)
    try:
        conversation = await asyncio.to_thread(store.update, conversation_id, payload)
    except ConversationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConversationCapacityError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    harden_database_permissions(store)
    return conversation


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    store: ConversationStore = Depends(get_conversations),
) -> dict[str, Any]:
    payload = await _bounded_json_model(request, ConversationDelete, 8 * 1024)
    try:
        deleted = await asyncio.to_thread(
            store.delete,
            conversation_id,
            expected_revision=payload.expected_revision,
        )
    except ConversationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    harden_database_permissions(store)
    return {"deleted": True, "id": conversation_id}


@app.post("/api/conversations/{conversation_id}/compact")
async def compact_conversation(
    conversation_id: str,
    request: Request,
    store: ConversationStore = Depends(get_conversations),
    ollama: OllamaClient = Depends(get_ollama),
) -> dict[str, Any]:
    payload = await _bounded_json_model(request, ConversationCompactRequest, 8 * 1024)
    conversation = await asyncio.to_thread(store.get, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    target_count = max(0, len(conversation["messages"]) - payload.keep_recent)
    cursor = min(int(conversation["summarized_message_count"]), target_count)
    if target_count <= cursor:
        return {
            "conversation": conversation,
            "compacted": False,
            "summary_method": conversation["summary_method"],
        }

    messages_to_merge = conversation["messages"][cursor:target_count]
    summary = deterministic_summary(conversation["summary"], messages_to_merge)
    method = "extractive"
    try:
        response = await asyncio.wait_for(
            ollama.proxy_json(
                "/api/chat",
                {
                    "model": payload.model or conversation["model"],
                    "messages": summary_prompt(conversation["summary"], messages_to_merge),
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 32_768,
                        "num_predict": 2_048,
                    },
                },
            ),
            timeout=60.0,
        )
        if response.status_code >= 400:
            raise RuntimeError("The local summary model rejected the request")
        response_payload = response.json()
        message_payload = (
            response_payload.get("message") if isinstance(response_payload, dict) else None
        )
        candidate = message_payload.get("content", "") if isinstance(message_payload, dict) else ""
        if not isinstance(candidate, str) or not candidate.strip():
            raise RuntimeError("The local summary model returned no visible summary")
        candidate = "".join(
            character if character in "\n\t" or ord(character) >= 32 else " "
            for character in candidate
        ).strip()
        if not candidate:
            raise RuntimeError("The local summary model returned no safe summary text")
        summary = candidate[:MAX_SUMMARY_CHARS].rstrip()
        method = "model"
    except (HTTPException, httpx.HTTPError, TimeoutError, ValueError, RuntimeError):
        # Compaction remains available during model outages. The extractive summary
        # labels itself as a fallback and samples every compacted turn.
        pass

    try:
        updated = await asyncio.to_thread(
            store.apply_summary,
            conversation_id,
            expected_revision=conversation["revision"],
            summary=summary,
            summarized_message_count=target_count,
            method=method,
        )
    except ConversationConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConversationCapacityError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    harden_database_permissions(store)
    return {"conversation": updated, "compacted": True, "summary_method": method}


@app.get("/api/search/status")
async def search_status(manager: ResearchManager = Depends(get_research)) -> dict[str, Any]:
    """Describe search capabilities without exposing provider credentials."""

    return manager.provider_status()


@app.post("/api/search", response_model=SearchResponse)
async def quick_search(
    request: Request, manager: ResearchManager = Depends(get_research)
) -> dict[str, Any]:
    payload = await _bounded_json_model(request, SearchRequest, MAX_SEARCH_JSON_BYTES)
    outcome = await manager.quick_search(payload.query, payload.mode, payload.limit)
    return outcome.public_dict()


@app.post("/api/research")
async def create_research(
    request: Request, manager: ResearchManager = Depends(get_research)
) -> dict[str, Any]:
    payload = await _bounded_json_model(request, ResearchRequest, MAX_RESEARCH_JSON_BYTES)
    try:
        return manager.serialize(
            manager.create(
                payload.question,
                payload.model,
                mode=payload.mode,
                depth=payload.depth,
            )
        )
    except ResearchCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.get("/api/research/{task_id}")
async def get_research_task(
    task_id: str, manager: ResearchManager = Depends(get_research)
) -> dict[str, Any]:
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Research task not found")
    return manager.serialize(task)


@app.delete("/api/research/{task_id}")
async def cancel_research_task(
    task_id: str, manager: ResearchManager = Depends(get_research)
) -> dict[str, Any]:
    task = await manager.cancel(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Research task not found")
    return manager.serialize(task)


@app.get("/api/re/toolchain")
async def reverse_toolchain(current: Settings = Depends(get_settings)) -> dict[str, Any]:
    return await re_toolchain_status(current)


@app.post("/api/re/inspect")
async def reverse_inspect(
    binary: UploadFile = File(...), current: Settings = Depends(get_settings)
) -> dict[str, Any]:
    return await inspect_upload(binary, current)


@app.delete("/api/re/inspect/{artifact_id}")
async def reverse_delete_inspection(
    artifact_id: str, current: Settings = Depends(get_settings)
) -> dict[str, Any]:
    return await delete_inspection(artifact_id, current)


@app.post("/api/re/triage")
async def reverse_triage(
    request: Request, current: Settings = Depends(get_settings)
) -> dict[str, str]:
    payload = await _bounded_json_model(request, TriageRequest, 4 * 1024 * 1024)
    return {"analysis": await ai_triage(payload.metadata, payload.model, current)}


@app.get("/api/re/mcp")
async def reverse_mcp_status(current: Settings = Depends(get_settings)) -> dict[str, Any]:
    return await mcp_status(current)


@app.post("/api/re/mcp/investigate")
async def reverse_mcp_investigate(
    request: Request, current: Settings = Depends(get_settings)
) -> dict[str, Any]:
    payload = await _bounded_json_model(request, McpInvestigationRequest, 32 * 1024)
    return await investigate_with_mcp(
        payload.binary_name,
        payload.question,
        payload.model,
        current,
    )


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models(ollama: OllamaClient = Depends(get_ollama)) -> Response:
    try:
        installed = await ollama.tags()
    except HTTPException as exc:
        return _openai_error(exc.status_code, str(exc.detail), "service_unavailable")
    names = {item.get("name") or item.get("model") for item in installed}
    data = [
        {
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }
        for name in sorted(name for name in names if name)
    ]
    for alias, target in MODEL_ALIASES.items():
        if target in names:
            data.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "localllm",
                }
            )
    return JSONResponse(content={"object": "list", "data": data})


@app.get("/v1/models/{model:path}", dependencies=[Depends(require_api_key)])
async def retrieve_model(model: str, ollama: OllamaClient = Depends(get_ollama)) -> Response:
    resolved = resolve_model(model)
    try:
        installed = await ollama.tags()
    except HTTPException as exc:
        return _openai_error(exc.status_code, str(exc.detail), "service_unavailable")
    names = {item.get("name") or item.get("model") for item in installed}
    if resolved not in names:
        return _openai_error(404, f"Model {model!r} was not found", "invalid_request_error")
    response = await ollama.get_model(resolved)
    return _passthrough(response)


async def _cancel_and_wait(task: asyncio.Future[Any]) -> None:
    """Cancel a losing task, consume its result, and close a raced stream if necessary."""

    task.cancel()
    try:
        result = await task
    except asyncio.CancelledError:
        return
    except Exception:
        return
    if isinstance(result, OllamaStream):
        try:
            await result.aclose()
        except Exception:
            pass


async def _await_upstream_or_disconnect(
    request: Request, operation: Awaitable[UpstreamResult]
) -> UpstreamResult:
    upstream_task = asyncio.ensure_future(operation)
    try:
        while True:
            if await request.is_disconnected():
                await _cancel_and_wait(upstream_task)
                raise _ClientDisconnected
            done, _pending = await asyncio.wait({upstream_task}, timeout=0.05)
            if upstream_task in done:
                return await upstream_task
    except BaseException:
        if not upstream_task.done():
            await _cancel_and_wait(upstream_task)
        raise


async def _proxy_openai(
    request: Request, endpoint: str, payload: dict[str, Any], ollama: OllamaClient
) -> Response:
    try:
        if payload.get("stream"):
            stream = await _await_upstream_or_disconnect(
                request, ollama.proxy_stream(endpoint, payload)
            )
            return await _streaming_passthrough(stream)
        response = await _await_upstream_or_disconnect(
            request, ollama.proxy_json(endpoint, payload)
        )
        return _passthrough(response)
    except _ClientDisconnected:
        return _openai_error(499, "Client closed request", "request_cancelled")
    except HTTPException as exc:
        return _openai_error(exc.status_code, str(exc.detail), "service_unavailable")


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(
    request: Request, ollama: OllamaClient = Depends(get_ollama)
) -> Response:
    payload = await _bounded_json_object(request, MAX_OPENAI_JSON_BYTES)
    return await _proxy_openai(request, "/v1/chat/completions", payload, ollama)


@app.post("/v1/responses", dependencies=[Depends(require_api_key)])
async def responses(request: Request, ollama: OllamaClient = Depends(get_ollama)) -> Response:
    payload = await _bounded_json_object(request, MAX_OPENAI_JSON_BYTES)
    return await _proxy_openai(request, "/v1/responses", payload, ollama)


@app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
async def embeddings(request: Request, ollama: OllamaClient = Depends(get_ollama)) -> Response:
    payload = await _bounded_json_object(request, 8 * 1024 * 1024)
    return await _proxy_openai(request, "/v1/embeddings", payload, ollama)


project_root = find_project_root()
references_dir = project_root / "references"
if references_dir.exists():
    app.mount("/references", StaticFiles(directory=references_dir), name="references")
web_dist = project_root / "apps" / "web" / "dist"
if web_dist.exists():
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
