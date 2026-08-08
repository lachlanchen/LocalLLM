from __future__ import annotations

import ipaddress
import json
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .catalog import MODEL_ALIASES, MODEL_CATALOG, resolve_model
from .config import Settings, get_settings
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


class ModelAction(BaseModel):
    model: str


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8, max_length=4000)
    model: str = "localllm-deep"


class TriageRequest(BaseModel):
    metadata: dict[str, Any]
    model: str = "localllm-deep"


class McpInvestigationRequest(BaseModel):
    binary_name: str = Field(min_length=1, max_length=500)
    question: str = Field(min_length=8, max_length=4000)
    model: str = "localllm-deep"


class OpenAIAuthenticationError(Exception):
    """Authentication failure that must retain an OpenAI-compatible body."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    settings.data_dir.chmod(0o700)
    app.state.settings = settings
    app.state.ollama = OllamaClient(settings)
    app.state.research = ResearchManager(settings)
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
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
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
        "default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; base-uri 'none'; "
        "frame-ancestors 'none'"
    )
    return response


def get_ollama(request: Request) -> OllamaClient:
    return request.app.state.ollama


def get_research(request: Request) -> ResearchManager:
    return request.app.state.research


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
            return JSONResponse(
                status_code=response.status_code, content={"error": normalized}
            )
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
    action: ModelAction, ollama: OllamaClient = Depends(get_ollama)
) -> StreamingResponse:
    allowed = {model["id"] for model in MODEL_CATALOG}
    resolved = resolve_model(action.model)
    if resolved not in allowed:
        raise HTTPException(status_code=400, detail="Model is not in the curated catalog")
    return StreamingResponse(ollama.pull(resolved), media_type="text/event-stream")


@app.post("/api/chat/completions")
async def app_chat(
    payload: dict[str, Any] = Body(...), ollama: OllamaClient = Depends(get_ollama)
) -> Response:
    return await _proxy_openai("/v1/chat/completions", payload, ollama)


@app.post("/api/research")
async def create_research(
    payload: ResearchRequest, manager: ResearchManager = Depends(get_research)
) -> dict[str, Any]:
    try:
        return manager.serialize(manager.create(payload.question, payload.model))
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
    payload: TriageRequest, current: Settings = Depends(get_settings)
) -> dict[str, str]:
    return {"analysis": await ai_triage(payload.metadata, payload.model, current)}


@app.get("/api/re/mcp")
async def reverse_mcp_status(current: Settings = Depends(get_settings)) -> dict[str, Any]:
    return await mcp_status(current)


@app.post("/api/re/mcp/investigate")
async def reverse_mcp_investigate(
    payload: McpInvestigationRequest, current: Settings = Depends(get_settings)
) -> dict[str, Any]:
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
                    "target": target,
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


async def _proxy_openai(endpoint: str, payload: dict[str, Any], ollama: OllamaClient) -> Response:
    try:
        if payload.get("stream"):
            return await _streaming_passthrough(await ollama.proxy_stream(endpoint, payload))
        return _passthrough(await ollama.proxy_json(endpoint, payload))
    except HTTPException as exc:
        return _openai_error(exc.status_code, str(exc.detail), "service_unavailable")


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(
    payload: dict[str, Any] = Body(...), ollama: OllamaClient = Depends(get_ollama)
) -> Response:
    return await _proxy_openai("/v1/chat/completions", payload, ollama)


@app.post("/v1/responses", dependencies=[Depends(require_api_key)])
async def responses(
    payload: dict[str, Any] = Body(...), ollama: OllamaClient = Depends(get_ollama)
) -> Response:
    return await _proxy_openai("/v1/responses", payload, ollama)


@app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
async def embeddings(
    payload: dict[str, Any] = Body(...), ollama: OllamaClient = Depends(get_ollama)
) -> Response:
    return await _proxy_openai("/v1/embeddings", payload, ollama)


project_root = find_project_root()
references_dir = project_root / "references"
if references_dir.exists():
    app.mount("/references", StaticFiles(directory=references_dir), name="references")
web_dist = project_root / "apps" / "web" / "dist"
if web_dist.exists():
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
