from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import math
import os
import re
import secrets
import stat
import struct
import time
import zlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx

CANARY_SCHEMA_VERSION = 1
CANARY_ROLES = ("text", "code", "vision", "embedding")
ROLE_ALIASES: dict[str, str] = {
    "text": "localllm-fast",
    "code": "localllm-code",
    "vision": "localllm-vision",
    "embedding": "localllm-embed",
}
CANARY_STATUSES = frozenset({"passed", "failed", "timed_out", "cancelled"})
MAX_RECEIPT_BYTES = 64 * 1024
MAX_HTTP_RESPONSE_BYTES = 2 * 1024 * 1024
MIN_ROLE_TIMEOUT_SECONDS = 1.0
MAX_ROLE_TIMEOUT_SECONDS = 600.0
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,199}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+\-]{0,127}$")
_IMMUTABLE_RELEASE_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{8}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_ROLE_RESULT_FIELDS = frozenset(
    {"role", "status", "latency_ms", "alias", "resolved_model", "digest", "timestamp"}
)
_RECEIPT_FIELDS = frozenset({"schema_version", "release_id", "status", "timestamp", "roles"})


class CanaryContractError(ValueError):
    """A caller supplied an unsafe verifier or receipt contract."""


class CanaryProbeError(RuntimeError):
    """A content-free canary failure sentinel."""


def utc_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise CanaryContractError("invalid canary timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise CanaryContractError("invalid canary timestamp") from exc
    return parsed


def _validate_release_id(value: object) -> str:
    if not isinstance(value, str) or not _RELEASE_ID.fullmatch(value):
        raise CanaryContractError("invalid canary release ID")
    return value


def validate_loopback_base_url(value: str) -> str:
    if len(value) > 512:
        raise CanaryContractError("base URL is too long")
    parsed = urlsplit(value.strip())
    try:
        port = parsed.port
        host = parsed.hostname
        address = ipaddress.ip_address(host or "")
    except ValueError as exc:
        raise CanaryContractError("base URL must use a literal loopback address") from exc
    if (
        parsed.scheme != "http"
        or not address.is_loopback
        or parsed.username is not None
        or parsed.password is not None
        or port is None
        or parsed.path.rstrip("/") != "/v1"
        or parsed.query
        or parsed.fragment
    ):
        raise CanaryContractError("base URL must be an HTTP loopback /v1 endpoint")
    netloc = (
        f"[{address.compressed}]:{port}" if address.version == 6 else f"{address.compressed}:{port}"
    )
    return urlunsplit(("http", netloc, "/v1", "", ""))


def validate_roles(roles: Sequence[str]) -> tuple[str, ...]:
    selected = set(roles)
    if not selected or not selected <= set(CANARY_ROLES):
        raise CanaryContractError("roles must select one or more supported inference roles")
    return tuple(role for role in CANARY_ROLES if role in selected)


def validate_role_timeouts(roles: Sequence[str], timeouts: Mapping[str, float]) -> dict[str, float]:
    selected = validate_roles(roles)
    if set(timeouts) != set(selected):
        raise CanaryContractError("every selected role must have exactly one timeout")
    normalized: dict[str, float] = {}
    for role in selected:
        value = float(timeouts[role])
        if not math.isfinite(value) or not (
            MIN_ROLE_TIMEOUT_SECONDS <= value <= MAX_ROLE_TIMEOUT_SECONDS
        ):
            raise CanaryContractError("role timeout is outside the bounded range")
        normalized[role] = value
    return normalized


def _validate_api_key(value: str) -> str:
    candidate = value.strip()
    if not candidate or len(candidate.encode("utf-8")) > 4096:
        raise CanaryContractError("API key is missing or too long")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in candidate):
        raise CanaryContractError("API key contains control characters")
    return candidate


def read_private_api_key(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CanaryContractError("API key file is unavailable") from exc
    try:
        entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_uid != os.getuid()
            or entry.st_nlink != 1
            or stat.S_IMODE(entry.st_mode) & 0o077
            or entry.st_size > 4096
        ):
            raise CanaryContractError("API key file must be owner-private and regular")
        content = os.read(descriptor, 4097)
        if len(content) > 4096:
            raise CanaryContractError("API key file is too large")
    finally:
        os.close(descriptor)
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanaryContractError("API key file is not UTF-8") from exc
    return _validate_api_key(decoded)


def api_key_from_environment(environment: Mapping[str, str], name: str = "LOCALLLM_API_KEY") -> str:
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", name):
        raise CanaryContractError("API key environment variable name is invalid")
    return _validate_api_key(environment.get(name, ""))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def embedded_vision_fixture_data_url() -> str:
    """Return deterministic red/green/blue vertical bands without filesystem input."""

    width = 96
    height = 48
    row = b"\x00" + (b"\xff\x00\x00" * 32) + (b"\x00\xff\x00" * 32) + (b"\x00\x00\xff" * 32)
    raw = row * height
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _bounded_json_bytes(body: bytes) -> Any:
    if len(body) > MAX_HTTP_RESPONSE_BYTES:
        raise CanaryProbeError("response_too_large")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError) as exc:
        raise CanaryProbeError("invalid_response") from exc


async def _request_json(
    client: httpx.AsyncClient, method: str, url: str, payload: dict[str, Any] | None = None
) -> Any:
    body = bytearray()
    request_options: dict[str, Any] = {}
    if payload is not None:
        request_options["json"] = payload
    async with client.stream(method, url, **request_options) as response:
        if response.status_code < 200 or response.status_code >= 300:
            raise CanaryProbeError("request_failed")
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > MAX_HTTP_RESPONSE_BYTES:
                raise CanaryProbeError("response_too_large")
            body.extend(chunk)
    return _bounded_json_bytes(bytes(body))


def _inventory_for_alias(
    alias: str, models_payload: object, capabilities_payload: object
) -> tuple[str, str, str]:
    if not isinstance(models_payload, dict) or not isinstance(models_payload.get("data"), list):
        raise CanaryProbeError("invalid_models")
    available_ids = {
        item.get("id")
        for item in models_payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if alias not in available_ids:
        raise CanaryProbeError("alias_unavailable")
    if not isinstance(capabilities_payload, dict) or not isinstance(
        capabilities_payload.get("models"), list
    ):
        raise CanaryProbeError("invalid_capabilities")
    if capabilities_payload.get("schema_version") != 2:
        raise CanaryProbeError("unsupported_capabilities")
    service = capabilities_payload.get("service")
    if not isinstance(service, dict):
        raise CanaryProbeError("invalid_capabilities")
    try:
        release_id = _validate_release_id(service.get("release_id"))
    except CanaryContractError as exc:
        raise CanaryProbeError("missing_release") from exc
    matches = [
        item
        for item in capabilities_payload["models"]
        if isinstance(item, dict)
        and isinstance(item.get("aliases"), list)
        and alias in item["aliases"]
    ]
    if len(matches) != 1:
        raise CanaryProbeError("ambiguous_alias")
    resolved = matches[0].get("id")
    digest = matches[0].get("digest")
    if (
        not isinstance(resolved, str)
        or not _MODEL_ID.fullmatch(resolved)
        or resolved not in available_ids
        or not isinstance(digest, str)
        or not _DIGEST.fullmatch(digest)
    ):
        raise CanaryProbeError("missing_provenance")
    return resolved, digest, release_id


def _chat_content(payload: object) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        raise CanaryProbeError("invalid_chat")
    choices = payload["choices"]
    if not choices or not isinstance(choices[0], dict):
        raise CanaryProbeError("invalid_chat")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise CanaryProbeError("invalid_chat")
    return message["content"]


def _normalized_answer(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def _validate_response_model(payload: object, resolved_model: str) -> None:
    if not isinstance(payload, dict) or payload.get("model") != resolved_model:
        raise CanaryProbeError("unexpected_response_model")


def _validate_chat_result(role: str, payload: object, resolved_model: str) -> None:
    _validate_response_model(payload, resolved_model)
    content = _chat_content(payload)
    expected = {
        "text": "42",
        "code": "30",
        "vision": "RED,GREEN,BLUE",
    }[role]
    if _normalized_answer(content) != expected:
        raise CanaryProbeError("unexpected_chat")


def _validate_embedding_result(payload: object, resolved_model: str) -> None:
    _validate_response_model(payload, resolved_model)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise CanaryProbeError("invalid_embedding")
    data = payload["data"]
    if len(data) != 2 or any(not isinstance(item, dict) for item in data):
        raise CanaryProbeError("invalid_embedding")
    vectors: list[list[float]] = []
    for item in data:
        vector = item.get("embedding")
        if (
            not isinstance(vector, list)
            or len(vector) != 1024
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in vector
            )
            or not any(float(value) != 0.0 for value in vector)
        ):
            raise CanaryProbeError("invalid_embedding")
        vectors.append([float(value) for value in vector])
    if not any(left != right for left, right in zip(vectors[0], vectors[1], strict=True)):
        raise CanaryProbeError("input_insensitive_embedding")


def _chat_payload(role: str, alias: str) -> dict[str, Any]:
    if role == "vision":
        content: str | list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Inspect the supplied image rather than inferring its contents from this text. "
                    "Name the three vertical band colors from left to right, separated by commas, "
                    "and output only those color names."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": embedded_vision_fixture_data_url()},
            },
        ]
    elif role == "code":
        content = (
            "Using Python semantics, evaluate sum(i * i for i in range(1, 5)). "
            "Output only the decimal integer result."
        )
    else:
        content = "Add seventeen and twenty-five. Output only the decimal integer result."
    return {
        "model": alias,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "think": False,
        "stream": False,
        "max_tokens": 32,
        "keep_alive": 0,
        "options": {"num_ctx": 2048, "num_predict": 32},
    }


async def _probe_role(
    client: httpx.AsyncClient,
    base_url: str,
    role: str,
    observed_release_ids: set[str],
) -> tuple[str, str, str]:
    alias = ROLE_ALIASES[role]
    origin = base_url.removesuffix("/v1")
    models_payload = await _request_json(client, "GET", f"{base_url}/models")
    capabilities_payload = await _request_json(client, "GET", f"{origin}/api/node/capabilities")
    resolved, digest, release_id = _inventory_for_alias(alias, models_payload, capabilities_payload)
    observed_release_ids.add(release_id)
    if len(observed_release_ids) > 1:
        raise CanaryProbeError("release_changed")
    inference_error: BaseException | None = None
    if role == "embedding":
        try:
            result = await _request_json(
                client,
                "POST",
                f"{base_url}/embeddings",
                {
                    "model": alias,
                    "input": ["a red circle", "a blue ocean wave"],
                    "keep_alive": 0,
                    "options": {"num_ctx": 2048},
                },
            )
        except (CanaryProbeError, httpx.HTTPError, TypeError, ValueError) as exc:
            result = None
            inference_error = exc
    else:
        try:
            result = await _request_json(
                client, "POST", f"{base_url}/chat/completions", _chat_payload(role, alias)
            )
        except (CanaryProbeError, httpx.HTTPError, TypeError, ValueError) as exc:
            result = None
            inference_error = exc
    post_capabilities = await _request_json(client, "GET", f"{origin}/api/node/capabilities")
    post_resolved, post_digest, post_release_id = _inventory_for_alias(
        alias, models_payload, post_capabilities
    )
    observed_release_ids.add(post_release_id)
    if len(observed_release_ids) > 1:
        raise CanaryProbeError("release_changed")
    if (post_resolved, post_digest, post_release_id) != (resolved, digest, release_id):
        raise CanaryProbeError("provenance_changed")
    if inference_error is not None:
        raise CanaryProbeError("inference_failed") from inference_error
    if role == "embedding":
        _validate_embedding_result(result, resolved)
    else:
        _validate_chat_result(role, result, resolved)
    return resolved, digest, release_id


def _role_result(
    role: str,
    status_value: str,
    latency_ms: int,
    timestamp: str,
    resolved_model: str | None = None,
    digest: str | None = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "status": status_value,
        "latency_ms": max(0, min(latency_ms, 600_000)),
        "alias": ROLE_ALIASES[role],
        "resolved_model": resolved_model,
        "digest": digest,
        "timestamp": timestamp,
    }


async def verify_node_inference(
    base_url: str,
    api_key: str,
    roles: Sequence[str],
    timeouts: Mapping[str, float],
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    normalized_url = validate_loopback_base_url(base_url)
    normalized_key = _validate_api_key(api_key)
    selected = validate_roles(roles)
    normalized_timeouts = validate_role_timeouts(selected, timeouts)
    results: list[dict[str, Any]] = []
    observed_release_ids: set[str] = set()
    headers = {"Authorization": f"Bearer {normalized_key}", "Accept": "application/json"}
    client_options: dict[str, Any] = {
        "headers": headers,
        "timeout": None,
        "trust_env": False,
        "follow_redirects": False,
    }
    if transport is not None:
        client_options["transport"] = transport
    async with httpx.AsyncClient(**client_options) as client:
        for role in selected:
            started = time.monotonic()
            try:
                resolved, digest, release_id = await asyncio.wait_for(
                    _probe_role(client, normalized_url, role, observed_release_ids),
                    timeout=normalized_timeouts[role],
                )
            except asyncio.TimeoutError:
                results.append(
                    _role_result(
                        role,
                        "timed_out",
                        round((time.monotonic() - started) * 1000),
                        utc_timestamp(),
                    )
                )
            except (CanaryProbeError, httpx.HTTPError, TypeError, ValueError):
                results.append(
                    _role_result(
                        role,
                        "failed",
                        round((time.monotonic() - started) * 1000),
                        utc_timestamp(),
                    )
                )
            else:
                results.append(
                    _role_result(
                        role,
                        "passed",
                        round((time.monotonic() - started) * 1000),
                        utc_timestamp(),
                        resolved,
                        digest,
                    )
                )
            if len(observed_release_ids) > 1:
                break
    if len(observed_release_ids) > 1:
        return failed_receipt(selected)
    receipt = {
        "schema_version": CANARY_SCHEMA_VERSION,
        "release_id": next(iter(observed_release_ids), "unknown"),
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
        "timestamp": utc_timestamp(),
        "roles": results,
    }
    return validate_canary_receipt(receipt)


def cancelled_receipt() -> dict[str, Any]:
    return {
        "schema_version": CANARY_SCHEMA_VERSION,
        "release_id": "unknown",
        "status": "cancelled",
        "timestamp": utc_timestamp(),
        "roles": [],
    }


def failed_receipt(roles: Sequence[str], release_id: str = "unknown") -> dict[str, Any]:
    selected = validate_roles(roles)
    normalized_release_id = _validate_release_id(release_id)
    timestamp = utc_timestamp()
    return validate_canary_receipt(
        {
            "schema_version": CANARY_SCHEMA_VERSION,
            "release_id": normalized_release_id,
            "status": "failed",
            "timestamp": timestamp,
            "roles": [_role_result(role, "failed", 0, timestamp) for role in selected],
        }
    )


def validate_canary_receipt(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _RECEIPT_FIELDS:
        raise CanaryContractError("invalid canary receipt")
    if payload.get("schema_version") != CANARY_SCHEMA_VERSION:
        raise CanaryContractError("unsupported canary receipt schema")
    status_value = payload.get("status")
    if status_value not in {"passed", "failed", "cancelled"}:
        raise CanaryContractError("invalid canary receipt status")
    timestamp = payload.get("timestamp")
    receipt_timestamp = _parse_timestamp(timestamp)
    release_id = _validate_release_id(payload.get("release_id"))
    raw_roles = payload.get("roles")
    if not isinstance(raw_roles, list) or len(raw_roles) > len(CANARY_ROLES):
        raise CanaryContractError("invalid canary receipt roles")
    if status_value != "cancelled" and not raw_roles:
        raise CanaryContractError("non-cancelled receipt must contain roles")
    normalized_roles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_roles:
        if not isinstance(raw, dict) or set(raw) != _ROLE_RESULT_FIELDS:
            raise CanaryContractError("invalid canary role result")
        role = raw.get("role")
        role_status = raw.get("status")
        alias = raw.get("alias")
        latency_ms = raw.get("latency_ms")
        resolved_model = raw.get("resolved_model")
        digest = raw.get("digest")
        role_timestamp = raw.get("timestamp")
        if role not in CANARY_ROLES or role in seen or alias != ROLE_ALIASES[role]:
            raise CanaryContractError("invalid canary role identity")
        if role_status not in CANARY_STATUSES:
            raise CanaryContractError("invalid canary role status")
        if (
            not isinstance(latency_ms, int)
            or isinstance(latency_ms, bool)
            or not 0 <= latency_ms <= 600_000
        ):
            raise CanaryContractError("invalid canary latency")
        parsed_role_timestamp = _parse_timestamp(role_timestamp)
        if parsed_role_timestamp > receipt_timestamp:
            raise CanaryContractError("canary role timestamp follows receipt timestamp")
        if role_status == "passed":
            if (
                not isinstance(resolved_model, str)
                or not _MODEL_ID.fullmatch(resolved_model)
                or not isinstance(digest, str)
                or not _DIGEST.fullmatch(digest)
            ):
                raise CanaryContractError("passed canary lacks model provenance")
        elif resolved_model is not None or digest is not None:
            raise CanaryContractError("failed canary must not claim model provenance")
        normalized_roles.append(
            {
                "role": role,
                "status": role_status,
                "latency_ms": latency_ms,
                "alias": alias,
                "resolved_model": resolved_model,
                "digest": digest,
                "timestamp": role_timestamp,
            }
        )
        seen.add(role)
    normalized_roles.sort(key=lambda item: CANARY_ROLES.index(item["role"]))
    if status_value == "passed" and any(item["status"] != "passed" for item in normalized_roles):
        raise CanaryContractError("passed receipt contains a failed role")
    if status_value == "passed" and release_id == "unknown":
        raise CanaryContractError("passed receipt must be bound to a release")
    if status_value == "failed" and all(item["status"] == "passed" for item in normalized_roles):
        raise CanaryContractError("failed receipt contains no failed role")
    if status_value == "cancelled" and any(
        item["status"] != "cancelled" for item in normalized_roles
    ):
        raise CanaryContractError("cancelled receipt contains a non-cancelled role")
    return {
        "schema_version": CANARY_SCHEMA_VERSION,
        "release_id": release_id,
        "status": status_value,
        "timestamp": timestamp,
        "roles": normalized_roles,
    }


def canonical_canary_receipt_bytes(receipt: object) -> bytes:
    return _canonical_json_bytes(validate_canary_receipt(receipt))


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative_receipt_path(receipt_path: Path, data_dir: Path) -> Path:
    absolute_data = _absolute_without_resolving(data_dir)
    absolute_receipt = _absolute_without_resolving(receipt_path)
    try:
        relative = absolute_receipt.relative_to(absolute_data)
    except ValueError as exc:
        raise CanaryContractError(
            "canary receipt must stay inside the LocalLLM data directory"
        ) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise CanaryContractError("invalid canary receipt path")
    return relative


def _expected_receipt_relative_path(release_id: str) -> Path:
    return Path("node-canaries") / f"{_validate_release_id(release_id)}.json"


def validate_canary_output_path(receipt_path: Path, data_dir: Path) -> str:
    """Validate a dedicated output shape before any expensive inference starts."""

    relative = _relative_receipt_path(receipt_path, data_dir)
    if relative.parent != Path("node-canaries") or relative.suffix != ".json":
        raise CanaryContractError("canary output must use data/node-canaries/<release-id>.json")
    release_id = _validate_release_id(relative.stem)
    if release_id == "unknown" or relative != _expected_receipt_relative_path(release_id):
        raise CanaryContractError("canary output must name a concrete release")
    parent_descriptor, filename = _open_receipt_parent(receipt_path, data_dir)
    try:
        if filename != relative.name:
            raise CanaryContractError("canary output path changed during validation")
        try:
            existing = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_uid != os.getuid()
            or existing.st_nlink != 1
        ):
            raise CanaryContractError("existing canary output is not a safe regular file")
    finally:
        os.close(parent_descriptor)
    return release_id


def _validate_receipt_storage_path(receipt_path: Path, data_dir: Path, release_id: str) -> Path:
    output_release_id = validate_canary_output_path(receipt_path, data_dir)
    relative = _relative_receipt_path(receipt_path, data_dir)
    if output_release_id != release_id:
        raise CanaryContractError("canary receipt path does not match its release")
    if relative != _expected_receipt_relative_path(release_id):
        raise CanaryContractError(
            "canary receipt path must use the dedicated release-bound location"
        )
    return relative


def _open_private_directory(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    entry = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(entry.st_mode)
        or entry.st_uid != os.getuid()
        or stat.S_IMODE(entry.st_mode) & 0o077
    ):
        os.close(descriptor)
        raise CanaryContractError("canary receipt directory must be owner-private")
    return descriptor


def _open_receipt_parent(receipt_path: Path, data_dir: Path) -> tuple[int, str]:
    relative = _relative_receipt_path(receipt_path, data_dir)
    try:
        descriptor = _open_private_directory(_absolute_without_resolving(data_dir))
    except OSError as exc:
        raise CanaryContractError("LocalLLM data directory is unavailable") from exc
    try:
        for part in relative.parent.parts:
            if part == ".":
                continue
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            child = os.open(part, flags, dir_fd=descriptor)
            entry = os.fstat(child)
            if (
                not stat.S_ISDIR(entry.st_mode)
                or entry.st_uid != os.getuid()
                or stat.S_IMODE(entry.st_mode) & 0o077
            ):
                os.close(child)
                raise CanaryContractError("canary receipt parent must be owner-private")
            os.close(descriptor)
            descriptor = child
        return descriptor, relative.name
    except BaseException:
        os.close(descriptor)
        raise


def atomic_write_canary_receipt(receipt: object, receipt_path: Path, data_dir: Path) -> None:
    normalized = validate_canary_receipt(receipt)
    _validate_receipt_storage_path(receipt_path, data_dir, normalized["release_id"])
    content = _canonical_json_bytes(normalized)
    if len(content) > MAX_RECEIPT_BYTES:
        raise CanaryContractError("canary receipt is too large")
    parent_descriptor, filename = _open_receipt_parent(receipt_path, data_dir)
    temporary = f".{filename}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    temporary_created = False
    try:
        try:
            existing = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_uid != os.getuid()
            or existing.st_nlink != 1
        ):
            raise CanaryContractError("existing canary receipt is not a safe regular file")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_descriptor)
        temporary_created = True
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(
            temporary,
            filename,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_created = False
        os.fsync(parent_descriptor)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _read_canary_receipt(
    receipt_path: Path, data_dir: Path, expected_release_id: str
) -> dict[str, Any]:
    _validate_receipt_storage_path(receipt_path, data_dir, expected_release_id)
    parent_descriptor, filename = _open_receipt_parent(receipt_path, data_dir)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(filename, flags, dir_fd=parent_descriptor)
        try:
            entry = os.fstat(descriptor)
            if (
                not stat.S_ISREG(entry.st_mode)
                or entry.st_uid != os.getuid()
                or entry.st_nlink != 1
                or stat.S_IMODE(entry.st_mode) & 0o077
                or entry.st_size > MAX_RECEIPT_BYTES
            ):
                raise CanaryContractError("canary receipt is not owner-private and regular")
            content = os.read(descriptor, MAX_RECEIPT_BYTES + 1)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    if len(content) > MAX_RECEIPT_BYTES:
        raise CanaryContractError("canary receipt is too large")
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError) as exc:
        raise CanaryContractError("canary receipt is invalid") from exc
    normalized = validate_canary_receipt(payload)
    if content != _canonical_json_bytes(normalized):
        raise CanaryContractError("canary receipt is not canonical")
    return normalized


def functional_readiness_document(
    receipt_path: Path | None,
    data_dir: Path,
    required_roles: Sequence[str],
    max_age_seconds: int,
    current_release_id: str = "dev",
    current_model_provenance: Mapping[str, Mapping[str, str | None]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    selected = validate_roles(required_roles)
    normalized_current_release_id = _validate_release_id(current_release_id)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    base: dict[str, Any] = {
        "required_for_catalog_readiness": False,
        "ready": False,
        "status": "not_configured" if receipt_path is None else "missing",
        "fresh": False,
        "max_age_seconds": max_age_seconds,
        "timestamp": None,
        "release_id": None,
        "age_seconds": None,
        "required_roles": list(selected),
        "roles": [],
    }
    if receipt_path is None:
        return base
    if not _IMMUTABLE_RELEASE_ID.fullmatch(normalized_current_release_id):
        return {**base, "status": "invalid"}
    try:
        receipt = _read_canary_receipt(receipt_path, data_dir, normalized_current_release_id)
    except FileNotFoundError:
        return base
    except (OSError, CanaryContractError):
        return {**base, "status": "invalid"}
    timestamp = _parse_timestamp(receipt["timestamp"])
    if (timestamp - current).total_seconds() > 300:
        return {**base, "status": "invalid"}
    role_by_name = {item["role"]: item for item in receipt["roles"]}
    required_present = all(role in role_by_name for role in selected)
    required_timestamps = [
        _parse_timestamp(role_by_name[role]["timestamp"])
        for role in selected
        if role in role_by_name
    ]
    if any(role_timestamp > current for role_timestamp in required_timestamps):
        return {**base, "status": "invalid"}
    evidence_timestamp = min(required_timestamps) if required_timestamps else timestamp
    age_seconds = math.floor((current - evidence_timestamp).total_seconds())
    if age_seconds < -300:
        return {**base, "status": "invalid"}
    age_seconds = max(0, age_seconds)
    fresh = age_seconds <= max_age_seconds
    release_matches = receipt["release_id"] == normalized_current_release_id
    required_passed = required_present and all(
        role_by_name[role]["status"] == "passed" for role in selected
    )
    provenance = current_model_provenance or {}
    model_matches = required_passed and all(
        isinstance(provenance.get(role), Mapping)
        and provenance[role].get("resolved_model") == role_by_name[role]["resolved_model"]
        and provenance[role].get("digest") == role_by_name[role]["digest"]
        for role in selected
    )
    if not release_matches:
        status_value: Literal[
            "release_mismatch", "model_mismatch", "stale", "incomplete", "failed", "passed"
        ] = "release_mismatch"
    elif not fresh:
        status_value = "stale"
    elif not required_present:
        status_value = "incomplete"
    elif receipt["status"] != "passed" or not required_passed:
        status_value = "failed"
    elif not model_matches:
        status_value = "model_mismatch"
    else:
        status_value = "passed"
    return {
        **base,
        "ready": status_value == "passed",
        "status": status_value,
        "fresh": fresh,
        "timestamp": receipt["timestamp"],
        "release_id": receipt["release_id"],
        "age_seconds": age_seconds,
        "roles": receipt["roles"],
    }
