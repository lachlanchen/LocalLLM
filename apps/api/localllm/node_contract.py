from __future__ import annotations

from typing import Any

from .catalog import MODEL_ALIASES, MODEL_CATALOG, resolve_model
from .ollama import OllamaProbe

READINESS_SCHEMA_VERSION = 1
NODE_CAPABILITIES_SCHEMA_VERSION = 1

_CATALOG_BY_ID = {model["id"]: model for model in MODEL_CATALOG}
_ALIASES_BY_TARGET = {
    target: sorted(alias for alias, candidate in MODEL_ALIASES.items() if candidate == target)
    for target in set(MODEL_ALIASES.values())
}

_PROTOCOLS: tuple[dict[str, Any], ...] = (
    {
        "id": "openai.models.list.v1",
        "method": "GET",
        "path": "/v1/models",
        "authentication": "bearer",
        "streaming": False,
    },
    {
        "id": "openai.models.retrieve.v1",
        "method": "GET",
        "path": "/v1/models/{model}",
        "authentication": "bearer",
        "streaming": False,
    },
    {
        "id": "openai.chat-completions.v1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "authentication": "bearer",
        "streaming": True,
    },
    {
        "id": "openai.responses.v1",
        "method": "POST",
        "path": "/v1/responses",
        "authentication": "bearer",
        "streaming": True,
    },
    {
        "id": "openai.embeddings.v1",
        "method": "POST",
        "path": "/v1/embeddings",
        "authentication": "bearer",
        "streaming": False,
    },
)


def required_model_states(
    required_models: list[str], installed_models: set[str]
) -> list[dict[str, Any]]:
    return [
        {
            "id": model,
            "resolved_id": resolve_model(model),
            "available": resolve_model(model) in installed_models,
        }
        for model in required_models
    ]


def is_ready(probe: OllamaProbe, required_models: list[str]) -> bool:
    if not probe.ok:
        return False
    installed = set(probe.models)
    return all(resolve_model(model) in installed for model in required_models)


def readiness_document(
    probe: OllamaProbe, required_models: list[str], service_version: str
) -> dict[str, Any]:
    installed = set(probe.models) if probe.ok else set()
    requirements = required_model_states(required_models, installed)
    models_ready = probe.ok and all(item["available"] for item in requirements)
    ready = probe.ok and models_ready
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "ok": ready,
        "status": "ready" if ready else "not_ready",
        "service": {"name": "localllm-api", "version": service_version},
        "checks": {
            "process": {"ok": True},
            "ollama": {
                "ok": probe.ok,
                "code": "ready" if probe.ok else probe.error_code,
            },
            "required_models": {
                "ok": models_ready,
                "models": requirements,
                "missing": [item["id"] for item in requirements if not item["available"]],
            },
        },
    }


def _installed_model_contract(installed_model: str) -> dict[str, Any]:
    catalog_entry = _CATALOG_BY_ID.get(installed_model)
    return {
        "id": installed_model,
        "aliases": list(_ALIASES_BY_TARGET.get(installed_model, [])),
        "catalogued": catalog_entry is not None,
        "modalities": list(catalog_entry["modalities"]) if catalog_entry else [],
        "context_tokens": int(catalog_entry["context"]) if catalog_entry else None,
    }


def node_capabilities_document(
    probe: OllamaProbe, required_models: list[str], service_version: str
) -> dict[str, Any]:
    installed = set(probe.models) if probe.ok else set()
    return {
        "schema_version": NODE_CAPABILITIES_SCHEMA_VERSION,
        "service": {
            "name": "localllm-api",
            "version": service_version,
            "node_kind": "local-inference",
        },
        "ready": is_ready(probe, required_models),
        "runtime": {
            "provider": "ollama",
            "ready": probe.ok,
            "error_code": None if probe.ok else probe.error_code,
        },
        "required_models": required_model_states(required_models, installed),
        "protocols": [dict(protocol) for protocol in _PROTOCOLS],
        "models": [_installed_model_contract(model) for model in sorted(installed)],
    }
