from typing import Any

from fastapi.testclient import TestClient

from localllm.catalog import MODEL_CATALOG
from localllm.main import app


class FakeOllama:
    async def health(self) -> dict[str, Any]:
        return {"ok": True, "version": "test"}

    async def tags(self) -> list[dict[str, Any]]:
        return [{"name": "qwen3:8b-q4_K_M", "size": 1234}]


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

        authorized = client.get(
            "/v1/models", headers={"Authorization": "Bearer local-dev-key"}
        )
        assert authorized.status_code == 200
        ids = {model["id"] for model in authorized.json()["data"]}
        assert "qwen3:8b-q4_K_M" in ids
        assert "localllm-fast" in ids


def test_health_reports_gateway_and_runtime() -> None:
    with TestClient(app) as client:
        client.app.state.ollama = FakeOllama()
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["service"] == "localllm-api"
        assert response.json()["ollama"]["version"] == "test"

