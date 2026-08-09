from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localllm.main import app
from localllm.system import _command, storage_status


def test_storage_status_has_nonnegative_values(tmp_path: Path) -> None:
    status = storage_status(tmp_path)
    assert status["total"] > 0
    assert status["free"] >= 0


@pytest.mark.asyncio
async def test_system_command_timeout_terminates_and_reaps_child() -> None:
    code, output = await _command("python3", "-c", "import time; time.sleep(30)", timeout=0.01)

    assert code == 124
    assert output == "timed out"


def test_binary_upload_content_length_is_rejected_before_multipart_parsing() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/re/inspect",
            headers={"Content-Length": str(66 * 1024 * 1024)},
            content=b"",
        )

    assert response.status_code == 413


def test_cdn_dependent_fastapi_docs_are_disabled() -> None:
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 200


def test_cross_site_mutation_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/re/inspect",
            files={"binary": ("sample.bin", b"safe test bytes")},
            headers={"Origin": "https://malicious.example", "Sec-Fetch-Site": "cross-site"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Cross-site request blocked"


def test_cross_site_api_get_is_rejected_before_expensive_route_work() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/system/status",
            headers={"Origin": "https://malicious.example", "Sec-Fetch-Site": "cross-site"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Cross-site request blocked"


@pytest.mark.parametrize(
    "origin",
    ["http://127.0.0.1:8008", "http://localhost:8008"],
)
def test_same_origin_loopback_mutation_reaches_route_validation(origin: str) -> None:
    with TestClient(app) as client:
        response = client.post("/api/research", headers={"Origin": origin}, json={})

    assert response.status_code == 422


def test_security_headers_and_trusted_host() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")
        untrusted = client.get("/healthz", headers={"Host": "malicious.example"})

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert untrusted.status_code == 400
