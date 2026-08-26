from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localllm.main import _static_web_cache_control, app
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
    assert "font-src 'self' data:" in response.headers["content-security-policy"]
    assert untrusted.status_code == 400


@pytest.mark.parametrize(
    ("path", "status_code", "content_type", "expected"),
    [
        ("/", 200, "text/html; charset=utf-8", "no-store"),
        ("/conversation/one", 200, "text/html", "no-store"),
        (
            "/assets/index-a1b2c3.js",
            200,
            "text/javascript; charset=utf-8",
            "public, max-age=31536000, immutable",
        ),
        ("/assets/retired.js", 404, "application/json", "no-store"),
        ("/manifest.webmanifest", 200, "application/manifest+json", "no-cache"),
        ("/sw.js", 200, "text/javascript", "no-cache"),
        ("/favicon.svg", 200, "image/svg+xml", "no-cache"),
        ("/api/system/status", 200, "application/json", None),
        ("/v1/models", 401, "application/json", None),
    ],
)
def test_static_web_cache_policy_is_release_safe(
    path: str,
    status_code: int,
    content_type: str,
    expected: str | None,
) -> None:
    assert _static_web_cache_control(path, status_code, content_type) == expected


def test_static_web_cache_headers_are_applied_at_the_response_boundary() -> None:
    with TestClient(app) as client:
        html = client.get("/")
        current_asset = client.get("/assets/index-BL5prAB3.js")
        retired_asset = client.get("/assets/retired-build.js")

    if html.headers.get("content-type", "").startswith("text/html"):
        assert html.headers["cache-control"] == "no-store"
    if current_asset.status_code == 200:
        assert current_asset.headers["cache-control"] == (
            "public, max-age=31536000, immutable"
        )
    assert retired_asset.status_code == 404
    assert retired_asset.headers["cache-control"] == "no-store"
