from pathlib import Path

from fastapi.testclient import TestClient

from localllm.main import app
from localllm.system import storage_status


def test_storage_status_has_nonnegative_values(tmp_path: Path) -> None:
    status = storage_status(tmp_path)
    assert status["total"] > 0
    assert status["free"] >= 0


def test_cross_site_mutation_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/re/inspect",
            files={"binary": ("sample.bin", b"safe test bytes")},
            headers={"Origin": "https://malicious.example", "Sec-Fetch-Site": "cross-site"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Cross-site request blocked"


def test_security_headers_and_trusted_host() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")
        untrusted = client.get("/healthz", headers={"Host": "malicious.example"})

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert untrusted.status_code == 400
