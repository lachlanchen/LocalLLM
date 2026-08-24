from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from localllm.node_canary import (
    CanaryContractError,
    atomic_write_canary_receipt,
    canonical_canary_receipt_bytes,
    functional_readiness_document,
    read_private_api_key,
    validate_canary_receipt,
    validate_loopback_base_url,
)

TIMESTAMP = "2026-08-25T00:00:00Z"
DIGESTS = {
    "text": "1" * 64,
    "code": "2" * 64,
    "vision": "3" * 64,
    "embedding": "4" * 64,
}
ALIASES = {
    "text": "localllm-fast",
    "code": "localllm-code",
    "vision": "localllm-vision",
    "embedding": "localllm-embed",
}
RELEASE_ID = "01234567-89abcdef"


def receipt_path(data_dir: Path, release_id: str = RELEASE_ID) -> Path:
    canary_dir = data_dir / "node-canaries"
    canary_dir.mkdir(mode=0o700, exist_ok=True)
    return canary_dir / f"{release_id}.json"


def receipt(roles: tuple[str, ...] = ("text", "code", "vision", "embedding")) -> dict:
    return {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "status": "passed",
        "timestamp": TIMESTAMP,
        "roles": [
            {
                "role": role,
                "status": "passed",
                "latency_ms": index + 1,
                "alias": ALIASES[role],
                "resolved_model": f"test-{role}:latest",
                "digest": DIGESTS[role],
                "timestamp": TIMESTAMP,
            }
            for index, role in enumerate(roles)
        ],
    }


def current_provenance(
    roles: tuple[str, ...] = ("text", "code", "vision", "embedding"),
) -> dict[str, dict[str, str]]:
    return {
        role: {
            "resolved_model": f"test-{role}:latest",
            "digest": DIGESTS[role],
        }
        for role in roles
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8008/v1",
        "http://localhost:8008/v1",
        "http://192.0.2.1:8008/v1",
        "http://user:secret@127.0.0.1:8008/v1",
        "http://127.0.0.1:8008/v1?key=secret",
        "http://127.0.0.1/v1",
    ],
)
def test_verifier_rejects_nonliteral_or_ambiguous_base_urls(url: str) -> None:
    with pytest.raises(CanaryContractError):
        validate_loopback_base_url(url)


def test_verifier_accepts_literal_ipv4_and_ipv6_loopback_urls() -> None:
    assert validate_loopback_base_url("http://127.0.0.1:8008/v1/") == ("http://127.0.0.1:8008/v1")
    assert validate_loopback_base_url("http://[::1]:8008/v1") == "http://[::1]:8008/v1"


def test_private_api_key_file_rejects_symlink_or_open_permissions(tmp_path: Path) -> None:
    private = tmp_path / "key"
    private.write_text("private-key\n", encoding="utf-8")
    private.chmod(0o600)
    assert read_private_api_key(private) == "private-key"

    private.chmod(0o640)
    with pytest.raises(CanaryContractError):
        read_private_api_key(private)
    private.chmod(0o600)
    link = tmp_path / "key-link"
    link.symlink_to(private)
    with pytest.raises(CanaryContractError):
        read_private_api_key(link)


@pytest.mark.parametrize(
    ("level", "field"),
    [
        ("receipt", "prompt"),
        ("receipt", "api_key"),
        ("role", "completion"),
        ("role", "vector"),
        ("role", "raw_error"),
        ("role", "output_path"),
    ],
)
def test_receipt_validation_rejects_content_secret_error_or_path_fields(
    level: str, field: str
) -> None:
    payload = receipt(("text",))
    target = payload if level == "receipt" else payload["roles"][0]
    target[field] = "private-value"
    with pytest.raises(CanaryContractError):
        validate_canary_receipt(payload)


def test_passed_receipt_requires_release_binding_and_consistent_timestamps() -> None:
    payload = receipt(("text",))
    payload["release_id"] = "unknown"
    with pytest.raises(CanaryContractError, match="bound to a release"):
        validate_canary_receipt(payload)

    payload = receipt(("text",))
    payload["roles"][0]["timestamp"] = "2026-08-25T00:00:01Z"
    with pytest.raises(CanaryContractError, match="timestamp"):
        validate_canary_receipt(payload)


def test_atomic_receipt_is_canonical_private_and_reports_freshness(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    destination = receipt_path(tmp_path)
    payload = receipt()
    atomic_write_canary_receipt(payload, destination, tmp_path)

    assert destination.read_bytes() == canonical_canary_receipt_bytes(payload)
    assert destination.stat().st_mode & 0o777 == 0o600
    document = functional_readiness_document(
        destination,
        tmp_path,
        ("text", "code", "vision", "embedding"),
        3600,
        RELEASE_ID,
        current_provenance(),
        now=datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc),
    )
    assert document["status"] == "passed"
    assert document["ready"] is True
    assert document["fresh"] is True
    assert document["age_seconds"] == 1800
    serialized = json.dumps(document)
    assert "prompt" not in serialized
    assert "completion" not in serialized
    assert "vector" not in serialized
    assert "key" not in serialized


def test_receipt_staleness_missing_roles_and_corruption_fail_closed(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    destination = receipt_path(tmp_path)
    atomic_write_canary_receipt(receipt(("text",)), destination, tmp_path)

    incomplete = functional_readiness_document(
        destination,
        tmp_path,
        ("text", "vision"),
        3600,
        RELEASE_ID,
        current_provenance(("text",)),
        now=datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc),
    )
    assert incomplete["status"] == "incomplete"
    assert incomplete["ready"] is False

    stale = functional_readiness_document(
        destination,
        tmp_path,
        ("text",),
        60,
        RELEASE_ID,
        current_provenance(("text",)),
        now=datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc),
    )
    assert stale["status"] == "stale"
    assert stale["fresh"] is False

    destination.write_text('{"prompt":"must not surface"}', encoding="utf-8")
    destination.chmod(0o600)
    invalid = functional_readiness_document(
        destination,
        tmp_path,
        ("text",),
        60,
        RELEASE_ID,
        current_provenance(("text",)),
        now=datetime.now(timezone.utc),
    )
    assert invalid["status"] == "invalid"
    assert "must not surface" not in json.dumps(invalid)


def test_receipt_from_another_release_cannot_certify_current_api(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    successor_release = "fedcba98-76543210"
    destination = receipt_path(tmp_path, successor_release)
    destination.write_bytes(canonical_canary_receipt_bytes(receipt(("text",))))
    destination.chmod(0o600)

    document = functional_readiness_document(
        destination,
        tmp_path,
        ("text",),
        3600,
        successor_release,
        current_provenance(("text",)),
        now=datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc),
    )

    assert document["status"] == "release_mismatch"
    assert document["ready"] is False
    assert document["fresh"] is True
    assert document["release_id"] == RELEASE_ID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resolved_model", "repulled-text:latest"),
        ("digest", "9" * 64),
    ],
)
def test_receipt_model_provenance_must_match_current_catalog(
    tmp_path: Path, field: str, value: str
) -> None:
    tmp_path.chmod(0o700)
    destination = receipt_path(tmp_path)
    atomic_write_canary_receipt(receipt(("text",)), destination, tmp_path)
    provenance = current_provenance(("text",))
    provenance["text"][field] = value

    document = functional_readiness_document(
        destination,
        tmp_path,
        ("text",),
        3600,
        RELEASE_ID,
        provenance,
        now=datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc),
    )

    assert document["status"] == "model_mismatch"
    assert document["ready"] is False
    assert document["fresh"] is True


def test_receipt_write_rejects_escape_symlink_parent_and_symlink_target(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    payload = receipt(("text",))

    with pytest.raises(CanaryContractError):
        atomic_write_canary_receipt(
            payload, outside / "node-canaries" / f"{RELEASE_ID}.json", data_dir
        )

    parent_link = data_dir / "node-canaries"
    parent_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises((CanaryContractError, OSError)):
        atomic_write_canary_receipt(payload, parent_link / f"{RELEASE_ID}.json", data_dir)

    parent_link.unlink()
    parent_link.mkdir(mode=0o700)
    target = parent_link / f"{RELEASE_ID}.json"
    outside_target = outside / "target.json"
    outside_target.write_text("untouched", encoding="utf-8")
    target.symlink_to(outside_target)
    with pytest.raises(CanaryContractError):
        atomic_write_canary_receipt(payload, target, data_dir)
    assert outside_target.read_text(encoding="utf-8") == "untouched"


def test_receipt_writer_cannot_replace_conversation_database(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    database = data_dir / "conversations.sqlite3"
    database.write_bytes(b"database-must-remain-untouched")
    database.chmod(0o600)

    with pytest.raises(CanaryContractError, match="node-canaries"):
        atomic_write_canary_receipt(receipt(("text",)), database, data_dir)

    assert database.read_bytes() == b"database-must-remain-untouched"


def test_future_receipt_and_noncanonical_receipt_are_invalid(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    destination = receipt_path(tmp_path)
    future = receipt(("text",))
    future_time = datetime.now(timezone.utc) + timedelta(hours=2)
    stamp = future_time.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    future["timestamp"] = stamp
    future["roles"][0]["timestamp"] = stamp
    atomic_write_canary_receipt(future, destination, tmp_path)
    assert (
        functional_readiness_document(
            destination,
            tmp_path,
            ("text",),
            3600,
            RELEASE_ID,
            current_provenance(("text",)),
            now=datetime.now(timezone.utc),
        )["status"]
        == "invalid"
    )

    destination.write_text(json.dumps(receipt(("text",)), indent=2), encoding="utf-8")
    destination.chmod(0o600)
    assert (
        functional_readiness_document(
            destination,
            tmp_path,
            ("text",),
            3600,
            RELEASE_ID,
            current_provenance(("text",)),
            now=datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc),
        )["status"]
        == "invalid"
    )


def test_freshness_uses_oldest_required_role_and_rejects_future_role(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    destination = receipt_path(tmp_path)
    old_role = receipt(("text",))
    old_role["timestamp"] = "2026-08-25T00:30:00Z"
    atomic_write_canary_receipt(old_role, destination, tmp_path)

    stale = functional_readiness_document(
        destination,
        tmp_path,
        ("text",),
        900,
        RELEASE_ID,
        current_provenance(("text",)),
        now=datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc),
    )
    assert stale["status"] == "stale"
    assert stale["age_seconds"] == 1800

    aggregate_future = receipt(("text",))
    aggregate_future["timestamp"] = "2026-08-25T02:00:00Z"
    atomic_write_canary_receipt(aggregate_future, destination, tmp_path)
    invalid_aggregate = functional_readiness_document(
        destination,
        tmp_path,
        ("text",),
        900,
        RELEASE_ID,
        current_provenance(("text",)),
        now=datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc),
    )
    assert invalid_aggregate["status"] == "invalid"

    future = receipt(("text",))
    future["timestamp"] = "2026-08-25T00:31:00Z"
    future["roles"][0]["timestamp"] = "2026-08-25T00:31:00Z"
    atomic_write_canary_receipt(future, destination, tmp_path)
    invalid = functional_readiness_document(
        destination,
        tmp_path,
        ("text",),
        900,
        RELEASE_ID,
        current_provenance(("text",)),
        now=datetime(2026, 8, 25, 0, 30, tzinfo=timezone.utc),
    )
    assert invalid["status"] == "invalid"


def test_receipt_path_environment_contract_does_not_follow_data_directory_symlink(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(CanaryContractError):
        atomic_write_canary_receipt(
            receipt(("text",)), linked / "node-canaries" / f"{RELEASE_ID}.json", linked
        )


def test_existing_receipt_with_multiple_links_is_rejected(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    destination = receipt_path(tmp_path)
    destination.write_text("old", encoding="utf-8")
    destination.chmod(0o600)
    os.link(destination, tmp_path / "second-link")
    with pytest.raises(CanaryContractError):
        atomic_write_canary_receipt(receipt(("text",)), destination, tmp_path)
