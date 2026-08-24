from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from localllm.node_canary import canonical_canary_receipt_bytes, utc_timestamp


def _load_script() -> ModuleType:
    script_path = Path(__file__).parents[3] / "scripts" / "verify-node-inference.py"
    specification = importlib.util.spec_from_file_location(
        "localllm_verify_node_inference_script", script_path
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _passed_text_receipt() -> dict:
    timestamp = utc_timestamp()
    return {
        "schema_version": 1,
        "release_id": "release-test",
        "status": "passed",
        "timestamp": timestamp,
        "roles": [
            {
                "role": "text",
                "status": "passed",
                "latency_ms": 1,
                "alias": "localllm-fast",
                "resolved_model": "qwen3:8b-q4_K_M",
                "digest": "a" * 64,
                "timestamp": timestamp,
            }
        ],
    }


def test_cli_persists_only_with_explicit_output_and_never_serializes_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    private_key = "private-cli-key-must-not-escape"
    calls: list[tuple[str, str, tuple[str, ...], dict[str, float]]] = []
    stdout_receipts: list[object] = []
    passed_receipt = _passed_text_receipt()

    async def fake_verify(base_url, api_key, roles, timeouts):
        calls.append((base_url, api_key, roles, timeouts))
        return passed_receipt

    monkeypatch.setattr(script, "verify_node_inference", fake_verify)
    monkeypatch.setattr(script, "_write_stdout", stdout_receipts.append)

    common = [
        "--base-url",
        "http://127.0.0.1:18008/v1",
        "--roles",
        "text",
        "--timeout-seconds",
        "7",
        "--data-dir",
        str(data_dir),
    ]
    environment = {"LOCALLLM_API_KEY": private_key}
    assert script.main(common, environment) == 0
    assert list(data_dir.iterdir()) == []

    destination = data_dir / "node-canaries" / "release-test.json"
    destination.parent.mkdir(mode=0o700)
    assert script.main([*common, "--output", str(destination)], environment) == 0

    assert destination.read_bytes() == canonical_canary_receipt_bytes(passed_receipt)
    assert destination.stat().st_mode & 0o777 == 0o600
    assert calls == [
        (
            "http://127.0.0.1:18008/v1",
            private_key,
            ("text",),
            {"text": 7.0},
        ),
        (
            "http://127.0.0.1:18008/v1",
            private_key,
            ("text",),
            {"text": 7.0},
        ),
    ]
    serialized = json.dumps(stdout_receipts)
    assert private_key not in serialized
    assert "prompt" not in serialized
    assert "completion" not in serialized
    assert "vector" not in serialized


def test_cli_rejects_key_value_argv_without_echoing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script()
    private_key = "private-key-on-argv-must-not-echo"
    stdout_receipts: list[object] = []
    monkeypatch.setattr(script, "_write_stdout", stdout_receipts.append)

    exit_code = script.main(
        ["--roles", "text", "--api-key", private_key],
        {"LOCALLLM_API_KEY": "safe-environment-key"},
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert private_key not in captured.out
    assert private_key not in captured.err
    assert private_key not in json.dumps(stdout_receipts)


def test_cli_rejects_database_output_before_starting_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    database = data_dir / "conversations.sqlite3"
    database.write_bytes(b"database-must-remain-untouched")
    database.chmod(0o600)
    inference_started = False

    async def fake_verify(base_url, api_key, roles, timeouts):
        nonlocal inference_started
        inference_started = True
        return _passed_text_receipt()

    monkeypatch.setattr(script, "verify_node_inference", fake_verify)
    monkeypatch.setattr(script, "_write_stdout", lambda receipt: None)

    exit_code = script.main(
        [
            "--roles",
            "text",
            "--data-dir",
            str(data_dir),
            "--output",
            str(database),
        ],
        {"LOCALLLM_API_KEY": "private-key"},
    )

    assert exit_code == 2
    assert inference_started is False
    assert database.read_bytes() == b"database-must-remain-untouched"


def test_cli_refuses_receipt_when_observed_release_does_not_match_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    canary_dir = data_dir / "node-canaries"
    canary_dir.mkdir(mode=0o700)
    destination = canary_dir / "different-release.json"

    async def fake_verify(base_url, api_key, roles, timeouts):
        return _passed_text_receipt()

    monkeypatch.setattr(script, "verify_node_inference", fake_verify)
    monkeypatch.setattr(script, "_write_stdout", lambda receipt: None)

    exit_code = script.main(
        [
            "--roles",
            "text",
            "--data-dir",
            str(data_dir),
            "--output",
            str(destination),
        ],
        {"LOCALLLM_API_KEY": "private-key"},
    )

    assert exit_code == 2
    assert not destination.exists()
