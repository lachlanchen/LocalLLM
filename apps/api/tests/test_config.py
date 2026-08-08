from __future__ import annotations

import pytest

from localllm.config import Settings


def test_allowed_origins_accept_json_environment_value(monkeypatch) -> None:
    monkeypatch.setenv("LOCALLLM_ALLOWED_ORIGINS", '["https://one.test","https://two.test"]')

    settings = Settings(_env_file=None)

    assert settings.allowed_origins == ["https://one.test", "https://two.test"]


def test_allowed_origins_accept_comma_separated_environment_value(monkeypatch) -> None:
    monkeypatch.setenv("LOCALLLM_ALLOWED_ORIGINS", "https://one.test, https://two.test")

    settings = Settings(_env_file=None)

    assert settings.allowed_origins == ["https://one.test", "https://two.test"]


def test_non_loopback_binding_is_rejected() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        Settings(host="0.0.0.0", api_key="local-dev-key", _env_file=None)


def test_non_loopback_binding_rejects_even_explicit_api_key() -> None:
    with pytest.raises(ValueError, match="authenticated tunnel"):
        Settings(host="0.0.0.0", api_key="a-real-private-key", _env_file=None)


def test_ipv6_binding_is_rejected_until_trusted_host_supports_it() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        Settings(host="::1", _env_file=None)


def test_alternate_port_is_rejected_to_keep_runtime_and_status_aligned() -> None:
    with pytest.raises(ValueError, match="fixed loopback endpoint"):
        Settings(port=9000, _env_file=None)
