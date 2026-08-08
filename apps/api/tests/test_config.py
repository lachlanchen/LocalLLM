from __future__ import annotations

from localllm.config import Settings


def test_allowed_origins_accept_json_environment_value(monkeypatch) -> None:
    monkeypatch.setenv("LOCALLLM_ALLOWED_ORIGINS", '["https://one.test","https://two.test"]')

    settings = Settings(_env_file=None)

    assert settings.allowed_origins == ["https://one.test", "https://two.test"]


def test_allowed_origins_accept_comma_separated_environment_value(monkeypatch) -> None:
    monkeypatch.setenv("LOCALLLM_ALLOWED_ORIGINS", "https://one.test, https://two.test")

    settings = Settings(_env_file=None)

    assert settings.allowed_origins == ["https://one.test", "https://two.test"]
