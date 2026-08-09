from __future__ import annotations

import pytest

from localllm.config import Settings, prepare_private_data_dir


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
    with pytest.raises(ValueError, match="authenticated access-control"):
        Settings(host="0.0.0.0", api_key="a-real-private-key", _env_file=None)


def test_ipv6_binding_is_rejected_until_trusted_host_supports_it() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        Settings(host="::1", _env_file=None)


def test_alternate_port_is_rejected_to_keep_runtime_and_status_aligned() -> None:
    with pytest.raises(ValueError, match="fixed loopback endpoint"):
        Settings(port=9000, _env_file=None)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",
        "http://localhost:11434",
        "http://ollama.example:11434",
        "http://user:secret@127.0.0.1:11434",
        "http://127.0.0.1:11435",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?token=secret",
    ],
)
def test_remote_or_ambiguous_ollama_runtime_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="Ollama runtime"):
        Settings(ollama_base_url=url, _env_file=None)


def test_local_ollama_runtime_is_normalized() -> None:
    settings = Settings(ollama_base_url="http://127.0.0.1:11434/", _env_file=None)

    assert settings.ollama_base_url == "http://127.0.0.1:11434"


def test_search_provider_credentials_use_localllm_environment_namespace(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LOCALLLM_SEARCH_BRAVE_API_KEY", "brave-secret")
    monkeypatch.setenv("LOCALLLM_SEARCH_SERPAPI_API_KEY", "scholar-secret")

    settings = Settings(_env_file=None)

    assert settings.search_brave_api_key == "brave-secret"
    assert settings.search_serpapi_api_key == "scholar-secret"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("search_max_results", 51),
        ("search_max_concurrency", 0),
        ("search_provider_timeout_seconds", 1),
        ("search_response_limit_bytes", 50_000),
    ],
)
def test_search_resource_limits_are_bounded(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})


def test_image_generation_is_default_off_and_resource_settings_are_bounded() -> None:
    settings = Settings(_env_file=None)

    assert settings.image_generation_enabled is False
    assert settings.image_generation_gpu == 0
    with pytest.raises(ValueError):
        Settings(image_generation_gpu=16, _env_file=None)
    with pytest.raises(ValueError):
        Settings(image_generation_timeout_seconds=59, _env_file=None)


def test_private_data_directory_does_not_follow_a_final_symlink(tmp_path) -> None:
    target = tmp_path / "unrelated"
    target.mkdir(mode=0o755)
    link = tmp_path / "data"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="not a symlink"):
        prepare_private_data_dir(link)

    assert target.stat().st_mode & 0o777 == 0o755
