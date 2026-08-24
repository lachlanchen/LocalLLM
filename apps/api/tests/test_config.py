from __future__ import annotations

import pytest

from localllm.config import (
    DEFAULT_NODE_CANARY_ROLES,
    DEFAULT_REQUIRED_MODELS,
    Settings,
    get_settings,
    prepare_private_data_dir,
)


def test_cached_application_settings_use_the_hermetic_test_key() -> None:
    assert get_settings().api_key == "local-dev-key"


def test_allowed_origins_accept_json_environment_value(monkeypatch) -> None:
    monkeypatch.setenv("LOCALLLM_ALLOWED_ORIGINS", '["https://one.test","https://two.test"]')

    settings = Settings(_env_file=None)

    assert settings.allowed_origins == ["https://one.test", "https://two.test"]


def test_allowed_origins_accept_comma_separated_environment_value(monkeypatch) -> None:
    monkeypatch.setenv("LOCALLLM_ALLOWED_ORIGINS", "https://one.test, https://two.test")

    settings = Settings(_env_file=None)

    assert settings.allowed_origins == ["https://one.test", "https://two.test"]


def test_required_models_default_to_the_core_pull_contract() -> None:
    settings = Settings(_env_file=None)

    assert settings.required_models == list(DEFAULT_REQUIRED_MODELS)


def test_functional_canary_defaults_match_the_practical_core_contract() -> None:
    settings = Settings(_env_file=None)

    assert settings.release_id == "dev"
    assert settings.node_canary_roles == list(DEFAULT_NODE_CANARY_ROLES)
    assert settings.node_canary_receipt_path is None
    assert settings.node_canary_max_age_seconds == 86_400


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('["text","code"]', ["text", "code"]),
        ("vision, embedding", ["vision", "embedding"]),
    ],
)
def test_node_canary_roles_accept_json_or_csv(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    monkeypatch.setenv("LOCALLLM_NODE_CANARY_ROLES", raw)

    assert Settings(_env_file=None).node_canary_roles == expected


@pytest.mark.parametrize("roles", [[], ["audio"], ["text", "text"]])
def test_node_canary_roles_reject_empty_unknown_or_duplicate_values(
    roles: list[str],
) -> None:
    with pytest.raises(ValueError, match="canary role"):
        Settings(node_canary_roles=roles, _env_file=None)


@pytest.mark.parametrize("release_id", ["", "contains spaces", "../../release", "x" * 129])
def test_release_id_uses_a_bounded_nonsecret_identifier_shape(release_id: str) -> None:
    with pytest.raises(ValueError, match="release ID"):
        Settings(release_id=release_id, _env_file=None)


def test_canary_receipt_path_requires_immutable_release_bound_location(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    release_id = "01234567-89abcdef"
    canary_dir = data_dir / "node-canaries"
    canary_dir.mkdir(mode=0o700)
    inside = data_dir / "node-canaries" / f"{release_id}.json"

    settings = Settings(
        data_dir=data_dir,
        release_id=release_id,
        node_canary_receipt_path=inside,
        _env_file=None,
    )
    assert settings.node_canary_receipt_path == inside

    with pytest.raises(ValueError, match="receipt"):
        Settings(
            data_dir=data_dir,
            release_id=release_id,
            node_canary_receipt_path=tmp_path / "outside.json",
            _env_file=None,
        )

    canary_dir.chmod(0o755)
    with pytest.raises(ValueError, match="owner-private"):
        Settings(
            data_dir=data_dir,
            release_id=release_id,
            node_canary_receipt_path=inside,
            _env_file=None,
        )

    with pytest.raises(ValueError, match="release-bound"):
        Settings(
            data_dir=data_dir,
            release_id=release_id,
            node_canary_receipt_path=data_dir / "conversations.sqlite3",
            _env_file=None,
        )


@pytest.mark.parametrize("release_id", ["dev", "unknown", "release-test"])
def test_configured_canary_rejects_reusable_release_identity(
    tmp_path, release_id: str
) -> None:
    data_dir = tmp_path / "data"
    with pytest.raises(ValueError, match="immutable"):
        Settings(
            data_dir=data_dir,
            release_id=release_id,
            node_canary_receipt_path=data_dir / "node-canaries" / f"{release_id}.json",
            _env_file=None,
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('["localllm-fast","localllm-vision"]', ["localllm-fast", "localllm-vision"]),
        (
            "localllm-fast, qwen3-vl:8b-instruct-q4_K_M",
            [
                "localllm-fast",
                "qwen3-vl:8b-instruct-q4_K_M",
            ],
        ),
        ("", []),
    ],
)
def test_required_models_accept_json_csv_or_an_explicit_empty_list(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    monkeypatch.setenv("LOCALLLM_REQUIRED_MODELS", raw)

    settings = Settings(_env_file=None)

    assert settings.required_models == expected


@pytest.mark.parametrize(
    "required_models",
    [["https://remote.example/model"], ["model with spaces"], ["model\nname"]],
)
def test_required_models_reject_non_local_identifier_shapes(required_models: list[str]) -> None:
    with pytest.raises(ValueError, match="local model ID syntax"):
        Settings(required_models=required_models, _env_file=None)


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


def test_search_application_key_uses_dedicated_environment_variable_and_stays_masked(
    monkeypatch,
) -> None:
    secret = "independent-search-credential-0123456789"
    monkeypatch.setenv("LOCALLLM_SEARCH_API_KEY", secret)

    settings = Settings(_env_file=None)

    assert settings.search_api_key.get_secret_value() == secret
    assert secret not in repr(settings)
    assert secret not in settings.model_dump_json()


def test_search_application_key_must_not_reuse_openai_compatible_key() -> None:
    secret = "shared-credential"
    with pytest.raises(ValueError, match="must be distinct") as exc_info:
        Settings(
            api_key=secret,
            search_api_key=secret,
            _env_file=None,
        )
    assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    "search_api_key",
    ["contains space", "contains\ttab", "contains\nnewline", "unicode-密钥", "x" * 513],
)
def test_search_application_key_rejects_ambiguous_header_values(search_api_key: str) -> None:
    with pytest.raises(ValueError, match="Search API key") as exc_info:
        Settings(search_api_key=search_api_key, _env_file=None)
    assert search_api_key not in str(exc_info.value)


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
