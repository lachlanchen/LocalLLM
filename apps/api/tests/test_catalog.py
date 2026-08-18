import re
from pathlib import Path

from localllm.catalog import MODEL_ALIASES, MODEL_CATALOG, resolve_model


def test_catalog_ids_are_unique() -> None:
    ids = [model["id"] for model in MODEL_CATALOG]
    assert len(ids) == len(set(ids))


def test_every_alias_resolves_to_catalog() -> None:
    catalog_ids = {model["id"] for model in MODEL_CATALOG}
    assert set(MODEL_ALIASES.values()) <= catalog_ids
    assert resolve_model("localllm-fast") in catalog_ids


def test_unknown_model_is_preserved() -> None:
    assert resolve_model("custom/model") == "custom/model"


def test_code_alias_has_exact_curated_metadata() -> None:
    coder = next(model for model in MODEL_CATALOG if model["id"] == "qwen3-coder:30b-a3b-q4_K_M")
    assert MODEL_ALIASES["localllm-code"] == coder["id"]
    assert coder["size_gb"] == 19.0
    assert coder["context"] == 262144
    assert coder["modalities"] == ["text", "tools"]


def test_pull_profiles_keep_code_optional_but_include_it_in_all() -> None:
    script = (Path(__file__).parents[3] / "scripts" / "pull-models.sh").read_text()

    def array(name: str) -> list[str]:
        match = re.search(rf"^{name}=\(\n(?P<body>.*?)^\)$", script, re.MULTILINE | re.DOTALL)
        assert match is not None
        return [line.strip() for line in match.group("body").splitlines() if line.strip()]

    coder = "qwen3-coder:30b-a3b-q4_K_M"
    assert array("code_models") == [coder]
    assert coder in array("all_models")
    assert coder not in array("core_models")
    assert 'code) selected_models=("${code_models[@]}") ;;' in script
