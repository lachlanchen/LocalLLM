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
