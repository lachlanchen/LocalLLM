"""Hermetic process-local settings for the API test suite."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# pydantic-settings otherwise discovers the repository's deployment .env while
# test modules import the FastAPI application. Remove every inherited LocalLLM
# setting in this pytest process, disable dotenv loading on the Settings class,
# and keep all test data in a disposable directory before application imports.
for _key in tuple(os.environ):
    if _key.startswith("LOCALLLM_"):
        del os.environ[_key]

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="localllm-api-pytest-"))
atexit.register(shutil.rmtree, _TEST_DATA_DIR, ignore_errors=True)
os.environ["LOCALLLM_API_KEY"] = "local-dev-key"
os.environ["LOCALLLM_DATA_DIR"] = str(_TEST_DATA_DIR)

from localllm.config import Settings, get_settings  # noqa: E402

Settings.model_config["env_file"] = None


@pytest.fixture(autouse=True)
def hermetic_api_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCALLLM_API_KEY", "local-dev-key")
    monkeypatch.setenv("LOCALLLM_DATA_DIR", str(_TEST_DATA_DIR))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
