from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_prefix="LOCALLLM_",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8008
    api_key: str = "local-dev-key"
    data_dir: Path = Path("./data")
    allowed_origins: Annotated[list[str], NoDecode] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8008",
        "http://localhost:8008",
    ]
    allowed_hosts: Annotated[list[str], NoDecode] = [
        "127.0.0.1",
        "localhost",
        "testserver",
    ]
    ollama_base_url: str = "http://127.0.0.1:11434"
    ghidra_home: Path = Path("./.local/opt/ghidra_12.0.3_PUBLIC")
    oghidra_home: Path = Path("./.local/tools/OGhidra")
    pyghidra_mcp_url: str = "http://127.0.0.1:18765/mcp"

    # Search credentials are optional. Providers with no key remain available, while
    # configured providers are federated and their failures are reported per request.
    search_brave_api_key: str = ""
    search_tavily_api_key: str = ""
    search_serper_api_key: str = ""
    search_serpapi_api_key: str = ""
    search_openalex_api_key: str = ""
    search_semantic_scholar_api_key: str = ""
    search_crossref_email: str = ""
    search_max_results: int = Field(default=30, ge=1, le=50)
    search_max_concurrency: int = Field(default=4, ge=1, le=8)
    search_provider_timeout_seconds: float = Field(default=12.0, ge=2.0, le=30.0)
    search_response_limit_bytes: int = Field(default=2_000_000, ge=100_000, le=5_000_000)

    @field_validator("allowed_origins", "allowed_hosts", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def require_local_ollama(cls, value: str) -> str:
        parsed = urlsplit(value.strip())
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Ollama runtime URL must be the local endpoint") from exc
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or port != 11434
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama runtime must remain local at http://127.0.0.1:11434")
        return "http://127.0.0.1:11434"

    @model_validator(mode="after")
    def require_loopback_binding(self) -> Settings:
        if self.host != "127.0.0.1" or self.port != 8008:
            raise ValueError(
                "LocalLLM Studio is loopback-only and uses the fixed loopback endpoint "
                "127.0.0.1:8008; "
                "add a separately authenticated access-control layer before any remote use"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
