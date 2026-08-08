from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator, model_validator
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

    @field_validator("allowed_origins", "allowed_hosts", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def require_loopback_binding(self) -> Settings:
        if self.host != "127.0.0.1" or self.port != 8008:
            raise ValueError(
                "LocalLLM Studio is loopback-only and uses the fixed loopback endpoint "
                "127.0.0.1:8008; "
                "use an authenticated tunnel for remote access"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
