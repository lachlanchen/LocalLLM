from __future__ import annotations

import json
import os
import re
import stat
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,199}$")
_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+\-]{0,127}$")
_IMMUTABLE_RELEASE_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{8}$")
DEFAULT_REQUIRED_MODELS = (
    "localllm-fast",
    "localllm-deep",
    "localllm-vision",
    "localllm-embed",
)
DEFAULT_NODE_CANARY_ROLES = ("text", "vision", "embedding")
_NODE_CANARY_ROLES = frozenset({"text", "code", "vision", "embedding"})
_SEARCH_CREDENTIAL_NAME = "localllm-search-api-key"
_SPEECH_CREDENTIAL_NAME = "localllm-speech-api-key"


def _systemd_private_credential(path: Path, name: str, label: str) -> bool:
    directory_value = os.environ.get("CREDENTIALS_DIRECTORY")
    if directory_value is None:
        return False
    directory = Path(directory_value)
    if not directory.is_absolute():
        raise ValueError("Systemd credentials directory must be absolute")
    expected = directory / name
    if path.parent == directory and path != expected:
        raise ValueError(f"{label} API key must use the fixed systemd credential name")
    return path == expected


def _validate_private_key_metadata(
    entry: os.stat_result, *, systemd_managed: bool, label: str
) -> None:
    common_valid = stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1
    owner_private = (
        entry.st_uid == os.getuid()
        and entry.st_gid == os.getgid()
        and stat.S_IMODE(entry.st_mode) & 0o077 == 0
    )
    # A system service with User= materializes LoadCredential= as root:root 0440
    # plus a service-user read ACL. This exact representation is acceptable only
    # at the fixed path beneath the manager-supplied CREDENTIALS_DIRECTORY.
    systemd_private = (
        systemd_managed
        and entry.st_uid == 0
        and entry.st_gid == 0
        and stat.S_IMODE(entry.st_mode) == 0o440
    )
    if not common_valid or not (owner_private or systemd_private):
        raise ValueError(
            f"{label} API key file must be one private regular file with an approved owner"
        )


def _validate_search_key_metadata(entry: os.stat_result, *, systemd_managed: bool) -> None:
    _validate_private_key_metadata(entry, systemd_managed=systemd_managed, label="Search")


def _read_private_application_key(
    path_value: object, *, credential_name: str, label: str
) -> SecretStr:
    try:
        path = Path(path_value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{label} API key file must be an absolute private file") from exc
    if not path.is_absolute():
        raise ValueError(f"{label} API key file must be an absolute private file")
    systemd_managed = _systemd_private_credential(path, credential_name, label)

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} API key file could not be opened safely") from exc
    try:
        entry = os.fstat(descriptor)
        _validate_private_key_metadata(entry, systemd_managed=systemd_managed, label=label)
        if entry.st_size < 1 or entry.st_size > 512:
            raise ValueError(f"{label} API key file must contain 1 to 512 bytes")
        chunks: list[bytes] = []
        size = 0
        while size <= 512:
            chunk = os.read(descriptor, min(513 - size, 513))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        payload = b"".join(chunks)
        if len(payload) < 1 or len(payload) > 512:
            raise ValueError(f"{label} API key file must contain 1 to 512 bytes")
        after = os.fstat(descriptor)
        before_identity = (
            entry.st_dev,
            entry.st_ino,
            entry.st_mode,
            entry.st_nlink,
            entry.st_uid,
            entry.st_gid,
            entry.st_size,
            entry.st_mtime_ns,
            entry.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(payload) != entry.st_size:
            raise ValueError(f"{label} API key file changed while it was being read")
        if any(byte < 0x21 or byte > 0x7E for byte in payload):
            raise ValueError(
                f"{label} API key file must contain visible ASCII without whitespace"
            )
        return SecretStr(payload.decode("ascii"))
    finally:
        os.close(descriptor)


def _read_private_search_key(path_value: object) -> SecretStr:
    """Read one exact private search credential without following the final path."""

    return _read_private_application_key(
        path_value, credential_name=_SEARCH_CREDENTIAL_NAME, label="Search"
    )


def _read_private_speech_key(path_value: object) -> SecretStr:
    """Read one exact private speech credential without following the final path."""

    return _read_private_application_key(
        path_value, credential_name=_SPEECH_CREDENTIAL_NAME, label="Speech"
    )


def prepare_private_data_dir(path: Path) -> Path:
    """Create/open the configured data directory without following a final symlink."""

    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("LocalLLM data directory must be a real directory, not a symlink") from exc
    try:
        entry = os.fstat(descriptor)
        if not stat.S_ISDIR(entry.st_mode):
            raise ValueError("LocalLLM data path is not a directory")
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
    return path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_prefix="LOCALLLM_",
        extra="ignore",
        hide_input_in_errors=True,
    )

    host: str = "127.0.0.1"
    port: int = 8008
    release_id: str = "dev"
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
    # Readiness is an operator contract, not merely a process check. The default
    # matches ``scripts/pull-models.sh core``; role-specific compute nodes can
    # narrow this list, while an explicit empty list requires only Ollama's
    # model catalog to be reachable.
    required_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_REQUIRED_MODELS)
    )
    # Functional evidence is deliberately separate from lightweight catalog
    # readiness. The practical core profile does not promise the optional 19 GB
    # coding specialist; a full workstation can opt into all four roles.
    node_canary_roles: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_NODE_CANARY_ROLES)
    )
    node_canary_receipt_path: Path | None = None
    node_canary_max_age_seconds: int = Field(default=86_400, ge=60, le=604_800)
    ghidra_home: Path = Path("./.local/opt/ghidra_12.0.3_PUBLIC")
    oghidra_home: Path = Path("./.local/tools/OGhidra")
    pyghidra_mcp_url: str = "http://127.0.0.1:18765/mcp"

    # Agent-side code execution is intentionally a separate, operator-controlled
    # capability. Building the sandbox image does not enable it.
    agent_code_execution_enabled: bool = False

    # Image generation is an optional, isolated worker. Installing its runtime or
    # model does not enable the API or reserve a GPU.
    image_generation_enabled: bool = False
    image_generation_gpu: int = Field(default=0, ge=0, le=15)
    image_generation_timeout_seconds: int = Field(default=300, ge=60, le=900)

    # Speech transcription is an optional local role. It reuses an operator-selected
    # cached Faster-Whisper model through an isolated persistent worker and never
    # makes model downloads at request time.
    speech_enabled: bool = False
    speech_model_path: Path | None = None
    speech_python_path: Path = Path("/home/lachlan/miniconda3/envs/whisperx/bin/python")
    speech_ffprobe_path: Path = Path("/usr/bin/ffprobe")
    speech_device: Literal["cpu", "cuda"] = "cpu"
    speech_device_index: int = Field(default=1, ge=0, le=15)
    speech_compute_type: Literal["int8", "int8_float16", "float16", "float32"] = "int8"
    speech_timeout_seconds: int = Field(default=180, ge=30, le=600)
    speech_worker_start_timeout_seconds: int = Field(default=180, ge=30, le=600)
    speech_api_key: SecretStr = SecretStr("")
    speech_api_key_file: Path | None = None

    # Application authorization for the quick-search route is independent from both
    # the OpenAI-compatible API key and outbound provider credentials. Leaving this
    # empty preserves the original loopback-only local workflow.
    search_api_key: SecretStr = SecretStr("")
    # Production services can keep the application credential in a systemd
    # LoadCredential= file and pass only its path via LOCALLLM_SEARCH_API_KEY_FILE.
    # Inline and file-backed values are mutually exclusive so precedence is never
    # environment-order dependent.
    search_api_key_file: Path | None = None

    # Search provider credentials are optional. Providers with no key remain available,
    # while configured providers are federated and their failures are reported per request.
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

    @field_validator(
        "allowed_origins", "allowed_hosts", "required_models", "node_canary_roles", mode="before"
    )
    @classmethod
    def parse_list_setting(cls, value: object) -> object:
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("required_models")
    @classmethod
    def validate_required_models(cls, value: list[str]) -> list[str]:
        if len(value) > 32:
            raise ValueError("At most 32 required models may be configured")
        normalized: list[str] = []
        seen: set[str] = set()
        for model in value:
            candidate = model.strip()
            if "://" in candidate or not _MODEL_ID.fullmatch(candidate):
                raise ValueError("Required model identifiers must use the local model ID syntax")
            if candidate not in seen:
                normalized.append(candidate)
                seen.add(candidate)
        return normalized

    @field_validator("node_canary_roles")
    @classmethod
    def validate_node_canary_roles(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one node canary role must be configured")
        normalized: list[str] = []
        seen: set[str] = set()
        for role in value:
            candidate = role.strip().lower()
            if candidate not in _NODE_CANARY_ROLES:
                raise ValueError("Node canary roles must use supported role names")
            if candidate in seen:
                raise ValueError("Node canary roles must be unique")
            normalized.append(candidate)
            seen.add(candidate)
        return normalized

    @field_validator("release_id")
    @classmethod
    def validate_release_id(cls, value: str) -> str:
        candidate = value.strip()
        if not _RELEASE_ID.fullmatch(candidate):
            raise ValueError("LocalLLM release ID must use the bounded non-secret ID syntax")
        return candidate

    @field_validator("search_api_key")
    @classmethod
    def validate_search_api_key(cls, value: SecretStr) -> SecretStr:
        candidate = value.get_secret_value()
        if not candidate:
            return value
        try:
            encoded = candidate.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("Search API key must use visible ASCII characters") from exc
        if len(encoded) > 512 or any(byte < 0x21 or byte > 0x7E for byte in encoded):
            raise ValueError(
                "Search API key must contain at most 512 visible ASCII characters without whitespace"
            )
        return value

    @field_validator("speech_api_key")
    @classmethod
    def validate_speech_api_key(cls, value: SecretStr) -> SecretStr:
        candidate = value.get_secret_value()
        if not candidate:
            return value
        try:
            encoded = candidate.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("Speech API key must use visible ASCII characters") from exc
        if len(encoded) > 512 or any(byte < 0x21 or byte > 0x7E for byte in encoded):
            raise ValueError(
                "Speech API key must contain at most 512 visible ASCII characters without whitespace"
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def load_file_backed_search_api_key(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("search_api_key_file") is None:
            return value
        if "search_api_key" in value:
            raise ValueError("Search API key and Search API key file are mutually exclusive")
        loaded = dict(value)
        loaded["search_api_key"] = _read_private_search_key(value["search_api_key_file"])
        return loaded

    @model_validator(mode="before")
    @classmethod
    def load_file_backed_speech_api_key(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("speech_api_key_file") is None:
            return value
        if "speech_api_key" in value:
            raise ValueError("Speech API key and Speech API key file are mutually exclusive")
        loaded = dict(value)
        loaded["speech_api_key"] = _read_private_speech_key(value["speech_api_key_file"])
        return loaded

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
        search_api_key = self.search_api_key.get_secret_value()
        if search_api_key and search_api_key == self.api_key:
            raise ValueError("Search API key must be distinct from the OpenAI-compatible API key")
        speech_api_key = self.speech_api_key.get_secret_value()
        if speech_api_key and speech_api_key in {self.api_key, search_api_key}:
            raise ValueError("Speech API key must be distinct from other application API keys")
        if self.speech_enabled:
            if not speech_api_key:
                raise ValueError("Enabled speech transcription requires a dedicated Speech API key")
            if self.speech_model_path is None or not self.speech_model_path.is_absolute():
                raise ValueError("Enabled speech transcription requires an absolute cached model path")
            if not self.speech_python_path.is_absolute() or not self.speech_ffprobe_path.is_absolute():
                raise ValueError("Speech worker executables must use absolute paths")
        if self.node_canary_receipt_path is not None:
            if not _IMMUTABLE_RELEASE_ID.fullmatch(self.release_id):
                raise ValueError(
                    "Configured node canary receipts require an immutable commit/archive release ID"
                )
            data_dir = Path(os.path.abspath(os.fspath(self.data_dir)))
            receipt_path = Path(os.path.abspath(os.fspath(self.node_canary_receipt_path)))
            expected_path = data_dir / "node-canaries" / f"{self.release_id}.json"
            if receipt_path != expected_path:
                raise ValueError(
                    "Node canary receipt must use the dedicated release-bound data path"
                )
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(expected_path.parent, flags)
            except OSError as exc:
                raise ValueError(
                    "Node canary receipt directory must already exist and be owner-private"
                ) from exc
            try:
                entry = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(entry.st_mode)
                    or entry.st_uid != os.getuid()
                    or stat.S_IMODE(entry.st_mode) & 0o077
                ):
                    raise ValueError(
                        "Node canary receipt directory must already exist and be owner-private"
                    )
            finally:
                os.close(descriptor)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
