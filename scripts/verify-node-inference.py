#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from localllm.node_canary import (
    CANARY_ROLES,
    DEFAULT_OLLAMA_BASE_URL,
    MAX_ROLE_TIMEOUT_SECONDS,
    MIN_ROLE_TIMEOUT_SECONDS,
    CanaryContractError,
    api_key_from_environment,
    atomic_write_canary_receipt,
    cancelled_receipt,
    canonical_canary_receipt_bytes,
    failed_receipt,
    read_private_api_key,
    validate_canary_output_path,
    validate_loopback_ollama_base_url,
    validate_roles,
    verify_node_inference,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CanaryContractError("invalid verifier arguments")


def _parser(environment: Mapping[str, str]) -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description=(
            "Run bounded, content-free LocalLLM inference canaries. The API key is read "
            "from an environment variable or owner-private file and is never accepted as a value argument."
        )
    )
    parser.add_argument(
        "--base-url",
        default=environment.get(
            "LOCALLLM_NODE_CANARY_BASE_URL", "http://127.0.0.1:8008/v1"
        ),
        help="literal loopback HTTP /v1 endpoint",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="literal loopback HTTP Ollama origin used only for exact-tag cleanup",
    )
    parser.add_argument(
        "--roles",
        default=",".join(CANARY_ROLES),
        help="comma-separated subset of text,code,vision,embedding",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help=(
            f"default timeout for each role ({MIN_ROLE_TIMEOUT_SECONDS:g}-"
            f"{MAX_ROLE_TIMEOUT_SECONDS:g} seconds)"
        ),
    )
    parser.add_argument(
        "--role-timeout",
        action="append",
        default=[],
        metavar="ROLE=SECONDS",
        help="override one selected role timeout; may be repeated",
    )
    parser.add_argument(
        "--api-key-env",
        default="LOCALLLM_API_KEY",
        help="name of the environment variable containing the API key",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="owner-private regular file containing the API key",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(environment.get("LOCALLLM_DATA_DIR", "data")),
        help="LocalLLM private data directory used to constrain --output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional canonical --data-dir/node-canaries/<release-id>.json receipt",
    )
    return parser


def _selected_roles(raw: str) -> tuple[str, ...]:
    if len(raw) > 128:
        raise CanaryContractError("role selection is too long")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) != len(set(values)):
        raise CanaryContractError("roles must not be repeated")
    return validate_roles(values)


def _role_timeouts(
    roles: Sequence[str], default_timeout: float, overrides: Sequence[str]
) -> dict[str, float]:
    if not MIN_ROLE_TIMEOUT_SECONDS <= default_timeout <= MAX_ROLE_TIMEOUT_SECONDS:
        raise CanaryContractError("default timeout is outside the bounded range")
    result = {role: default_timeout for role in roles}
    seen: set[str] = set()
    for raw in overrides:
        role, separator, raw_seconds = raw.partition("=")
        if not separator or role not in result or role in seen:
            raise CanaryContractError("invalid role timeout override")
        try:
            seconds = float(raw_seconds)
        except ValueError as exc:
            raise CanaryContractError("invalid role timeout override") from exc
        if not MIN_ROLE_TIMEOUT_SECONDS <= seconds <= MAX_ROLE_TIMEOUT_SECONDS:
            raise CanaryContractError(
                "role timeout override is outside the bounded range"
            )
        result[role] = seconds
        seen.add(role)
    return result


def _absolute_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_stdout(receipt: object) -> None:
    sys.stdout.buffer.write(canonical_canary_receipt_bytes(receipt))
    sys.stdout.buffer.flush()


def main(
    argv: Sequence[str] | None = None, environment: Mapping[str, str] | None = None
) -> int:
    current_environment = os.environ if environment is None else environment
    roles: tuple[str, ...] = CANARY_ROLES
    output: Path | None = None
    data_dir = PROJECT_ROOT / "data"
    persistence_failed = False
    exit_code = 2
    try:
        arguments = _parser(current_environment).parse_args(argv)
        roles = _selected_roles(arguments.roles)
        timeouts = _role_timeouts(
            roles, arguments.timeout_seconds, arguments.role_timeout
        )
        ollama_base_url = validate_loopback_ollama_base_url(
            arguments.ollama_base_url
        )
        data_dir = _absolute_project_path(arguments.data_dir)
        output = (
            _absolute_project_path(arguments.output)
            if arguments.output is not None
            else None
        )
        if output is not None:
            validate_canary_output_path(output, data_dir)
        if arguments.api_key_file is not None:
            api_key = read_private_api_key(
                _absolute_project_path(arguments.api_key_file)
            )
        else:
            api_key = api_key_from_environment(
                current_environment, arguments.api_key_env
            )
        receipt = asyncio.run(
            verify_node_inference(
                arguments.base_url,
                api_key,
                roles,
                timeouts,
                ollama_base_url=ollama_base_url,
            )
        )
        exit_code = 0 if receipt["status"] == "passed" else 1
    except KeyboardInterrupt:
        receipt = cancelled_receipt()
        exit_code = 130
    except (CanaryContractError, OSError, RuntimeError, ValueError):
        receipt = failed_receipt(roles)
        exit_code = 2

    if output is not None:
        try:
            atomic_write_canary_receipt(receipt, output, data_dir)
        except (CanaryContractError, OSError):
            persistence_failed = True
    _write_stdout(receipt)
    return 2 if persistence_failed else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
