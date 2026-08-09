"""Resource-limited DDGS worker used by the federated search broker.

The DDGS/primp stack eagerly buffers and decodes search-engine responses. Running it
out of process makes timeout cancellation effective and gives that unavoidable eager
allocation a hard address-space ceiling. Only a small normalized JSON payload crosses
back into the API process.
"""

from __future__ import annotations

import json
import resource
import sys
from typing import Any

MAX_FIELD_CHARS = 8_192
MAX_RESULTS = 20
MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
CPU_LIMIT_SECONDS = 15
FILE_DESCRIPTOR_LIMIT = 64


def _apply_resource_limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS))
    resource.setrlimit(
        resource.RLIMIT_NOFILE,
        (FILE_DESCRIPTOR_LIMIT, FILE_DESCRIPTOR_LIMIT),
    )


def _bounded_string(value: object) -> str:
    return str(value or "")[:MAX_FIELD_CHARS]


def _safe_results(records: object, limit: int) -> list[dict[str, str]]:
    if not isinstance(records, list):
        return []
    output: list[dict[str, str]] = []
    for raw in records[: min(MAX_RESULTS, limit)]:
        if not isinstance(raw, dict):
            continue
        output.append(
            {
                "title": _bounded_string(raw.get("title")),
                "href": _bounded_string(raw.get("href") or raw.get("url")),
                "body": _bounded_string(raw.get("body") or raw.get("snippet")),
            }
        )
    return output


def _emit(payload: dict[str, Any], output_limit: int) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > output_limit:
        encoded = b'{"ok":false,"error":"output_limit"}'
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def main() -> int:
    if len(sys.argv) != 5:
        return 2
    backend = sys.argv[1]
    if backend not in {"duckduckgo", "brave", "yahoo", "mojeek"}:
        return 2
    try:
        limit = max(1, min(MAX_RESULTS, int(sys.argv[2])))
        timeout = max(2, min(10, int(sys.argv[3])))
        output_limit = max(100_000, min(5_000_000, int(sys.argv[4])))
    except ValueError:
        return 2

    _apply_resource_limits()
    query = sys.stdin.buffer.read(4_096).decode("utf-8", errors="replace").strip()
    if not query:
        _emit({"ok": False, "error": "empty_query"}, output_limit)
        return 0

    try:
        # Import only after the resource ceiling is active. The worker receives no
        # service environment or API keys from its parent.
        from ddgs import DDGS

        with DDGS(timeout=timeout) as client:
            records = client.text(
                query,
                max_results=limit,
                backend=backend,
            )
        _emit({"ok": True, "results": _safe_results(records, limit)}, output_limit)
    except Exception:
        _emit({"ok": False, "error": "provider_failure"}, output_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
