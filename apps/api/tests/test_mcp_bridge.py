from __future__ import annotations

from types import SimpleNamespace

import pytest

from localllm.mcp_bridge import _tool_payload, validate_loopback_mcp_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:18765/mcp",
        "http://localhost:18765/mcp",
        "http://[::1]:18765/mcp",
    ],
)
def test_mcp_url_accepts_explicit_loopback(url: str) -> None:
    validate_loopback_mcp_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0:18765/mcp",
        "http://192.168.1.5:18765/mcp",
        "https://example.com/mcp",
        "file:///tmp/mcp",
        "http://user:secret@127.0.0.1:18765/mcp",
    ],
)
def test_mcp_url_rejects_non_loopback_or_credentials(url: str) -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        validate_loopback_mcp_url(url)


def test_tool_payload_prefers_structured_content() -> None:
    result = SimpleNamespace(
        isError=False,
        structuredContent={"imports": [{"name": "libusb"}]},
        content=[],
    )

    assert _tool_payload(result) == {"imports": [{"name": "libusb"}]}


def test_tool_payload_surfaces_mcp_error() -> None:
    result = SimpleNamespace(
        isError=True,
        structuredContent=None,
        content=[SimpleNamespace(type="text", text="blocked")],
    )

    with pytest.raises(RuntimeError, match="blocked"):
        _tool_payload(result)
