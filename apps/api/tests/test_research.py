from __future__ import annotations

import pytest

from localllm.research import ResearchManager


def test_clean_extracted_text_removes_embedded_payloads() -> None:
    payload = "A" * 800
    text = f"Useful evidence. data:image/png;base64,{payload} More evidence."

    cleaned = ResearchManager._clean_extracted_text(text)

    assert "Useful evidence." in cleaned
    assert "More evidence." in cleaned
    assert payload not in cleaned
    assert "omitted" in cleaned


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/private",
        "http://[::1]/private",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.8/internal",
        "ftp://example.com/file",
        "https://user:password@example.com/",
        "https://example.com:8443/",
    ],
)
async def test_research_fetch_rejects_non_public_targets(url: str) -> None:
    assert not await ResearchManager._is_public_http_url(url)


@pytest.mark.asyncio
async def test_research_fetch_accepts_public_literal_address() -> None:
    assert await ResearchManager._is_public_http_url("https://1.1.1.1/document")
