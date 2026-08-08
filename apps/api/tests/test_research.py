from __future__ import annotations

from localllm.research import ResearchManager


def test_clean_extracted_text_removes_embedded_payloads() -> None:
    payload = "A" * 800
    text = f"Useful evidence. data:image/png;base64,{payload} More evidence."

    cleaned = ResearchManager._clean_extracted_text(text)

    assert "Useful evidence." in cleaned
    assert "More evidence." in cleaned
    assert payload not in cleaned
    assert "omitted" in cleaned
