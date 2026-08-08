from __future__ import annotations

import asyncio
import ipaddress
import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from localllm.config import Settings
from localllm.research import (
    ResearchCapacityError,
    ResearchManager,
    ResearchSource,
    ResearchTask,
)


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


@pytest.mark.asyncio
async def test_research_connects_to_validated_ip_with_original_host_and_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    async def resolve(_url: str):
        return [ipaddress.ip_address("93.184.216.34")]

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(
            url=str(request.url),
            host=request.headers.get("host"),
            connection=request.headers.get("connection"),
            sni=request.extensions.get("sni_hostname"),
        )
        return httpx.Response(200, text="evidence")

    monkeypatch.setattr(ResearchManager, "_resolve_public_addresses", resolve)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False
    ) as client:
        response = await ResearchManager._get_pinned_response(
            client, "https://research.example/report"
        )
        assert response is not None
        await response.aclose()

    assert seen == {
        "url": "https://93.184.216.34/report",
        "host": "research.example",
        "connection": "close",
        "sni": "research.example",
    }


def make_manager(tmp_path: Path) -> ResearchManager:
    return ResearchManager(Settings(data_dir=tmp_path, ollama_base_url="http://ollama.test"))


def make_task(task_id: str = "research-test") -> ResearchTask:
    return ResearchTask(
        id=task_id,
        question="What does the evidence establish?",
        model="qwen3:4b-q4_K_M",
    )


@pytest.mark.asyncio
async def test_manager_tracks_and_cancels_background_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_release(_task: ResearchTask) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(manager, "_run", wait_for_release)
    task = manager.create("What evidence should this test inspect?", "localllm-pocket")
    await started.wait()

    assert task.id in manager._runners
    cancelled = await manager.cancel(task.id)
    await asyncio.sleep(0)

    assert cancelled is task
    assert task.status == "cancelled"
    assert task.stage == "Research cancelled"
    assert task.id not in manager._runners
    persisted = json.loads((manager.directory / f"{task.id}.json").read_text())
    assert persisted["status"] == "cancelled"


@pytest.mark.asyncio
async def test_manager_bounds_queue_and_shutdown_cancels_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(tmp_path)
    release = asyncio.Event()

    async def wait_for_release(_task: ResearchTask) -> None:
        await release.wait()

    monkeypatch.setattr(manager, "_run", wait_for_release)
    created = [
        manager.create(f"Research capacity question number {index}", "localllm-pocket")
        for index in range(manager.max_pending_tasks)
    ]

    with pytest.raises(ResearchCapacityError, match="queue is full"):
        manager.create("One research request too many", "localllm-pocket")

    await manager.shutdown()
    assert manager._runners == {}
    assert all(task.status == "cancelled" for task in created)


def mock_research_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    manager: ResearchManager,
    sources: list[ResearchSource],
    answers: list[str],
) -> list[list[dict[str, str]]]:
    prompts: list[list[dict[str, str]]] = []
    answer_iterator: Iterator[str] = iter(answers)

    async def plan(_task: ResearchTask) -> list[str]:
        return ["test query"]

    async def fetch(_client: object, _source: ResearchSource) -> None:
        return None

    async def public_url(_url: str) -> bool:
        return True

    async def model_chat(
        _task: ResearchTask, messages: list[dict[str, str]]
    ) -> str:
        prompts.append(messages)
        return next(answer_iterator)

    monkeypatch.setattr(manager, "_plan_queries", plan)
    monkeypatch.setattr(manager, "_search_sync", lambda _queries: sources)
    monkeypatch.setattr(manager, "_fetch_source", fetch)
    monkeypatch.setattr(manager, "_is_public_http_url", public_url)
    monkeypatch.setattr(manager, "_model_chat", model_chat)
    return prompts


@pytest.mark.asyncio
async def test_research_persists_exact_numbered_evidence_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(tmp_path)
    sources = [
        ResearchSource("Discarded", "https://example.com/empty", ""),
        ResearchSource("Snippet source", "https://example.com/snippet", "Useful snippet"),
        ResearchSource(
            "Extracted source", "https://example.org/page", "", "Extracted page evidence"
        ),
    ]
    prompts = mock_research_pipeline(
        monkeypatch, manager, sources, ["Supported claim [1][2]\n\n## Sources"]
    )
    task = make_task()

    await manager._run(task)

    assert task.status == "complete"
    assert [source.title for source in task.sources] == ["Snippet source", "Extracted source"]
    evidence_prompt = prompts[0][1]["content"]
    assert "SOURCE [1]\nTitle: Snippet source" in evidence_prompt
    assert "SOURCE [2]\nTitle: Extracted source" in evidence_prompt
    assert "Discarded" not in evidence_prompt
    persisted = json.loads((manager.directory / f"{task.id}.json").read_text())
    assert [source["title"] for source in persisted["sources"]] == [
        "Snippet source",
        "Extracted source",
    ]
    assert all("content" not in source for source in persisted["sources"])
    assert "[1] [Snippet source](https://example.com/snippet)" in task.report
    assert "[2] [Extracted source](https://example.org/page)" in task.report


@pytest.mark.asyncio
async def test_research_fails_without_usable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(tmp_path)
    prompts = mock_research_pipeline(
        monkeypatch,
        manager,
        [ResearchSource("Empty", "https://example.com/empty", "   ", "")],
        [],
    )
    task = make_task("no-evidence")

    await manager._run(task)

    assert task.status == "failed"
    assert task.sources == []
    assert task.error == "No usable public web evidence was found"
    assert prompts == []
    persisted = json.loads((manager.directory / f"{task.id}.json").read_text())
    assert persisted["status"] == "failed"


def test_interrupted_persisted_research_is_failed_on_reload(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    task = make_task("deadbeefcafe")
    task.status = "running"
    manager._persist(task)
    manager.tasks.clear()

    restored = manager.get(task.id)

    assert restored is not None
    assert restored.status == "failed"
    assert restored.stage == "Research interrupted"
    assert "restarted" in (restored.error or "")
    persisted = json.loads((manager.directory / f"{task.id}.json").read_text())
    assert persisted["status"] == "failed"
    assert persisted["sources"] == []


def test_research_lookup_rejects_non_opaque_task_ids(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    assert manager.get("../outside") is None
    assert manager.get("not-a-task") is None


def test_canonical_sources_replaces_styled_model_appendix() -> None:
    source = ResearchSource(
        "Official docs",
        "https://example.com/docs",
        "",
    )

    for heading in ("### **Sources**", "## _References_"):
        result = ResearchManager._with_canonical_sources(
            f"Supported statement [1]\n\n{heading}\n\n1. old link",
            [source],
        )

        assert "old link" not in result
        assert result.count("## Sources") == 1
        assert "[1] [Official docs](https://example.com/docs)" in result


def test_canonical_sources_sanitizes_untrusted_markdown_components() -> None:
    source = ResearchSource(
        "Trusted] title\n[99] [Injected",
        "https://example.com/report_(draft)?next=a b",
        "",
    )

    result = ResearchManager._with_canonical_sources("Supported [1]", [source])

    assert result.count("\n[99]") == 0
    assert "Trusted\\] title \\[99\\] \\[Injected" in result
    assert "report_%28draft%29?next=a%20b" in result


@pytest.mark.asyncio
async def test_research_drops_unsafe_snippet_only_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(tmp_path)
    sources = [
        ResearchSource("Unsafe", "http://127.0.0.1/admin", "Search-engine snippet"),
        ResearchSource("Public", "https://example.com/report", "Public evidence"),
    ]
    prompts = mock_research_pipeline(
        monkeypatch,
        manager,
        sources,
        ["Supported public finding [1]\n\n## Sources"],
    )

    async def validate(url: str) -> bool:
        return url.startswith("https://example.com/")

    monkeypatch.setattr(manager, "_is_public_http_url", validate)
    task = make_task("c0ffee123456")

    await manager._run(task)

    assert task.status == "complete"
    assert [source.title for source in task.sources] == ["Public"]
    assert "Unsafe" not in prompts[0][1]["content"]


def test_citation_validation_ignores_model_written_source_appendix() -> None:
    report = "Unsupported body\n\n### **Sources**\n\n[1] https://example.com"

    assert not ResearchManager._citations_are_valid(report, 1)


def test_citation_validation_requires_each_prose_or_list_unit() -> None:
    covered = (
        "## Finding\n\nSupported paragraph [1]\n\n"
        "1. **Controls**\n   - Supported bullet that wraps\n"
        "     onto another line [2]\n\n## Sources"
    )
    missing_paragraph = "Supported paragraph [1]\n\nUnsupported paragraph\n\n## Sources"
    missing_bullet = "Supported paragraph [1]\n\n- Unsupported bullet\n\n## Sources"

    assert ResearchManager._citations_are_valid(covered, 2)
    assert not ResearchManager._citations_are_valid(missing_paragraph, 2)
    assert not ResearchManager._citations_are_valid(missing_bullet, 2)


@pytest.mark.asyncio
async def test_research_repairs_and_revalidates_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(tmp_path)
    prompts = mock_research_pipeline(
        monkeypatch,
        manager,
        [ResearchSource("Evidence", "https://example.com/evidence", "Reliable fact")],
        ["Draft without citations", "Repaired supported claim [1]\n\n## Sources"],
    )
    task = make_task("repaired")

    await manager._run(task)

    assert task.status == "complete"
    assert task.report.startswith("Repaired supported claim [1]")
    assert len(prompts) == 2
    assert "NUMBERED EVIDENCE\nSOURCE [1]" in prompts[1][1]["content"]


@pytest.mark.asyncio
async def test_research_fails_when_repaired_citations_remain_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(tmp_path)
    mock_research_pipeline(
        monkeypatch,
        manager,
        [ResearchSource("Evidence", "https://example.com/evidence", "Reliable fact")],
        ["Unsupported claim [9]", "Still references a missing source [2]"],
    )
    task = make_task("invalid-repair")

    await manager._run(task)

    assert task.status == "failed"
    assert task.error == (
        "The local model could not produce a report with valid source citations"
    )
    persisted = json.loads((manager.directory / f"{task.id}.json").read_text())
    assert persisted["status"] == "failed"
