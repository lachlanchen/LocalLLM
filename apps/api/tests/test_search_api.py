from __future__ import annotations

from fastapi.testclient import TestClient

from localllm.main import app
from localllm.research import ResearchTask
from localllm.search import ProviderDiagnostic, ResearchSource, SearchOutcome


def normalized_source() -> ResearchSource:
    return ResearchSource(
        title="Verified source",
        url="https://example.com/report",
        snippet="Evidence",
        provider="crossref",
        providers=["crossref", "semantic_scholar"],
        kind="paper",
        authors=["Ada Lovelace"],
        year=2024,
        published_date="2024-01-02",
        doi="10.1000/example",
        citation_count=9,
        score=4.2,
        query="verified research",
        provenance=[{"provider": "crossref", "query": "verified research"}],
    )


def test_search_status_exposes_capabilities_without_credentials() -> None:
    with TestClient(app) as client:
        response = client.get("/api/search/status")

    assert response.status_code == 200
    body = response.json()
    assert body["modes"] == ["web", "papers", "both"]
    assert {provider["name"] for provider in body["providers"]} >= {
        "duckduckgo",
        "crossref",
        "semantic_scholar",
        "arxiv",
        "europe_pmc",
        "openalex",
        "google_scholar_serpapi",
    }
    assert "api_key" not in response.text.lower()


def test_quick_search_endpoint_returns_normalized_ranked_sources() -> None:
    outcome = SearchOutcome(
        query="verified research",
        mode="papers",
        sources=[normalized_source()],
        providers=[ProviderDiagnostic("crossref", "paper", True, 1, 25)],
    )
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int):
            assert (query, mode, limit) == ("verified research", "papers", 7)
            return outcome

        manager.quick_search = search
        response = client.post(
            "/api/search",
            json={"query": "verified research", "mode": "papers", "limit": 7},
        )

    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert source["providers"] == ["crossref", "semantic_scholar"]
    assert source["kind"] == "paper"
    assert source["doi"] == "10.1000/example"
    assert source["citation_count"] == 9
    assert source["provenance"][0]["provider"] == "crossref"


def test_search_endpoint_bounds_mode_query_and_result_count() -> None:
    with TestClient(app) as client:
        bad_mode = client.post("/api/search", json={"query": "valid query", "mode": "scrape"})
        too_many = client.post(
            "/api/search", json={"query": "valid query", "mode": "web", "limit": 999}
        )
        short = client.post("/api/search", json={"query": "x", "mode": "web"})

    assert bad_mode.status_code == 422
    assert too_many.status_code == 422
    assert short.status_code == 422


def test_search_endpoint_caps_chunked_json_and_does_not_echo_extra_values() -> None:
    secret = "DO-NOT-ECHO-THIS-VALUE"

    def oversized_chunks():
        yield b'{"query":"valid query","junk":"'
        yield b"x" * (20 * 1024)
        yield b'"}'

    with TestClient(app) as client:
        oversized = client.post(
            "/api/search",
            content=oversized_chunks(),
            headers={"Content-Type": "application/json"},
        )
        extra = client.post(
            "/api/search",
            json={"query": "valid query", "junk": secret},
        )

    assert oversized.status_code == 413
    assert extra.status_code == 422
    assert secret not in extra.text


def test_research_endpoint_rejects_oversized_declared_body_before_parsing() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/research",
            content=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(64 * 1024),
            },
        )

    assert response.status_code == 413


def test_research_endpoint_rejects_whitespace_only_question() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/research",
            json={"question": "            ", "model": "localllm-pocket"},
        )

    assert response.status_code == 422


def test_research_request_forwards_deterministic_mode_and_depth() -> None:
    captured: dict[str, str] = {}
    with TestClient(app) as client:
        manager = client.app.state.research

        def create(question: str, model: str, mode: str, depth: str):
            captured.update(question=question, model=model, mode=mode, depth=depth)
            return ResearchTask(
                id="cafebabefeed",
                question=question,
                model=model,
                mode=mode,
                depth=depth,
            )

        manager.create = create
        response = client.post(
            "/api/research",
            json={
                "question": "What does the literature establish?",
                "model": "localllm-fast",
                "mode": "papers",
                "depth": "deep",
            },
        )

    assert response.status_code == 200
    assert captured == {
        "question": "What does the literature establish?",
        "model": "localllm-fast",
        "mode": "papers",
        "depth": "deep",
    }
    assert response.json()["mode"] == "papers"
    assert response.json()["depth"] == "deep"
