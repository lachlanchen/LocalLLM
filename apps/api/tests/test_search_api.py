from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from localllm.config import Settings, get_settings
from localllm.main import SearchAuthenticationError, app, require_search_api_key
from localllm.research import ResearchTask
from localllm.search import ProviderDiagnostic, ResearchSource, SearchOutcome

SEARCH_API_KEY = "search-only-credential-0123456789abcdef"
OPENAI_API_KEY = "openai-only-credential-0123456789abcdef"


@contextmanager
def configured_search_auth() -> Iterator[Settings]:
    settings = Settings(
        api_key=OPENAI_API_KEY,
        search_api_key=SEARCH_API_KEY,
        _env_file=None,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        yield settings
    finally:
        app.dependency_overrides.clear()


def empty_search_outcome(query: str, mode: str) -> SearchOutcome:
    return SearchOutcome(query=query, mode=mode, sources=[], providers=[])


def assert_private_search_response(response) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers


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
    assert response.headers["cache-control"] == "no-store"
    assert body["authentication"] == {
        "required": False,
        "scheme": "bearer",
        "scope": "quick-search",
    }


def test_search_status_reports_auth_requirement_without_revealing_credential() -> None:
    with configured_search_auth():
        with TestClient(app) as client:
            response = client.get("/api/search/status")

    assert response.status_code == 200
    assert response.json()["authentication"] == {
        "required": True,
        "scheme": "bearer",
        "scope": "quick-search",
    }
    assert SEARCH_API_KEY not in response.text
    assert OPENAI_API_KEY not in response.text
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    "headers",
    [
        None,
        {"Authorization": "Bearer wrong-search-credential"},
        {"Authorization": f"bearer {SEARCH_API_KEY}"},
        {"Authorization": f"Bearer  {SEARCH_API_KEY}"},
        {"Authorization": f"Bearer\t{SEARCH_API_KEY}"},
        {"Authorization": f"Bearer {SEARCH_API_KEY} "},
        {"Authorization": f"Basic {SEARCH_API_KEY}"},
        {"Authorization": "Bearer"},
    ],
)
def test_configured_search_rejects_missing_wrong_or_malformed_bearer(
    headers: dict[str, str] | None,
) -> None:
    with configured_search_auth():
        with TestClient(app) as client:
            response = client.post(
                "/api/search",
                headers=headers,
                json={"query": "verified research", "mode": "papers"},
            )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid search API key"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers
    assert SEARCH_API_KEY not in response.text
    assert OPENAI_API_KEY not in response.text


def test_configured_search_rejects_duplicate_authorization_headers() -> None:
    with configured_search_auth():
        with TestClient(app) as client:
            response = client.post(
                "/api/search",
                headers=[
                    ("Authorization", f"Bearer {SEARCH_API_KEY}"),
                    ("Authorization", f"Bearer {SEARCH_API_KEY}"),
                ],
                json={"query": "verified research", "mode": "papers"},
            )

    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize(
    "authorization",
    [
        b"Bearer\x00search-only-credential-0123456789abcdef",
        b"Bearer\rsearch-only-credential-0123456789abcdef",
        b"Bearer\nsearch-only-credential-0123456789abcdef",
        b"Bearer\x7fsearch-only-credential-0123456789abcdef",
    ],
)
def test_search_dependency_rejects_control_characters_in_raw_authorization(
    authorization: bytes,
) -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/search",
            "headers": [(b"authorization", authorization)],
        }
    )
    settings = Settings(
        api_key=OPENAI_API_KEY,
        search_api_key=SEARCH_API_KEY,
        _env_file=None,
    )

    with pytest.raises(SearchAuthenticationError):
        require_search_api_key(request, settings)


def test_search_credential_is_scoped_to_search_and_openai_key_cannot_authorize_search() -> None:
    with configured_search_auth():
        with TestClient(app) as client:
            search_with_openai_key = client.post(
                "/api/search",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"query": "verified research", "mode": "papers"},
            )
            openai_with_search_key = client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {SEARCH_API_KEY}"},
            )

    assert search_with_openai_key.status_code == 401
    assert openai_with_search_key.status_code == 401
    assert openai_with_search_key.json()["error"]["type"] == "authentication_error"
    assert openai_with_search_key.headers["www-authenticate"] == "Bearer"
    assert "cache-control" not in openai_with_search_key.headers


def test_configured_search_accepts_one_exact_bearer_and_marks_response_no_store() -> None:
    with configured_search_auth():
        with TestClient(app) as client:
            manager = client.app.state.research

            async def search(query: str, mode: str, limit: int):
                assert (query, mode, limit) == ("verified research", "papers", 5)
                return empty_search_outcome(query, mode)

            manager.quick_search = search
            response = client.post(
                "/api/search",
                headers={"Authorization": f"Bearer {SEARCH_API_KEY}"},
                json={"query": "verified research", "mode": "papers", "limit": 5},
            )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers
    assert response.json()["query"] == "verified research"


def test_unconfigured_search_preserves_loopback_unauthenticated_behavior() -> None:
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int):
            return empty_search_outcome(query, mode)

        manager.quick_search = search
        response = client.post(
            "/api/search",
            json={"query": "verified research", "mode": "web"},
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_authenticated_search_validation_error_is_never_cacheable() -> None:
    with configured_search_auth():
        with TestClient(app) as client:
            response = client.post(
                "/api/search",
                headers={"Authorization": f"Bearer {SEARCH_API_KEY}"},
                json={"query": "x", "mode": "web"},
            )

    assert response.status_code == 422
    assert_private_search_response(response)


def test_authenticated_search_malformed_json_is_never_cacheable() -> None:
    with configured_search_auth():
        with TestClient(app) as client:
            response = client.post(
                "/api/search",
                headers={
                    "Authorization": f"Bearer {SEARCH_API_KEY}",
                    "Content-Type": "application/json",
                },
                content=b'{"query":',
            )

    assert response.status_code == 400
    assert response.json() == {"detail": "Request body must be valid JSON"}
    assert_private_search_response(response)


def test_authenticated_search_oversized_declared_body_is_never_cacheable() -> None:
    with configured_search_auth():
        with TestClient(app) as client:
            response = client.post(
                "/api/search",
                headers={
                    "Authorization": f"Bearer {SEARCH_API_KEY}",
                    "Content-Type": "application/json",
                    "Content-Length": str(20 * 1024),
                },
                content=b"{}",
            )

    assert response.status_code == 413
    assert_private_search_response(response)


def test_authenticated_search_oversized_chunked_body_is_never_cacheable() -> None:
    def oversized_chunks():
        yield b'{"query":"verified research","padding":"'
        yield b"x" * (20 * 1024)
        yield b'"}'

    with configured_search_auth():
        with TestClient(app) as client:
            response = client.post(
                "/api/search",
                headers={
                    "Authorization": f"Bearer {SEARCH_API_KEY}",
                    "Content-Type": "application/json",
                },
                content=oversized_chunks(),
            )

    assert response.status_code == 413
    assert_private_search_response(response)


def test_authenticated_search_provider_http_error_cannot_set_cache_or_cookie() -> None:
    with configured_search_auth():
        with TestClient(app) as client:
            manager = client.app.state.research

            async def search(query: str, mode: str, limit: int):
                raise HTTPException(
                    status_code=502,
                    detail="Search provider unavailable",
                    headers={
                        "Cache-Control": "public, max-age=3600",
                        "Set-Cookie": "provider-session=must-not-survive",
                    },
                )

            manager.quick_search = search
            response = client.post(
                "/api/search",
                headers={"Authorization": f"Bearer {SEARCH_API_KEY}"},
                json={"query": "verified research", "mode": "both"},
            )

    assert response.status_code == 502
    assert response.json() == {"detail": "Search provider unavailable"}
    assert_private_search_response(response)


def test_authenticated_search_unhandled_internal_error_is_never_cacheable() -> None:
    with configured_search_auth():
        with TestClient(app, raise_server_exceptions=False) as client:
            manager = client.app.state.research

            async def search(query: str, mode: str, limit: int):
                raise RuntimeError("simulated internal provider failure")

            manager.quick_search = search
            response = client.post(
                "/api/search",
                headers={"Authorization": f"Bearer {SEARCH_API_KEY}"},
                json={"query": "verified research", "mode": "both"},
            )

    assert response.status_code == 500
    assert_private_search_response(response)


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


def test_quick_search_redacts_embedded_url_secrets_at_the_provider_boundary() -> None:
    captured: list[str] = []
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int, **_kwargs):
            captured.append(query)
            return SearchOutcome(query=query, mode=mode, sources=[], providers=[])

        manager.search.search = search
        response = client.post(
            "/api/search",
            json={
                "query": (
                    "Search [source](https://example.org/private?token=TOPSECRET) "
                    "and path:/home/alice/secret.txt"
                ),
                "mode": "web",
            },
        )

    assert response.status_code == 200
    assert captured == ["Search example.org and local path"]
    assert "TOPSECRET" not in response.text
    assert "alice" not in response.text.casefold()


def test_quick_search_suppresses_wrapped_labeled_and_encoded_private_values() -> None:
    queries = [
        "Verify 【/home/alice/My Project/TOPSECRET plan.txt】",
        "Verify URL:corpserver/private/TOPSECRET",
        "Verify example.com%252Fprivate%252FTOPSECRET?token=SIGNED",
        "Verify [path](/home/alice/My (Project) TOPSECRET/file.txt)",
        "Verify sms:+85212345678?body=TOPSECRET",
        "Verify github.com:private/TOPSECRET.git",
        "Verify ethereum:TOPSECRET",
        "Verify gitlab.com:TOPSECRET.git",
        "Verify data:text/plain;base64,VE9QU0VDUkVU",
        "Verify https://example.com/private,TOPSECRET",
    ]
    captured: list[str] = []
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int, **_kwargs):
            captured.append(query)
            return SearchOutcome(query=query, mode=mode, sources=[], providers=[])

        manager.search.search = search
        for query in queries:
            response = client.post("/api/search", json={"query": query, "mode": "web"})
            assert response.status_code == 200

    assert captured == [
        "Verify local path",
        "Verify network resource",
        "Verify example.com",
        "Verify local path",
        "Verify public resource",
        "Verify github.com",
        "Verify public resource",
        "Verify gitlab.com",
        "Verify public resource",
        "Verify example.com",
    ]
    assert all("TOPSECRET" not in query and "alice" not in query.casefold() for query in captured)


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
