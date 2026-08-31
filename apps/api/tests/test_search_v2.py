from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

import localllm.main as main_module
import localllm.search_v2 as search_v2_module
from localllm.config import Settings, get_settings
from localllm.main import app
from localllm.search import FederatedSearch, ProviderDiagnostic, ResearchSource, SearchOutcome
from localllm.search_v2 import (
    POLICY_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    RESPONSE_SCHEMA_VERSION,
    SearchRequestV2,
    SearchResponseV2,
    canonical_arxiv_identifier,
    canonical_policy_preimage,
    canonical_returned_identity_preimage,
    canonical_source_identity_preimage,
    compute_policy_digest,
    compute_returned_identity_binding,
    compute_source_identity_digest,
    execute_search_v2,
    parse_search_v2_request,
)

SEARCH_API_KEY = "search-v2-only-credential-0123456789abcdef"
OPENAI_API_KEY = "openai-only-credential-0123456789abcdef"
QUERY_PLAN_DIGEST = "sha256:" + "1" * 64
POLICY_VECTOR_DIGEST = "sha256:ccf5b13b08f247de0033a2c1d4c9bd3866ae0a8ce2b9cf411907080f39ec629c"
RETURNED_VECTOR_DIGEST = "sha256:3ddb7b8783c4dd600be7eaeed485f1af2821624232eaab6526426a4027375dd8"
SOURCE_IDENTITY_VECTOR_DIGEST = (
    "sha256:d3ed58ed7c051d5948cfb3cb212a5b7d3ac06a4544c5e9c0b2a2f7d840db4857"
)
TWO_RECORD_BINDING_VECTOR_DIGEST = (
    "sha256:fc7e740594137849892f6d9cd6ac3ad82901eafe8361d43741f8a626c42774dd"
)


@contextmanager
def configured_search_auth() -> Iterator[None]:
    settings = Settings(
        api_key=OPENAI_API_KEY,
        search_api_key=SEARCH_API_KEY,
        _env_file=None,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        yield
    finally:
        app.dependency_overrides.clear()


def v2_payload(
    *,
    query: str = "verified research",
    mode: str = "papers",
    limit: int = 4,
    strategy: str = "ranked",
    allowed_domains: list[str] | None = None,
    exact_identifiers: list[dict[str, str]] | None = None,
    query_plan_digest: str = QUERY_PLAN_DIGEST,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": REQUEST_SCHEMA_VERSION,
        "query": query,
        "mode": mode,
        "limit": limit,
        "constraints": {
            "schemaVersion": POLICY_SCHEMA_VERSION,
            "strategy": strategy,
            "allowedDomains": allowed_domains or [],
            "exactIdentifiers": exact_identifiers or [],
            "queryPlanDigest": query_plan_digest,
            "policyDigest": "sha256:" + "0" * 64,
        },
    }
    provisional = SearchRequestV2.model_validate(payload)
    payload["constraints"]["policyDigest"] = compute_policy_digest(provisional)
    return payload


def resign_policy(payload: dict[str, Any]) -> None:
    """Sign the raw policy independently of the production Pydantic validators."""

    constraints = payload["constraints"]
    document = {
        "schemaVersion": payload["schemaVersion"],
        "query": payload["query"],
        "mode": payload["mode"],
        "limit": payload["limit"],
        "constraints": {
            "schemaVersion": constraints["schemaVersion"],
            "strategy": constraints["strategy"],
            "allowedDomains": constraints["allowedDomains"],
            "exactIdentifiers": constraints["exactIdentifiers"],
        },
    }
    preimage = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    constraints["policyDigest"] = "sha256:" + hashlib.sha256(preimage).hexdigest()


def source(
    url: str,
    *,
    doi: str | None = None,
    provider: str = "crossref",
    query: str = "verified research",
) -> ResearchSource:
    return ResearchSource(
        title="Verified source",
        url=url,
        snippet="Bounded public evidence",
        provider=provider,
        providers=[provider],
        kind="paper",
        authors=["Ada Lovelace"],
        year=2024,
        published_date="2024-01-02",
        doi=doi,
        citation_count=9,
        score=4.2,
        query=query,
        provenance=[{"provider": provider, "query": query}],
    )


def empty_outcome(query: str, mode: str = "papers") -> SearchOutcome:
    return SearchOutcome(query=query, mode=mode, sources=[], providers=[])


async def drain_exact_cleanup_reapers() -> None:
    while search_v2_module._exact_cleanup_reapers:
        await asyncio.gather(*tuple(search_v2_module._exact_cleanup_reapers))


def assert_private(response) -> None:
    assert response.headers["cache-control"] == "no-store"
    assert "set-cookie" not in response.headers


def test_ranked_v2_echoes_request_and_binds_ordered_canonical_source_identities() -> None:
    payload = v2_payload(allowed_domains=["example.com"])
    outcome = SearchOutcome(
        query="verified research",
        mode="papers",
        sources=[
            source("https://papers.example.com/report", doi="10.1000/example"),
            source("https://example.com.evil.test/report", doi="10.1000/attack"),
        ],
        providers=[ProviderDiagnostic("crossref", "paper", True, 2, 15)],
    )
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            assert (query, mode, limit) == ("verified research", "papers", 12)
            return outcome

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 200
    assert_private(response)
    body = response.json()
    assert body["schemaVersion"] == RESPONSE_SCHEMA_VERSION
    assert body["policyCompliant"] is True
    assert body["request"] == payload
    assert len(body["sources"]) == 1
    returned = body["sources"][0]
    assert returned["canonicalUrl"] == "https://papers.example.com/report"
    assert returned["domain"] == "papers.example.com"
    assert returned["identifiers"] == [{"kind": "doi", "value": "10.1000/example"}]
    assert returned["identityDigest"].startswith("sha256:")
    assert returned["rank"] == 1
    assert returned["matchedAllowedDomains"] == ["example.com"]
    assert returned["matchedExactIdentifiers"] == []
    assert body["returnedIdentityBinding"] == compute_returned_identity_binding(
        payload["constraints"]["queryPlanDigest"],
        payload["constraints"]["policyDigest"],
        [
            {
                "rank": 1,
                "identityDigest": returned["identityDigest"],
                "matchedAllowedDomains": ["example.com"],
                "matchedExactIdentifiers": [],
            }
        ],
    )
    assert body["resolvedIdentifiers"] == []
    assert body["unresolvedIdentifiers"] == []


def test_ranked_allowed_domains_overfetches_before_filtering_and_truncating() -> None:
    payload = v2_payload(limit=1, allowed_domains=["example.com"])
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            assert (query, mode, limit) == ("verified research", "papers", 3)
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[
                    source("https://outside.test/first"),
                    source("https://docs.example.com/second"),
                ],
                providers=[],
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 200
    assert [item["canonicalUrl"] for item in response.json()["sources"]] == [
        "https://docs.example.com/second"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_limit", [10, 12, 30])
async def test_ranked_domain_budget_reaches_one_provider_rank_13_with_a_hard_cap(
    request_limit: int,
) -> None:
    results = [
        source(
            f"https://outside-{rank}.test/paper",
            provider="bounded_provider",
        )
        for rank in range(1, 13)
    ]
    results.append(
        source(
            "https://docs.example.com/rank-13",
            provider="bounded_provider",
        )
    )

    class RankedProvider:
        name = "bounded_provider"
        kind = "paper"
        calls: list[tuple[str, int]] = []

        async def search(self, query: str, limit: int) -> list[ResearchSource]:
            self.calls.append((query, limit))
            return results[:limit]

    provider = RankedProvider()
    federation = FederatedSearch(Settings(_env_file=None))
    federation._academic = [provider]

    class FederatedManager:
        async def quick_search(
            self,
            query: str,
            mode: str,
            limit: int,
            *,
            provider_candidate_limit: int | None = None,
        ) -> SearchOutcome:
            assert limit == 30

            async def public(_url: str) -> bool:
                return True

            return await federation.search(
                query,
                mode,
                limit,
                public_url_validator=public,
                provider_candidate_limit=provider_candidate_limit,
            )

    payload = SearchRequestV2.model_validate(
        v2_payload(limit=request_limit, allowed_domains=["example.com"])
    )
    response = await execute_search_v2(FederatedManager(), payload)

    assert provider.calls == [("verified research", 20)]
    assert [item.canonical_url for item in response.sources] == ["https://docs.example.com/rank-13"]


def test_query_plan_digest_is_opaque_and_excluded_from_policy_digest() -> None:
    first = v2_payload(query_plan_digest="sha256:" + "1" * 64)
    second = v2_payload(query_plan_digest="sha256:" + "2" * 64)
    assert first["constraints"]["policyDigest"] == second["constraints"]["policyDigest"]

    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            return empty_outcome(query, mode)

        manager.quick_search = search
        first_response = client.post("/api/search/v2", json=first)
        second_response = client.post("/api/search/v2", json=second)

    assert first_response.status_code == second_response.status_code == 200
    assert (
        first_response.json()["returnedIdentityBinding"]
        != second_response.json()["returnedIdentityBinding"]
    )


def test_cross_language_canonical_digest_vectors_are_byte_exact() -> None:
    policy_preimage = (
        '{"constraints":{"allowedDomains":["arxiv.org","example.com"],'
        '"exactIdentifiers":[{"kind":"arxiv","value":"2005.11401v1"},'
        '{"kind":"doi","value":"10.1000/example"}],'
        '"schemaVersion":"localllm-grounded-search-policy-v1","strategy":"exact"},'
        '"limit":2,"mode":"papers","query":"量子 evidence",'
        '"schemaVersion":"localllm-grounded-search-request-v2"}'
    ).encode()
    payload = SearchRequestV2.model_validate(
        {
            "schemaVersion": REQUEST_SCHEMA_VERSION,
            "query": "量子 evidence",
            "mode": "papers",
            "limit": 2,
            "constraints": {
                "schemaVersion": POLICY_SCHEMA_VERSION,
                "strategy": "exact",
                "allowedDomains": ["arxiv.org", "example.com"],
                "exactIdentifiers": [
                    {"kind": "arxiv", "value": "2005.11401v1"},
                    {"kind": "doi", "value": "10.1000/example"},
                ],
                "queryPlanDigest": QUERY_PLAN_DIGEST,
                "policyDigest": POLICY_VECTOR_DIGEST,
            },
        }
    )
    assert canonical_policy_preimage(payload) == policy_preimage
    assert compute_policy_digest(payload) == POLICY_VECTOR_DIGEST

    returned_identities = [
        {
            "rank": 1,
            "identityDigest": "sha256:" + "2" * 64,
            "matchedAllowedDomains": ["arxiv.org"],
            "matchedExactIdentifiers": [
                {
                    "requested": {"kind": "arxiv", "value": "2005.11401v1"},
                    "returned": {"kind": "arxiv", "value": "2005.11401v1"},
                    "matchType": "exact",
                }
            ],
        }
    ]
    returned_preimage = (
        b'{"policyDigest":"sha256:ccf5b13b08f247de0033a2c1d4c9bd3866ae0a8ce2b9cf411907080f39ec629c",'
        b'"queryPlanDigest":"sha256:1111111111111111111111111111111111111111111111111111111111111111",'
        b'"returnedIdentities":[{"identityDigest":"sha256:'
        b'2222222222222222222222222222222222222222222222222222222222222222",'
        b'"matchedAllowedDomains":["arxiv.org"],"matchedExactIdentifiers":['
        b'{"matchType":"exact","requested":{"kind":"arxiv","value":"2005.11401v1"},'
        b'"returned":{"kind":"arxiv","value":"2005.11401v1"}}],"rank":1}]}'
    )
    assert (
        canonical_returned_identity_preimage(
            QUERY_PLAN_DIGEST,
            POLICY_VECTOR_DIGEST,
            returned_identities,
        )
        == returned_preimage
    )
    assert (
        compute_returned_identity_binding(
            QUERY_PLAN_DIGEST,
            POLICY_VECTOR_DIGEST,
            returned_identities,
        )
        == RETURNED_VECTOR_DIGEST
    )


def test_source_identity_digest_has_a_hard_coded_byte_vector() -> None:
    identifiers = [
        {"kind": "arxiv", "value": "2005.11401v1"},
        {"kind": "doi", "value": "10.1000/example"},
    ]
    preimage = (
        b'{"canonicalUrl":"https://example.com/paper","domain":"example.com",'
        b'"identifiers":[{"kind":"arxiv","value":"2005.11401v1"},'
        b'{"kind":"doi","value":"10.1000/example"}]}'
    )
    assert (
        canonical_source_identity_preimage(
            "https://example.com/paper",
            "example.com",
            identifiers,
        )
        == preimage
    )
    assert (
        compute_source_identity_digest(
            "https://example.com/paper",
            "example.com",
            identifiers,
        )
        == SOURCE_IDENTITY_VECTOR_DIGEST
    )


def test_two_record_binding_vector_is_order_sensitive_and_tamper_evident() -> None:
    records = [
        {
            "rank": 1,
            "identityDigest": "sha256:" + "2" * 64,
            "matchedAllowedDomains": ["arxiv.org"],
            "matchedExactIdentifiers": [
                {
                    "requested": {"kind": "arxiv", "value": "2005.11401v1"},
                    "returned": {"kind": "arxiv", "value": "2005.11401v1"},
                    "matchType": "exact",
                }
            ],
        },
        {
            "rank": 2,
            "identityDigest": "sha256:" + "3" * 64,
            "matchedAllowedDomains": ["example.com"],
            "matchedExactIdentifiers": [
                {
                    "requested": {"kind": "doi", "value": "10.1000/example"},
                    "returned": {"kind": "doi", "value": "10.1000/example"},
                    "matchType": "exact",
                }
            ],
        },
    ]

    def binding(value: list[dict[str, Any]]) -> str:
        return compute_returned_identity_binding(
            QUERY_PLAN_DIGEST,
            POLICY_VECTOR_DIGEST,
            value,
        )

    assert binding(records) == TWO_RECORD_BINDING_VECTOR_DIGEST
    assert binding(list(reversed(records))) != TWO_RECORD_BINDING_VECTOR_DIGEST

    for path, replacement in (
        ((0, "rank"), 2),
        ((0, "matchedAllowedDomains"), ["example.com"]),
        ((1, "identityDigest"), "sha256:" + "4" * 64),
    ):
        tampered = json.loads(json.dumps(records))
        tampered[path[0]][path[1]] = replacement
        assert binding(tampered) != TWO_RECORD_BINDING_VECTOR_DIGEST

    tampered_match = json.loads(json.dumps(records))
    tampered_match[0]["matchedExactIdentifiers"][0]["matchType"] = "arxiv-root"
    assert binding(tampered_match) != TWO_RECORD_BINDING_VECTOR_DIGEST


def test_strict_response_model_rejects_a_stale_returned_identity_binding() -> None:
    request = SearchRequestV2.model_validate(v2_payload())
    response = {
        "schemaVersion": RESPONSE_SCHEMA_VERSION,
        "policyCompliant": True,
        "request": request.model_dump(by_alias=True, mode="json"),
        "sources": [],
        "providers": [],
        "warnings": [],
        "resolvedIdentifiers": [],
        "unresolvedIdentifiers": [],
        "returnedIdentityBinding": compute_returned_identity_binding(
            request.constraints.query_plan_digest,
            request.constraints.policy_digest,
            [],
        ),
    }
    SearchResponseV2.model_validate(response)
    response["returnedIdentityBinding"] = "sha256:" + "f" * 64

    with pytest.raises(ValidationError) as raised:
        SearchResponseV2.model_validate(response)

    assert "returnedIdentityBinding mismatch" in str(raised.value)


def test_v2_uses_the_exact_search_scoped_bearer_and_private_response_boundary() -> None:
    payload = v2_payload()
    with configured_search_auth():
        with TestClient(app) as client:
            manager = client.app.state.research

            async def search(query: str, mode: str, limit: int) -> SearchOutcome:
                return empty_outcome(query, mode)

            manager.quick_search = search
            missing = client.post("/api/search/v2", json=payload)
            openai = client.post(
                "/api/search/v2",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=payload,
            )
            duplicate = client.post(
                "/api/search/v2",
                headers=[
                    ("Authorization", f"Bearer {SEARCH_API_KEY}"),
                    ("Authorization", f"Bearer {SEARCH_API_KEY}"),
                ],
                json=payload,
            )
            accepted = client.post(
                "/api/search/v2",
                headers={"Authorization": f"Bearer {SEARCH_API_KEY}"},
                json=payload,
            )

    for rejected in (missing, openai, duplicate):
        assert rejected.status_code == 401
        assert rejected.json() == {"detail": "Invalid search API key"}
        assert rejected.headers["www-authenticate"] == "Bearer"
        assert_private(rejected)
    assert accepted.status_code == 200
    assert_private(accepted)
    assert SEARCH_API_KEY not in accepted.text
    assert OPENAI_API_KEY not in accepted.text


def test_v2_private_boundary_strips_provider_cache_and_cookie_headers() -> None:
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            raise HTTPException(
                status_code=502,
                detail="Search provider unavailable",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Set-Cookie": "provider-session=must-not-survive",
                },
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=v2_payload())

    assert response.status_code == 502
    assert_private(response)


def test_v2_has_an_independent_16_kib_declared_and_chunked_body_limit() -> None:
    def oversized_chunks():
        yield b'{"schemaVersion":"localllm-grounded-search-request-v2","padding":"'
        yield b"x" * (17 * 1024)
        yield b'"}'

    with TestClient(app) as client:
        declared = client.post(
            "/api/search/v2",
            content=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(17 * 1024),
            },
        )
        chunked = client.post(
            "/api/search/v2",
            content=oversized_chunks(),
            headers={"Content-Type": "application/json"},
        )

    assert declared.status_code == chunked.status_code == 413
    assert_private(declared)
    assert_private(chunked)


@pytest.mark.asyncio
async def test_route_parser_itself_enforces_the_16_kib_stream_cap() -> None:
    messages = [
        {
            "type": "http.request",
            "body": b"x" * (16 * 1024 + 1),
            "more_body": False,
        }
    ]

    async def receive() -> dict[str, Any]:
        return messages.pop(0)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/search/v2",
            "headers": [(b"content-type", b"application/json")],
        },
        receive=receive,
    )
    with pytest.raises(HTTPException) as raised:
        await parse_search_v2_request(request)

    assert raised.value.status_code == 413
    assert raised.value.detail == "JSON request exceeds the size limit"


@pytest.mark.asyncio
async def test_route_parser_rejects_oversized_declared_content_length_before_reading() -> None:
    receive_called = False

    async def receive() -> dict[str, Any]:
        nonlocal receive_called
        receive_called = True
        return {"type": "http.request", "body": b"{}", "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/search/v2",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(16 * 1024 + 1).encode("ascii")),
            ],
        },
        receive=receive,
    )
    with pytest.raises(HTTPException) as raised:
        await parse_search_v2_request(request)

    assert raised.value.status_code == 413
    assert raised.value.detail == "JSON request exceeds the size limit"
    assert receive_called is False


def test_v2_forbids_extras_and_duplicate_json_members_without_echoing_values() -> None:
    payload = v2_payload()
    payload["unexpected"] = "DO-NOT-ECHO"
    encoded = json.dumps(v2_payload(), separators=(",", ":"))
    duplicate = encoded.replace(
        '"query":"verified research",',
        '"query":"verified research","query":"verified research",',
        1,
    )
    with TestClient(app) as client:
        extra_response = client.post("/api/search/v2", json=payload)
        duplicate_response = client.post(
            "/api/search/v2",
            content=duplicate,
            headers={"Content-Type": "application/json"},
        )

    assert extra_response.status_code == 422
    assert "DO-NOT-ECHO" not in extra_response.text
    assert duplicate_response.status_code == 400
    assert "duplicate JSON object members" in duplicate_response.text
    assert_private(extra_response)
    assert_private(duplicate_response)


def test_v2_recomputes_and_rejects_a_mismatched_policy_digest() -> None:
    payload = v2_payload()
    payload["constraints"]["policyDigest"] = "sha256:" + "f" * 64
    with TestClient(app) as client:
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 422
    assert "policyDigest" in response.text
    assert_private(response)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("allowedDomains", ["Example.com"]),
        ("allowedDomains", ["z.example.com", "a.example.com"]),
        ("allowedDomains", ["example.com", "example.com"]),
        ("allowedDomains", ["com"]),
        ("queryPlanDigest", "SHA256:" + "1" * 64),
    ],
)
def test_v2_rejects_noncanonical_domain_order_case_duplicates_and_digest_case(
    field: str, value: Any
) -> None:
    payload = v2_payload()
    payload["constraints"][field] = value
    resign_policy(payload)
    with TestClient(app) as client:
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 422
    assert_private(response)


@pytest.mark.parametrize("domain", ["co.uk", "github.io"])
def test_v2_rejects_bare_public_and_private_suffix_domains(domain: str) -> None:
    with pytest.raises(ValidationError, match="canonical public DNS names"):
        v2_payload(allowed_domains=[domain])


@pytest.mark.parametrize(
    "domain",
    [
        "example.co.uk",
        "tenant.github.io",
        "city.kawasaki.jp",
        "foo.city.kawasaki.jp",
        "xn--bcher-kva.de",
    ],
)
def test_v2_accepts_registrable_psl_exception_and_punycode_domains(domain: str) -> None:
    payload = v2_payload(allowed_domains=[domain])

    assert payload["constraints"]["allowedDomains"] == [domain]


@pytest.mark.parametrize(
    "query",
    [
        " verified research",
        "verified  research",
        "Search https://example.com/private?token=SECRET",
        "Search /home/alice/private.txt",
        "Cafe\u0301 evidence",
    ],
)
def test_v2_rejects_noncanonical_or_privacy_rewritten_provider_queries(query: str) -> None:
    payload = v2_payload()
    payload["query"] = query
    resign_policy(payload)
    with TestClient(app) as client:
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 422
    assert "SECRET" not in response.text
    assert "alice" not in response.text.casefold()
    assert_private(response)


@pytest.mark.parametrize(
    ("strategy", "identifiers", "limit"),
    [
        ("exact", [], 4),
        ("ranked", [{"kind": "doi", "value": "10.1000/example"}], 4),
        (
            "exact",
            [
                {"kind": "doi", "value": "10.1000/a"},
                {"kind": "doi", "value": "10.1000/b"},
            ],
            1,
        ),
    ],
)
def test_v2_enforces_strategy_identifier_and_limit_invariants(
    strategy: str, identifiers: list[dict[str, str]], limit: int
) -> None:
    payload = v2_payload()
    payload["constraints"]["strategy"] = strategy
    payload["constraints"]["exactIdentifiers"] = identifiers
    payload["limit"] = limit
    resign_policy(payload)
    with TestClient(app) as client:
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 422


def test_v2_rejects_exact_identifier_strategy_in_web_mode() -> None:
    payload = v2_payload(
        strategy="exact",
        exact_identifiers=[{"kind": "doi", "value": "10.1000/example"}],
    )
    payload["mode"] = "web"
    resign_policy(payload)
    with TestClient(app) as client:
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 422
    assert "papers or both" in response.text


def test_v2_rejects_noncanonical_or_duplicate_exact_identifier_lists() -> None:
    cases = [
        [{"kind": "doi", "value": "10.1000/UPPER"}],
        [{"kind": "doi", "value": "10.48550/arxiv.2005.11401"}],
        [{"kind": "doi", "value": "10.48550/arxiv.2005.1140"}],
        [
            {"kind": "doi", "value": "10.1000/a"},
            {"kind": "arxiv", "value": "2005.11401"},
        ],
        [
            {"kind": "arxiv", "value": "2005.11401"},
            {"kind": "arxiv", "value": "2005.11401"},
        ],
    ]
    with TestClient(app) as client:
        for identifiers in cases:
            payload = v2_payload()
            payload["constraints"]["strategy"] = "exact"
            payload["constraints"]["exactIdentifiers"] = identifiers
            resign_policy(payload)
            response = client.post("/api/search/v2", json=payload)
            assert response.status_code == 422


@pytest.mark.parametrize(
    "identifier",
    [
        "0704.0001",
        "1412.9999v2",
        "1501.00001",
        "2005.11401v1",
        "hep-th/9901001",
        "math.gt/0309136v2",
    ],
)
def test_strict_arxiv_grammar_accepts_only_canonical_valid_identities(identifier: str) -> None:
    assert canonical_arxiv_identifier(identifier) == identifier


@pytest.mark.parametrize(
    "identifier",
    [
        "0703.0001",
        "0704.0000",
        "0713.0001",
        "1412.00001",
        "1413.0001",
        "1501.0001",
        "1501.00000",
        "2005.1140",
        "2005.11401v0",
        "2005.11401v01",
        "hep-th/0000000",
        "hep-th/990100",
        "hep--th/9901001",
        "math./0309136",
    ],
)
def test_strict_arxiv_grammar_rejects_wrong_era_width_month_and_sequence(
    identifier: str,
) -> None:
    assert canonical_arxiv_identifier(identifier) is None


@pytest.mark.parametrize(
    ("identifier", "canonical"),
    [
        ("Math.GT/0309136", "math.gt/0309136"),
        ("2005.11401V2", "2005.11401v2"),
    ],
)
def test_arxiv_parser_exposes_noncanonical_case_for_request_rejection(
    identifier: str, canonical: str
) -> None:
    assert canonical_arxiv_identifier(identifier) == canonical
    assert canonical != identifier


def test_uppercase_arxiv_identity_is_rejected_by_the_actual_request_contract() -> None:
    payload = v2_payload(
        strategy="exact",
        exact_identifiers=[{"kind": "arxiv", "value": "2005.11401v2"}],
    )
    payload["constraints"]["exactIdentifiers"] = [{"kind": "arxiv", "value": "2005.11401V2"}]
    resign_policy(payload)
    with TestClient(app) as client:
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 422
    assert "canonical lowercase identity form" in response.text


@pytest.mark.asyncio
async def test_exact_search_admission_is_process_wide_and_times_out_with_429(
    monkeypatch,
) -> None:
    admission = search_v2_module._ProcessWideExactSearchAdmission(1)
    monkeypatch.setattr(search_v2_module, "_exact_search_admission", admission)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_ADMISSION_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_OVERALL_TIMEOUT_SECONDS", 1.0)
    payload = SearchRequestV2.model_validate(
        v2_payload(
            strategy="exact",
            exact_identifiers=[{"kind": "doi", "value": "10.1000/example"}],
        )
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingManager:
        active = 0
        peak = 0

        async def quick_search(self, query: str, mode: str, limit: int) -> SearchOutcome:
            self.active += 1
            self.peak = max(self.peak, self.active)
            entered.set()
            try:
                await release.wait()
                return empty_outcome(query, mode)
            finally:
                self.active -= 1

    manager = BlockingManager()
    first = asyncio.create_task(execute_search_v2(manager, payload))
    try:
        await asyncio.wait_for(entered.wait(), timeout=0.5)
        with pytest.raises(HTTPException) as rejected:
            await execute_search_v2(manager, payload)
        assert rejected.value.status_code == 429
        assert admission.in_use == 1
        assert manager.peak == 1
    finally:
        release.set()
        await first

    assert admission.in_use == 0
    assert manager.active == 0


@pytest.mark.asyncio
async def test_exact_admission_does_not_take_a_late_release_after_deadline(monkeypatch) -> None:
    admission = search_v2_module._ProcessWideExactSearchAdmission(1)
    assert await admission.acquire(0) is True
    clock = {"now": 0.0}

    async def release_after_deadline(_delay: float) -> None:
        clock["now"] = 2.0
        admission.release()

    monkeypatch.setattr(admission, "_now", lambda: clock["now"])
    monkeypatch.setattr(admission, "_pause", release_after_deadline)

    assert await admission.acquire(1.0) is False
    assert admission.in_use == 0


@pytest.mark.asyncio
async def test_exact_search_overall_deadline_cancels_children_and_releases_admission(
    monkeypatch,
) -> None:
    admission = search_v2_module._ProcessWideExactSearchAdmission(1)
    monkeypatch.setattr(search_v2_module, "_exact_search_admission", admission)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_ADMISSION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_OVERALL_TIMEOUT_SECONDS", 0.03)
    payload = SearchRequestV2.model_validate(
        v2_payload(
            strategy="exact",
            exact_identifiers=[{"kind": "doi", "value": "10.1000/example"}],
        )
    )
    cancelled = asyncio.Event()
    never = asyncio.Event()

    class BlockingManager:
        active = 0

        async def quick_search(self, query: str, mode: str, limit: int) -> SearchOutcome:
            self.active += 1
            try:
                await never.wait()
                return empty_outcome(query, mode)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                self.active -= 1

    manager = BlockingManager()
    with pytest.raises(HTTPException) as timed_out:
        await execute_search_v2(manager, payload)

    assert timed_out.value.status_code == 504
    await drain_exact_cleanup_reapers()
    assert cancelled.is_set()
    assert manager.active == 0
    assert admission.in_use == 0
    await asyncio.sleep(0)
    assert not any(
        task.get_name().startswith("search-v2-exact-") and not task.done()
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_hard_deadline_returns_504_while_reaper_retains_lease_for_stubborn_cleanup(
    monkeypatch,
) -> None:
    admission = search_v2_module._ProcessWideExactSearchAdmission(1)
    monkeypatch.setattr(search_v2_module, "_exact_search_admission", admission)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_ADMISSION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_OVERALL_TIMEOUT_SECONDS", 0.01)
    payload = SearchRequestV2.model_validate(
        v2_payload(
            strategy="exact",
            exact_identifiers=[{"kind": "doi", "value": "10.1000/example"}],
        )
    )
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    never = asyncio.Event()

    class StubbornCleanupManager:
        active = 0
        cleanup_interrupted = False

        async def quick_search(self, query: str, mode: str, limit: int) -> SearchOutcome:
            self.active += 1
            entered.set()
            try:
                await never.wait()
                return empty_outcome(query, mode)
            except asyncio.CancelledError:
                cleanup_started.set()
                try:
                    await allow_cleanup.wait()
                except asyncio.CancelledError:
                    self.cleanup_interrupted = True
                    raise
                cleanup_finished.set()
                raise
            finally:
                self.active -= 1

    manager = StubbornCleanupManager()
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    request_task = asyncio.create_task(execute_search_v2(manager, payload))
    try:
        await asyncio.wait_for(entered.wait(), timeout=0.5)
        done, _pending = await asyncio.wait({request_task}, timeout=0.1)
        elapsed = loop.time() - started_at

        assert request_task in done
        error = request_task.exception()
        assert isinstance(error, HTTPException)
        assert error.status_code == 504
        assert elapsed < 0.1
        await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
        assert cleanup_finished.is_set() is False
        assert manager.active == 1
        assert manager.cleanup_interrupted is False
        assert admission.in_use == 1
        assert len(search_v2_module._exact_cleanup_reapers) == 1
    finally:
        allow_cleanup.set()
        await asyncio.gather(request_task, return_exceptions=True)
        await asyncio.wait_for(cleanup_finished.wait(), timeout=0.5)
        await drain_exact_cleanup_reapers()

    assert manager.active == 0
    assert manager.cleanup_interrupted is False
    assert admission.in_use == 0
    assert not any(
        task.get_name().startswith("search-v2-exact-") and not task.done()
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_repeated_caller_and_reaper_cancellation_cannot_interrupt_child_cleanup(
    monkeypatch,
) -> None:
    admission = search_v2_module._ProcessWideExactSearchAdmission(1)
    monkeypatch.setattr(search_v2_module, "_exact_search_admission", admission)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_ADMISSION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_OVERALL_TIMEOUT_SECONDS", 1.0)
    payload = SearchRequestV2.model_validate(
        v2_payload(
            strategy="exact",
            exact_identifiers=[{"kind": "doi", "value": "10.1000/example"}],
        )
    )
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    never = asyncio.Event()

    class DelayedCleanupManager:
        active = 0
        cleanup_interrupted = False
        child_task: asyncio.Task[Any] | None = None

        async def quick_search(self, query: str, mode: str, limit: int) -> SearchOutcome:
            self.active += 1
            self.child_task = asyncio.current_task()
            entered.set()
            try:
                await never.wait()
                return empty_outcome(query, mode)
            except asyncio.CancelledError:
                cleanup_started.set()
                try:
                    await allow_cleanup.wait()
                except asyncio.CancelledError:
                    self.cleanup_interrupted = True
                    raise
                cleanup_finished.set()
                raise
            finally:
                self.active -= 1

    manager = DelayedCleanupManager()
    outer = asyncio.create_task(execute_search_v2(manager, payload))
    await asyncio.wait_for(entered.wait(), timeout=0.5)

    outer.cancel()
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
    assert admission.in_use == 1
    assert manager.child_task is not None
    assert manager.child_task.done() is False
    assert len(search_v2_module._exact_cleanup_reapers) == 1

    reaper = next(iter(search_v2_module._exact_cleanup_reapers))
    await asyncio.sleep(0)
    assert reaper.done() is False
    reaper.cancel()
    await asyncio.sleep(0)
    reaper.cancel()
    await asyncio.sleep(0)
    assert reaper.done() is False
    assert manager.cleanup_interrupted is False
    assert cleanup_finished.is_set() is False
    assert admission.in_use == 1

    allow_cleanup.set()
    await drain_exact_cleanup_reapers()

    assert cleanup_finished.is_set()
    assert manager.cleanup_interrupted is False
    assert manager.active == 0
    assert manager.child_task.done()
    assert manager.child_task.cancelled()
    assert admission.in_use == 0
    await asyncio.sleep(0)
    assert not any(
        task.get_name().startswith("search-v2-exact-") and not task.done()
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_prestart_reaper_cancellation_transfers_the_lease_to_a_successor() -> None:
    admission = search_v2_module._ProcessWideExactSearchAdmission(1)
    assert await admission.acquire(0) is True
    owned = asyncio.get_running_loop().create_future()

    search_v2_module._transfer_exact_cleanup(admission, [owned])
    first_reaper = next(iter(search_v2_module._exact_cleanup_reapers))
    first_reaper.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    successors = search_v2_module._exact_cleanup_reapers
    assert admission.in_use == 1
    assert len(successors) == 1
    assert first_reaper not in successors
    assert owned.done() is False

    owned.set_result(None)
    await drain_exact_cleanup_reapers()

    assert admission.in_use == 0
    assert search_v2_module._exact_cleanup_reapers == set()


@pytest.mark.asyncio
async def test_exact_lookup_error_cancels_and_joins_sibling_tasks(monkeypatch) -> None:
    admission = search_v2_module._ProcessWideExactSearchAdmission(1)
    monkeypatch.setattr(search_v2_module, "_exact_search_admission", admission)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_ADMISSION_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(search_v2_module, "EXACT_SEARCH_OVERALL_TIMEOUT_SECONDS", 1.0)
    payload = SearchRequestV2.model_validate(
        v2_payload(
            limit=2,
            strategy="exact",
            exact_identifiers=[
                {"kind": "doi", "value": "10.1000/a"},
                {"kind": "doi", "value": "10.1000/b"},
            ],
        )
    )
    both_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    never = asyncio.Event()

    class FailingManager:
        active = 0

        async def quick_search(self, query: str, mode: str, limit: int) -> SearchOutcome:
            self.active += 1
            if self.active == 2:
                both_started.set()
            try:
                await both_started.wait()
                if query == "doi:10.1000/a":
                    raise RuntimeError("deterministic first lookup failure")
                await never.wait()
                return empty_outcome(query, mode)
            except asyncio.CancelledError:
                if query == "doi:10.1000/b":
                    sibling_cancelled.set()
                raise
            finally:
                self.active -= 1

    manager = FailingManager()
    with pytest.raises(RuntimeError, match="deterministic first lookup failure"):
        await execute_search_v2(manager, payload)

    assert sibling_cancelled.is_set()
    assert manager.active == 0
    assert admission.in_use == 0
    await asyncio.sleep(0)
    assert not any(
        task.get_name().startswith("search-v2-exact-") and not task.done()
        for task in asyncio.all_tasks()
    )


def test_exact_doi_federation_hard_filters_identity_and_domain_with_truthful_coverage() -> None:
    identifier = {"kind": "doi", "value": "10.1000/example"}
    payload = v2_payload(
        strategy="exact",
        exact_identifiers=[identifier],
        allowed_domains=["example.com"],
    )
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            assert (query, mode, limit) == ("doi:10.1000/example", "papers", 12)
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[
                    source(
                        "https://papers.example.com/right",
                        doi="10.1000/example",
                        query=query,
                    ),
                    source(
                        "https://papers.example.com/wrong",
                        doi="10.1000/other",
                        query=query,
                    ),
                    source(
                        "https://example.com.evil.test/attack",
                        doi="10.1000/example",
                        query=query,
                    ),
                ],
                providers=[],
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert [item["canonicalUrl"] for item in body["sources"]] == [
        "https://papers.example.com/right"
    ]
    assert body["resolvedIdentifiers"] == [identifier]
    assert body["unresolvedIdentifiers"] == []
    assert body["sources"][0]["matchedAllowedDomains"] == ["example.com"]
    assert body["sources"][0]["matchedExactIdentifiers"] == [
        {
            "requested": identifier,
            "returned": identifier,
            "matchType": "exact",
        }
    ]


def test_conflicting_metadata_and_url_dois_fail_closed_as_unresolved() -> None:
    identifier = {"kind": "doi", "value": "10.1000/metadata"}
    payload = v2_payload(strategy="exact", exact_identifiers=[identifier])
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[
                    source(
                        "https://doi.org/10.1000/url",
                        doi="10.1000/metadata",
                        query=query,
                    )
                ],
                providers=[],
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["resolvedIdentifiers"] == []
    assert response.json()["unresolvedIdentifiers"] == [identifier]


def test_conflicting_arxiv_alias_and_url_roots_fail_closed_as_unresolved() -> None:
    identifier = {"kind": "arxiv", "value": "2005.11401v1"}
    payload = v2_payload(strategy="exact", exact_identifiers=[identifier])
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[
                    source(
                        "https://arxiv.org/abs/2309.01431v1",
                        doi="10.48550/arxiv.2005.11401v1",
                        query=query,
                    )
                ],
                providers=[],
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["resolvedIdentifiers"] == []
    assert response.json()["unresolvedIdentifiers"] == [identifier]


def test_malformed_arxiv_doi_alias_cannot_become_a_second_doi_identity() -> None:
    identifier = {"kind": "arxiv", "value": "2005.11401"}
    payload = v2_payload(strategy="exact", exact_identifiers=[identifier])
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[
                    source(
                        "https://arxiv.org/abs/2005.11401",
                        doi="10.48550/arxiv.2005.1140",
                        query=query,
                    )
                ],
                providers=[],
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["resolvedIdentifiers"] == []
    assert response.json()["unresolvedIdentifiers"] == [identifier]


def test_arxiv_doi_alias_is_returned_only_as_version_preserving_arxiv_identity() -> None:
    identifier = {"kind": "arxiv", "value": "2005.11401v1"}
    payload = v2_payload(strategy="exact", exact_identifiers=[identifier])
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[
                    source(
                        "https://doi.org/10.48550/arxiv.2005.11401v1",
                        doi="10.48550/arxiv.2005.11401v1",
                        query=query,
                    )
                ],
                providers=[],
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 200
    returned = response.json()["sources"][0]
    assert returned["canonicalUrl"] == "https://arxiv.org/abs/2005.11401v1"
    assert returned["domain"] == "arxiv.org"
    assert returned["doi"] is None
    assert returned["identifiers"] == [identifier]
    assert returned["matchedExactIdentifiers"] == [
        {
            "requested": identifier,
            "returned": identifier,
            "matchType": "exact",
        }
    ]
    assert response.json()["resolvedIdentifiers"] == [identifier]


@pytest.mark.parametrize("returned", ["2005.11401", "2005.11401v2"])
def test_requested_arxiv_version_is_not_satisfied_by_unversioned_or_other_version(
    returned: str,
) -> None:
    identifier = {"kind": "arxiv", "value": "2005.11401v1"}
    payload = v2_payload(strategy="exact", exact_identifiers=[identifier])
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[source(f"https://arxiv.org/abs/{returned}", query=query)],
                providers=[],
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["resolvedIdentifiers"] == []
    assert response.json()["unresolvedIdentifiers"] == [identifier]


def test_unproven_conflicting_arxiv_versions_fail_closed() -> None:
    identifiers = [
        {"kind": "arxiv", "value": "2005.11401v1"},
        {"kind": "arxiv", "value": "2005.11401v2"},
    ]
    payload = v2_payload(strategy="exact", exact_identifiers=identifiers)
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[source("https://arxiv.org/abs/2005.11401v1", query=query)],
                providers=[],
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 409
    assert "could not all be proven" in response.text
    assert_private(response)


def test_backend_may_prove_both_conflicting_arxiv_versions_as_distinct_sources() -> None:
    identifiers = [
        {"kind": "arxiv", "value": "2005.11401v1"},
        {"kind": "arxiv", "value": "2005.11401v2"},
    ]
    payload = v2_payload(strategy="exact", exact_identifiers=identifiers)
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            version = query.rsplit("v", 1)[-1]
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[source(f"https://arxiv.org/abs/2005.11401v{version}", query=query)],
                providers=[],
            )

        manager.quick_search = search
        response = client.post("/api/search/v2", json=payload)

    assert response.status_code == 200
    assert response.json()["resolvedIdentifiers"] == identifiers
    assert response.json()["unresolvedIdentifiers"] == []
    assert [item["identifiers"] for item in response.json()["sources"]] == [
        [identifiers[0]],
        [identifiers[1]],
    ]


def test_legacy_search_response_shape_remains_v1_only() -> None:
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[],
                providers=[],
                warnings=[],
            )

        manager.quick_search = search
        response = client.post(
            "/api/search",
            json={"query": "verified research", "mode": "papers", "limit": 4},
        )

    assert response.status_code == 200
    assert response.json() == {
        "query": "verified research",
        "mode": "papers",
        "sources": [],
        "providers": [],
        "warnings": [],
    }
    assert "schemaVersion" not in response.json()


def test_legacy_populated_search_response_has_an_exact_byte_golden() -> None:
    legacy_source = ResearchSource(
        title="Verified source",
        url="https://example.com/report",
        snippet="Evidence",
        provider="crossref",
        providers=["crossref"],
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
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[legacy_source],
                providers=[
                    ProviderDiagnostic(
                        "crossref",
                        "paper",
                        True,
                        1,
                        25,
                        queries=["verified research"],
                    )
                ],
                warnings=["bounded warning"],
            )

        manager.quick_search = search
        response = client.post(
            "/api/search",
            json={"query": "verified research", "mode": "papers", "limit": 4},
        )

    expected = (
        b'{"query":"verified research","mode":"papers","sources":[{'
        b'"title":"Verified source","url":"https://example.com/report",'
        b'"snippet":"Evidence","provider":"crossref","providers":["crossref"],'
        b'"kind":"paper","authors":["Ada Lovelace"],"year":2024,'
        b'"published_date":"2024-01-02","doi":"10.1000/example",'
        b'"citation_count":9,"score":4.2,"query":"verified research",'
        b'"provenance":[{"provider":"crossref","query":"verified research"}]}],'
        b'"providers":[{"name":"crossref","kind":"paper","ok":true,'
        b'"result_count":1,"duration_ms":25,"error":null,'
        b'"queries":["verified research"]}],"warnings":["bounded warning"]}'
    )
    assert response.status_code == 200
    assert response.content == expected


def test_legacy_search_error_has_an_exact_byte_golden() -> None:
    with TestClient(app) as client:
        manager = client.app.state.research

        async def search(query: str, mode: str, limit: int) -> SearchOutcome:
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
            json={"query": "verified research", "mode": "papers", "limit": 4},
        )

    assert response.status_code == 502
    assert response.content == b'{"detail":"Search provider unavailable"}'
    assert_private(response)


def test_route_response_model_rejects_undeclared_v2_fields_fail_closed(monkeypatch) -> None:
    async def invalid_response(_manager, payload):
        return {
            "schemaVersion": RESPONSE_SCHEMA_VERSION,
            "policyCompliant": True,
            "request": payload.model_dump(by_alias=True, mode="json"),
            "sources": [],
            "providers": [],
            "warnings": [],
            "resolvedIdentifiers": [],
            "unresolvedIdentifiers": [],
            "returnedIdentityBinding": compute_returned_identity_binding(
                payload.constraints.query_plan_digest,
                payload.constraints.policy_digest,
                [],
            ),
            "undeclaredProviderPayload": "must-not-leak-or-disappear",
        }

    monkeypatch.setattr(main_module, "execute_search_v2", invalid_response)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/search/v2", json=v2_payload())

    assert response.status_code == 500
    assert "must-not-leak-or-disappear" not in response.text
    assert_private(response)
