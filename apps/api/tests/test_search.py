from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

import localllm.search as search_module
from localllm.config import Settings
from localllm.ddgs_worker import MAX_FIELD_CHARS, _safe_results
from localllm.search import (
    ArxivProvider,
    CrossrefProvider,
    DuckDuckGoProvider,
    FederatedSearch,
    GitHubRepositoriesProvider,
    GitHubUsersProvider,
    HackerNewsAlgoliaProvider,
    ProviderResponseError,
    ResearchSource,
    SemanticScholarProvider,
    WikipediaProvider,
    _bounded_records,
    _canonical_arxiv_entry_url,
    _canonical_url,
    _load_bounded_json,
    _matches_query_site,
    _normalise_doi,
    _plain_text,
    _query_arxiv_ids,
    _query_site_hosts,
    _source,
    _structured_keyword_query,
)


class FakeProvider:
    def __init__(
        self,
        name: str,
        kind: str,
        results: list[ResearchSource] | None = None,
        error: Exception | None = None,
    ):
        self.name = name
        self.kind = kind
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        await asyncio.sleep(0)
        return self.results[:limit]


def settings(**values: Any) -> Settings:
    return Settings(_env_file=None, **values)


def source(
    title: str,
    url: str,
    provider: str,
    *,
    kind: str = "web",
    doi: str | None = None,
    citations: int | None = None,
) -> ResearchSource:
    return ResearchSource(
        title=title,
        url=url,
        snippet=f"Evidence about deterministic research and {title}",
        provider=provider,
        providers=[provider],
        kind=kind,
        doi=doi,
        citation_count=citations,
        query="deterministic research",
        provenance=[{"provider": provider, "query": "deterministic research"}],
    )


def test_canonical_url_removes_fragments_and_tracking_but_rejects_credentials() -> None:
    assert (
        _canonical_url("HTTPS://Example.COM:443//paper/?utm_source=spam&id=7#instructions")
        == "https://example.com/paper?id=7"
    )
    assert _canonical_url("https://user:secret@example.com/report") == ""
    assert _canonical_url("https://example.com:8443/report") == ""
    assert _canonical_url("https://example.com/" + "x" * 5_000) == ""
    assert _normalise_doi("https://doi.org/10.1000/Test.1") == "10.1000/test.1"
    assert _normalise_doi("10.1234/" + "x" * 10_000) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://arxiv.org/abs/2005.11401v4", "https://arxiv.org/abs/2005.11401v4"),
        ("https://arxiv.org:443/abs/math.GT/0309136", "https://arxiv.org/abs/math.GT/0309136"),
    ],
)
def test_arxiv_atom_identity_is_pinned_to_https(raw: str, expected: str) -> None:
    assert _canonical_arxiv_entry_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "http://arxiv.org.evil.example/abs/2005.11401",
        "https://user@arxiv.org/abs/2005.11401",
        "http://arxiv.org:443/abs/2005.11401",
        "https://arxiv.org:80/abs/2005.11401",
        "https://arxiv.org/abs/2005.11401?download=1",
        "https://arxiv.org./abs/2005.11401",
        "https://arxiv.org.../abs/2005.11401",
        "https://arxiv.org/pdf/2005.11401",
        "https://arxiv.org/abs/../2005.11401",
    ],
)
def test_arxiv_atom_identity_rejects_lookalikes_and_ambiguous_urls(raw: str) -> None:
    assert _canonical_arxiv_entry_url(raw) == ""


def test_arxiv_query_extracts_only_explicit_prefixed_or_bare_identity_lists() -> None:
    assert _query_arxiv_ids(
        "Find arXiv:2005.11401 and DOI 10.48550/arXiv.2309.01431; repeat arXiv:2005.11401"
    ) == ("2005.11401", "2309.01431")
    assert _query_arxiv_ids("2005.11401 2309.01431") == ("2005.11401", "2309.01431")
    assert _query_arxiv_ids("arXiv:hep-th/9901001v2") == ("hep-th/9901001v2",)
    assert _query_arxiv_ids("A decimal 2005.11401 appears in an ordinary sentence") == ()
    assert _query_arxiv_ids("arxiv.org.evil/abs/2005.11401") == ()


@pytest.mark.parametrize(
    "query",
    [
        "https://evil-arxiv.org/abs/2005.11401",
        "https://evil.example/arxiv.org/abs/2005.11401",
        "https://evil.example/10.48550/arxiv.2005.11401",
        "https://evil.example?next=arxiv.org/abs/2005.11401",
        "https://evil.example\\arxiv.org/abs/2005.11401",
        "https://evil.example/(arxiv.org/abs/2005.11401)",
        "https://evil.example/?next=(arxiv.org/abs/2005.11401)",
        "https://evil.example/?next=[arXiv:2005.11401]",
        'https://evil.example/#"10.48550/arxiv.2005.11401"',
        "arXiv:2005.11401_suffix",
        "arXiv:2005.11401.evil",
        "arXiv:2005.11401%2Fpdf",
        "arXiv:2005.11401)evil",
        "https://arxiv.org/abs/2005.11401)evil",
    ],
)
def test_arxiv_query_rejects_lookalike_authorities_paths_and_suffixes(query: str) -> None:
    assert _query_arxiv_ids(query) == ()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Find arXiv:2005.11401.", ("2005.11401",)),
        ("Find (arXiv : 2005.11401).", ("2005.11401",)),
        (
            'Compare [arXiv:2005.11401] and "10.48550/arXiv.2309.01431".',
            ("2005.11401", "2309.01431"),
        ),
        ("2005.11401,2309.01431", ("2005.11401", "2309.01431")),
        ("2005.11401;2309.01431", ("2005.11401", "2309.01431")),
    ],
)
def test_arxiv_query_accepts_sentence_terminators_and_compact_bare_lists(
    query: str, expected: tuple[str, ...]
) -> None:
    assert _query_arxiv_ids(query) == expected


def test_arxiv_query_deduplicates_before_applying_identifier_bound() -> None:
    query = " ".join(["arXiv:hep-th/9901001", "arXiv:HEP-TH/9901001"] * 11 + ["arXiv:2309.01431"])

    assert _query_arxiv_ids(query) == ("hep-th/9901001", "2309.01431")


def test_plain_text_normalization_is_linear_and_prebounded_for_malformed_markup() -> None:
    started = time.monotonic()
    normalized = _plain_text("<" * 200_000, 100)

    assert normalized == "<" * 100
    assert time.monotonic() - started < 1.0


def test_provider_metadata_numbers_and_record_arrays_are_bounded() -> None:
    with pytest.raises(ValueError):
        _load_bounded_json(b'{"value":NaN}')
    with pytest.raises(ValueError):
        _load_bounded_json(b'{"value":1e9999}')
    with pytest.raises(ValueError):
        _load_bounded_json(b'{"value":' + b"9" * 1_000 + b"}")

    records = [{"index": index} for index in range(100)]
    assert _bounded_records(records, 5) == records[:5]
    assert _bounded_records("not-an-array", 5) == []

    item = _source(
        provider="test",
        kind="paper",
        query="bounded numbers",
        title="Evidence",
        url="https://example.com/evidence",
        citation_count=10**1_000,
        authors="not-an-author-array",  # type: ignore[arg-type]
    )
    assert item is not None
    assert item.citation_count is None
    assert item.authors == []
    assert FederatedSearch._rank("bounded numbers", [item]) == [item]

    worker_records = _safe_results(
        [
            {
                "title": "x" * (MAX_FIELD_CHARS * 2),
                "href": "https://example.com/" + "y" * (MAX_FIELD_CHARS * 2),
                "body": "z" * (MAX_FIELD_CHARS * 2),
            }
        ]
        * 100,
        5,
    )
    assert len(worker_records) == 5
    assert all(
        len(value) <= MAX_FIELD_CHARS for record in worker_records for value in record.values()
    )


def test_ranking_uses_exact_tokens_instead_of_substring_matches() -> None:
    relevant = source(
        "SQLite WAL documentation",
        "https://sqlite.org/wal.html",
        "hacker_news_algolia",
    )
    substring_only = source(
        "Wallpaper manager",
        "https://example.com/wallpaper",
        "github_repositories",
    )

    ranked = FederatedSearch._rank("SQLite WAL", [substring_only, relevant])

    assert ranked[0] is relevant
    assert relevant.score > substring_only.score


def test_structured_query_removes_chat_formatting_instructions() -> None:
    assert (
        _structured_keyword_query(
            "In two concise sentences, what is a large language model and name one common use? "
            "Cite the supplied sources."
        )
        == "large language model common use"
    )
    assert _structured_keyword_query("量子计算是什么？") == "量子计算是什么？"
    assert _structured_keyword_query("Help me search lachlan Chen") == "lachlan Chen"
    assert _structured_keyword_query("Please search QAOA") == "QAOA"


@pytest.mark.asyncio
async def test_federation_rejects_partial_name_hits_from_provider_noise() -> None:
    engine = FederatedSearch(settings())
    relevant = ResearchSource(
        title="Lachlan Chen - Google Scholar",
        url="https://scholar.google.com/citations?user=example",
        snippet="Research profile for Lachlan Chen.",
        provider="yahoo_html",
        providers=["yahoo_html"],
        query="Help me search lachlan Chen",
    )
    partial = ResearchSource(
        title="Anduril Industries",
        url="https://en.wikipedia.org/wiki/Anduril_Industries",
        snippet="A company whose long article happens to mention someone named Chen.",
        provider="wikipedia",
        providers=["wikipedia"],
        query="Help search lachlan Chen",
    )
    engine._general = [FakeProvider("web_test", "web", [partial, relevant])]
    engine._keyless_web = []

    async def public(_url: str) -> bool:
        return True

    outcome = await engine.search(
        "Help me search lachlan Chen", "web", 8, public_url_validator=public
    )

    assert [item.title for item in outcome.sources] == ["Lachlan Chen - Google Scholar"]


def test_positive_site_operator_is_exact_and_subdomain_aware() -> None:
    hosts = _query_site_hosts(
        "site:Docs.Python.org Python 3.14 site:python.org -site:untrusted.example"
    )

    assert hosts == ("docs.python.org", "python.org")
    assert _matches_query_site(
        source("Official", "https://docs.python.org/3/whatsnew/3.14.html", "test"), hosts
    )
    assert _matches_query_site(
        source("Release", "https://www.python.org/downloads/release/python-3140/", "test"),
        hosts,
    )
    assert not _matches_query_site(
        source("Boundary attack", "https://notpython.org/result", "test"), hosts
    )
    assert not _matches_query_site(
        source("Suffix attack", "https://docs.python.org.evil.example/result", "test"), hosts
    )


@pytest.mark.asyncio
async def test_federation_enforces_site_operator_ignored_by_provider() -> None:
    engine = FederatedSearch(settings())
    provider = FakeProvider(
        "web_test",
        "web",
        [
            source(
                "Unrelated high-prior result",
                "https://news.example/unrelated",
                "hacker_news_algolia",
            ),
            source(
                "What's new in Python 3.14",
                "https://docs.python.org/3/whatsnew/3.14.html",
                "brave_html",
            ),
            source(
                "Malicious hostname suffix",
                "https://docs.python.org.evil.example/result",
                "brave_html",
            ),
        ],
    )
    engine._general = [provider]
    engine._keyless_web = []

    async def public(_url: str) -> bool:
        return True

    outcome = await engine.search(
        "site:docs.python.org Python 3.14 what is new",
        "web",
        4,
        public_url_validator=public,
    )

    assert [item.url for item in outcome.sources] == [
        "https://docs.python.org/3/whatsnew/3.14.html"
    ]


def test_deduplication_merges_academic_provenance_by_doi() -> None:
    first = source(
        "A useful paper",
        "https://doi.org/10.1000/example",
        "crossref",
        kind="paper",
        doi="10.1000/example",
        citations=2,
    )
    second = source(
        "A useful paper",
        "https://example.org/paper",
        "semantic_scholar",
        kind="paper",
        doi="10.1000/example",
        citations=19,
    )

    merged = FederatedSearch._deduplicate([first, second])

    assert len(merged) == 1
    assert merged[0].providers == ["crossref", "semantic_scholar"]
    assert merged[0].citation_count == 19
    assert len(merged[0].provenance) == 2


def test_same_title_without_strong_identity_is_not_false_corroboration() -> None:
    first = source(
        "A deliberately identical long result title",
        "https://example.com/first",
        "duckduckgo",
    )
    second = source(
        "A deliberately identical long result title",
        "https://example.org/second",
        "brave",
    )

    merged = FederatedSearch._deduplicate([first, second])

    assert len(merged) == 2
    assert [item.providers for item in merged] == [["duckduckgo"], ["brave"]]


@pytest.mark.asyncio
async def test_federated_search_falls_back_and_filters_unsafe_urls() -> None:
    engine = FederatedSearch(settings(search_brave_api_key="configured"))
    broken = FakeProvider("configured", "web", error=RuntimeError("provider offline"))
    fallback = FakeProvider(
        "duckduckgo",
        "web",
        [
            source("Public result", "https://example.com/report", "duckduckgo"),
            source("Unsafe result", "http://127.0.0.1/admin", "duckduckgo"),
        ],
    )
    engine._general = [broken]
    engine._keyless_web = [fallback]

    async def public(url: str) -> bool:
        return url.startswith("https://example.com/")

    outcome = await engine.search("deterministic research", "web", 10, public_url_validator=public)

    assert [item.title for item in outcome.sources] == ["Public result"]
    assert [item.name for item in outcome.providers] == ["configured", "duckduckgo"]
    assert not outcome.providers[0].ok
    assert outcome.providers[1].ok
    assert (
        "Some search connectors did not answer; successful fallbacks still supplied the evidence: configured"
        in outcome.warnings
    )


@pytest.mark.asyncio
async def test_keyless_search_defers_rate_prone_second_wave_when_primary_is_sufficient() -> None:
    engine = FederatedSearch(settings())
    primary = FakeProvider(
        "yahoo_html",
        "web",
        [
            source(f"Deterministic research result {index}", f"https://example.com/{index}", "yahoo_html")
            for index in range(4)
        ],
    )
    secondary = FakeProvider(
        "duckduckgo",
        "web",
        [source("Deterministic research fallback", "https://fallback.example/result", "duckduckgo")],
    )
    engine._general = []
    engine._keyless_web = [primary, secondary]

    async def public(_url: str) -> bool:
        return True

    outcome = await engine.search(
        "deterministic research", "web", 4, public_url_validator=public
    )

    assert len(outcome.sources) == 4
    assert primary.calls == [("deterministic research", 4)]
    assert secondary.calls == []


@pytest.mark.asyncio
async def test_successful_search_is_reused_from_bounded_memory_cache() -> None:
    engine = FederatedSearch(settings())
    provider = FakeProvider(
        "web_test",
        "web",
        [source("Deterministic research result", "https://example.com/result", "web_test")],
    )
    engine._general = [provider]
    engine._keyless_web = []

    async def public(_url: str) -> bool:
        return True

    first = await engine.search(
        "deterministic research", "web", 4, public_url_validator=public
    )
    first.sources[0].content = "caller-owned mutation"
    second = await engine.search(
        "deterministic research", "web", 4, public_url_validator=public
    )

    assert provider.calls == [("deterministic research", 4)]
    assert second.sources[0].content == ""
    assert len(engine._search_cache) == 1


@pytest.mark.asyncio
async def test_fallback_uses_safe_deduplicated_web_count() -> None:
    engine = FederatedSearch(settings(search_brave_api_key="configured"))
    duplicate_title = "One canonical configured-provider result"
    configured = FakeProvider(
        "configured",
        "web",
        [
            source(duplicate_title, "https://example.com/one", "brave"),
            source(duplicate_title, "https://example.com/two", "brave"),
            source("Private result one", "http://127.0.0.1/one", "brave"),
            source("Private result two", "http://10.0.0.1/two", "brave"),
        ],
    )
    fallback = FakeProvider(
        "duckduckgo",
        "web",
        [source("Independent fallback", "https://fallback.example/result", "duckduckgo")],
    )
    engine._general = [configured]
    engine._keyless_web = [fallback]

    async def public(url: str) -> bool:
        return url.startswith(("https://example.com/", "https://fallback.example/"))

    outcome = await engine.search("deterministic research", "web", 10, public_url_validator=public)

    assert fallback.calls == [("deterministic research", 10)]
    assert {item.title for item in outcome.sources} == {
        duplicate_title,
        "Independent fallback",
    }


@pytest.mark.asyncio
async def test_both_mode_federates_web_and_papers_with_stable_ranking() -> None:
    engine = FederatedSearch(settings())
    engine._general = [
        FakeProvider(
            "web_test",
            "web",
            [source("Generic result", "https://example.com/generic", "duckduckgo")],
        )
    ]
    engine._academic = [
        FakeProvider(
            "paper_test",
            "paper",
            [
                source(
                    "Deterministic research benchmark",
                    "https://doi.org/10.1000/benchmark",
                    "crossref",
                    kind="paper",
                    doi="10.1000/benchmark",
                    citations=500,
                )
            ],
        )
    ]
    engine._keyless_web = [FakeProvider("duckduckgo", "web", [])]

    async def public(_url: str) -> bool:
        return True

    outcome = await engine.search(
        "deterministic research benchmark", "both", 10, public_url_validator=public
    )

    assert [item.kind for item in outcome.sources] == ["paper", "web"]
    assert {item.name for item in outcome.providers} == {
        "web_test",
        "paper_test",
        "duckduckgo",
    }
    assert all(item.score > 0 for item in outcome.sources)


def test_both_mode_reserves_space_for_web_and_paper_evidence() -> None:
    papers = [
        source(
            f"Highly ranked paper {index}",
            f"https://doi.org/10.1000/paper{index}",
            "crossref",
            kind="paper",
            doi=f"10.1000/paper{index}",
            citations=10_000,
        )
        for index in range(10)
    ]
    web = [
        source(
            f"Independent web evidence {index}",
            f"https://example.com/evidence/{index}",
            "duckduckgo",
        )
        for index in range(4)
    ]
    ranked = FederatedSearch._rank("highly ranked paper", [*papers, *web])

    selected = FederatedSearch._select_diverse(ranked, "both", 4)

    assert len(selected) == 4
    assert {item.kind for item in selected} == {"web", "paper"}


@pytest.mark.asyncio
async def test_provider_diagnostic_never_leaks_query_string_credentials() -> None:
    engine = FederatedSearch(settings(search_openalex_api_key="TOP-SECRET-KEY"))
    request = httpx.Request(
        "GET", "https://api.openalex.org/works?api_key=TOP-SECRET-KEY&search=test"
    )
    response = httpx.Response(401, request=request)
    failure = httpx.HTTPStatusError(
        "request failed for secret-bearing URL", request=request, response=response
    )
    engine._academic = [FakeProvider("openalex", "paper", error=failure)]

    async def public(_url: str) -> bool:
        return True

    outcome = await engine.search("test query", "papers", 5, public_url_validator=public)
    serialized = json.dumps(outcome.public_dict())

    assert outcome.providers[0].error == "HTTP 401"
    assert "TOP-SECRET-KEY" not in serialized
    assert str(request.url) not in serialized


@pytest.mark.asyncio
async def test_provider_rank_is_preserved_in_provenance() -> None:
    engine = FederatedSearch(settings())
    provider = FakeProvider(
        "paper_test",
        "paper",
        [
            source("First provider result", "https://example.com/first", "paper_test"),
            source("Second provider result", "https://example.com/second", "paper_test"),
        ],
    )
    engine._academic = [provider]

    async def public(_url: str) -> bool:
        return True

    outcome = await engine.search("provider result", "papers", 5, public_url_validator=public)
    ranks = {item.title: item.provenance[0]["provider_rank"] for item in outcome.sources}

    assert ranks == {"First provider result": 1, "Second provider result": 2}


@pytest.mark.asyncio
async def test_legacy_federation_keeps_the_twelve_record_provider_budget() -> None:
    engine = FederatedSearch(settings(search_max_results=30))
    provider = FakeProvider("paper_test", "paper")
    engine._academic = [provider]

    async def public(_url: str) -> bool:
        return True

    await engine.search("provider budget", "papers", 30, public_url_validator=public)

    assert provider.calls == [("provider budget", 12)]


def test_status_marks_keyed_scholar_and_openalex_as_opt_in() -> None:
    unconfigured = FederatedSearch(settings()).status()
    keyed = FederatedSearch(
        settings(search_serpapi_api_key="scholar", search_openalex_api_key="openalex")
    ).status()

    plain = {item["name"]: item for item in unconfigured["providers"]}
    enabled = {item["name"]: item for item in keyed["providers"]}
    assert plain["google_scholar_serpapi"]["enabled"] is False
    assert plain["openalex"]["enabled"] is False
    assert enabled["google_scholar_serpapi"]["enabled"] is True
    assert enabled["openalex"]["enabled"] is True
    assert "no scraping" in enabled["google_scholar_serpapi"]["description"]


def test_crossref_contact_is_not_sent_to_other_providers() -> None:
    provider = SemanticScholarProvider(settings(search_crossref_email="private@example.test"))

    assert "private@example.test" not in json.dumps(provider._headers())
    assert provider._headers()["Accept-Encoding"] == "identity"


@pytest.mark.asyncio
async def test_duckduckgo_fallback_isolated_worker_pins_backend_and_sanitizes_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeStdin:
        def write(self, data: bytes) -> None:
            captured["stdin"] = data

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeStdout:
        def __init__(self) -> None:
            self.chunks = [
                json.dumps(
                    {
                        "ok": True,
                        "results": [
                            {
                                "title": "Pinned backend result",
                                "href": "https://example.com/result",
                                "body": "Evidence",
                            }
                        ],
                    }
                ).encode(),
                b"",
            ]

        async def read(self, _size: int) -> bytes:
            return self.chunks.pop(0)

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = FakeStdin()
            self.stdout = FakeStdout()
            self.returncode: int | None = None

        async def wait(self) -> int:
            self.returncode = 0
            return 0

        def kill(self) -> None:
            captured["killed"] = True
            self.returncode = -9

    async def create_subprocess(*args: str, **kwargs: Any) -> FakeProcess:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(search_module.asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setenv("LOCALLLM_SEARCH_BRAVE_API_KEY", "must-not-reach-worker")
    provider = DuckDuckGoProvider(settings())

    results = await provider.search("research", 5)

    assert captured["args"][3] == "duckduckgo"
    assert captured["stdin"] == b"research"
    assert "LOCALLLM_SEARCH_BRAVE_API_KEY" not in captured["env"]
    assert "HTTP_PROXY" not in captured["env"]
    assert results[0].provider == "duckduckgo"


@pytest.mark.asyncio
async def test_provider_deadline_includes_wait_for_shared_slot() -> None:
    engine = FederatedSearch(settings())
    engine.settings.search_provider_timeout_seconds = -2.95
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    provider = FakeProvider("queued", "web")

    _results, diagnostic = await engine._call_provider(provider, "bounded queue", 5, semaphore)

    assert diagnostic.ok is False
    assert diagnostic.error == "provider request timed out"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_wikipedia_provider_uses_mediawiki_api_and_normalizes_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = WikipediaProvider(settings())
    captured: dict[str, Any] = {}

    async def response(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        captured.update({"method": method, "url": url, **kwargs})
        return {
            "query": {
                "pages": [
                    {
                        "pageid": 42,
                        "title": "<b>Local language model</b>",
                        "fullurl": "https://en.wikipedia.org/wiki/Language_model?utm_source=test",
                        "extract": "<p>A probability distribution over token sequences.</p>",
                    },
                    {"pageid": 7, "title": "Fallback URL", "extract": "Useful context"},
                    {"pageid": 9, "title": "", "fullurl": "https://example.com/empty"},
                ]
            }
        }

    monkeypatch.setattr(provider, "_json", response)
    results = await provider.search("local language model", 5)

    assert captured["method"] == "GET"
    assert captured["url"] == "https://en.wikipedia.org/w/api.php"
    assert captured["params"]["generator"] == "search"
    assert captured["params"]["gsrlimit"] == 5
    assert captured["params"]["formatversion"] == 2
    assert "github.com/lachlanchen/LocalLLM" in captured["headers"]["User-Agent"]
    assert [item.title for item in results] == ["Local language model", "Fallback URL"]
    assert results[0].url == "https://en.wikipedia.org/wiki/Language_model"
    assert results[0].snippet == "A probability distribution over token sequences."
    assert results[0].provider == "wikipedia"
    assert results[0].provenance[0]["record_id"] == "42"
    assert results[1].url == "https://en.wikipedia.org/?curid=7"


@pytest.mark.asyncio
async def test_github_repository_provider_uses_documented_api_and_normalizes_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GitHubRepositoriesProvider(settings())
    captured: dict[str, Any] = {}

    async def response(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        captured.update({"method": method, "url": url, **kwargs})
        return {
            "items": [
                {
                    "id": 123,
                    "node_id": "R_repo",
                    "full_name": "QwenLM/Qwen3",
                    "html_url": "https://github.com/QwenLM/Qwen3?utm_campaign=test",
                    "description": "<em>Open</em> large language models",
                },
                {"id": 999, "name": "missing-url", "description": "ignored"},
            ]
        }

    monkeypatch.setattr(provider, "_json", response)
    results = await provider.search("Qwen local model", 50)

    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.github.com/search/repositories"
    assert captured["params"] == {"q": "Qwen local model", "per_page": 20, "page": 1}
    assert captured["headers"]["Accept"] == "application/vnd.github+json"
    assert captured["headers"]["X-GitHub-Api-Version"] == "2022-11-28"
    assert len(results) == 1
    assert results[0].title == "QwenLM/Qwen3"
    assert results[0].url == "https://github.com/QwenLM/Qwen3"
    assert results[0].snippet == "Open large language models"
    assert results[0].provider == "github_repositories"
    assert results[0].provenance[0]["record_id"] == "R_repo"


@pytest.mark.asyncio
async def test_github_user_provider_uses_documented_api_and_normalizes_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GitHubUsersProvider(settings())
    captured: dict[str, Any] = {}

    async def response(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        captured.update({"method": method, "url": url, **kwargs})
        return {
            "items": [
                {
                    "id": 123,
                    "node_id": "U_profile",
                    "login": "lachlanchen",
                    "html_url": "https://github.com/lachlanchen?tab=repositories",
                },
                {"id": 999, "login": "missing-url"},
            ]
        }

    monkeypatch.setattr(provider, "_json", response)
    results = await provider.search("Help me search lachlan Chen", 50)

    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.github.com/search/users"
    assert captured["params"] == {"q": "lachlan Chen", "per_page": 20, "page": 1}
    assert captured["headers"]["Accept"] == "application/vnd.github+json"
    assert len(results) == 1
    assert results[0].title == "lachlanchen (GitHub profile)"
    assert results[0].url == "https://github.com/lachlanchen?tab=repositories"
    assert results[0].provider == "github_users"
    assert results[0].provenance[0]["record_id"] == "U_profile"


@pytest.mark.asyncio
async def test_hacker_news_algolia_provider_normalizes_external_and_discussion_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HackerNewsAlgoliaProvider(settings())

    async def response(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "hits": [
                {
                    "objectID": "314159",
                    "title": "Local inference release",
                    "url": "https://example.org/release?utm_medium=feed",
                    "story_text": "<p>Release notes and benchmarks.</p>",
                    "author": "ada",
                    "created_at": "2025-03-14T12:00:00Z",
                },
                {
                    "objectID": "271828",
                    "story_title": "A technical discussion",
                    "comment_text": "<em>Measured</em> performance details",
                    "author": "grace",
                    "created_at": "2024-02-01T00:00:00Z",
                },
            ]
        }

    monkeypatch.setattr(provider, "_json", response)
    results = await provider.search("local inference", 4)

    assert [item.provider for item in results] == [
        "hacker_news_algolia",
        "hacker_news_algolia",
    ]
    assert results[0].url == "https://example.org/release"
    assert results[0].snippet == "Release notes and benchmarks."
    assert results[0].authors == ["ada"]
    assert results[0].year == 2025
    assert results[1].url == "https://news.ycombinator.com/item?id=271828"
    assert results[1].title == "A technical discussion"


@pytest.mark.asyncio
async def test_structured_keyless_failure_is_diagnostic_and_does_not_block_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FederatedSearch(settings())
    wikipedia = WikipediaProvider(settings())

    async def unavailable(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ProviderResponseError("wikipedia returned invalid JSON")

    monkeypatch.setattr(wikipedia, "_json", unavailable)
    engine._keyless_web = [
        wikipedia,
        FakeProvider(
            "github_repositories",
            "web",
            [
                source(
                    "Independent repository result",
                    "https://github.com/example/project",
                    "github_repositories",
                )
            ],
        ),
    ]

    async def public(_url: str) -> bool:
        return True

    outcome = await engine.search("independent project", "web", 5, public_url_validator=public)

    assert [item.title for item in outcome.sources] == ["Independent repository result"]
    assert [(item.name, item.ok) for item in outcome.providers] == [
        ("wikipedia", False),
        ("github_repositories", True),
    ]
    assert outcome.providers[0].error == "wikipedia returned invalid JSON"
    assert (
        "Some search connectors did not answer; successful fallbacks still supplied the evidence: wikipedia"
        in outcome.warnings
    )


def test_status_exposes_explicit_structured_keyless_provenance() -> None:
    providers = {item["name"]: item for item in FederatedSearch(settings()).status()["providers"]}

    for name in ("wikipedia", "github_users", "github_repositories", "hacker_news_algolia"):
        assert providers[name]["kind"] == "web"
        assert providers[name]["configured"] is True
        assert providers[name]["enabled"] is True
        assert providers[name]["requires_key"] is False


@pytest.mark.asyncio
async def test_crossref_parser_normalizes_paper_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = CrossrefProvider(settings())

    async def response(*_args, **_kwargs):
        return {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/TEST",
                        "title": ["<b>Paper title</b>"],
                        "author": [{"given": "Ada", "family": "Lovelace"}],
                        "published": {"date-parts": [[2024, 2, 3]]},
                        "URL": "https://doi.org/10.1234/TEST",
                        "abstract": "<jats:p>Useful abstract</jats:p>",
                        "is-referenced-by-count": 42,
                    }
                ]
            }
        }

    monkeypatch.setattr(provider, "_json", response)
    results = await provider.search("paper", 5)

    assert results[0].title == "Paper title"
    assert results[0].authors == ["Ada Lovelace"]
    assert results[0].doi == "10.1234/test"
    assert results[0].year == 2024
    assert results[0].citation_count == 42


@pytest.mark.asyncio
async def test_arxiv_parser_uses_atom_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ArxivProvider(settings())
    atom = """<?xml version='1.0'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry><id>http://arxiv.org/abs/2401.00001</id><title>Test paper</title>
      <summary>Grounded abstract</summary><published>2024-01-02T00:00:00Z</published>
      <author><name>Ada Lovelace</name></author></entry>
    </feed>"""

    async def response(*_args, **_kwargs):
        return atom

    monkeypatch.setattr(provider, "_text", response)
    results = await provider.search("test", 5)

    assert results[0].url == "https://arxiv.org/abs/2401.00001"
    assert results[0].snippet == "Grounded abstract"
    assert results[0].authors == ["Ada Lovelace"]
    assert results[0].year == 2024


@pytest.mark.asyncio
async def test_arxiv_provider_uses_exact_id_list_for_verbose_identifier_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ArxivProvider(settings())
    atom = """<?xml version='1.0'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry><id>http://arxiv.org/abs/2005.11401v4</id>
      <title>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</title>
      <published>2020-05-22T00:00:00Z</published></entry>
      <entry><id>http://arxiv.org/abs/2309.01431v2</id>
      <title>Benchmarking Large Language Models in Retrieval-Augmented Generation</title>
      <published>2023-09-04T00:00:00Z</published></entry>
    </feed>"""
    captured: dict[str, object] = {}

    async def response(url: str, **kwargs):
        captured.update({"url": url, **kwargs})
        return atom

    monkeypatch.setattr(provider, "_text", response)
    results = await provider.search(
        "Find exactly arXiv:2005.11401 and arXiv:2309.01431, then summarize them.", 8
    )

    assert captured == {
        "url": "https://export.arxiv.org/api/query",
        "params": {"start": 0, "max_results": 2, "id_list": "2005.11401,2309.01431"},
    }
    assert [source.url for source in results] == [
        "https://arxiv.org/abs/2005.11401v4",
        "https://arxiv.org/abs/2309.01431v2",
    ]


@pytest.mark.asyncio
async def test_arxiv_provider_filters_unrequested_and_wrong_version_atom_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ArxivProvider(settings())
    atom = """<?xml version='1.0'?>
    <feed xmlns='http://www.w3.org/2005/Atom'>
      <entry><id>https://arxiv.org/abs/9999.99999v1</id>
      <title>Unrequested paper</title></entry>
      <entry><id>https://arxiv.org/abs/2005.11401v4</id>
      <title>Wrong requested version</title></entry>
      <entry><id>https://arxiv.org.../abs/2005.11401v3</id>
      <title>Invalid repeated-dot authority</title></entry>
      <entry><id>https://arxiv.org/abs/2005.11401v3</id>
      <title>Exact requested version</title></entry>
    </feed>"""

    async def response(*_args, **_kwargs):
        return atom

    monkeypatch.setattr(provider, "_text", response)
    results = await provider.search("arXiv:2005.11401v3", 1)

    assert [source.url for source in results] == ["https://arxiv.org/abs/2005.11401v3"]
    assert [source.provenance[0]["record_id"] for source in results] == ["2005.11401v3"]
