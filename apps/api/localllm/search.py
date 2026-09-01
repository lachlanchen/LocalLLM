from __future__ import annotations

import asyncio
import copy
import html
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal, Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from .config import Settings

SearchMode = Literal["web", "papers", "both"]
_PUBLISHED_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def canonical_published_date(value: object) -> str | None:
    """Keep only a real canonical calendar date at a public source boundary."""

    if not isinstance(value, str) or _PUBLISHED_DATE_PATTERN.fullmatch(value) is None:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return value if parsed.isoformat() == value else None


@dataclass
class ResearchSource:
    """Normalized evidence record shared by quick search and deep research."""

    title: str
    url: str
    snippet: str
    content: str = ""
    provider: str = "unknown"
    providers: list[str] = field(default_factory=list)
    kind: str = "web"
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    published_date: str | None = None
    doi: str | None = None
    citation_count: int | None = None
    score: float = 0.0
    query: str = ""
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def public_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_content:
            payload.pop("content", None)
        return payload


@dataclass
class ProviderDiagnostic:
    name: str
    kind: str
    ok: bool
    result_count: int
    duration_ms: int
    error: str | None = None
    queries: list[str] = field(default_factory=list)


@dataclass
class SearchOutcome:
    query: str
    mode: SearchMode
    sources: list[ResearchSource]
    providers: list[ProviderDiagnostic]
    warnings: list[str] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "sources": [source.public_dict() for source in self.sources],
            "providers": [asdict(provider) for provider in self.providers],
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class ProviderDescription:
    name: str
    kind: str
    configured: bool
    enabled: bool
    requires_key: bool
    description: str


class SearchProvider(Protocol):
    name: str
    kind: str

    async def search(self, query: str, limit: int) -> list[ResearchSource]: ...


class ProviderResponseError(RuntimeError):
    """A provider returned a malformed, oversized, or unsuccessful response."""


_TRACKING_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}
MAX_SOURCE_URL_CHARS = 4_096
MAX_DOI_CHARS = 512
MAX_CITATION_COUNT = 1_000_000_000_000
MAX_PROVIDER_RECORDS = 20
MAX_JSON_NUMBER_CHARS = 128
DDGS_WORKER_MAX_OUTPUT_BYTES = 2_000_000
SEARCH_CACHE_MAX_ENTRIES = 64
SEARCH_CACHE_TTL_SECONDS = 120.0
_STRUCTURED_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "be",
    "brief",
    "briefly",
    "cite",
    "cited",
    "citation",
    "citations",
    "concise",
    "could",
    "describe",
    "explain",
    "find",
    "for",
    "from",
    "give",
    "help",
    "how",
    "in",
    "include",
    "is",
    "me",
    "name",
    "of",
    "on",
    "one",
    "please",
    "search",
    "sentence",
    "sentences",
    "source",
    "sources",
    "supplied",
    "tell",
    "that",
    "the",
    "these",
    "this",
    "to",
    "two",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _strip_angle_tags(text: str) -> str:
    """Strip ``<...>`` spans in linear time while preserving unmatched text."""

    output: list[str] = []
    pending: list[str] | None = None
    for character in text:
        if pending is None:
            if character == "<":
                pending = [character]
            else:
                output.append(character)
            continue
        pending.append(character)
        if character == ">":
            # Match the old ``<[^>]+>`` behavior: ``<>`` is text, not a tag.
            if len(pending) == 2:
                output.extend(pending)
            pending = None
    if pending:
        output.extend(pending)
    return "".join(output)


def _plain_text(value: object, limit: int = 4000) -> str:
    # Provider fields are untrusted. Cap before normalization so malformed markup
    # cannot turn a bounded response into disproportionate CPU or allocation work.
    input_limit = max(8_192, min(32_768, limit * 8))
    raw = str(value or "")[:input_limit]
    text = _strip_angle_tags(html.unescape(raw))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _bounded_records(value: object, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    bounded_limit = max(0, min(MAX_PROVIDER_RECORDS, limit))
    return [record for record in value[:bounded_limit] if isinstance(record, dict)]


def _bounded_json_integer(value: str) -> int:
    if len(value.lstrip("-")) > MAX_JSON_NUMBER_CHARS:
        raise ValueError("JSON integer exceeded the digit limit")
    return int(value)


def _bounded_json_float(value: str) -> float:
    if len(value) > MAX_JSON_NUMBER_CHARS:
        raise ValueError("JSON float exceeded the character limit")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON float must be finite")
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _load_bounded_json(value: bytes | bytearray | str) -> Any:
    return json.loads(
        value,
        parse_int=_bounded_json_integer,
        parse_float=_bounded_json_float,
        parse_constant=_reject_json_constant,
    )


def _bounded_citation_count(value: object) -> int | None:
    try:
        if isinstance(value, str) and len(value.strip()) > 32:
            raise ValueError("citation count is too long")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("citation count must be finite")
        parsed = int(value) if value is not None else None
        if parsed is None or parsed > MAX_CITATION_COUNT:
            return None
        return max(0, parsed)
    except (OverflowError, TypeError, ValueError):
        return None


def _structured_keyword_query(value: str) -> str:
    """Remove conversational formatting instructions for metadata search APIs."""

    normalized = re.sub(r"\s+", " ", value).strip()
    tokens = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", normalized, flags=re.UNICODE)
    keywords = [
        token
        for token in tokens
        if len(token) > 1 and token.casefold() not in _STRUCTURED_QUERY_STOPWORDS
    ]
    if len(keywords) >= 2:
        return " ".join(keywords[:24])
    # Preserve a genuinely single-token CJK question or exact identifier. When
    # chat instructions surround one topical token, keep that token alone.
    if len(keywords) == 1 and len(tokens) > 1:
        return keywords[0]
    return normalized[:800]


def _lexical_tokens(value: str) -> set[str]:
    """Return exact Unicode word tokens for scoring, never substrings."""

    return {
        token
        for token in re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)
        if len(token) >= 3
    }


def _safe_year(value: object) -> int | None:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", str(value or ""))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1900 <= year <= datetime.now(timezone.utc).year + 1 else None


def _normalise_doi(value: object) -> str | None:
    doi = str(value or "").strip()
    if len(doi) > MAX_DOI_CHARS + 64:
        return None
    doi = doi.lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    doi = doi.rstrip(".,;)")
    return doi if len(doi) <= MAX_DOI_CHARS and re.fullmatch(r"10\.\d{4,9}/\S+", doi) else None


_ARXIV_IDENTIFIER = r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})(?:v[1-9]\d*)?"
_ARXIV_ID_PATH = re.compile(rf"^/abs/(?P<identifier>{_ARXIV_IDENTIFIER})$")
_ARXIV_QUERY_LEFT_BOUNDARY = r"(?:^|(?<=\s))[([{<\"'“‘]*"
_ARXIV_QUERY_RIGHT_BOUNDARY = r"[])}>\"'”’]*[.,;:!?]?(?=$|\s)"
_ARXIV_QUERY_PATTERNS = tuple(
    re.compile(expression, flags=re.IGNORECASE)
    for expression in (
        rf"{_ARXIV_QUERY_LEFT_BOUNDARY}arxiv\s*:\s*"
        rf"(?P<identifier>{_ARXIV_IDENTIFIER}){_ARXIV_QUERY_RIGHT_BOUNDARY}",
        rf"{_ARXIV_QUERY_LEFT_BOUNDARY}(?:https?://)?arxiv\.org/abs/"
        rf"(?P<identifier>{_ARXIV_IDENTIFIER}){_ARXIV_QUERY_RIGHT_BOUNDARY}",
        rf"{_ARXIV_QUERY_LEFT_BOUNDARY}(?:(?:https?://)?doi\.org/)?"
        rf"10\.48550/arxiv\.(?P<identifier>{_ARXIV_IDENTIFIER})"
        rf"{_ARXIV_QUERY_RIGHT_BOUNDARY}",
    )
)
_ARXIV_BARE_ID_LIST = re.compile(
    rf"^\s*(?P<identifiers>{_ARXIV_IDENTIFIER}"
    rf"(?:\s*(?:[,;]\s*|\s+){_ARXIV_IDENTIFIER})*)\s*\.?\s*$",
    flags=re.IGNORECASE,
)
_ARXIV_VERSION_SUFFIX = re.compile(r"v[1-9]\d*$", flags=re.IGNORECASE)


def _query_arxiv_ids(value: str) -> tuple[str, ...]:
    """Extract explicit arXiv identities without interpreting ordinary numbers as IDs."""

    query = value[:800]
    located = sorted(
        (match.start(), match.group("identifier"))
        for pattern in _ARXIV_QUERY_PATTERNS
        for match in pattern.finditer(query)
    )
    if located:
        identifiers = [identifier for _offset, identifier in located]
    else:
        bare = _ARXIV_BARE_ID_LIST.fullmatch(query)
        identifiers = (
            re.findall(_ARXIV_IDENTIFIER, bare.group("identifiers"), re.IGNORECASE) if bare else []
        )
    distinct: dict[str, str] = {}
    for identifier in identifiers:
        distinct.setdefault(identifier.casefold(), identifier)
    return tuple(distinct.values())[:20]


def _arxiv_response_matches_request(returned: str, requested: tuple[str, ...]) -> bool:
    returned_folded = returned.casefold()
    returned_base = _ARXIV_VERSION_SUFFIX.sub("", returned_folded)
    return any(
        returned_folded == candidate.casefold()
        or (
            _ARXIV_VERSION_SUFFIX.search(candidate) is None
            and returned_base == candidate.casefold()
        )
        for candidate in requested
    )


def _canonical_arxiv_entry_url(value: object) -> str:
    """Pin Atom entry identities to arxiv.org HTTPS before general normalization."""

    parsed = urlparse(str(value or "").strip())
    try:
        port = parsed.port
    except ValueError:
        return ""
    match = _ARXIV_ID_PATH.fullmatch(parsed.path)
    if (
        parsed.scheme not in {"http", "https"}
        or (parsed.hostname or "").casefold() != "arxiv.org"
        or parsed.username
        or parsed.password
        or not (
            port is None
            or (parsed.scheme == "http" and port == 80)
            or (parsed.scheme == "https" and port == 443)
        )
        or parsed.query
        or parsed.fragment
        or match is None
    ):
        return ""
    return f"https://arxiv.org/abs/{match.group('identifier')}"


def _canonical_url(url: str) -> str:
    url = url.strip()
    if len(url) > MAX_SOURCE_URL_CHARS:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    if parsed.username or parsed.password or port not in {None, 80, 443}:
        return ""
    hostname = parsed.hostname.lower().rstrip(".")
    netloc = hostname
    if ":" in hostname:
        netloc = f"[{hostname}]"
    if (
        port
        and not (parsed.scheme == "http" and port == 80)
        and not (parsed.scheme == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_KEYS
    ]
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    canonical = urlunparse((parsed.scheme.lower(), netloc, path, "", urlencode(query), ""))
    return canonical if len(canonical) <= MAX_SOURCE_URL_CHARS else ""


_SITE_OPERATOR_RE = re.compile(
    r"(?<![\w-])site:([a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?)\.?(?=$|[\s,;!?)}\]\"'])",
    flags=re.IGNORECASE,
)


def _query_relevance_terms(query: str) -> set[str]:
    """Extract topical terms without chat instructions or site operators."""

    lexical_query = _SITE_OPERATOR_RE.sub(" ", query)
    return _lexical_tokens(_structured_keyword_query(lexical_query))


def _source_is_relevant(query: str, source: ResearchSource) -> bool:
    """Reject weak provider hits before their priors can outrank real evidence."""

    terms = _query_relevance_terms(query)
    if not terms:
        return True
    haystack = f"{source.title} {source.snippet} {source.url}"
    haystack_terms = _lexical_tokens(haystack)
    minimum_hits = 1 if len(terms) == 1 else 2
    if len(terms & haystack_terms) >= minimum_hits:
        return True
    compact_query = "".join(
        character
        for character in _structured_keyword_query(_SITE_OPERATOR_RE.sub(" ", query)).casefold()
        if character.isalnum()
    )
    compact_haystack = "".join(character for character in haystack.casefold() if character.isalnum())
    return len(terms) >= 2 and len(compact_query) >= 6 and compact_query in compact_haystack


def _query_site_hosts(query: str) -> tuple[str, ...]:
    """Return valid positive ``site:`` host constraints without broadening them.

    Providers differ in whether they understand search operators. Enforcing the
    operator again after federation prevents an unrelated high-prior connector
    from outranking the domain the caller explicitly requested.
    """

    hosts: list[str] = []
    for match in _SITE_OPERATOR_RE.finditer(query[:800]):
        hostname = match.group(1).casefold().rstrip(".")
        labels = hostname.split(".")
        if (
            not hostname
            or len(hostname) > 253
            or any(
                not label
                or len(label) > 63
                or label.startswith("-")
                or label.endswith("-")
                or re.fullmatch(r"[a-z0-9-]+", label) is None
                for label in labels
            )
        ):
            continue
        if hostname not in hosts:
            hosts.append(hostname)
    return tuple(hosts)


def _matches_query_site(source: ResearchSource, hosts: tuple[str, ...]) -> bool:
    if not hosts:
        return True
    hostname = (urlparse(source.url).hostname or "").casefold().rstrip(".")
    return any(hostname == host or hostname.endswith(f".{host}") for host in hosts)


def _source(
    *,
    provider: str,
    kind: str,
    query: str,
    title: object,
    url: object,
    snippet: object = "",
    authors: list[str] | None = None,
    published_date: object = None,
    doi: object = None,
    citation_count: object = None,
    record_id: object = None,
) -> ResearchSource | None:
    canonical = _canonical_url(str(url or ""))
    clean_title = _plain_text(title, 500)
    if not canonical or not clean_title:
        return None
    clean_doi = _normalise_doi(doi)
    published = _plain_text(published_date, 40) or None
    citations = _bounded_citation_count(citation_count)
    author_values = authors if isinstance(authors, list) else []
    clean_authors = [_plain_text(author, 160) for author in author_values[:20]]
    clean_authors = [author for author in clean_authors if author][:20]
    return ResearchSource(
        title=clean_title,
        url=canonical,
        snippet=_plain_text(snippet, 4000),
        provider=provider,
        providers=[provider],
        kind=kind,
        authors=clean_authors,
        year=_safe_year(published),
        published_date=published,
        doi=clean_doi,
        citation_count=citations,
        query=query,
        provenance=[
            {
                "provider": provider,
                "query": query,
                "record_id": _plain_text(record_id, 300) or None,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )


class HTTPProvider:
    name = "provider"
    kind = "web"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.timeout = httpx.Timeout(settings.search_provider_timeout_seconds, connect=5.0)

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": "LocalLLM-Research/0.2 (+local research assistant)",
            "Accept": "application/json, application/atom+xml;q=0.9, */*;q=0.5",
            # Bound decoded memory as well as wire bytes. Providers that ignore this
            # request and send compressed content are rejected below.
            "Accept-Encoding": "identity",
        }
        headers.update(extra or {})
        return headers

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float | None:
        if attempt > 0 or response.status_code not in {429, 500, 502, 503, 504}:
            return None
        try:
            requested = float(response.headers.get("retry-after", ""))
        except ValueError:
            requested = 0.5
        return min(2.0, max(0.25, requested))

    async def _json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
        body: bytearray | None = None
        async with httpx.AsyncClient(
            timeout=self.timeout,
            trust_env=False,
            limits=limits,
            follow_redirects=False,
        ) as client:
            for attempt in range(2):
                retry_delay: float | None = None
                async with client.stream(
                    method, url, params=params, headers=self._headers(headers), json=payload
                ) as response:
                    retry_delay = self._retry_delay(response, attempt)
                    if retry_delay is None:
                        response.raise_for_status()
                        if response.headers.get("content-encoding", "identity").lower() not in {
                            "",
                            "identity",
                        }:
                            raise ProviderResponseError(f"{self.name} returned compressed content")
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > self.settings.search_response_limit_bytes:
                                raise ProviderResponseError(
                                    f"{self.name} response exceeded the size limit"
                                )
                            body.extend(chunk)
                if body is not None:
                    break
                await asyncio.sleep(retry_delay or 0)
        if body is None:
            raise ProviderResponseError(f"{self.name} returned no response")
        try:
            data = _load_bounded_json(body)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ProviderResponseError(f"{self.name} returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ProviderResponseError(f"{self.name} returned an unexpected JSON shape")
        return data

    async def _text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        limits = httpx.Limits(max_connections=1, max_keepalive_connections=0)
        body: bytearray | None = None
        async with httpx.AsyncClient(
            timeout=self.timeout,
            trust_env=False,
            limits=limits,
            follow_redirects=False,
            headers=self._headers(),
        ) as client:
            for attempt in range(2):
                retry_delay: float | None = None
                async with client.stream("GET", url, params=params) as response:
                    retry_delay = self._retry_delay(response, attempt)
                    if retry_delay is None:
                        response.raise_for_status()
                        if response.headers.get("content-encoding", "identity").lower() not in {
                            "",
                            "identity",
                        }:
                            raise ProviderResponseError(f"{self.name} returned compressed content")
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > self.settings.search_response_limit_bytes:
                                raise ProviderResponseError(
                                    f"{self.name} response exceeded the size limit"
                                )
                            body.extend(chunk)
                if body is not None:
                    break
                await asyncio.sleep(retry_delay or 0)
        if body is None:
            raise ProviderResponseError(f"{self.name} returned no response")
        return bytes(body).decode("utf-8", errors="replace")


class BraveProvider(HTTPProvider):
    name = "brave"
    kind = "web"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        data = await self._json(
            "GET",
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(limit, 20), "safesearch": "moderate"},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.settings.search_brave_api_key,
            },
        )
        web = data.get("web")
        results = web.get("results") if isinstance(web, dict) else []
        return [
            item
            for raw in _bounded_records(results, limit)
            if (
                item := _source(
                    provider=self.name,
                    kind=self.kind,
                    query=query,
                    title=raw.get("title"),
                    url=raw.get("url"),
                    snippet=raw.get("description"),
                )
            )
            is not None
        ]


class TavilyProvider(HTTPProvider):
    name = "tavily"
    kind = "web"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        data = await self._json(
            "POST",
            "https://api.tavily.com/search",
            payload={
                "api_key": self.settings.search_tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": min(limit, 20),
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        return [
            item
            for raw in _bounded_records(data.get("results"), limit)
            if (
                item := _source(
                    provider=self.name,
                    kind=self.kind,
                    query=query,
                    title=raw.get("title"),
                    url=raw.get("url"),
                    snippet=raw.get("content"),
                )
            )
            is not None
        ]


class SerperProvider(HTTPProvider):
    name = "serper"
    kind = "web"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        data = await self._json(
            "POST",
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.settings.search_serper_api_key},
            payload={"q": query, "num": min(limit, 20)},
        )
        return [
            item
            for raw in _bounded_records(data.get("organic"), limit)
            if (
                item := _source(
                    provider=self.name,
                    kind=self.kind,
                    query=query,
                    title=raw.get("title"),
                    url=raw.get("link"),
                    snippet=raw.get("snippet"),
                )
            )
            is not None
        ]


class WikipediaProvider(HTTPProvider):
    """English Wikipedia results from the official MediaWiki Action API."""

    name = "wikipedia"
    kind = "web"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        query = _structured_keyword_query(query)
        data = await self._json(
            "GET",
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 0,
                "gsrlimit": min(limit, 20),
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "redirects": 1,
                "format": "json",
                "formatversion": 2,
            },
            # Wikimedia requires an identifiable tool-specific agent with contact
            # information; the generic provider agent is intentionally rejected.
            headers={
                "User-Agent": ("LocalLLM-Research/0.2 (https://github.com/lachlanchen/LocalLLM)")
            },
        )
        query_payload = data.get("query")
        pages = query_payload.get("pages") if isinstance(query_payload, dict) else None

        sources: list[ResearchSource] = []
        for raw in _bounded_records(pages, limit):
            page_id = raw.get("pageid")
            url = raw.get("fullurl")
            if not url and isinstance(page_id, int) and page_id > 0:
                url = f"https://en.wikipedia.org/?curid={page_id}"
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=raw.get("title"),
                url=url,
                snippet=raw.get("extract"),
                record_id=page_id,
            )
            if item:
                sources.append(item)
        return sources


class GitHubRepositoriesProvider(HTTPProvider):
    """Public repositories from GitHub's documented REST search endpoint."""

    name = "github_repositories"
    kind = "web"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        query = _structured_keyword_query(query)
        data = await self._json(
            "GET",
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": min(limit, 20), "page": 1},
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        records = data.get("items")

        sources: list[ResearchSource] = []
        for raw in _bounded_records(records, limit):
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=raw.get("full_name") or raw.get("name"),
                url=raw.get("html_url"),
                snippet=raw.get("description"),
                record_id=raw.get("node_id") or raw.get("id"),
            )
            if item:
                sources.append(item)
        return sources


class GitHubUsersProvider(HTTPProvider):
    """Public user identities from GitHub's documented REST search endpoint."""

    name = "github_users"
    kind = "web"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        query = _structured_keyword_query(query)
        data = await self._json(
            "GET",
            "https://api.github.com/search/users",
            params={"q": query, "per_page": min(limit, 20), "page": 1},
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        records = data.get("items")

        sources: list[ResearchSource] = []
        for raw in _bounded_records(records, limit):
            login = _plain_text(raw.get("login"), 160)
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=f"{login} (GitHub profile)" if login else "",
                url=raw.get("html_url"),
                snippet=f"Public GitHub user profile for {login}." if login else "",
                record_id=raw.get("node_id") or raw.get("id"),
            )
            if item:
                sources.append(item)
        return sources


class HackerNewsAlgoliaProvider(HTTPProvider):
    """Hacker News stories and discussions from the public Algolia search API."""

    name = "hacker_news_algolia"
    kind = "web"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        query = _structured_keyword_query(query)
        data = await self._json(
            "GET",
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "hitsPerPage": min(limit, 20), "page": 0},
        )
        records = data.get("hits")

        sources: list[ResearchSource] = []
        for raw in _bounded_records(records, limit):
            record_id = _plain_text(raw.get("objectID"), 40)
            discussion_url = (
                f"https://news.ycombinator.com/item?id={record_id}"
                if re.fullmatch(r"\d+", record_id)
                else None
            )
            title = raw.get("title") or raw.get("story_title")
            snippet = raw.get("story_text") or raw.get("comment_text") or title
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=title,
                url=raw.get("url") or raw.get("story_url") or discussion_url,
                snippet=snippet,
                authors=[str(raw.get("author") or "")],
                published_date=raw.get("created_at"),
                record_id=record_id,
            )
            # An invalid external URL should not discard an otherwise valid HN record.
            if item is None and discussion_url:
                item = _source(
                    provider=self.name,
                    kind=self.kind,
                    query=query,
                    title=title,
                    url=discussion_url,
                    snippet=snippet,
                    authors=[str(raw.get("author") or "")],
                    published_date=raw.get("created_at"),
                    record_id=record_id,
                )
            if item:
                sources.append(item)
        return sources


class DDGSWebProvider:
    """One DDGS engine isolated behind a bounded, cancellable worker process."""

    kind = "web"
    name = "ddgs"
    backend = "duckduckgo"

    def __init__(self, settings: Settings):
        self.settings = settings

    async def _worker_records(self, query: str, limit: int) -> list[dict[str, Any]]:
        output_limit = min(
            DDGS_WORKER_MAX_OUTPUT_BYTES,
            self.settings.search_response_limit_bytes,
        )
        worker_timeout = self.settings.search_provider_timeout_seconds
        environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "localllm.ddgs_worker",
            self.backend,
            str(min(limit, MAX_PROVIDER_RECORDS)),
            str(max(2, min(10, math.ceil(worker_timeout)))),
            str(output_limit),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
        )

        async def collect() -> bytes:
            if process.stdin is None or process.stdout is None:
                raise ProviderResponseError(f"{self.name} worker pipes were unavailable")
            try:
                process.stdin.write(query.encode("utf-8"))
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()

            body = bytearray()
            while chunk := await process.stdout.read(65_536):
                if len(body) + len(chunk) > output_limit:
                    raise ProviderResponseError(
                        f"{self.name} worker output exceeded the size limit"
                    )
                body.extend(chunk)
            return_code = await process.wait()
            if return_code != 0:
                raise ProviderResponseError(f"{self.name} fallback worker failed")
            return bytes(body)

        try:
            output = await asyncio.wait_for(collect(), timeout=worker_timeout)
        except asyncio.TimeoutError as exc:
            raise ProviderResponseError(f"{self.name} fallback worker timed out") from exc
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()

        try:
            payload = _load_bounded_json(output)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ProviderResponseError(f"{self.name} worker returned invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ProviderResponseError(f"{self.name} fallback worker failed")
        return _bounded_records(payload.get("results"), limit)

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        sources: list[ResearchSource] = []
        for raw in await self._worker_records(query, limit):
            item = _source(
                provider=self.name,
                kind="web",
                query=query,
                title=raw.get("title"),
                url=raw.get("href") or raw.get("url"),
                snippet=raw.get("body") or raw.get("snippet"),
            )
            if item:
                sources.append(item)
        return sources


class DuckDuckGoProvider(DDGSWebProvider):
    name = "duckduckgo"
    backend = "duckduckgo"


class BraveHtmlProvider(DDGSWebProvider):
    name = "brave_html"
    backend = "brave"


class YahooHtmlProvider(DDGSWebProvider):
    name = "yahoo_html"
    backend = "yahoo"


class MojeekHtmlProvider(DDGSWebProvider):
    name = "mojeek_html"
    backend = "mojeek"


class CrossrefProvider(HTTPProvider):
    name = "crossref"
    kind = "paper"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        params: dict[str, Any] = {
            "query.bibliographic": query,
            "rows": min(limit, 20),
            "select": "DOI,title,author,published,URL,abstract,is-referenced-by-count,type",
        }
        if self.settings.search_crossref_email:
            params["mailto"] = self.settings.search_crossref_email
        data = await self._json("GET", "https://api.crossref.org/works", params=params)
        message = data.get("message")
        records = message.get("items") if isinstance(message, dict) else []
        sources: list[ResearchSource] = []
        for raw in _bounded_records(records, limit):
            doi = _normalise_doi(raw.get("DOI"))
            title_value = raw.get("title") or []
            title = title_value[0] if isinstance(title_value, list) and title_value else title_value
            authors = []
            author_records = raw.get("author")
            if not isinstance(author_records, list):
                author_records = []
            for author in author_records[:20]:
                if isinstance(author, dict):
                    authors.append(
                        " ".join(filter(None, [author.get("given"), author.get("family")]))
                    )
            date_parts = (raw.get("published") or {}).get("date-parts") or []
            published = "-".join(str(part) for part in date_parts[0]) if date_parts else None
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=title,
                url=(f"https://doi.org/{doi}" if doi else raw.get("URL")),
                snippet=raw.get("abstract") or raw.get("type"),
                authors=authors,
                published_date=published,
                doi=doi,
                citation_count=raw.get("is-referenced-by-count"),
                record_id=doi,
            )
            if item:
                sources.append(item)
        return sources


class SemanticScholarProvider(HTTPProvider):
    name = "semantic_scholar"
    kind = "paper"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        headers = {}
        if self.settings.search_semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.search_semantic_scholar_api_key
        data = await self._json(
            "GET",
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query.replace("-", " "),
                "limit": min(limit, 20),
                "fields": (
                    "title,url,abstract,authors,year,publicationDate,citationCount,externalIds,"
                    "openAccessPdf"
                ),
            },
            headers=headers,
        )
        sources: list[ResearchSource] = []
        for raw in _bounded_records(data.get("data"), limit):
            external = raw.get("externalIds") or {}
            doi = _normalise_doi(external.get("DOI"))
            open_pdf = raw.get("openAccessPdf") or {}
            url = open_pdf.get("url") or raw.get("url")
            author_records = raw.get("authors")
            if not isinstance(author_records, list):
                author_records = []
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=raw.get("title"),
                url=url,
                snippet=raw.get("abstract"),
                authors=[
                    author.get("name", "")
                    for author in author_records[:20]
                    if isinstance(author, dict)
                ],
                published_date=raw.get("publicationDate") or raw.get("year"),
                doi=doi,
                citation_count=raw.get("citationCount"),
                record_id=raw.get("paperId"),
            )
            if item:
                sources.append(item)
        return sources


class EuropePMCProvider(HTTPProvider):
    name = "europe_pmc"
    kind = "paper"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        data = await self._json(
            "GET",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": query,
                "format": "json",
                "pageSize": min(limit, 20),
                "resultType": "core",
            },
        )
        sources: list[ResearchSource] = []
        result_list = data.get("resultList")
        records = result_list.get("result") if isinstance(result_list, dict) else []
        for raw in _bounded_records(records, limit):
            doi = _normalise_doi(raw.get("doi"))
            source_id = raw.get("pmcid") or raw.get("pmid") or raw.get("id")
            if doi:
                url = f"https://doi.org/{doi}"
            elif source_id:
                url = f"https://europepmc.org/article/{raw.get('source', 'MED')}/{source_id}"
            else:
                continue
            author_list = raw.get("authorList")
            author_records = author_list.get("author") if isinstance(author_list, dict) else []
            authors = [
                author.get("fullName", "") for author in _bounded_records(author_records, 20)
            ]
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=raw.get("title"),
                url=url,
                snippet=raw.get("abstractText"),
                authors=authors,
                published_date=raw.get("firstPublicationDate") or raw.get("pubYear"),
                doi=doi,
                citation_count=raw.get("citedByCount"),
                record_id=source_id,
            )
            if item:
                sources.append(item)
        return sources


class ArxivProvider(HTTPProvider):
    name = "arxiv"
    kind = "paper"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        identifiers = _query_arxiv_ids(query)
        params: dict[str, object] = {
            "start": 0,
            "max_results": min(limit, len(identifiers) or 20, 20),
        }
        if identifiers:
            params["id_list"] = ",".join(identifiers)
        else:
            params.update({"search_query": f"all:{query}", "sortBy": "relevance"})
        document = await self._text(
            "https://export.arxiv.org/api/query",
            params=params,
        )
        try:
            root = ET.fromstring(document)
        except ET.ParseError as exc:
            raise ProviderResponseError("arxiv returned invalid Atom XML") from exc
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        sources: list[ResearchSource] = []
        for entry in root.findall("atom:entry", namespace)[:MAX_PROVIDER_RECORDS]:
            entry_id = entry.findtext("atom:id", default="", namespaces=namespace)
            canonical_entry_url = _canonical_arxiv_entry_url(entry_id)
            identity_match = _ARXIV_ID_PATH.fullmatch(urlparse(canonical_entry_url).path)
            returned_identifier = identity_match.group("identifier") if identity_match else ""
            if identifiers and not _arxiv_response_matches_request(
                returned_identifier, identifiers
            ):
                continue
            authors = [
                author.findtext("atom:name", default="", namespaces=namespace)
                for author in entry.findall("atom:author", namespace)[:20]
            ]
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=entry.findtext("atom:title", default="", namespaces=namespace),
                url=canonical_entry_url,
                snippet=entry.findtext("atom:summary", default="", namespaces=namespace),
                authors=authors,
                published_date=entry.findtext("atom:published", default="", namespaces=namespace),
                record_id=returned_identifier,
            )
            if item:
                sources.append(item)
                if len(sources) >= limit:
                    break
        return sources


class OpenAlexProvider(HTTPProvider):
    name = "openalex"
    kind = "paper"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        data = await self._json(
            "GET",
            "https://api.openalex.org/works",
            params={
                "api_key": self.settings.search_openalex_api_key,
                "search": query,
                "per-page": min(limit, 20),
            },
        )
        sources: list[ResearchSource] = []
        for raw in _bounded_records(data.get("results"), limit):
            doi = _normalise_doi(raw.get("doi"))
            primary = raw.get("primary_location") or {}
            best_oa = raw.get("best_oa_location") or {}
            url = best_oa.get("landing_page_url") or primary.get("landing_page_url")
            url = url or (f"https://doi.org/{doi}" if doi else raw.get("id"))
            authorship_records = raw.get("authorships")
            if not isinstance(authorship_records, list):
                authorship_records = []
            authors = [
                ((authorship.get("author") or {}).get("display_name") or "")
                for authorship in authorship_records[:20]
                if isinstance(authorship, dict) and isinstance(authorship.get("author") or {}, dict)
            ]
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=raw.get("display_name") or raw.get("title"),
                url=url,
                snippet=(raw.get("type") or "") + "; " + (raw.get("language") or ""),
                authors=authors,
                published_date=raw.get("publication_date") or raw.get("publication_year"),
                doi=doi,
                citation_count=raw.get("cited_by_count"),
                record_id=raw.get("id"),
            )
            if item:
                sources.append(item)
        return sources


class GoogleScholarSerpApiProvider(HTTPProvider):
    """Google Scholar compatibility through SerpAPI; never scrapes Scholar HTML."""

    name = "google_scholar_serpapi"
    kind = "paper"

    async def search(self, query: str, limit: int) -> list[ResearchSource]:
        data = await self._json(
            "GET",
            "https://serpapi.com/search.json",
            params={
                "engine": "google_scholar",
                "q": query,
                "num": min(limit, 20),
                "api_key": self.settings.search_serpapi_api_key,
            },
        )
        sources: list[ResearchSource] = []
        for raw in _bounded_records(data.get("organic_results"), limit):
            publication = raw.get("publication_info") or {}
            summary = publication.get("summary") or ""
            cited = (raw.get("inline_links") or {}).get("cited_by") or {}
            item = _source(
                provider=self.name,
                kind=self.kind,
                query=query,
                title=raw.get("title"),
                url=raw.get("link"),
                snippet=raw.get("snippet"),
                published_date=summary,
                citation_count=cited.get("total"),
                record_id=raw.get("result_id"),
            )
            if item:
                sources.append(item)
        return sources


_PROVIDER_PRIOR = {
    "openalex": 1.2,
    "semantic_scholar": 1.15,
    "crossref": 1.1,
    "europe_pmc": 1.1,
    "arxiv": 1.05,
    "google_scholar_serpapi": 1.0,
    "brave": 0.9,
    "tavily": 0.9,
    "serper": 0.85,
    "github_users": 0.84,
    "wikipedia": 0.82,
    "github_repositories": 0.8,
    "hacker_news_algolia": 0.78,
    "duckduckgo": 0.75,
    "brave_html": 0.72,
    "yahoo_html": 0.7,
    "mojeek_html": 0.68,
}


class FederatedSearch:
    """Bounded provider fan-out, normalization, deduplication, and ranking."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._provider_semaphore = asyncio.Semaphore(settings.search_max_concurrency)
        self._validation_semaphore = asyncio.Semaphore(settings.search_max_concurrency)
        self._search_cache: OrderedDict[
            tuple[str, SearchMode, int, int], tuple[float, SearchOutcome]
        ] = OrderedDict()
        self._general: list[SearchProvider] = []
        if settings.search_brave_api_key:
            self._general.append(BraveProvider(settings))
        if settings.search_tavily_api_key:
            self._general.append(TavilyProvider(settings))
        if settings.search_serper_api_key:
            self._general.append(SerperProvider(settings))
        self._keyless_web: list[SearchProvider] = [
            YahooHtmlProvider(settings),
            WikipediaProvider(settings),
            GitHubUsersProvider(settings),
            GitHubRepositoriesProvider(settings),
            HackerNewsAlgoliaProvider(settings),
            DuckDuckGoProvider(settings),
            BraveHtmlProvider(settings),
            MojeekHtmlProvider(settings),
        ]

        # These APIs permit useful unauthenticated access. OpenAlex is now opt-in
        # because it requires a configured key; Semantic Scholar accepts an optional key.
        self._academic: list[SearchProvider] = [
            CrossrefProvider(settings),
            SemanticScholarProvider(settings),
            EuropePMCProvider(settings),
            ArxivProvider(settings),
        ]
        if settings.search_openalex_api_key:
            self._academic.append(OpenAlexProvider(settings))
        if settings.search_serpapi_api_key:
            self._academic.append(GoogleScholarSerpApiProvider(settings))

    def status(self) -> dict[str, Any]:
        configured = {
            "brave": bool(self.settings.search_brave_api_key),
            "tavily": bool(self.settings.search_tavily_api_key),
            "serper": bool(self.settings.search_serper_api_key),
            "wikipedia": True,
            "github_users": True,
            "github_repositories": True,
            "hacker_news_algolia": True,
            "duckduckgo": True,
            "brave_html": True,
            "yahoo_html": True,
            "mojeek_html": True,
            "crossref": True,
            "semantic_scholar": bool(self.settings.search_semantic_scholar_api_key),
            "europe_pmc": True,
            "arxiv": True,
            "openalex": bool(self.settings.search_openalex_api_key),
            "google_scholar_serpapi": bool(self.settings.search_serpapi_api_key),
        }
        descriptions = [
            ProviderDescription(
                "brave", "web", configured["brave"], configured["brave"], True, "Brave Search API"
            ),
            ProviderDescription(
                "tavily",
                "web",
                configured["tavily"],
                configured["tavily"],
                True,
                "Tavily Search API",
            ),
            ProviderDescription(
                "serper",
                "web",
                configured["serper"],
                configured["serper"],
                True,
                "Serper Google Search API",
            ),
            ProviderDescription(
                "wikipedia",
                "web",
                True,
                True,
                False,
                "English Wikipedia through the official MediaWiki Action API",
            ),
            ProviderDescription(
                "github_users",
                "web",
                True,
                True,
                False,
                "Public user profiles through the GitHub REST search API",
            ),
            ProviderDescription(
                "github_repositories",
                "web",
                True,
                True,
                False,
                "Public repository metadata through the GitHub REST search API",
            ),
            ProviderDescription(
                "hacker_news_algolia",
                "web",
                True,
                True,
                False,
                "Hacker News stories and discussions through the public Algolia API",
            ),
            ProviderDescription(
                "duckduckgo",
                "web",
                True,
                True,
                False,
                "Keyless web-search fallback in a resource-limited DDGS worker",
            ),
            ProviderDescription(
                "brave_html",
                "web",
                True,
                True,
                False,
                "Keyless Brave public-search fallback via a resource-limited DDGS worker",
            ),
            ProviderDescription(
                "yahoo_html",
                "web",
                True,
                True,
                False,
                "Keyless Yahoo public-search fallback via a resource-limited DDGS worker",
            ),
            ProviderDescription(
                "mojeek_html",
                "web",
                True,
                True,
                False,
                "Keyless Mojeek public-search fallback via a resource-limited DDGS worker",
            ),
            ProviderDescription("crossref", "paper", True, True, False, "Crossref works metadata"),
            ProviderDescription(
                "semantic_scholar",
                "paper",
                configured["semantic_scholar"],
                True,
                False,
                "Semantic Scholar Academic Graph; key improves limits",
            ),
            ProviderDescription(
                "europe_pmc", "paper", True, True, False, "Europe PMC literature API"
            ),
            ProviderDescription("arxiv", "paper", True, True, False, "arXiv Atom API"),
            ProviderDescription(
                "openalex",
                "paper",
                configured["openalex"],
                configured["openalex"],
                True,
                "OpenAlex works API",
            ),
            ProviderDescription(
                "google_scholar_serpapi",
                "paper",
                configured["google_scholar_serpapi"],
                configured["google_scholar_serpapi"],
                True,
                "Optional Google Scholar results via SerpAPI; no scraping",
            ),
        ]
        return {
            "providers": [asdict(item) for item in descriptions],
            "modes": ["web", "papers", "both"],
            "limits": {
                "max_results": self.settings.search_max_results,
                "max_concurrency": self.settings.search_max_concurrency,
                "provider_timeout_seconds": self.settings.search_provider_timeout_seconds,
            },
        }

    async def _call_provider(
        self,
        provider: SearchProvider,
        query: str,
        limit: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[ResearchSource], ProviderDiagnostic]:
        started = time.monotonic()
        try:

            async def call_with_slot() -> list[ResearchSource]:
                async with semaphore:
                    return await provider.search(query, limit)

            # Include queueing for the shared slot in the deadline. Otherwise a burst
            # of local search requests can create an indefinitely growing waiter set.
            results = await asyncio.wait_for(
                call_with_slot(),
                timeout=self.settings.search_provider_timeout_seconds + 3.0,
            )
            for rank, source in enumerate(results, 1):
                for record in source.provenance:
                    if record.get("provider") == provider.name and "provider_rank" not in record:
                        record["provider_rank"] = rank
            diagnostic = ProviderDiagnostic(
                name=provider.name,
                kind=provider.kind,
                ok=True,
                result_count=len(results),
                duration_ms=int((time.monotonic() - started) * 1000),
                queries=[query],
            )
            return results, diagnostic
        except Exception as exc:
            # Never publish exception strings from HTTP clients: they commonly contain
            # request URLs and query-string credentials (for example SerpAPI/OpenAlex).
            if isinstance(exc, httpx.HTTPStatusError):
                message = f"HTTP {exc.response.status_code}"
            elif isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
                message = "provider request timed out"
            elif isinstance(exc, ProviderResponseError):
                message = str(exc)
            else:
                message = type(exc).__name__
            diagnostic = ProviderDiagnostic(
                name=provider.name,
                kind=provider.kind,
                ok=False,
                result_count=0,
                duration_ms=int((time.monotonic() - started) * 1000),
                error=message,
                queries=[query],
            )
            return [], diagnostic

    @staticmethod
    def _keyless_waves(
        providers: list[SearchProvider],
    ) -> tuple[list[SearchProvider], list[SearchProvider]]:
        """Run the reliable first wave before rate-prone anonymous HTML engines."""

        primary_names = {
            "yahoo_html",
            "wikipedia",
            "github_users",
            "github_repositories",
            "hacker_news_algolia",
        }
        primary = [provider for provider in providers if provider.name in primary_names]
        if not primary:
            return list(providers), []
        secondary = [provider for provider in providers if provider.name not in primary_names]
        return primary, secondary

    @staticmethod
    def _deduplicate(sources: list[ResearchSource]) -> list[ResearchSource]:
        merged: dict[str, ResearchSource] = {}
        for source in sources:
            title_key = "".join(
                character for character in source.title.casefold() if character.isalnum()
            )
            if source.doi:
                key = f"doi:{source.doi}"
            elif (
                source.kind == "paper"
                and len(title_key) >= 24
                and source.year is not None
                and source.authors
            ):
                author_key = "".join(
                    character for character in source.authors[0].casefold() if character.isalnum()
                )
                key = f"work:{title_key}:{source.year}:{author_key}"
            else:
                key = f"url:{source.url}"
            existing = merged.get(key)
            if existing is None:
                merged[key] = source
                continue
            existing.providers = list(dict.fromkeys([*existing.providers, *source.providers]))
            existing.provenance.extend(source.provenance)
            if len(source.snippet) > len(existing.snippet):
                existing.snippet = source.snippet
            if not existing.doi:
                existing.doi = source.doi
            if not existing.authors:
                existing.authors = source.authors
            if not existing.published_date:
                existing.published_date = source.published_date
                existing.year = source.year
            existing.citation_count = (
                max(existing.citation_count or 0, source.citation_count or 0) or None
            )
        return list(merged.values())

    @staticmethod
    def _rank(query: str, sources: list[ResearchSource]) -> list[ResearchSource]:
        terms = _query_relevance_terms(query)
        current_year = datetime.now(timezone.utc).year
        for source in sources:
            source.citation_count = _bounded_citation_count(source.citation_count)
            provider_names = source.providers or [source.provider]
            haystack_terms = _lexical_tokens(f"{source.title} {source.snippet}")
            title_terms = _lexical_tokens(source.title)
            overlap = len(terms & haystack_terms) / max(1, len(terms))
            title_overlap = len(terms & title_terms)
            citations = math.log1p(source.citation_count or 0) / 8.0
            provider_ranks = [
                record.get("provider_rank")
                for record in source.provenance
                if isinstance(record.get("provider_rank"), int) and record["provider_rank"] > 0
            ]
            reciprocal_rank = max((0.6 / rank for rank in provider_ranks), default=0.0)
            recency = 0.0
            if source.year:
                recency = max(0.0, 1.0 - min(30, current_year - source.year) / 30) * 0.25
            source.score = round(
                2.4 * overlap
                + 0.18 * title_overlap
                + max((_PROVIDER_PRIOR.get(item, 0.5) for item in provider_names), default=0.5)
                + 0.3 * max(0, len(provider_names) - 1)
                + min(0.8, citations)
                + recency
                + (0.2 if source.doi else 0.0)
                + (0.15 if source.snippet else 0.0)
                + reciprocal_rank,
                4,
            )
        return sorted(sources, key=lambda item: (-item.score, item.title.casefold(), item.url))

    @staticmethod
    def _select_diverse(
        ranked: list[ResearchSource], mode: SearchMode, limit: int
    ) -> list[ResearchSource]:
        """Keep both evidence lanes represented before filling by global score."""

        if mode != "both" or limit < 2:
            return ranked[:limit]
        web = [source for source in ranked if source.kind == "web"]
        papers = [source for source in ranked if source.kind == "paper"]
        if not web or not papers:
            return ranked[:limit]
        reserve = max(1, limit // 4)
        selected = [*web[:reserve], *papers[:reserve]]
        selected_ids = {id(source) for source in selected}
        selected.extend(source for source in ranked if id(source) not in selected_ids)
        # Restore score ordering after enforcing lane membership.
        selected = sorted(
            selected[:limit], key=lambda item: (-item.score, item.title.casefold(), item.url)
        )
        return selected

    async def search(
        self,
        query: str,
        mode: SearchMode,
        limit: int,
        *,
        public_url_validator: Callable[[str], Awaitable[bool]],
        provider_candidate_limit: int | None = None,
    ) -> SearchOutcome:
        query = re.sub(r"\s+", " ", query).strip()[:800]
        if len(query) < 3:
            return SearchOutcome(
                query=query,
                mode=mode,
                sources=[],
                providers=[],
                warnings=["Search query is too short"],
            )
        limit = min(max(1, limit), self.settings.search_max_results)
        semaphore = self._provider_semaphore
        providers: list[SearchProvider] = []
        if mode in {"web", "both"}:
            providers.extend(self._general)
        if mode in {"papers", "both"}:
            providers.extend(self._academic)
        # Explicitly named keyless engines provide model-independent failover while
        # retaining which public search frontend produced each result.
        secondary_keyless: list[SearchProvider] = []
        if mode in {"web", "both"} and not self._general:
            primary_keyless, secondary_keyless = self._keyless_waves(self._keyless_web)
            providers.extend(primary_keyless)

        # ``None`` preserves the legacy per-provider budget for existing
        # callers. Strict v2 may opt into a larger candidate pool, still under
        # the provider-wide record ceiling used by every adapter.
        per_provider_limit = (
            min(limit, 12)
            if provider_candidate_limit is None
            else min(MAX_PROVIDER_RECORDS, max(1, provider_candidate_limit))
        )
        cache_key = (query, mode, limit, per_provider_limit)
        now = time.monotonic()
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            cached_at, outcome = cached
            if now - cached_at <= SEARCH_CACHE_TTL_SECONDS:
                self._search_cache.move_to_end(cache_key)
                return copy.deepcopy(outcome)
            del self._search_cache[cache_key]
        calls = [
            self._call_provider(provider, query, per_provider_limit, semaphore)
            for provider in providers
        ]
        results = await asyncio.gather(*calls) if calls else []
        sources = [source for batch, _diagnostic in results for source in batch]
        diagnostics = [diagnostic for _batch, diagnostic in results]

        async def validate(url: str) -> bool:
            try:

                async def validate_with_slot() -> bool:
                    async with self._validation_semaphore:
                        return await public_url_validator(url)

                return await asyncio.wait_for(
                    validate_with_slot(),
                    timeout=self.settings.search_provider_timeout_seconds,
                )
            except Exception:
                return False

        async def safe_deduplicated(
            candidates: list[ResearchSource],
        ) -> list[ResearchSource]:
            deduplicated = self._deduplicate(candidates)
            site_hosts = _query_site_hosts(query)
            if site_hosts:
                deduplicated = [
                    source for source in deduplicated if _matches_query_site(source, site_hosts)
                ]
            deduplicated = [
                source for source in deduplicated if _source_is_relevant(query, source)
            ]
            candidate_limit = max(24, min(90, limit * 3))
            candidates_to_validate = self._select_diverse(
                self._rank(query, deduplicated), mode, candidate_limit
            )
            public_flags = await asyncio.gather(
                *(validate(source.url) for source in candidates_to_validate)
            )
            return [
                source
                for source, is_public in zip(candidates_to_validate, public_flags, strict=True)
                if is_public
            ]

        safe_sources = await safe_deduplicated(sources)

        # Count only canonical, public, deduplicated configured-provider results.
        # Otherwise duplicate or private hits could suppress the keyless fallback and
        # leave the caller with no usable web evidence after validation.
        web_count = sum(1 for source in safe_sources if source.kind == "web")
        fallback_providers = self._keyless_web if self._general else secondary_keyless
        if mode in {"web", "both"} and fallback_providers and web_count < min(4, limit):
            fallback_results = await asyncio.gather(
                *(
                    self._call_provider(provider, query, per_provider_limit, semaphore)
                    for provider in fallback_providers
                )
            )
            diagnostics.extend(item[1] for item in fallback_results)
            safe_fallback = await safe_deduplicated(
                [source for batch, _diagnostic in fallback_results for source in batch]
            )
            safe_sources = self._deduplicate([*safe_sources, *safe_fallback])

        ranked = self._select_diverse(self._rank(query, safe_sources), mode, limit)
        warnings = []
        failed = [provider.name for provider in diagnostics if not provider.ok]
        if failed:
            prefix = (
                "Some search connectors did not answer; successful fallbacks still supplied the evidence: "
                if ranked
                else "Search connectors unavailable: "
            )
            warnings.append(prefix + ", ".join(failed))
        if not ranked:
            warnings.append("No usable public search results were returned")
        outcome = SearchOutcome(query, mode, ranked, diagnostics, warnings)
        if ranked:
            self._search_cache[cache_key] = (time.monotonic(), copy.deepcopy(outcome))
            self._search_cache.move_to_end(cache_key)
            while len(self._search_cache) > SEARCH_CACHE_MAX_ENTRIES:
                self._search_cache.popitem(last=False)
        return outcome
