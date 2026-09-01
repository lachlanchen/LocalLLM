from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import math
import re
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from tld import get_tld

from .query_privacy import redact_url_tokens
from .search import (
    MAX_DOI_CHARS,
    ProviderDiagnostic,
    ResearchSource,
    SearchOutcome,
    _canonical_url,
    _normalise_doi,
    canonical_published_date,
)

REQUEST_SCHEMA_VERSION = "localllm-grounded-search-request-v2"
POLICY_SCHEMA_VERSION = "localllm-grounded-search-policy-v1"
RESPONSE_SCHEMA_VERSION = "localllm-grounded-search-response-v2"
MAX_SEARCH_V2_JSON_BYTES = 16 * 1024
MAX_ALLOWED_DOMAINS = 16
MAX_EXACT_IDENTIFIERS = 8
MAX_EXACT_SEARCH_REQUESTS = 2
EXACT_SEARCH_ADMISSION_TIMEOUT_SECONDS = 0.5
EXACT_SEARCH_OVERALL_TIMEOUT_SECONDS = 45.0

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_DNS_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_DOI_ARXIV_PREFIX = "10.48550/arxiv."
_MODERN_ARXIV_PATTERN = re.compile(
    r"(?P<year_month>\d{4})\.(?P<sequence>\d{4,5})(?P<version>v[1-9]\d*)?"
)
_LEGACY_ARXIV_PATTERN = re.compile(
    r"(?P<category>[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*)/"
    r"(?P<sequence>\d{7})(?P<version>v[1-9]\d*)?"
)
_ARXIV_VERSION_PATTERN = re.compile(r"(?P<root>.+?)(?P<version>v[1-9]\d*)$")


class _ProcessWideExactSearchAdmission:
    """Event-loop-neutral process-wide admission for bounded exact fanout."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("exact search admission capacity must be positive")
        self.capacity = capacity
        self._in_use = 0
        self._lock = threading.Lock()

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    @staticmethod
    def _now() -> float:
        return asyncio.get_running_loop().time()

    @staticmethod
    async def _pause(delay: float) -> None:
        await asyncio.sleep(delay)

    async def acquire(self, timeout_seconds: float) -> bool:
        deadline = self._now() + max(0.0, timeout_seconds)
        first_attempt = True
        while True:
            if not first_attempt and self._now() >= deadline:
                return False
            with self._lock:
                if self._in_use < self.capacity:
                    self._in_use += 1
                    return True
            remaining = deadline - self._now()
            if remaining <= 0:
                return False
            first_attempt = False
            await self._pause(min(0.01, remaining))

    def release(self) -> None:
        with self._lock:
            if self._in_use <= 0:
                raise RuntimeError("exact search admission released without a lease")
            self._in_use -= 1


_exact_search_admission = _ProcessWideExactSearchAdmission(MAX_EXACT_SEARCH_REQUESTS)
_exact_cleanup_reapers: set[asyncio.Task[None]] = set()


class _StrictV2Model(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        strict=True,
        populate_by_name=False,
        serialize_by_alias=True,
    )


def canonical_arxiv_identifier(value: str) -> str | None:
    """Return the v2 canonical arXiv identity, or ``None`` for an invalid ID.

    The modern arXiv sequence width changed in January 2015.  Treating four-
    and five-digit sequences as interchangeable would let a backend satisfy a
    request with a different identity, so this parser deliberately does not use
    the looser grammar retained by the legacy search route.
    """

    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    folded = value.casefold()
    modern = _MODERN_ARXIV_PATTERN.fullmatch(folded)
    if modern is not None:
        year_month_text = modern.group("year_month")
        year_month = int(year_month_text)
        month = year_month % 100
        sequence = modern.group("sequence")
        if month < 1 or month > 12 or int(sequence) == 0:
            return None
        if 704 <= year_month <= 1412:
            expected_width = 4
        elif year_month >= 1501:
            expected_width = 5
        else:
            return None
        return folded if len(sequence) == expected_width else None

    legacy = _LEGACY_ARXIV_PATTERN.fullmatch(folded)
    if legacy is None or int(legacy.group("sequence")) == 0:
        return None
    return folded


def _canonical_doi(value: str) -> str | None:
    if not isinstance(value, str) or not value or len(value) > MAX_DOI_CHARS:
        return None
    # DOI names are defined over visible ASCII in this contract.  This also
    # excludes control characters that Python's ``\S`` accepts.
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        return None
    normalized = _normalise_doi(value)
    return normalized


def _arxiv_from_doi(doi: str) -> str | None:
    if not doi.startswith(_DOI_ARXIV_PREFIX):
        return None
    return canonical_arxiv_identifier(doi[len(_DOI_ARXIV_PREFIX) :])


def _canonical_identifier(kind: str, value: str) -> tuple[str, str] | None:
    if kind == "arxiv":
        normalized = canonical_arxiv_identifier(value)
        return ("arxiv", normalized) if normalized is not None else None
    if kind == "doi":
        normalized = _canonical_doi(value)
        if normalized is None:
            return None
        if normalized.startswith(_DOI_ARXIV_PREFIX):
            arxiv = _arxiv_from_doi(normalized)
            return ("arxiv", arxiv) if arxiv is not None else None
        return ("doi", normalized)
    return None


class ExactIdentifierV2(_StrictV2Model):
    kind: Literal["doi", "arxiv"]
    value: str = Field(min_length=1, max_length=MAX_DOI_CHARS)

    @model_validator(mode="after")
    def require_canonical_identity(self) -> ExactIdentifierV2:
        canonical = _canonical_identifier(self.kind, self.value)
        if canonical is None:
            raise ValueError("exact identifier is invalid")
        if canonical != (self.kind, self.value):
            if self.kind == "doi" and canonical[0] == "arxiv":
                raise ValueError("10.48550/arxiv DOI aliases must use kind 'arxiv'")
            raise ValueError("exact identifier must use canonical lowercase identity form")
        return self


def _canonical_domain(value: str) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 253:
        return None
    if value != value.casefold() or value.endswith("."):
        return None
    labels = value.split(".")
    if len(labels) < 2 or any(_DNS_LABEL_PATTERN.fullmatch(label) is None for label in labels):
        return None
    try:
        suffix = get_tld(
            f"http://{value}",
            as_object=True,
            fail_silently=True,
            fix_protocol=True,
        )
    except (TypeError, ValueError):
        return None
    # Reject a bare public suffix (for example ``com`` or ``co.uk``), which
    # would turn an allowlist item into a broad cross-tenant wildcard.
    if suffix is None or suffix.fld == suffix.tld:
        return None
    return value


class SearchConstraintsV2(_StrictV2Model):
    schema_version: Literal[POLICY_SCHEMA_VERSION] = Field(alias="schemaVersion")
    strategy: Literal["ranked", "exact"]
    allowed_domains: list[str] = Field(
        alias="allowedDomains",
        min_length=0,
        max_length=MAX_ALLOWED_DOMAINS,
    )
    exact_identifiers: list[ExactIdentifierV2] = Field(
        alias="exactIdentifiers",
        min_length=0,
        max_length=MAX_EXACT_IDENTIFIERS,
    )
    query_plan_digest: str = Field(alias="queryPlanDigest", pattern=_DIGEST_PATTERN.pattern)
    policy_digest: str = Field(alias="policyDigest", pattern=_DIGEST_PATTERN.pattern)

    @field_validator("allowed_domains")
    @classmethod
    def require_canonical_domains(cls, values: list[str]) -> list[str]:
        if any(_canonical_domain(value) != value for value in values):
            raise ValueError("allowedDomains must contain canonical public DNS names")
        if values != sorted(values):
            raise ValueError("allowedDomains must be sorted")
        if len(values) != len(set(values)):
            raise ValueError("allowedDomains must not contain duplicates")
        return values

    @field_validator("exact_identifiers")
    @classmethod
    def require_sorted_unique_identifiers(
        cls, values: list[ExactIdentifierV2]
    ) -> list[ExactIdentifierV2]:
        identities = [(identifier.kind, identifier.value) for identifier in values]
        if identities != sorted(identities):
            raise ValueError("exactIdentifiers must be sorted by kind then value")
        if len(identities) != len(set(identities)):
            raise ValueError("exactIdentifiers must not contain duplicates")
        return values


def _canonical_provider_query(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError:
        return ""
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in normalized):
        return ""
    return re.sub(r"\s+", " ", redact_url_tokens(normalized)).strip()


class SearchRequestV2(_StrictV2Model):
    schema_version: Literal[REQUEST_SCHEMA_VERSION] = Field(alias="schemaVersion")
    query: str = Field(min_length=3, max_length=800)
    mode: Literal["web", "papers", "both"]
    limit: int = Field(ge=1, le=30)
    constraints: SearchConstraintsV2

    @field_validator("query")
    @classmethod
    def require_canonical_provider_safe_query(cls, value: str) -> str:
        if _canonical_provider_query(value) != value:
            raise ValueError(
                "query must be NFC, single-space canonical, and unchanged by provider privacy redaction"
            )
        return value

    @model_validator(mode="after")
    def require_strategy_identifier_contract(self) -> SearchRequestV2:
        identifiers = self.constraints.exact_identifiers
        if self.constraints.strategy == "exact" and not identifiers:
            raise ValueError("exact strategy requires at least one exact identifier")
        if self.constraints.strategy == "exact" and self.mode == "web":
            raise ValueError("exact identifier strategy requires papers or both mode")
        if self.constraints.strategy == "ranked" and identifiers:
            raise ValueError("ranked strategy must not carry exact identifiers")
        if self.limit < len(identifiers):
            raise ValueError("limit must cover every exact identifier")
        return self


class MatchedExactIdentifierV2(_StrictV2Model):
    requested: ExactIdentifierV2
    returned: ExactIdentifierV2
    match_type: Literal["exact", "arxiv-root"] = Field(alias="matchType")


class SearchProvenanceResponseV2(_StrictV2Model):
    provider: str
    query: str
    record_id: str | None = None
    retrieved_at: str | None = None
    provider_rank: int | None = Field(default=None, ge=1)


class SearchSourceResponseV2(_StrictV2Model):
    rank: int = Field(ge=1, le=30)
    title: str
    url: str
    snippet: str
    provider: str
    providers: list[str]
    kind: Literal["web", "paper"]
    authors: list[str]
    year: int | None
    published_date: str | None
    doi: str | None
    citation_count: int | None
    score: float
    query: str
    provenance: list[SearchProvenanceResponseV2]
    canonical_url: str = Field(alias="canonicalUrl")
    domain: str
    identifiers: list[ExactIdentifierV2]
    identity_digest: str = Field(alias="identityDigest", pattern=_DIGEST_PATTERN.pattern)
    matched_allowed_domains: list[str] = Field(alias="matchedAllowedDomains")
    matched_exact_identifiers: list[MatchedExactIdentifierV2] = Field(
        alias="matchedExactIdentifiers"
    )

    @field_validator("published_date")
    @classmethod
    def require_canonical_published_date(cls, value: str | None) -> str | None:
        if value is not None and canonical_published_date(value) != value:
            raise ValueError("published_date must be a canonical calendar date or null")
        return value


class SearchProviderResponseV2(_StrictV2Model):
    name: str
    kind: Literal["web", "paper"]
    ok: bool
    result_count: int
    duration_ms: int
    error: str | None = None
    queries: list[str]


class SearchResponseV2(_StrictV2Model):
    schema_version: Literal[RESPONSE_SCHEMA_VERSION] = Field(alias="schemaVersion")
    policy_compliant: Literal[True] = Field(alias="policyCompliant")
    request: SearchRequestV2
    sources: list[SearchSourceResponseV2]
    providers: list[SearchProviderResponseV2]
    warnings: list[str]
    resolved_identifiers: list[ExactIdentifierV2] = Field(alias="resolvedIdentifiers")
    unresolved_identifiers: list[ExactIdentifierV2] = Field(alias="unresolvedIdentifiers")
    returned_identity_binding: str = Field(
        alias="returnedIdentityBinding", pattern=_DIGEST_PATTERN.pattern
    )

    @model_validator(mode="after")
    def require_bound_policy_consistency(self) -> SearchResponseV2:
        constraints = self.request.constraints
        if not hmac.compare_digest(constraints.policy_digest, compute_policy_digest(self.request)):
            raise ValueError("response request has a mismatched policyDigest")
        if len(self.sources) > self.request.limit:
            raise ValueError("response source count exceeds request limit")

        returned_identities: list[dict[str, Any]] = []
        for expected_rank, source in enumerate(self.sources, 1):
            if source.rank != expected_rank:
                raise ValueError("response source ranks must be contiguous and ordered")
            if source.url != source.canonical_url or _canonical_url(source.url) != source.url:
                raise ValueError("response source URL is not canonical")
            domain = (urlparse(source.canonical_url).hostname or "").casefold().rstrip(".")
            if source.domain != domain:
                raise ValueError("response source domain does not match canonicalUrl")

            identifier_payloads = [
                identifier.model_dump(mode="json") for identifier in source.identifiers
            ]
            identifier_tuples = [
                (identifier.kind, identifier.value) for identifier in source.identifiers
            ]
            if identifier_tuples != sorted(set(identifier_tuples)):
                raise ValueError("response source identifiers must be sorted and unique")
            expected_doi = next(
                (identifier.value for identifier in source.identifiers if identifier.kind == "doi"),
                None,
            )
            if source.doi != expected_doi:
                raise ValueError("response DOI must agree with canonical identifiers")
            expected_identity_digest = compute_source_identity_digest(
                source.canonical_url,
                source.domain,
                identifier_payloads,
            )
            if not hmac.compare_digest(source.identity_digest, expected_identity_digest):
                raise ValueError("response source identityDigest mismatch")

            expected_domains = _matching_allowed_domains(
                source.domain,
                constraints.allowed_domains,
            )
            if source.matched_allowed_domains != expected_domains:
                raise ValueError("response allowed-domain proof mismatch")
            if constraints.allowed_domains and not expected_domains:
                raise ValueError("response source is outside allowedDomains")

            expected_exact_matches = _matched_exact_identifiers(
                constraints.exact_identifiers,
                identifier_tuples,
            )
            actual_exact_matches = [
                match.model_dump(by_alias=True, mode="json")
                for match in source.matched_exact_identifiers
            ]
            if actual_exact_matches != expected_exact_matches:
                raise ValueError("response exact-identifier proof mismatch")
            if constraints.strategy == "exact" and not expected_exact_matches:
                raise ValueError("exact response source has no requested identity match")

            returned_identities.append(
                {
                    "rank": source.rank,
                    "identityDigest": source.identity_digest,
                    "matchedAllowedDomains": source.matched_allowed_domains,
                    "matchedExactIdentifiers": actual_exact_matches,
                }
            )

        expected_resolved = [
            identifier
            for identifier in constraints.exact_identifiers
            if any(
                _identifier_matches(
                    identifier,
                    [(returned.kind, returned.value) for returned in source.identifiers],
                )
                for source in self.sources
            )
        ]
        expected_unresolved = [
            identifier
            for identifier in constraints.exact_identifiers
            if identifier not in expected_resolved
        ]
        if self.resolved_identifiers != expected_resolved:
            raise ValueError("response resolvedIdentifiers coverage mismatch")
        if self.unresolved_identifiers != expected_unresolved:
            raise ValueError("response unresolvedIdentifiers coverage mismatch")

        expected_binding = compute_returned_identity_binding(
            constraints.query_plan_digest,
            constraints.policy_digest,
            returned_identities,
        )
        if not hmac.compare_digest(self.returned_identity_binding, expected_binding):
            raise ValueError("response returnedIdentityBinding mismatch")
        return self


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_policy_document(payload: SearchRequestV2) -> dict[str, Any]:
    """Return the normalized request subset bound by ``policyDigest``.

    Both digest fields are deliberately absent.  ``queryPlanDigest`` is an
    opaque caller correlation value and participates only in the returned
    identity binding.
    """

    return {
        "schemaVersion": payload.schema_version,
        "query": payload.query,
        "mode": payload.mode,
        "limit": payload.limit,
        "constraints": {
            "schemaVersion": payload.constraints.schema_version,
            "strategy": payload.constraints.strategy,
            "allowedDomains": list(payload.constraints.allowed_domains),
            "exactIdentifiers": [
                identifier.model_dump(mode="json")
                for identifier in payload.constraints.exact_identifiers
            ],
        },
    }


def canonical_policy_preimage(payload: SearchRequestV2) -> bytes:
    return _canonical_json_bytes(canonical_policy_document(payload))


def compute_policy_digest(payload: SearchRequestV2) -> str:
    return "sha256:" + hashlib.sha256(canonical_policy_preimage(payload)).hexdigest()


def compute_source_identity_digest(
    canonical_url: str,
    domain: str,
    identifiers: Sequence[Mapping[str, str]],
) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_source_identity_preimage(canonical_url, domain, identifiers)
        ).hexdigest()
    )


def canonical_source_identity_preimage(
    canonical_url: str,
    domain: str,
    identifiers: Sequence[Mapping[str, str]],
) -> bytes:
    return _canonical_json_bytes(
        {
            "canonicalUrl": canonical_url,
            "domain": domain,
            "identifiers": [dict(identifier) for identifier in identifiers],
        }
    )


def compute_returned_identity_binding(
    query_plan_digest: str,
    policy_digest: str,
    returned_identities: Sequence[Mapping[str, Any]],
) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_returned_identity_preimage(
                query_plan_digest,
                policy_digest,
                returned_identities,
            )
        ).hexdigest()
    )


def canonical_returned_identity_preimage(
    query_plan_digest: str,
    policy_digest: str,
    returned_identities: Sequence[Mapping[str, Any]],
) -> bytes:
    return _canonical_json_bytes(
        {
            "queryPlanDigest": query_plan_digest,
            "policyDigest": policy_digest,
            "returnedIdentities": [dict(identity) for identity in returned_identities],
        }
    )


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey
        result[key] = value
    return result


def _bounded_json_integer(value: str) -> int:
    if len(value.lstrip("-")) > 128:
        raise ValueError("JSON integer exceeded the digit limit")
    return int(value)


def _bounded_json_float(value: str) -> float:
    if len(value) > 128:
        raise ValueError("JSON float exceeded the character limit")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON float must be finite")
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _json_structure_is_bounded(value: object) -> bool:
    stack: list[tuple[object, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > 2_000 or depth > 20:
            return False
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return True


async def parse_search_v2_request(request: Request) -> SearchRequestV2:
    """Read and validate v2 independently of FastAPI's permissive JSON parser."""

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_SEARCH_V2_JSON_BYTES:
                raise HTTPException(status_code=413, detail="JSON request exceeds the size limit")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc

    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_SEARCH_V2_JSON_BYTES:
            raise HTTPException(status_code=413, detail="JSON request exceeds the size limit")
        body.extend(chunk)
    try:
        decoded = json.loads(
            body,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_int=_bounded_json_integer,
            parse_float=_bounded_json_float,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJSONKey as exc:
        raise HTTPException(
            status_code=400,
            detail="Request body must not contain duplicate JSON object members",
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")
    if not _json_structure_is_bounded(decoded):
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")
    try:
        payload = SearchRequestV2.model_validate(decoded)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status_code=422, detail=errors[:20]) from exc
    expected_digest = compute_policy_digest(payload)
    if not hmac.compare_digest(payload.constraints.policy_digest, expected_digest):
        raise HTTPException(
            status_code=422,
            detail="constraints.policyDigest does not match the canonical normalized policy",
        )
    return payload


def _doi_from_url(canonical_url: str) -> tuple[bool, str | None]:
    parsed = urlparse(canonical_url)
    if (parsed.hostname or "").casefold().rstrip(".") not in {"doi.org", "dx.doi.org"}:
        return False, None
    candidate = unquote(parsed.path.lstrip("/"))
    return True, _canonical_doi(candidate)


def _arxiv_from_url(canonical_url: str) -> tuple[bool, str | None]:
    parsed = urlparse(canonical_url)
    if (parsed.hostname or "").casefold().rstrip(".") != "arxiv.org":
        return False, None
    if not parsed.path.startswith("/abs/"):
        return False, None
    if parsed.query or parsed.fragment:
        return True, None
    return True, canonical_arxiv_identifier(parsed.path[len("/abs/") :])


def _coalesce_arxiv_identities(candidates: Sequence[str]) -> tuple[bool, str | None]:
    if not candidates:
        return True, None
    by_root: dict[str, set[str | None]] = {}
    for candidate in candidates:
        match = _ARXIV_VERSION_PATTERN.fullmatch(candidate)
        root = match.group("root") if match else candidate
        version = match.group("version") if match else None
        by_root.setdefault(root, set()).add(version)
    if len(by_root) != 1:
        return False, None
    root, versions = next(iter(by_root.items()))
    explicit = sorted(version for version in versions if version is not None)
    if len(explicit) > 1:
        return False, None
    return True, f"{root}{explicit[0]}" if explicit else root


@dataclass(frozen=True)
class _DerivedSourceIdentity:
    canonical_url: str
    identifiers: tuple[tuple[str, str], ...]


def _derive_source_identity(source: ResearchSource) -> _DerivedSourceIdentity | None:
    """Derive metadata and URL identities independently and reject contradictions."""

    canonical_url = _canonical_url(source.url)
    if not canonical_url:
        return None

    metadata_raw = str(source.doi or "")
    metadata_doi = _canonical_doi(metadata_raw) if metadata_raw else None
    if metadata_raw and metadata_doi is None:
        return None

    is_doi_url, url_doi = _doi_from_url(canonical_url)
    if is_doi_url and url_doi is None:
        return None
    is_arxiv_url, url_arxiv = _arxiv_from_url(canonical_url)
    if is_arxiv_url and url_arxiv is None:
        return None

    doi_candidates: set[str] = set()
    arxiv_candidates: list[str] = []
    for candidate in (metadata_doi, url_doi):
        if candidate is None:
            continue
        if candidate.startswith(_DOI_ARXIV_PREFIX):
            arxiv_alias = _arxiv_from_doi(candidate)
            if arxiv_alias is None:
                return None
            arxiv_candidates.append(arxiv_alias)
        else:
            doi_candidates.add(candidate)
    if url_arxiv is not None:
        arxiv_candidates.append(url_arxiv)

    # Two independently asserted identities in one namespace must agree.  A
    # journal DOI plus an arXiv identity is allowed; two different DOIs or two
    # different arXiv roots/versions are an ambiguous provider record.
    if len(doi_candidates) > 1:
        return None
    arxiv_ok, arxiv_identity = _coalesce_arxiv_identities(arxiv_candidates)
    if not arxiv_ok:
        return None

    identifiers: set[tuple[str, str]] = {("doi", doi) for doi in doi_candidates}
    if arxiv_identity is not None:
        identifiers.add(("arxiv", arxiv_identity))

    if url_arxiv is not None:
        canonical_url = f"https://arxiv.org/abs/{url_arxiv}"
    elif url_doi is not None:
        url_doi_arxiv = _arxiv_from_doi(url_doi)
        if url_doi_arxiv is not None:
            canonical_url = f"https://arxiv.org/abs/{url_doi_arxiv}"
    return _DerivedSourceIdentity(canonical_url, tuple(sorted(identifiers)))


def _domain_allowed(domain: str, allowed_domains: Sequence[str]) -> bool:
    if not allowed_domains:
        return True
    return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowed_domains)


def _matching_allowed_domains(domain: str, allowed_domains: Sequence[str]) -> list[str]:
    return [
        allowed
        for allowed in allowed_domains
        if domain == allowed or domain.endswith(f".{allowed}")
    ]


@dataclass(frozen=True)
class _EnrichedSource:
    public: dict[str, Any]
    identifiers: tuple[tuple[str, str], ...]


def _enrich_source(
    source: ResearchSource,
    allowed_domains: Sequence[str],
    requested_identifiers: Sequence[ExactIdentifierV2],
) -> _EnrichedSource | None:
    identity = _derive_source_identity(source)
    if identity is None:
        return None
    canonical_url = identity.canonical_url
    domain = (urlparse(canonical_url).hostname or "").casefold().rstrip(".")
    if not domain or not _domain_allowed(domain, allowed_domains):
        return None
    identifier_tuples = identity.identifiers
    identifiers = [{"kind": kind, "value": value} for kind, value in identifier_tuples]
    identity_digest = compute_source_identity_digest(canonical_url, domain, identifiers)
    public = source.public_dict()
    public["url"] = canonical_url
    public["published_date"] = canonical_published_date(source.published_date)
    public["doi"] = next(
        (identifier["value"] for identifier in identifiers if identifier["kind"] == "doi"),
        None,
    )
    public.update(
        canonicalUrl=canonical_url,
        domain=domain,
        identifiers=identifiers,
        identityDigest=identity_digest,
        matchedAllowedDomains=_matching_allowed_domains(domain, allowed_domains),
        matchedExactIdentifiers=_matched_exact_identifiers(
            requested_identifiers,
            identifier_tuples,
        ),
    )
    return _EnrichedSource(
        public=public,
        identifiers=identifier_tuples,
    )


def _identifier_match_type(
    requested: ExactIdentifierV2,
    returned: tuple[str, str],
) -> Literal["exact", "arxiv-root"] | None:
    if requested.kind == "doi":
        return "exact" if (requested.kind, requested.value) == returned else None
    if returned[0] != "arxiv":
        return None
    requested_match = _ARXIV_VERSION_PATTERN.fullmatch(requested.value)
    requested_root = requested_match.group("root") if requested_match else requested.value
    requested_version = requested_match.group("version") if requested_match else None
    returned_match = _ARXIV_VERSION_PATTERN.fullmatch(returned[1])
    returned_root = returned_match.group("root") if returned_match else returned[1]
    returned_version = returned_match.group("version") if returned_match else None
    if returned_root != requested_root:
        return None
    if requested_version is not None:
        return "exact" if returned_version == requested_version else None
    return "exact" if returned_version is None else "arxiv-root"


def _identifier_matches(
    requested: ExactIdentifierV2,
    returned: Sequence[tuple[str, str]],
) -> bool:
    return any(_identifier_match_type(requested, identifier) is not None for identifier in returned)


def _matched_exact_identifiers(
    requested: Sequence[ExactIdentifierV2],
    returned: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for request_identifier in requested:
        for returned_identifier in returned:
            match_type = _identifier_match_type(request_identifier, returned_identifier)
            if match_type is None:
                continue
            matches.append(
                {
                    "requested": _identifier_payload(request_identifier),
                    "returned": {
                        "kind": returned_identifier[0],
                        "value": returned_identifier[1],
                    },
                    "matchType": match_type,
                }
            )
    return matches


def _exact_backend_query(identifier: ExactIdentifierV2) -> str:
    return f"doi:{identifier.value}" if identifier.kind == "doi" else f"arXiv:{identifier.value}"


async def _join_futures_resiliently(
    futures: Sequence[asyncio.Future[Any]],
) -> bool:
    """Join owned work without letting repeated caller cancellation reach it.

    Return whether cancellation arrived while joining.  The caller can then
    propagate that cancellation only after every owned future is truly done.
    """

    joined = asyncio.gather(*futures, return_exceptions=True)
    cancellation_seen = False
    while True:
        try:
            await asyncio.shield(joined)
            return cancellation_seen
        except asyncio.CancelledError:
            cancellation_seen = True


async def _reap_exact_search(
    admission: _ProcessWideExactSearchAdmission,
    futures: Sequence[asyncio.Future[Any]],
) -> None:
    try:
        await _join_futures_resiliently(futures)
    finally:
        admission.release()


def _forget_exact_cleanup_reaper(
    admission: _ProcessWideExactSearchAdmission,
    futures: Sequence[asyncio.Future[Any]],
    reaper: asyncio.Task[None],
) -> None:
    _exact_cleanup_reapers.discard(reaper)
    # A task cancelled before its coroutine's first instruction never executes
    # the reaper's ``finally``.  Transfer to a successor instead of reopening
    # capacity while owned provider work is still alive.
    if reaper.cancelled():
        if any(not future.done() for future in futures):
            _transfer_exact_cleanup(admission, futures)
        else:
            admission.release()
        return
    try:
        reaper.exception()
    except asyncio.CancelledError:  # pragma: no cover - guarded above
        admission.release()


def _transfer_exact_cleanup(
    admission: _ProcessWideExactSearchAdmission,
    futures: Sequence[asyncio.Future[Any]],
) -> None:
    reaper = asyncio.create_task(
        _reap_exact_search(admission, futures),
        name="search-v2-exact-cleanup-reaper",
    )
    _exact_cleanup_reapers.add(reaper)
    reaper.add_done_callback(partial(_forget_exact_cleanup_reaper, admission, futures))


async def _exact_outcomes(manager: Any, payload: SearchRequestV2) -> list[SearchOutcome]:
    # Bound per-request fanout in addition to process-wide admission and the
    # federation's own provider semaphore.
    lookup_slots = asyncio.Semaphore(2)
    candidate_limit = min(30, max(12, payload.limit))

    async def lookup(identifier: ExactIdentifierV2) -> SearchOutcome:
        async with lookup_slots:
            return await manager.quick_search(
                _exact_backend_query(identifier),
                "papers",
                candidate_limit,
            )

    tasks = [
        asyncio.create_task(lookup(identifier), name=f"search-v2-exact-{index}")
        for index, identifier in enumerate(payload.constraints.exact_identifiers)
    ]
    outcomes = asyncio.gather(*tasks)
    try:
        # Shield the group so caller cancellation cannot first cancel the
        # children implicitly and then cancel them a second time during teardown.
        return await asyncio.shield(outcomes)
    except BaseException as exc:
        for task in tasks:
            if not task.done():
                task.cancel()
        cancellation_seen = await _join_futures_resiliently([outcomes, *tasks])
        if cancellation_seen and not isinstance(exc, asyncio.CancelledError):
            raise asyncio.CancelledError from None
        raise


async def _admitted_exact_outcomes(manager: Any, payload: SearchRequestV2) -> list[SearchOutcome]:
    admission = _exact_search_admission
    acquired = await admission.acquire(EXACT_SEARCH_ADMISSION_TIMEOUT_SECONDS)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail="Exact search capacity is busy; retry later",
        )
    lease_transferred = False
    try:
        operation = asyncio.create_task(
            _exact_outcomes(manager, payload),
            name="search-v2-exact-operation",
        )
        deadline = asyncio.create_task(
            asyncio.sleep(EXACT_SEARCH_OVERALL_TIMEOUT_SECONDS),
            name="search-v2-exact-deadline",
        )
        first_completed = asyncio.create_task(
            asyncio.wait({operation, deadline}, return_when=asyncio.FIRST_COMPLETED),
            name="search-v2-exact-admission-wait",
        )
        try:
            await asyncio.shield(first_completed)
        except asyncio.CancelledError:
            if not operation.done():
                operation.cancel()
            if not deadline.done():
                deadline.cancel()
            _transfer_exact_cleanup(
                admission,
                [first_completed, operation, deadline],
            )
            lease_transferred = True
            raise

        if operation.done():
            if not deadline.done():
                deadline.cancel()
            cancellation_seen = await _join_futures_resiliently(
                [first_completed, operation, deadline]
            )
            if cancellation_seen:
                raise asyncio.CancelledError
            return operation.result()

        operation.cancel()
        _transfer_exact_cleanup(admission, [first_completed, operation, deadline])
        lease_transferred = True
        raise HTTPException(
            status_code=504,
            detail="Exact search exceeded its overall deadline",
        )
    finally:
        if not lease_transferred:
            admission.release()


def _ranked_candidate_limit(payload: SearchRequestV2) -> int:
    if not payload.constraints.allowed_domains:
        return payload.limit
    return min(30, max(payload.limit + 1, payload.limit * 3))


async def _ranked_outcome(manager: Any, payload: SearchRequestV2) -> SearchOutcome:
    candidate_limit = _ranked_candidate_limit(payload)
    quick_search = manager.quick_search
    try:
        supports_candidate_budget = (
            "provider_candidate_limit" in inspect.signature(quick_search).parameters
        )
    except (TypeError, ValueError):
        supports_candidate_budget = False
    if supports_candidate_budget:
        return await quick_search(
            payload.query,
            payload.mode,
            candidate_limit,
            provider_candidate_limit=candidate_limit,
        )
    return await quick_search(payload.query, payload.mode, candidate_limit)


def _deduplicate_enriched(sources: Sequence[_EnrichedSource]) -> list[_EnrichedSource]:
    distinct: dict[str, _EnrichedSource] = {}
    for source in sources:
        distinct.setdefault(source.public["identityDigest"], source)
    return list(distinct.values())


def _select_exact_sources(
    sources: Sequence[_EnrichedSource],
    requested: Sequence[ExactIdentifierV2],
    limit: int,
) -> list[_EnrichedSource]:
    matching = [
        source
        for source in sources
        if any(_identifier_matches(identifier, source.identifiers) for identifier in requested)
    ]
    selected: list[_EnrichedSource] = []
    selected_digests: set[str] = set()
    for identifier in requested:
        candidate = next(
            (
                source
                for source in matching
                if source.public["identityDigest"] not in selected_digests
                and _identifier_matches(identifier, source.identifiers)
            ),
            None,
        )
        if candidate is not None:
            selected.append(candidate)
            selected_digests.add(candidate.public["identityDigest"])
    selected.extend(
        source for source in matching if source.public["identityDigest"] not in selected_digests
    )
    return selected[:limit]


def _identifier_payload(identifier: ExactIdentifierV2) -> dict[str, str]:
    return {"kind": identifier.kind, "value": identifier.value}


def _conflicting_arxiv_versions(
    identifiers: Sequence[ExactIdentifierV2],
) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for identifier in identifiers:
        if identifier.kind != "arxiv":
            continue
        match = _ARXIV_VERSION_PATTERN.fullmatch(identifier.value)
        if match is not None:
            groups.setdefault(match.group("root"), set()).add(match.group("version"))
    return {root: versions for root, versions in groups.items() if len(versions) > 1}


def _provider_payload(diagnostic: ProviderDiagnostic) -> dict[str, Any]:
    return asdict(diagnostic)


async def execute_search_v2(manager: Any, payload: SearchRequestV2) -> SearchResponseV2:
    constraints = payload.constraints
    if constraints.strategy == "ranked":
        outcomes = [await _ranked_outcome(manager, payload)]
    else:
        outcomes = await _admitted_exact_outcomes(manager, payload)

    enriched = _deduplicate_enriched(
        [
            wrapped
            for outcome in outcomes
            for source in outcome.sources
            if (
                wrapped := _enrich_source(
                    source,
                    constraints.allowed_domains,
                    constraints.exact_identifiers,
                )
            )
            is not None
        ]
    )
    if constraints.strategy == "exact":
        selected = _select_exact_sources(
            enriched,
            constraints.exact_identifiers,
            payload.limit,
        )
    else:
        selected = enriched[: payload.limit]

    resolved: list[ExactIdentifierV2] = []
    unresolved: list[ExactIdentifierV2] = []
    for identifier in constraints.exact_identifiers:
        target = (
            resolved
            if any(_identifier_matches(identifier, source.identifiers) for source in selected)
            else unresolved
        )
        target.append(identifier)

    conflicts = _conflicting_arxiv_versions(constraints.exact_identifiers)
    unresolved_keys = {(identifier.kind, identifier.value) for identifier in unresolved}
    if any(
        any(("arxiv", f"{root}{version}") in unresolved_keys for version in versions)
        for root, versions in conflicts.items()
    ):
        raise HTTPException(
            status_code=409,
            detail="Conflicting requested arXiv versions could not all be proven by exact backend identities",
        )

    warnings = list(dict.fromkeys(warning for outcome in outcomes for warning in outcome.warnings))
    if unresolved:
        warnings.append(
            "Some exact identifiers could not be resolved to returned source identities"
        )

    for rank, source in enumerate(selected, 1):
        source.public["rank"] = rank
    returned_identities = [
        {
            "rank": source.public["rank"],
            "identityDigest": source.public["identityDigest"],
            "matchedAllowedDomains": source.public["matchedAllowedDomains"],
            "matchedExactIdentifiers": source.public["matchedExactIdentifiers"],
        }
        for source in selected
    ]
    response = {
        "schemaVersion": RESPONSE_SCHEMA_VERSION,
        "policyCompliant": True,
        "request": payload.model_dump(by_alias=True, mode="json"),
        "sources": [source.public for source in selected],
        "providers": [
            _provider_payload(diagnostic)
            for outcome in outcomes
            for diagnostic in outcome.providers
        ],
        "warnings": list(dict.fromkeys(warnings)),
        "resolvedIdentifiers": [_identifier_payload(identifier) for identifier in resolved],
        "unresolvedIdentifiers": [_identifier_payload(identifier) for identifier in unresolved],
        "returnedIdentityBinding": compute_returned_identity_binding(
            constraints.query_plan_digest,
            constraints.policy_digest,
            returned_identities,
        ),
    }
    # Treat any internal response-shape drift as a private 500 rather than
    # silently dropping or serializing an undeclared field at the HTTP boundary.
    return SearchResponseV2.model_validate(response)
