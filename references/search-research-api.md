# Search and Research API

LocalLLM keeps search orchestration outside the language model. A 4B, 8B, or MoE
model receives the same normalized, ranked, numbered evidence instead of being asked
to discover or invoke tools reliably on its own.

## Boundary and base URL

These management routes use `http://127.0.0.1:8008/api/...`; they are separate
from the OpenAI-compatible `/v1/*` surface. Search, Research, conversations,
grounded Chat, and Agent routes do **not** consult `LOCALLLM_API_KEY`.
`POST /api/search` and `POST /api/search/v2` instead support their own application
credential,
`LOCALLLM_SEARCH_API_KEY`, or the preferred file-backed
`LOCALLLM_SEARCH_API_KEY_FILE`. The settings are mutually exclusive. When one
is configured, each route requires exactly one
`Authorization: Bearer <search-key>` header. The scheme spelling,
single space, token, and header cardinality are exact; missing, duplicated,
malformed, or wrong credentials return the same HTTP 401 response with
`Cache-Control: no-store`. The key must use at most 512 visible ASCII characters
without whitespace and must differ from `LOCALLLM_API_KEY`. Leaving it empty
preserves the original unauthenticated loopback-only quick-search workflow.

Every image job/output read and mutation requires `LOCALLLM_API_KEY`; only
`GET /api/images/status` is loopback-public. Every route also relies on the fixed
loopback peer restriction plus browser origin/fetch-site checks, so a native
same-host process can still call ungated routes. Never proxy or tunnel raw port
8008. Publish only an exact reviewed path through a separately authenticated,
default-deny access-control layer. See
[OpenAI API compatibility](openai-api-compatibility.md) for the complete gateway
boundary.

For a protected systemd search worker, keep the strong random value in a
separate private credential source and expose only its runtime path to LocalLLM:

```systemd
[Service]
LoadCredential=localllm-search-api-key:/absolute/private/credential-source
Environment=LOCALLLM_SEARCH_API_KEY_FILE=%d/localllm-search-api-key
```

The application opens the final path with `O_NOFOLLOW`, requires one regular
single-link file, binds a systemd-managed credential to that exact basename,
accepts the host's exact private systemd materialization, and rejects empty,
whitespace-containing, non-ASCII, or oversized values. An owner-private regular
file may be used outside systemd. Configure the worker to inject the same value
in the Authorization header. Do not place the value itself in Git, manifests,
unit arguments, environment dumps, logs, screenshots, or documentation.

## Provider federation

| Provider | Lane | Default | Credential variable |
| --- | --- | --- | --- |
| Brave Search API | Web | Optional | `LOCALLLM_SEARCH_BRAVE_API_KEY` |
| Tavily Search API | Web | Optional | `LOCALLLM_SEARCH_TAVILY_API_KEY` |
| Serper Search API | Web | Optional | `LOCALLLM_SEARCH_SERPER_API_KEY` |
| English Wikipedia MediaWiki API | Web | Keyless fallback pool | None |
| GitHub repository-search REST API | Web | Keyless fallback pool | None |
| Hacker News Algolia API | Web | Keyless fallback pool | None |
| DuckDuckGo via DDGS engine `duckduckgo` | Web | Keyless fallback pool | None |
| Brave via DDGS engine `brave` | Web | Keyless fallback pool | None |
| Yahoo via DDGS engine `yahoo` | Web | Keyless fallback pool | None |
| Mojeek via DDGS engine `mojeek` | Web | Keyless fallback pool | None |
| Crossref REST API | Papers | Enabled | Optional polite-pool contact: `LOCALLLM_SEARCH_CROSSREF_EMAIL` |
| Semantic Scholar Academic Graph | Papers | Enabled at public limits | Optional: `LOCALLLM_SEARCH_SEMANTIC_SCHOLAR_API_KEY` |
| Europe PMC REST API | Papers | Enabled | None |
| arXiv Atom API | Papers | Enabled | None |
| OpenAlex Works API | Papers | Optional | `LOCALLLM_SEARCH_OPENALEX_API_KEY` |
| Google Scholar via SerpAPI | Papers | Optional | `LOCALLLM_SEARCH_SERPAPI_API_KEY` |

Google Scholar pages are never scraped. Google Scholar results are available only
from the third-party SerpAPI adapter when its operator-supplied key is configured;
this is not a direct Scholar API or a parity claim. SerpAPI receives the query under
its own account, terms, and network boundary. OpenAlex is also disabled until its
API key is configured.

The keyless web pool runs when no configured web API is available. It also runs
when configured providers yield fewer than `min(4, requested limit)` canonical,
public, deduplicated web results. Each DDGS adapter selects one named engine;
provider diagnostics and provenance therefore report the actual engine rather
than attributing all fallback results to DuckDuckGo. Because DDGS eagerly buffers
search-engine pages, each DDGS call runs in a separate worker with a 256 MiB
address-space limit, a 15-second CPU limit, a provider deadline, a bounded JSON
output pipe, and a minimal environment that excludes LocalLLM credentials and
proxy variables. Timeout or cancellation terminates the worker.

Optional resource controls are:

```dotenv
LOCALLLM_SEARCH_MAX_RESULTS=30
LOCALLLM_SEARCH_MAX_CONCURRENCY=4
LOCALLLM_SEARCH_PROVIDER_TIMEOUT_SECONDS=12
LOCALLLM_SEARCH_RESPONSE_LIMIT_BYTES=2000000
```

Restart `localllm-api.service` after changing credentials.

## Encoded request-body limits

The body limiter checks both a declared `Content-Length` and the bytes actually
received, including chunked bodies. Oversized requests receive HTTP 413.

| Route | Maximum encoded request body |
| --- | ---: |
| `POST /api/search` | 16 KiB |
| `POST /api/search/v2` | 16 KiB |
| `POST /api/research` | 32 KiB |
| `POST /api/agent/chat` | 25 MiB |

These transport caps are separate from the field and decoded-image limits
below. At most four request bodies are read concurrently by the bounded body
middleware.

## Quick search

`GET /api/search/status` returns enabled/configured providers and limits without
returning credentials. Its no-store `authentication` object reports only whether
the dedicated Bearer credential is required, the supported scheme, and its
quick-search scope. It is a configuration/capability description, not a live
provider health check; per-request diagnostics show actual success or failure.

`POST /api/search` accepts:

```json
{
  "query": "retrieval augmented generation citation accuracy",
  "mode": "papers",
  "limit": 12
}
```

Every response on this exact route is `Cache-Control: no-store`, including
authentication, body-limit, JSON-validation, provider, and internal-error
responses. The response boundary also removes every `Set-Cookie` field. This
policy is exact-path scoped and does not change neighboring management or `/v1`
routes.

`query` is 3–800 characters, `limit` is 1–30 and is also capped by
`LOCALLLM_SEARCH_MAX_RESULTS`, and `mode` is `web`, `papers`, or `both`. The
provider-bound copy of `query` applies the same URL/URI and local-path redaction
used by Chat and Deep Research; the original URL path, credentials, query, and
fragment are never sent to a provider. The response contains ranked `sources`,
per-provider diagnostics, and non-fatal warnings. Every source includes its provider
set, evidence kind, authors, publication metadata, DOI, provider citation count,
deterministic score, originating query, and provenance records. Provenance includes
provider-native rank where available.

Quick search returns normalized provider metadata, snippets, and abstracts. It
validates the public destination of every returned URL but does not fetch or extract
those pages. Use Deep Research when page reading is required.

Federation performs deterministic URL/DOI/title deduplication and reciprocal-rank,
query-overlap, corroboration, citation, and recency scoring. `both` mode reserves
space for both web and paper evidence before filling the remaining positions by
global score.

### Provider-neutral search v2

`POST /api/search/v2` is a separate, strict contract; the legacy route and its
response bytes are unchanged. It uses the same search-scoped Bearer dependency and
its own 16 KiB reader. Every exact-path response, including authentication,
validation, body-limit, provider, and internal errors, is `Cache-Control: no-store`
and has all `Set-Cookie` fields removed. Duplicate JSON member names and undeclared
request fields are rejected.

The required DTO is:

```json
{
  "schemaVersion": "localllm-grounded-search-request-v2",
  "query": "retrieval grounded citation accuracy",
  "mode": "papers",
  "limit": 4,
  "constraints": {
    "schemaVersion": "localllm-grounded-search-policy-v1",
    "strategy": "ranked",
    "allowedDomains": ["example.com"],
    "exactIdentifiers": [],
    "queryPlanDigest": "sha256:<64-lowercase-hex>",
    "policyDigest": "sha256:<64-lowercase-hex>"
  }
}
```

All fields are required and type-strict. `query` is NFC, 3–800 characters, uses
single spaces with no leading/trailing whitespace, and must be unchanged by the
provider privacy-redaction pass. `mode` is `web`, `papers`, or `both`; `limit` is an
integer from 1 through 30. `allowedDomains` contains at most 16 sorted, unique,
lowercase public DNS names (never a bare public suffix). Matching is hostname-exact
or dot-delimited subdomain matching; suffix lookalikes such as
`example.com.evil.invalid` do not match `example.com`.
For a ranked request with a domain policy, v2 asks federation for a candidate pool
of `min(30, max(limit + 1, limit * 3))` before the hard domain filter and then
truncates to `limit`. Each provider request retains the existing security cap of 20
records, and the actual pool can be smaller after provider yield, validation, and
deduplication. This bounded over-fetch can recover an allowed result at provider
rank 13 without claiming that every filtered slot can always be refilled.

`exactIdentifiers` contains at most eight sorted, unique `{kind,value}` records,
ordered by `kind` and then `value`. A DOI is its lowercase bare DOI. A
`10.48550/arxiv.*` DOI alias is noncanonical and must instead be one `arxiv`
identity; a suffix that is not a valid strict arXiv ID is rejected rather than
treated as an ordinary DOI, so the same identity is never represented as both DOI and arXiv. Legacy
arXiv IDs use lowercase `category/` plus exactly seven digits with a nonzero sequence.
Modern IDs from `0704` through `1412` use exactly four sequence digits; IDs from
`1501` onward use exactly five. The month is `01`–`12`, the sequence is nonzero,
and an optional version is `v` plus a positive integer without leading zeroes.
`ranked` forbids exact identifiers. `exact` requires one or more, requires
`papers` or `both`, and `limit` must cover their count.

The policy digest algorithm is deterministic:

1. Build an object with top-level `schemaVersion`, `query`, `mode`, `limit`, and a
   `constraints` object containing only `schemaVersion`, `strategy`,
   `allowedDomains`, and `exactIdentifiers`. Thus both schema versions are bound;
   Both digest fields, `queryPlanDigest` and `policyDigest`, are excluded.
2. Encode that object with UTF-8 JSON using recursively sorted object keys, no ASCII
   escaping, no insignificant whitespace (`,` and `:` separators), and no non-finite
   numbers. This is equivalent to Python `json.dumps(value, ensure_ascii=False,
   sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")`.
3. Set `policyDigest` to lowercase `sha256:` plus the SHA-256 hex digest of those
   bytes. LocalLLM recomputes it and rejects any mismatch. `queryPlanDigest` remains
   an opaque correlation input and is not interpreted.

A successful response has schema
`localllm-grounded-search-response-v2`, `policyCompliant: true`, an exact normalized
`request` echo, normalized provider diagnostics/warnings, `resolvedIdentifiers` and
`unresolvedIdentifiers`, and enriched sources. Each source includes its legacy
normalized metadata plus `rank`, `canonicalUrl`, `domain`, canonical `identifiers`,
`identityDigest`, `matchedAllowedDomains`, and `matchedExactIdentifiers`. An exact
match record binds `requested`, `returned`, and `matchType` (`exact` or
`arxiv-root`). The source identity digest uses the same canonical JSON/SHA-256
algorithm over `{canonicalUrl,domain,identifiers}`.
Metadata DOI claims, DOI-resolver URL identities, arXiv DOI aliases, and arxiv.org
URL identities are derived independently. Two different explicit identities in the
same namespace make the whole source ambiguous, so it is discarded before coverage;
a source cannot prove DOI A using a DOI URL for B or prove two arXiv roots at once.

The fixed source-identity preimage and digest vector is:

```text
{"canonicalUrl":"https://example.com/paper","domain":"example.com","identifiers":[{"kind":"arxiv","value":"2005.11401v1"},{"kind":"doi","value":"10.1000/example"}]}
```

`sha256:d3ed58ed7c051d5948cfb3cb212a5b7d3ac06a4544c5e9c0b2a2f7d840db4857`

`returnedIdentityBinding` uses the same digest algorithm over:

```json
{
  "queryPlanDigest": "...",
  "policyDigest": "...",
  "returnedIdentities": [
    {
      "rank": 1,
      "identityDigest": "...",
      "matchedAllowedDomains": ["example.com"],
      "matchedExactIdentifiers": []
    }
  ]
}
```

Interoperability vectors below are single UTF-8 lines with **no trailing LF**. The
canonical policy preimage is:

```text
{"constraints":{"allowedDomains":["arxiv.org","example.com"],"exactIdentifiers":[{"kind":"arxiv","value":"2005.11401v1"},{"kind":"doi","value":"10.1000/example"}],"schemaVersion":"localllm-grounded-search-policy-v1","strategy":"exact"},"limit":2,"mode":"papers","query":"量子 evidence","schemaVersion":"localllm-grounded-search-request-v2"}
```

Its digest is
`sha256:ccf5b13b08f247de0033a2c1d4c9bd3866ae0a8ce2b9cf411907080f39ec629c`.
Using that policy digest, a query-plan digest of `sha256:` followed by 64 `1`
characters, one identity digest of `sha256:` followed by 64 `2` characters, and the
shown exact arXiv match produces this returned-binding preimage:

```text
{"policyDigest":"sha256:ccf5b13b08f247de0033a2c1d4c9bd3866ae0a8ce2b9cf411907080f39ec629c","queryPlanDigest":"sha256:1111111111111111111111111111111111111111111111111111111111111111","returnedIdentities":[{"identityDigest":"sha256:2222222222222222222222222222222222222222222222222222222222222222","matchedAllowedDomains":["arxiv.org"],"matchedExactIdentifiers":[{"matchType":"exact","requested":{"kind":"arxiv","value":"2005.11401v1"},"returned":{"kind":"arxiv","value":"2005.11401v1"}}],"rank":1}]}
```

Its `returnedIdentityBinding` is
`sha256:3ddb7b8783c4dd600be7eaeed485f1af2821624232eaab6526426a4027375dd8`.

The order-sensitive two-record vector appends rank 2 with identity digest `sha256:`
plus 64 `3` characters, matched domain `example.com`, and an exact requested/returned
DOI of `10.1000/example`. Its exact preimage is:

```text
{"policyDigest":"sha256:ccf5b13b08f247de0033a2c1d4c9bd3866ae0a8ce2b9cf411907080f39ec629c","queryPlanDigest":"sha256:1111111111111111111111111111111111111111111111111111111111111111","returnedIdentities":[{"identityDigest":"sha256:2222222222222222222222222222222222222222222222222222222222222222","matchedAllowedDomains":["arxiv.org"],"matchedExactIdentifiers":[{"matchType":"exact","requested":{"kind":"arxiv","value":"2005.11401v1"},"returned":{"kind":"arxiv","value":"2005.11401v1"}}],"rank":1},{"identityDigest":"sha256:3333333333333333333333333333333333333333333333333333333333333333","matchedAllowedDomains":["example.com"],"matchedExactIdentifiers":[{"matchType":"exact","requested":{"kind":"doi","value":"10.1000/example"},"returned":{"kind":"doi","value":"10.1000/example"}}],"rank":2}]}
```

Its digest is
`sha256:fc7e740594137849892f6d9cd6ac3ad82901eafe8361d43741f8a626c42774dd`;
reversing the records changes the digest.

The ordered records are exactly those returned to the caller. Ranked results and
exact lookups are hard-filtered again by domain. DOI resolution uses a deterministic
federated `doi:<value>` query and then exact canonical-DOI filtering; arXiv resolution
uses a structured `arXiv:<value>` query and the same hard identity filter. A requested
version such as `v1` cannot be satisfied by an unversioned result or `v2`. Unproven
identifiers are reported honestly; conflicting explicit versions of one arXiv root
fail closed with HTTP 409 unless every requested version is proved by a distinct
returned identity. The strict response model turns undeclared internal output into a
private error rather than leaking or silently dropping it.

Exact federation is process-wide admitted: at most two exact requests run lookup
fanout concurrently, admission waits at most 0.5 seconds before HTTP 429, and each
admitted lookup set has a hard 45-second response deadline before HTTP 504. At most
two identifier lookups run inside one admitted request. Errors cancel each
unfinished sibling once and join it before returning. A deadline response or caller
cancellation may complete while cancellation cleanup is still running; a tracked,
cancellation-shielded reaper then retains the admission lease until every owned
lookup has actually exited, preventing delayed cleanup from reopening capacity.

## Deep research

`POST /api/research` creates an asynchronous task. `question` is 8–4,000
characters, and the request accepts `mode` and `depth` in addition to the model:

```json
{
  "question": "What evidence supports retrieval-grounded citation accuracy?",
  "model": "localllm-deep",
  "mode": "both",
  "depth": "deep"
}
```

`depth` is `quick`, `standard`, or `deep`. It deterministically controls query
variants and an up-to-6, up-to-12, or up-to-20 source budget, capped by
`LOCALLLM_SEARCH_MAX_RESULTS`. Task responses expose `mode`, `depth`,
`max_sources`, provider diagnostics/errors, progress, normalized sources, and the
final report.

Query variants are derived from a redacted planning copy of the question. URL
and URI credentials, paths, query strings, and fragments are removed; a bounded
hostname or authority label may remain, and local filesystem paths become the
inert phrase `local path`. DOI identifiers are preserved for scholarly search.
The original question remains in the private saved task record, so this is an
outbound-query control rather than a local data-erasure feature.

The manager admits at most three queued/running task runners and executes one
research pipeline at a time to avoid competing for the local GPUs. A fourth
submission returns HTTP 429 until a run finishes or is cancelled. It prunes
old terminal objects to keep the in-memory cache at 32 whenever possible.

- `GET /api/research/{task_id}` polls a task.
- `DELETE /api/research/{task_id}` cancels a queued/running task and returns its
  serialized state.
- Private brokers that require query-free exact routes use the authenticated
  v2 forms instead: `POST /api/research/v2/create`,
  `POST /api/research/v2/status`, and `POST /api/research/v2/cancel`. Create
  accepts the same research document; status and cancel accept only
  `{"task_id":"<12 lowercase hex>"}`. Every response is wrapped as
  `{"schema":"localllm/research-task/v2","task":{...}}`. These routes use
  the same bearer credential as `/api/search/v2`, reject extra fields, and keep
  the task identifier out of URLs and query strings so an exact-path LazyEdge
  role can expose them without adding wildcard routing. Every response from
  these three exact paths, including authentication, validation, capacity, and
  internal errors, is `Cache-Control: no-store` and strips `Set-Cookie` fields.
- Task states are `queued`, `running`, `complete`, `failed`, or `cancelled`.
  A task is first persisted under `data/research/` after it acquires the single
  run slot; a still-waiting queued task is memory-only. Progress and terminal
  states are persisted. If a saved task contains an interrupted queued/running
  state, polling after restart returns it as failed with an explicit interruption
  error; start a new run.
- New tasks fail closed when the saved archive cannot be inspected or once it
  reaches 500 JSON files or 256 MiB. Archive-capacity rejection is returned as
  HTTP 429; LocalLLM never deletes older reports automatically.

The synthesis pass receives escaped JSONL: one object per represented source,
with an immutable citation index and source-identity fields. Only extracted page
text and the snippet/abstract may be shortened to fit. Evidence is capped at
22,000 UTF-8 bytes, and the budget is reduced further by conservatively
subtracting the question plus a 4,096-token prompt reserve and 4,096-token
output reserve from a 32,768-token context request.

The accepted model-authored report body is deliberately limited to headings,
paragraphs, and list items. Tables, code, HTML, blockquotes, thematic breaks,
links/images, reference links, URLs, email addresses, domain-like text and
autolinks, and angle brackets
are rejected. Comparisons must use the words “less than” or “greater than.”
Only the exact first-line H1 title `Research Report` and its supported Chinese
variants, plus generic structural headings/numbered labels, are exempt;
substantive headings and labels are validated as claim units. Common CJK
structural headings and terminal punctuation are supported. Every parsed
non-exempt unit must end in a contiguous in-range citation cluster, and every
numeric bracket marker anywhere in the unit must be in range. One model repair
pass runs after a failure. If necessary, a deterministic salvage moves only
already-present valid markers to unit endings while dropping uncited,
out-of-range, empty-section, and unknown-heading content; it then reruns the
strict validator. If a weak model remains invalid, the service completes with
a deterministic evidence-inventory-only report containing no model-generated
conclusion; zero-evidence runs and any unsafe inventory-rendering failure still
fail. The service then regenerates the source appendix from captured validated
URLs.

This validation establishes report structure and citation-number range only.
It does not prove claim-level coverage, source support, or entailment.

## Network and evidence safety

- Search and fetch concurrency, provider time (including the wait for a shared
  provider slot), response bytes, extracted page
  bytes, source count, JSONL evidence bytes, task queue size, in-memory task
  count, and saved archive size are bounded.
- Search-provider destinations are fixed in code; user input cannot select an API
  endpoint.
- Every returned evidence URL is restricted to public HTTP(S) targets on ports 80 or
  443. Every DNS answer must be globally routable. Private, loopback,
  link-local, site-local, multicast, reserved, unspecified, IPv4-mapped-private,
  well-known/local-use NAT64, Teredo, and 6to4 transition/translation addresses
  are rejected.
- Page fetches connect to a validated IP while retaining the original Host header and
  TLS SNI. Every redirect is independently resolved and validated, preventing DNS
  rebinding and redirect-based SSRF.
- Structured provider and page clients request `Accept-Encoding: identity` and
  reject compressed replies. Their streamed byte caps therefore also bound
  decoded response memory rather than trusting a compressed wire size. The
  third-party DDGS client cannot expose a streaming cap, so it is confined to
  the separately killed, memory/CPU/output-limited worker described above.
- Provider exceptions are reduced to status/class diagnostics. Request URLs, headers,
  bodies, and query-string API keys are never returned to clients or stored in tasks.
- Extracted pages and search snippets are marked as untrusted data in model prompts.
  Embedded data URLs and large encoded payloads are removed before JSONL evidence
  packing. JSON escaping prevents record contents from introducing a new citation
  identity, but it does not make the text trustworthy.

Deep Research extracts supported HTML/XML text with Trafilatura. It can retain a
provider abstract/snippet when page text is unavailable, but it does not extract PDF
bodies in the current implementation.

Provider failure is non-fatal when another lane/provider returns evidence. The
response identifies failed providers, making partial coverage visible instead of
silently pretending the search was complete.

## Grounded chat stream

`POST /api/agent/chat` is the web application's bounded adaptive
search-then-chat route. It is separate from the OpenAI-compatible `/v1/*`
surface. A typical
request is:

```json
{
  "messages": [{"role": "user", "content": "Compare current RAG citation methods"}],
  "model": "localllm-fast",
  "mode": "auto",
  "limit": 12,
  "temperature": 0.35,
  "max_tokens": 2048
}
```

`mode` is `auto`, `local`, `web`, `papers`, or `all`. `auto` is the bundled UI
default and uses a deterministic local-first router over the latest user turn.
It remains local unless that turn explicitly asks for current/fresh/verified web
information, scholarly literature, or both; model size does not control this
decision. `local` always bypasses the query planner and every search provider.
The explicit `web`, `papers`, and `all` values override Auto. A pasted URL-shaped
token is an Auto Web signal, while a DOI is a Papers signal. Local filesystem
paths and dotted source/package/version tokens such as `package.json` and
`v1.2.3` do not trigger general web search by themselves.

For every resolved non-local request, the routed local model receives only the
current question—not the saved transcript—and returns a
JSON-schema-constrained plan of one to three passive queries. Web plans target
general public pages, Papers plans use scholarly metadata terms, and All requires
both lanes. Planner output is untrusted: query count and length are capped,
extra/tool fields and URLs are rejected, and unrelated or malformed output falls
back to deterministic language-aware variants. Before Chat's local planner
receives the question—and before either Chat or Deep Research sends a provider
query—URL/URI credentials, path, query, and fragment material is removed. Only
a bounded inert hostname or authority label may remain, and any model plan that
reproduces a removed term is rejected. The fallback preserves DOI identifiers
and the question language, including Chinese, Japanese, Korean, Arabic,
Cyrillic, and several common Latin-language patterns. This does not redact the
locally persisted original chat turn.

If a routed search request is an unresolved follow-up such as “What about its
latest release?”, the stream asks the user to name the model, project, device,
paper, or organization and stops before planner, provider, or answer-model calls.
This prevents a guess from becoming an unrelated external query.

Validated variants run with a concurrency cap of two, a per-variant deadline, and
an overall retrieval deadline. Results are merged across query and provider
provenance, deduplicated by DOI/canonical URL, reranked against the original user
question, and lane-balanced for All mode. A failed variant is non-fatal when another
variant returns evidence. Provider diagnostics aggregate attempts, successful
attempts, queries, result counts, duration, and `healthy`/`partial`/`unavailable`
state without exposing raw connector exception strings. The final evidence is
numbered, inserted as an explicitly untrusted internal message, and the request
fails closed without synthesis when no public evidence survives validation.
Grounded chat packs provider snippets/abstracts; unlike Deep Research, it does not
fetch result pages. The stream uses named Server-Sent Events:

| Event | Payload |
| --- | --- |
| `status` | `preparing`, Auto `routing`, optional `clarifying`, `planning`, `planned`, `searching`, `ranking`, and `generating` stages, including the resolved mode and bounded query/lane plan where relevant |
| `clarification` | a typed unresolved-reference reason, visible question, and resolved search mode; no external retrieval follows |
| `source` | one normalized, validated public source object |
| `warning` | a sanitized provider or citation warning |
| `reasoning` | optional model reasoning text when the runtime exposes it |
| `delta` | visible answer text |
| `done` | resolved/requested model, requested mode, Auto `resolved_mode` when applicable, sources, aggregate provider diagnostics, accepted `search_plan`, optional clarification, warnings, and `answer_truncated: true` when the visible persistence cap was reached |
| `error` | a sanitized terminal message |

Messages use the `system`/`user`/`assistant` roles and the final turn must be a
user message. A request allows up to 100 messages, 32,000 text characters per
message, 64 structured content parts per message, 80,000 text characters total,
and a 25 MiB encoded request body. Unknown request fields and unsupported content
part shapes are rejected rather than silently ignored.
`limit` is 1–20, `temperature` is 0–2, and `max_tokens` is 1–8,192.

The bundled Playground enforces the same 32,000-character limit before its
mandatory pre-inference conversation save. If that save fails for validation,
message/image quota, archive capacity, or local storage reasons, it makes no
model request and restores the exact draft, attachment, and prior transcript.
Visible assistant deltas are gracefully capped at 30,000 characters. At that
boundary the stream emits a warning and a normal `done` event, allowing the UI
to persist and resume the bounded answer below the 32,000-character store cap.
Direct `/v1/*` proxy responses are not subject to this grounded-chat persistence
cap.

Structured image parts accept only base64 data URLs for PNG, JPEG, or WebP,
validate file signatures and bounded raster dimensions, allow at most four
images, 8 MiB per image, 16 MiB decoded total, 16,384 pixels on either axis, and
40 megapixels. Animated raster formats are rejected, and remote image URLs are
never fetched. Image turns retain an explicitly selected vision model or safely
fall back to `localllm-vision` when a text-only model is requested.

Grounded chat emits a status event immediately, resolves Auto deterministically,
plans and performs retrieval only when the resulting mode is non-local, and then
streams answer deltas. Once response headers have been sent, a terminal
failure is reported as an SSE `error` event on the existing HTTP 200 stream.
At completion, the service checks visible text for missing or out-of-range
citations and emits a warning when necessary. This is not paragraph coverage or
an entailment check. Deep Research remains the stricter buffered workflow with
citation repair and fail-closed structural validation.

Primary provider and API sources are collected in the
[source ledger](sources.md#search-and-scholarly-metadata).
