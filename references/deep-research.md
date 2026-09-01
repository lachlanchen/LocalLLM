# Deep Research pipeline

Deep Research is an agent workflow, not a single model capability. LocalLLM implements a bounded, inspectable version:

```text
question + Web/Papers/Both + Quick/Standard/Deep
  → deterministic query variants and source budget
  → bounded federation across enabled web and scholarly providers
  → normalize, DOI/title/URL deduplicate, rank, and preserve lane diversity
  → validate strictly public destinations and fetch identity-encoded pages
  → Trafilatura extracts main text and tables
  → pack escaped JSONL evidence within a conservative UTF-8 budget
  → local model writes a strict evidence-bounded Markdown report
  → citation validation, one bounded model repair, and conservative cited-unit salvage
  → report and source metadata are saved locally
```

Provider execution, normalization, deduplication, ranking, and provenance do
not depend on model tool-calling ability. Qwen3 4B, 8B, and 30B-A3B therefore
receive the same evidence for a given provider response; model size affects the
synthesis, not whether search tools are invoked correctly.

## Pipeline stages

1. **Plan**: remove URL/URI credentials, paths, query strings, and fragments from the query-planning copy of the question, preserve DOI identifiers, create one to three bounded query variants according to the selected depth, and choose an up-to-six, up-to-twelve, or up-to-twenty-source budget. `LOCALLLM_SEARCH_MAX_RESULTS` can lower each budget.
2. **Search**: fan out across the enabled provider lanes. Keyless web coverage uses structured Wikipedia MediaWiki, GitHub repository-search, and Hacker News Algolia APIs plus explicitly selected DDGS engines for DuckDuckGo, Brave, Yahoo, and Mojeek. Since DDGS eagerly buffers its engine response, every DDGS call runs in a cancellable worker with hard memory, CPU, deadline, environment, and output limits. Keyless paper coverage uses Crossref, Semantic Scholar at public limits, Europe PMC, and arXiv. Configured Brave, Tavily, Serper, OpenAlex, or Google Scholar-via-SerpAPI routes join the same federation. Provider-native rank and query provenance remain attached to every result; individual provider failures are visible and non-fatal when another lane returns evidence.
3. **Normalize**: canonicalize URLs and DOIs, merge duplicate records by DOI/title/URL, score lexical overlap, reciprocal provider rank, corroboration, citations, recency, and source metadata, then reserve evidence slots for both web and paper lanes in `both` mode.
4. **Read**: resolve targets, require every resolved address to be globally routable, and reject private, loopback, link-local, IPv6 site-local, multicast, reserved, unspecified, IPv4-mapped-private, and known IPv6 transition/translation destinations. Connect to a validated address while preserving the HTTP Host and TLS SNI, and repeat resolution and validation for every redirect. Page requests ask for `identity` content encoding and reject compressed replies so the five-million-byte streamed response limit also bounds decoded memory. Only supported text/HTML/XML content is extracted; PDFs are not parsed. Embedded base64/data payloads are removed before model use.
5. **Synthesize**: encode each represented source as one escaped JSON object per line with an immutable citation index and source-identity metadata. If the budget is tight, only extracted text and the snippet/abstract may be shortened; source identity is never rewritten to make a record fit, and an overlarge identity record is skipped. The 32,768-token model window reserves 4,096 tokens for output and 4,096 for the fixed prompt, subtracts the question's UTF-8 byte length conservatively, and caps evidence at 22,000 UTF-8 bytes. This tokenizer-independent byte accounting intentionally underuses context rather than risking overflow.

   The prompt permits headings, paragraphs, and list items only. It requires factual units to end in an in-range citation cluster, tells the model to distinguish fact from inference, and asks it to write comparisons as “less than” or “greater than.” The validator and repair behavior are described below.
6. **Persist**: save task state, progress, provider diagnostics, source metadata, and the report under `data/research/`. Directories use mode `0700` and task files use `0600`. Extracted page bodies are never returned by the task API or written into task JSON; they are cleared after synthesis.

For authenticated private orchestration, the exact-path v2 protocol separates
task creation, status polling, and cancellation into three query-free POST
routes. The task runner and archive remain owned by LocalLLM; a remote agent or
edge only receives the bounded public task envelope and never a filesystem path
or extracted page body.

## Capacity and persistence limits

The manager admits at most three queued/running tasks and executes only one
research pipeline at a time. A fourth submission receives HTTP 429. It prunes
the oldest terminal entries to keep the in-memory task cache at 32 whenever
possible; persisted tasks remain available by ID. Before creating a task it
inspects `data/research/` and refuses new work once the archive has 500 JSON
files or 256 MiB. An unreadable or unstatable archive fails closed instead of
assuming capacity is available.

A task is first written when it acquires the run slot and begins running, then
again at progress and terminal transitions. A still-waiting queued task exists
only in memory, so a service restart can drop an unstarted queued ID. If a task
file contains an interrupted queued/running state, a subsequent poll converts
it to an explicit failed state; LocalLLM does not silently resume model work.

## Security boundary

Search-result titles, snippets, academic abstracts, and fetched webpages are
untrusted. The synthesis system prompt explicitly tells the model to treat
source text as evidence rather than instructions. This mitigates but does not
eliminate prompt injection. The orchestrator—not source text—controls provider
destinations, available tools, source count, maximum page size, and final
request construction. Provider exceptions are sanitized so credentials in
headers, request bodies, or query strings are never returned as diagnostics.
JSONL escaping keeps newlines and source-like text inside a record from creating
a new citation identity, but source text remains adversarial evidence rather
than trusted instructions.

## Citation boundary

The accepted report-body dialect contains Markdown headings, paragraphs, and
ordered or unordered list items only. Tables, inline or fenced code, indented
code, HTML, blockquotes, thematic breaks, body links and images, reference
links, URLs, email addresses, domain-like text and autolinks, and angle brackets
are rejected. Because the angle-bracket characters are forbidden, comparisons
must be written as “less than” or “greater than.” Only the exact first-line H1
title `Research Report` and its supported Chinese variants, plus a small
allowlist of generic structural headings or numbered bold labels, are exempt
from citation; every substantive heading or label is a claim unit.

Every parsed paragraph, non-structural list item, substantive heading, and
substantive numbered label must end with a contiguous citation cluster such as
`[1]` or `[2][3]`. Citations elsewhere in the unit do not satisfy the terminal
rule, and every numeric bracket marker anywhere in the unit must be in range for
the represented JSONL evidence. Common Chinese/Japanese structural labels and
CJK terminal punctuation are accepted.

One model repair pass runs when validation fails. If a small model still puts an
existing valid marker at the start of a unit, a deterministic final pass may
move that marker to the end; it drops uncited units, units with any out-of-range
marker, empty sections, and unknown heading claims. It never creates a citation
number, and the complete strict validator runs again. If a weak model still
cannot satisfy the dialect, the task completes with a deterministic,
evidence-inventory-only report made entirely from service-owned text. That
fallback exposes each validated source for direct review, clearly retains no
model-generated conclusion, and is itself passed through the same validator.
Zero-evidence runs and any failure to render that inventory safely still fail.
The source appendix is then regenerated from the captured validated URLs,
outside the model-authored body.

These are structural guarantees only. They do not establish claim-level
coverage, prove that a cited source supports the surrounding claim, or perform
entailment validation. Users should open the regenerated links and verify
high-impact claims. A production-grade system would add claim-level coverage
and entailment checks, source-quality scoring, date normalization, stronger
post-redirect source-identity handling, PDF extraction, and independent review.

## Internet and privacy

Research necessarily sends queries to the enabled external search and metadata
providers and fetches public URLs. The selected LLM still runs locally. Chat
uses the same retrieval broker when Web, Papers, or All is selected, or when
local-first Auto detects an explicit current/web/scholarly evidence request.
Local mode and Vision Lab remain offline. LocalLLM never scrapes Google Scholar
HTML: Google Scholar results require an operator-supplied SerpAPI key and use
that third party's account, terms, and network boundary.

Pasted URL-shaped input is never forwarded verbatim as a provider query. Chat,
direct Quick Search, and Deep Research reduce a URL or URI to a bounded
hostname/authority label and remove credentials, path segments, query parameters,
and fragments; local filesystem paths become the inert phrase `local path`. Chat additionally
rejects a local model query plan if it reproduces a removed private term. DOI
identifiers are not mistaken for URLs and remain intact for Papers searches.
This boundary governs outbound query planning only: the original question can
still be present in the local conversation database or saved research task.

Quick grounded Chat streams status immediately. Auto first resolves the latest
turn with a deterministic local-first router; explicit Local bypasses that router
and all retrieval. A resolved search route asks the selected local model—without
the saved transcript—for one to three schema-constrained search variants and
publishes the accepted query/lane plan before retrieval. Invalid, URL-bearing, or
tool-shaped planner output is replaced with deterministic language-aware queries.
An unresolved referential follow-up produces a clarification and stops before any
external request. Searches run under fixed concurrency and deadline bounds; their
sources and provider diagnostics are merged across variants before answer tokens
stream. At completion Chat checks that visible citations exist and are within the
returned evidence range, and shows a warning if not. Deep Research
is stricter: it buffers, validates, attempts one citation-repair pass, applies
the conservative cited-unit salvage described above when necessary, and falls
back to the validated evidence inventory when a small model remains invalid.

## Why the API resembles—but does not claim parity with—cloud Deep Research

Cloud research products combine frontier models, proprietary search/ranking, long-horizon planning, browser automation, document processing, and evaluation infrastructure. LocalLLM supplies a useful, transparent local pipeline; it does not claim equivalent recall, source ranking, or reasoning reliability.

Provider configuration and the exact management-route contract are documented
in [Search and Research API](search-research-api.md).
