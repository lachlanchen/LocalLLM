# Deep Research pipeline

Deep Research is an agent workflow, not a single model capability. LocalLLM implements a bounded, inspectable version:

```text
question
  → local model proposes three distinct search queries
  → DDGS returns candidates
  → HTTP client fetches up to ten pages
  → Trafilatura extracts main text and tables
  → local model writes an evidence-bounded report
  → report and source metadata are saved locally
```

## Stages shown in the UI

1. **Plan**: generate several search angles. If the model is unavailable or produces invalid JSON, deterministic fallback queries keep the run inspectable.
2. **Search**: use DDGS to discover candidate pages. No prompt content is sent to OpenAI or another model API.
3. **Read**: fetch pages with a descriptive user agent, reject private/reserved network targets and unsafe redirects, stream through a 5 MB response cap, extract main content, and strip embedded base64/data payloads before they can consume model context.
4. **Synthesize**: number every source, prefer primary evidence when sources conflict, require inline `[1]` citations, distinguish facts from inference, and preserve uncertainty. A citation-repair pass runs when the first draft has no valid numbered citations.
5. **Persist**: save the task, progress, sources, and report under `data/research/`.

## Security boundary

Fetched webpages are untrusted. The synthesis system prompt explicitly tells the model to treat source text as evidence rather than instructions. This mitigates but does not eliminate prompt injection. The orchestrator—not webpage text—controls available tools, source count, maximum page size, and final request construction.

## Citation boundary

The pipeline encourages citations but cannot guarantee that every generated statement is entailed by a source. Users should open the linked pages and verify high-impact claims. A production-grade research system would add claim-level entailment checks, source quality scoring, date normalization, duplicate/canonical URL handling, PDF extraction, and independent report review.

## Internet and privacy

Research necessarily sends search queries to the configured search path and fetches public URLs. The selected LLM still runs locally. Chat and Vision do not use this path automatically.

## Why the API resembles—but does not claim parity with—cloud Deep Research

Cloud research products combine frontier models, proprietary search/ranking, long-horizon planning, browser automation, document processing, and evaluation infrastructure. LocalLLM supplies a useful, transparent local pipeline; it does not claim equivalent recall, source ranking, or reasoning reliability.
