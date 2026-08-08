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
2. **Search**: use DDGS to discover candidate pages. Generated search queries, which may reveal the question's subject, go to DDGS-selected external search backends; no request is sent to a hosted model API.
3. **Read**: resolve targets, reject private/reserved addresses and unsafe redirects, connect to a validated address while preserving the HTTP host and TLS SNI, stream through a 5 MB response cap, extract main content, and strip embedded base64/data payloads before they can consume model context. Every redirect is resolved and validated again.
4. **Synthesize**: pack complete source blocks into the bounded evidence window and number exactly the sources actually sent to the model. The prompt asks the model to prefer primary evidence when sources conflict, cite every factual paragraph or bullet, distinguish facts from inference, and preserve uncertainty. Mechanical validation requires every parsed prose paragraph and non-structural list item to contain at least one bracketed citation, and checks that every citation number is in range. It excludes headings, fenced code, tables, thematic breaks, and structural numbered labels; it does not verify that every claim in a cited unit is supported or that the cited source entails it. A citation-repair pass runs when that check fails, and the run fails if the repaired draft still fails. The final source appendix is regenerated from captured URLs as canonical Markdown links rather than trusting model-written links.
5. **Persist**: save the task, progress, source metadata, and report under `data/research/`. Directories use mode `0700`, task files use `0600`, and extracted page bodies are neither returned by the public task API nor retained in the JSON report.

## Security boundary

Search-result titles and snippets and fetched webpages are untrusted. The synthesis system prompt explicitly tells the model to treat source text as evidence rather than instructions. This mitigates but does not eliminate prompt injection. The orchestrator—not source text—controls available tools, source count, maximum page size, and final request construction.

## Citation boundary

For a completed run, each parsed prose paragraph and non-structural list item has at least one citation, every bracketed citation number is in range for the packed evidence set, and the regenerated source links match the captured URLs. The check does not guarantee claim-level coverage within a cited paragraph or list item, or that a cited source entails it. Users should open the linked pages and verify high-impact claims. A production-grade research system would add claim-level coverage and entailment checks, source quality scoring, date normalization, duplicate/canonical URL handling, PDF extraction, and independent report review.

## Internet and privacy

Research necessarily sends search queries to DDGS-selected external backend(s) and fetches public URLs. The selected LLM still runs locally. Chat and Vision do not use this path automatically.

## Why the API resembles—but does not claim parity with—cloud Deep Research

Cloud research products combine frontier models, proprietary search/ranking, long-horizon planning, browser automation, document processing, and evaluation infrastructure. LocalLLM supplies a useful, transparent local pipeline; it does not claim equivalent recall, source ranking, or reasoning reliability.
