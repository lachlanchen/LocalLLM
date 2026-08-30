# OpenAI-compatible local API

## Base URL and authentication

```text
http://127.0.0.1:8008/v1
Authorization: Bearer local-dev-key
```

`LOCALLLM_API_KEY` gates `/v1/*` plus every separate image job/output read and
mutation; only image-generation status remains loopback-public. Setting the key
to an empty value disables both key checks. The shipped `local-dev-key` is an
interoperability placeholder, not a secret. Other management `/api/*` routes do
not consult this key. `POST /api/search` can instead require an independent
search-scoped token through `LOCALLLM_SEARCH_API_KEY`, or production can load
the token through the mutually exclusive `LOCALLLM_SEARCH_API_KEY_FILE` systemd
credential path. That token cannot authenticate `/v1`, and configuration
rejects reuse of the `/v1` key. Every route relies on the
loopback peer restriction; all browser requests to `/api/*` and `/v1/*`
additionally receive `Origin`/`Sec-Fetch-Site` checks. A native process in the
same host network namespace can invoke them. Do not proxy or tunnel port 8008
without adding authentication and authorization.

The gateway accepts only the exact local Ollama upstream
`http://127.0.0.1:11434`. Startup configuration rejects HTTPS, credentials,
alternate hosts or ports, paths, queries, and fragments. Ollama requests also
use `trust_env=False`, so an outbound proxy environment variable cannot turn
the local gateway into an implicit remote-runtime client. LocalLLM does not
call hosted OpenAI models.

## Endpoints

| Endpoint | Local behavior |
| --- | --- |
| `GET /v1/models` | returns installed Ollama models plus LocalLLM aliases whose targets are installed |
| `GET /v1/models/{id}` | forwards retrieval to Ollama after alias resolution |
| `POST /v1/chat/completions` | forwards text, image content parts, streaming, JSON mode, tools, and supported sampling fields |
| `POST /v1/responses` | forwards Ollama’s non-stateful Responses implementation, including streaming and functions |
| `POST /v1/embeddings` | forwards to an installed embedding-capable Ollama model |

## Request-body limits

| Endpoint | Maximum encoded request body |
| --- | ---: |
| `POST /v1/chat/completions` | 25 MiB |
| `POST /v1/responses` | 25 MiB |
| `POST /v1/embeddings` | 8 MiB |

The local convenience route `POST /api/chat/completions` also has a 25 MiB
cap. The middleware validates a declared `Content-Length` and counts bytes from
chunked bodies, so omitting the header does not bypass the limit. An oversized
`/v1/*` request receives HTTP 413 with an OpenAI-shaped
`invalid_request_error`; these transport caps do not promise that Ollama will
accept every smaller payload or field shape.

## Stable aliases

Aliases are resolved in the LocalLLM gateway before forwarding:

```text
localllm-pocket      → qwen3:4b-q4_K_M
localllm-fast        → qwen3:8b-q4_K_M
localllm-balanced    → qwen3:8b-q8_0
localllm-deep        → qwen3:30b-a3b-instruct-2507-q4_K_M
localllm-max         → qwen3:30b-a3b-instruct-2507-q8_0
localllm-code        → qwen3-coder:30b-a3b-q4_K_M
localllm-vision      → qwen3-vl:8b-instruct-q4_K_M
localllm-vision-max  → qwen3-vl:8b-instruct-q8_0
localllm-vision-xl   → qwen3-vl:30b-a3b-instruct-q4_K_M
localllm-embed       → bge-m3:latest
```

## Important limitations

- Compatibility means common request/response shapes, not identical model behavior.
- Ollama supports a subset of OpenAI fields and tools. Unsupported fields may be ignored or rejected upstream.
- Ollama’s Responses API does not provide cloud conversation storage. `previous_response_id` and `conversation` are not local state stores.
- LocalLLM does not persist `/v1/*` request bodies or proxy responses. The
  Playground uses a separate local SQLite management API under
  `/api/conversations`; it does not turn OpenAI `conversation` or
  `previous_response_id` fields into stateful APIs. Vision Lab remains in browser
  memory. Deep Research reports and Binary Studio uploads use separate
  persistent management routes under `data/`; user-service output can also
  remain in systemd-journald according to the host retention policy.
- Hosted OpenAI tools such as web search, file search, computer use, image generation, and code interpreter do not appear merely because the endpoint is named `/v1/responses`.
- The app’s grounded Chat (`/api/agent/chat`), quick search (`/api/search`),
  and Deep Research routes are separate local orchestration APIs, not OpenAI
  hosted tools. Their contract is documented in
  [Search and Research API](search-research-api.md).
- The plan-preview/confirmed-Python routes under `/api/agent/*` and the optional
  Z-Image-Turbo routes under `/api/images/*` are likewise management APIs. They
  are not exposed as `/v1/responses` tools; clients must opt into their separate
  contracts and safety gates.
- The `/v1` proxy forwards supported image content parts to Ollama; it does not
  apply the grounded agent's strict data-URL, signature, dimension, and remote-URL
  checks. For a local-only vision request, use a bounded local image encoded as a
  data URL and do not supply a remote image URL. Upstream support or rejection of
  other image URL shapes is Ollama behavior.
- Context size is configured at model/runtime level; it is not inferred from an
  OpenAI request. The installed user service defaults direct `/v1` requests to
  a bounded 65,536-token Ollama context through
  `LOCALLLM_OLLAMA_CONTEXT_LENGTH`, which can be changed and rendered by
  rerunning `scripts/install-user-services.sh`. Ollama's native `options`
  object is not a supported field on this OpenAI lane and cannot override that
  context per request.
- Thinking-capable Qwen3 tags can spend much of a small completion-token budget
  on hidden reasoning before emitting visible text. A very low `max_tokens` can
  therefore produce an empty visible message even though inference occurred;
  omit the limit or leave enough headroom. The bounded node canary supplies the
  supported OpenAI `"reasoning_effort": "none"` control. Ollama's native
  `think` and `options` fields are not supported OpenAI fields here and must not
  be relied on.
- Ollama's native `keep_alive` field is not part of this OpenAI lane. The node
  canary performs its bounded cleanup separately against the literal-loopback
  native Ollama origin, using the exact model tag resolved during preflight and
  no API authorization header.

## Verify with the official Python SDK

After installing the core models and starting LocalLLM, run the contract probe:

```bash
uv run --project apps/api --extra dev python scripts/verify-openai-api.py
```

It exercises model listing, Chat Completions (including streaming), Responses, a forced Chat Completions function call, JSON mode, and 1024-dimensional BGE-M3 embeddings through the official `openai` package. Once a Qwen3-VL alias is installed, include a local image to exercise the multimodal request shape as well:

```bash
uv run --project apps/api --extra dev python scripts/verify-openai-api.py \
  --image /absolute/path/to/interface.png
```

Streaming requests are preflighted before response headers are sent, so an upstream Ollama 4xx/5xx remains an OpenAI-shaped HTTP error instead of becoming a misleading `200` event stream.

## Official contract sources

OpenAI recommends the Responses API for new projects while continuing to support Chat Completions. The official migration guide describes Responses as a unified, multimodal, agent-oriented interface: [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses).

The exact local subset is grounded in Ollama’s [OpenAI compatibility documentation](https://docs.ollama.com/api/openai-compatibility). The project installer pins Ollama v0.32.6 and the Linux amd64 archive SHA-256 `dec2fa50d24e6868ca3c4c977d69d059399372105f951a9acc320a5a79aadcfc`.
