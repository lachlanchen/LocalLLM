# OpenAI-compatible local API

## Base URL and authentication

```text
http://127.0.0.1:8008/v1
Authorization: Bearer local-dev-key
```

`LOCALLLM_API_KEY` gates only `/v1/*`; setting it to an empty value disables
that check. The shipped `local-dev-key` is an interoperability placeholder, not
a secret. Management `/api/*` routes do not consult this key. Every route relies
on the loopback peer restriction; all browser requests to `/api/*` and `/v1/*`
additionally receive `Origin`/`Sec-Fetch-Site` checks. A
native process in the same host network namespace can invoke them. Do not proxy
or tunnel port 8008 without adding authentication and authorization.

## Endpoints

| Endpoint | Local behavior |
| --- | --- |
| `GET /v1/models` | returns installed Ollama models plus LocalLLM aliases whose targets are installed |
| `GET /v1/models/{id}` | forwards retrieval to Ollama after alias resolution |
| `POST /v1/chat/completions` | forwards text, image content parts, streaming, JSON mode, tools, and supported sampling fields |
| `POST /v1/responses` | forwards Ollama’s non-stateful Responses implementation, including streaming and functions |
| `POST /v1/embeddings` | forwards to an installed embedding-capable Ollama model |

## Stable aliases

Aliases are resolved in the LocalLLM gateway before forwarding:

```text
localllm-pocket      → qwen3:4b-q4_K_M
localllm-fast        → qwen3:8b-q4_K_M
localllm-balanced    → qwen3:8b-q8_0
localllm-deep        → qwen3:30b-a3b-instruct-2507-q4_K_M
localllm-max         → qwen3:30b-a3b-instruct-2507-q8_0
localllm-vision      → qwen3-vl:8b-instruct-q4_K_M
localllm-vision-max  → qwen3-vl:8b-instruct-q8_0
localllm-embed       → bge-m3:latest
```

## Important limitations

- Compatibility means common request/response shapes, not identical model behavior.
- Ollama supports a subset of OpenAI fields and tools. Unsupported fields may be ignored or rejected upstream.
- Ollama’s Responses API does not provide cloud conversation storage. `previous_response_id` and `conversation` are not local state stores.
- Hosted OpenAI tools such as web search, file search, computer use, image generation, and code interpreter do not appear merely because the endpoint is named `/v1/responses`.
- The app’s Deep Research pipeline is a separate local orchestration route, not an OpenAI hosted tool.
- Context size is configured at model/runtime level; it is not inferred from an OpenAI request.

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
