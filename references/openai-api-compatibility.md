# OpenAI-compatible local API

## Base URL and authentication

```text
http://127.0.0.1:8008/v1
Authorization: Bearer local-dev-key
```

Set `LOCALLLM_API_KEY` to replace the default. The management UI uses same-origin `/api/*` routes. The OpenAI-shaped `/v1/*` routes require the configured key.

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

## Official contract sources

OpenAI recommends the Responses API for new projects while continuing to support Chat Completions. The official migration guide describes Responses as a unified, multimodal, agent-oriented interface: [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses).

The exact local subset is grounded in Ollama’s [OpenAI compatibility documentation](https://docs.ollama.com/api/openai-compatibility). Ollama added `/v1/responses` in v0.13.3; this installation uses a newer project-local Ollama release.
