# Primary-source ledger

Accessed 2026-08-08 unless noted otherwise.

## Models and runtime

- Qwen official GGUF cards: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF), [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF).
- Ollama official catalogs: [Qwen3](https://ollama.com/library/qwen3/tags), [Qwen3-VL](https://ollama.com/library/qwen3-vl/tags).
- Embeddings: [Ollama BGE-M3](https://ollama.com/library/bge-m3), [BAAI BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).
- Ollama documentation: [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [Vision](https://docs.ollama.com/capabilities/vision), [API introduction](https://docs.ollama.com/api/introduction).
- Ollama source and releases: [ollama/ollama](https://github.com/ollama/ollama), [releases](https://github.com/ollama/ollama/releases).

## API shape

- OpenAI official documentation: [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses), [Create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

LocalLLM does not call hosted OpenAI models. These sources define the familiar client-facing shape used by the local gateway.

## Reverse engineering

- Ghidra official source: [NationalSecurityAgency/ghidra](https://github.com/NationalSecurityAgency/ghidra).
- Pinned release: [Ghidra 12.0.3](https://github.com/NationalSecurityAgency/ghidra/releases/tag/Ghidra_12.0.3_build), asset `ghidra_12.0.3_PUBLIC_20260210.zip`, SHA-256 `90d3fffb20b00030dcef8d2a24dd0f422d3a61e432b3ad43f77233ac6d667981`.
- LLNL agent integration: [LLNL/OGhidra](https://github.com/LLNL/OGhidra).
- Headless MCP integration: [clearbluejar/pyghidra-mcp](https://github.com/clearbluejar/pyghidra-mcp).
- Ghidra plugin bridge: [LaurieWired/GhidraMCP](https://github.com/LaurieWired/GhidraMCP).

## Interpretation policy

Sizes and context limits in the app are artifact metadata, not benchmark guarantees. Performance statements in this repository are framed as deployment guidance and should be measured on the actual topology after the NVIDIA driver state is healthy.
