# Primary-source ledger

Accessed 2026-08-08 unless noted otherwise.

## Models and runtime

- Qwen cards matching the selected families/revisions: [Qwen3-4B GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF), [Qwen3-8B GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF), [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), and [Qwen3-VL-8B-Instruct GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF).
- Ollama official catalogs for the selected GGUF tags: [Qwen3](https://ollama.com/library/qwen3/tags), [Qwen3-VL](https://ollama.com/library/qwen3-vl/tags).
- Embeddings: [Ollama BGE-M3](https://ollama.com/library/bge-m3), [BAAI BGE-M3 model card](https://huggingface.co/BAAI/bge-m3).
- Ollama documentation: [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [Vision](https://docs.ollama.com/capabilities/vision), [API introduction](https://docs.ollama.com/api/introduction).
- Runtime pin: [Ollama v0.32.6](https://github.com/ollama/ollama/releases/tag/v0.32.6), Linux amd64 archive SHA-256 `dec2fa50d24e6868ca3c4c977d69d059399372105f951a9acc320a5a79aadcfc`; [scheduler source at that tag](https://github.com/ollama/ollama/blob/v0.32.6/server/sched.go).
- Alternative runtime pin: [llama.cpp b10327](https://github.com/ggml-org/llama.cpp/releases/tag/b10327), commit [`69bf6437914596fbbc4caf09a7ac16f2acdd1a94`](https://github.com/ggml-org/llama.cpp/commit/69bf6437914596fbbc4caf09a7ac16f2acdd1a94); see its pinned [multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/69bf6437914596fbbc4caf09a7ac16f2acdd1a94/docs/multi-gpu.md).

## API shape

- OpenAI official documentation: [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses), [Create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

LocalLLM does not call hosted OpenAI models. These sources define the familiar client-facing shape used by the local gateway.

## Reverse engineering

- Ghidra official source: [NationalSecurityAgency/ghidra](https://github.com/NationalSecurityAgency/ghidra).
- Pinned release: [Ghidra 12.0.3](https://github.com/NationalSecurityAgency/ghidra/releases/tag/Ghidra_12.0.3_build), asset `ghidra_12.0.3_PUBLIC_20260210.zip`, SHA-256 `90d3fffb20b00030dcef8d2a24dd0f422d3a61e432b3ad43f77233ac6d667981`.
- LLNL agent integration: [LLNL/OGhidra at `93a4380fc748a393690be9bfd2c2156fade82757`](https://github.com/LLNL/OGhidra/tree/93a4380fc748a393690be9bfd2c2156fade82757), plus the repository-tracked local security patch.
- Headless MCP integration: [clearbluejar/pyghidra-mcp at `f29063b8636100b71e9c3aec61fe056827c556e4`](https://github.com/clearbluejar/pyghidra-mcp/tree/f29063b8636100b71e9c3aec61fe056827c556e4), package version 0.2.5.
- Ghidra plugin bridge: [LaurieWired/GhidraMCP](https://github.com/LaurieWired/GhidraMCP).

## Hardware and packet evidence

- NVIDIA hardware characteristics: [GeForce RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).
- Container base: [Docker Official Image for Ubuntu](https://hub.docker.com/_/ubuntu), exact `linux/amd64` base `ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea`.
- Packet tools: [TShark manual](https://www.wireshark.org/docs/man-pages/tshark.html) and [Wireshark USB capture setup](https://wiki.wireshark.org/CaptureSetup/USB). The audited image used TShark/Wireshark `4.2.2-1.1build3`, usbutils `1:017-3build1`, libusb runtime/development `2:1.0.27-1`, and pkg-config `1.8.1-2build1`.

## Interpretation policy

Sizes and context limits in the app are registry/catalog metadata, not benchmark guarantees. Ollama model tags are mutable; record resolved manifest digests when exact provenance matters. Performance statements in this repository are deployment guidance and should be measured on the actual topology after the NVIDIA driver state is healthy.
