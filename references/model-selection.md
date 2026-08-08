# Model selection for one or two RTX 4090 GPUs

## Decision

LocalLLM keeps four generation capability levels, a retrieval model, and two
quantization lanes for the generation models:

1. Qwen3 4B for small, low-latency experiments.
2. Qwen3 8B for daily chat, coding, and fast tool loops.
3. Qwen3 30B-A3B Instruct 2507 for deeper research and reverse engineering.
4. Qwen3-VL 8B Instruct for screenshots, OCR, diagrams, and image Q&A.
5. BGE-M3 for multilingual semantic search and the embeddings API.

Q4_K_M is the default. Q8_0 remains available in the comparison sets; it is not part of the default `core` pull and is not automatically better for every task.

## Curated Ollama tags

These are registry tags, not immutable content pins. A publisher can move any
tag, and `bge-m3:latest` is explicitly floating. Sizes and context values below
are catalog metadata observed on 2026-08-08. For evidence-grade repeatability,
record the resolved manifest digest after each pull and recheck it before use:

```bash
curl -fsS http://127.0.0.1:11434/api/tags | python3 -m json.tool
```

| Ollama tag | Size | Published context | Intended placement |
| --- | ---: | ---: | --- |
| `qwen3:4b-q4_K_M` | 2.6 GB | 40K | one GPU |
| `qwen3:4b-q8_0` | 4.4 GB | 40K | one GPU |
| `qwen3:8b-q4_K_M` | 5.2 GB | 40K | one GPU |
| `qwen3:8b-q8_0` | 8.9 GB | 40K | one GPU |
| `qwen3:30b-a3b-instruct-2507-q4_K_M` | 19 GB | 256K | one 4090 for moderate context; two when context/concurrency grows |
| `qwen3:30b-a3b-instruct-2507-q8_0` | 32 GB | 256K | two 4090s |
| `qwen3-vl:8b-instruct-q4_K_M` | 6.1 GB | 256K | one GPU |
| `qwen3-vl:8b-instruct-q8_0` | 9.8 GB | 256K | one GPU |
| `bge-m3:latest` | 1.2 GB | 8K | CPU or one GPU; 1024-dimensional embeddings |

The complete set is approximately 89.2 GB before filesystem overhead. The `core` set—8B Q4, 30B Q4, VL 8B Q4, and BGE-M3—is approximately 31.5 GB.

## Context is not free

Published maximum context is an architecture limit, not a recommendation to allocate it immediately. KV cache, vision tokens, batching, and runtime buffers consume additional VRAM. Begin at 16K–32K. Increase only for workloads that measurably benefit, while observing `ollama ps` and GPU memory.

## Dual-GPU reality

RTX 4090 cards do not have NVLink. Multi-GPU inference communicates over PCIe and primarily expands capacity; it does not promise a 2× token rate. Identical cards are appropriate for an even split, but topology, PCIe link width, CPU placement, and context size influence performance.

Pinned Ollama v0.32.6 first tries a single-GPU fit when possible.
`OLLAMA_SCHED_SPREAD=1` on the Ollama **server process** forces scheduling across
all visible GPUs; for a foreground run that starts its own Ollama process, use
`OLLAMA_SCHED_SPREAD=1 scripts/run.sh`. The generated Ollama systemd unit does
not read the project `.env`, so a persistent service needs a user-unit override.
Placement is not a performance guarantee: benchmark single- and two-card modes
on the actual PCIe topology.

## Why no 70B default

A dense 70B Q4 model can be made to fit near the total 48 GB budget, but leaves less headroom for cache and concurrency, crosses PCIe every generation step, and does not serve the fast daily/vision split as well. It remains an optional specialist rather than the base installation.

## Verifying placement

```bash
scripts/diagnose.sh
ollama ps
```

The app’s system panel uses NVML through `nvidia-smi`. Ollama also performs its own CUDA discovery; these can disagree during a driver/library mismatch. Repair the driver state before trusting benchmarks.

## Primary sources

- [Qwen3 4B GGUF model card](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [Qwen3 8B GGUF model card](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
- [Qwen3 30B-A3B Instruct 2507 model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [Qwen3-VL 8B Instruct GGUF model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- [Ollama Qwen3 tags](https://ollama.com/library/qwen3/tags)
- [Ollama Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags)
- [Ollama BGE-M3](https://ollama.com/library/bge-m3)
- [BAAI BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [Ollama v0.32.6 scheduler source](https://github.com/ollama/ollama/blob/v0.32.6/server/sched.go)
