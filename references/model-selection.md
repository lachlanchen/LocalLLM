# Model selection for one or two RTX 4090 GPUs

## Decision

LocalLLM keeps six generation capability lanes and one retrieval model:

1. Qwen3 4B for small, low-latency experiments.
2. Qwen3 8B for daily chat, coding, and fast tool loops.
3. Qwen3 30B-A3B Instruct 2507 for deeper research and reverse engineering.
4. Qwen3-Coder 30B-A3B for repository implementation, debugging, and coding tool loops.
5. Qwen3-VL 8B Instruct for fast screenshots, OCR, diagrams, and image Q&A.
6. Qwen3-VL 30B-A3B Instruct Q4 for the higher-capability Vision XL lane.
7. BGE-M3 for multilingual semantic search and the embeddings API.

Q4_K_M is the default across all six generation lanes. Q8_0 remains available
for the 4B, 8B, 30B-A3B text, and 8B vision comparison sets; Vision XL is
currently Q4-only. Q8 is not part of the default `core` pull and is not
automatically better for every task.

## Curated Ollama tags

These are registry tags, not immutable content pins. A publisher can move any
tag, and `bge-m3:latest` is explicitly floating. Sizes and context values below
are catalog metadata observed from 2026-08-08 through 2026-08-09. For
evidence-grade repeatability,
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
| `qwen3-coder:30b-a3b-q4_K_M` | 19 GB | 256K | one 4090 for moderate context; two when context/concurrency grows |
| `qwen3-vl:8b-instruct-q4_K_M` | 6.1 GB | 256K | one GPU |
| `qwen3-vl:8b-instruct-q8_0` | 9.8 GB | 256K | one GPU |
| `qwen3-vl:30b-a3b-instruct-q4_K_M` | 20 GB | 256K | one 4090 at modest context; two as cache/vision load grows |
| `bge-m3:latest` | 1.2 GB | 8K | CPU or one GPU; 1024-dimensional embeddings |

The complete `all` profile is eleven raw Ollama tags—six general text, one
coding specialist, three vision including the 30B-A3B Vision XL Q4 tag, and one
embedding model—and is approximately 128.2 GB before filesystem overhead. The
separate `code` profile pulls only the 19 GB coding specialist. The `core`
set—8B Q4, 30B Q4, VL 8B Q4, and BGE-M3—remains unchanged at approximately
31.5 GB; adding a specialist never silently expands that practical baseline.

## Context is not free

Published maximum context is an architecture limit, not a recommendation to allocate it immediately. KV cache, vision tokens, batching, and runtime buffers consume additional VRAM. The persistent service renders `LOCALLLM_OLLAMA_CONTEXT_LENGTH=65536` as the bounded default for direct Ollama/OpenAI-compatible requests; grounded app routes choose a model-aware value no larger than 65,536. Begin at 16K–32K for custom deployments and increase only for workloads that measurably benefit, while observing `ollama ps` and GPU memory. Rerun `scripts/install-user-services.sh` after changing the project setting.

The installed unit also sets `OLLAMA_NO_CLOUD=1`, admits at most two loaded models per GPU, allows one parallel sequence per runner, and caps the Ollama request queue at 32. These are service resource/privacy defaults, not model capability claims.

## Dual-GPU reality

RTX 4090 cards do not have NVLink. Multi-GPU inference communicates over PCIe and primarily expands capacity; it does not promise a 2× token rate. Identical cards are appropriate for an even split, but topology, PCIe link width, CPU placement, and context size influence performance.

Pinned Ollama v0.32.6 first tries a single-GPU fit when possible and distributes
a model that does not fit over its visible GPUs. `OLLAMA_SCHED_SPREAD=1` on the
Ollama **server process** forces scheduling across all visible GPUs. For the
persistent service, set `LOCALLLM_OLLAMA_SCHED_SPREAD=1` in `.env` and rerun
`scripts/install-user-services.sh`; the installer validates and maps that one
whitelisted value to `OLLAMA_SCHED_SPREAD` without exposing API/search secrets
to the Ollama process. A foreground process instead needs the native
`OLLAMA_SCHED_SPREAD=1` variable exported in its shell.

Ollama discovers accelerators when its server starts. On this dual-card host,
set `LOCALLLM_EXPECTED_GPU_COUNT=2` in `.env`, then rerun
`scripts/install-user-services.sh`. The generated user service waits until
`nvidia-smi` exposes both cards before starting a fresh Ollama process. The
installer then checks the pinned Ollama startup log for at least two
`inference compute` devices and fails visibly if runtime discovery still
disagrees. Keep the value at `0` on CPU-only or variable-GPU systems. Placement
is not a performance guarantee: benchmark single- and two-card modes on the
actual PCIe topology.

## Why no 70B default

A dense 70B Q4 model can be made to fit near the total 48 GB budget, but leaves less headroom for cache and concurrency, crosses PCIe every generation step, and does not serve the fast daily/vision split as well. It remains an optional specialist rather than the base installation.

## Verifying placement

```bash
scripts/diagnose.sh
ollama ps
journalctl --user --unit localllm-ollama.service --boot --no-pager \
  | rg 'inference compute'
nvidia-smi
```

Run the final two commands while the target model is loaded. `ollama ps` reports
aggregate CPU/GPU placement; `nvidia-smi` shows per-card memory and utilization,
while the startup journal records Ollama's inference-device inventory. The app’s
system panel also uses NVML through `nvidia-smi`. NVML and Ollama's own CUDA
discovery can disagree during a driver/library mismatch. Repair the driver state
before trusting benchmarks.

## Primary sources

- [Qwen3 4B GGUF model card](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [Qwen3 8B GGUF model card](https://huggingface.co/Qwen/Qwen3-8B-GGUF)
- [Qwen3 30B-A3B Instruct 2507 model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [Qwen3-Coder model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [Qwen3-VL 8B Instruct GGUF model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- [Qwen3-VL 30B-A3B Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
- [Ollama Qwen3 tags](https://ollama.com/library/qwen3/tags)
- [Ollama Qwen3-Coder tags](https://ollama.com/library/qwen3-coder/tags)
- [Ollama Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags)
- [Ollama BGE-M3](https://ollama.com/library/bge-m3)
- [BAAI BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [Ollama FAQ, including multi-GPU loading](https://docs.ollama.com/faq)
- [Ollama GPU discovery and selection](https://docs.ollama.com/gpu)
- [Ollama v0.32.6 scheduler source](https://github.com/ollama/ollama/blob/v0.32.6/server/sched.go)
