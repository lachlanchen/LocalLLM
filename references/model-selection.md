# Model selection for one or two RTX 4090 GPUs

## Decision

LocalLLM keeps four capability levels and two quantization lanes:

1. Qwen3 4B for small, low-latency experiments.
2. Qwen3 8B for daily chat, coding, and fast tool loops.
3. Qwen3 30B-A3B Instruct 2507 for deeper research and reverse engineering.
4. Qwen3-VL 8B Instruct for screenshots, OCR, diagrams, and image Q&A.

Q4_K_M is the default. Q8_0 is installed for controlled fidelity comparisons, not because it is automatically better for every task.

## Exact artifacts

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

The complete requested set is approximately 88 GB before filesystem overhead. The `core` set—8B Q4, 30B Q4, and VL 8B Q4—is approximately 30 GB.

## Context is not free

Published maximum context is an architecture limit, not a recommendation to allocate it immediately. KV cache, vision tokens, batching, and runtime buffers consume additional VRAM. Begin at 16K–32K. Increase only for workloads that measurably benefit, while observing `ollama ps` and GPU memory.

## Dual-GPU reality

RTX 4090 cards do not have NVLink. Multi-GPU inference communicates over PCIe and primarily expands capacity; it does not promise a 2× token rate. Identical cards are appropriate for an even split, but topology, PCIe link width, CPU placement, and context size influence performance.

Ollama selects GPU placement automatically. `OLLAMA_SCHED_SPREAD=1` can force spreading, but it is not enabled by default because small models are usually faster and more useful when contained on one card.

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
- [Qwen3 30B-A3B GGUF model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF)
- [Ollama Qwen3 tags](https://ollama.com/library/qwen3/tags)
- [Ollama Qwen3-VL tags](https://ollama.com/library/qwen3-vl/tags)

