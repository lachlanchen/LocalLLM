# llama.cpp CUDA runtime

LocalLLM uses Ollama by default and also includes a source-revision-pinned, project-local `llama.cpp` build recipe for direct GGUF experiments. The two runtimes are independent: installing or starting `llama.cpp` does not replace Ollama or modify its model store. Compiler, CMake, CUDA, and host-library versions are not pinned, so this is not a claim of bit-for-bit reproducible binaries.

## Pinned source revision

The installer checks out the official `ggml-org/llama.cpp` release `b10327`, commit `69bf6437914596fbbc4caf09a7ac16f2acdd1a94`, published on 2026-08-08. By default it builds CUDA code for compute capability `8.9`, the NVIDIA Ada architecture used by RTX 4090 cards.

Source, build files, and installed binaries stay under ignored project-local directories:

```text
.local/tools/llama.cpp/
.local/build/llama.cpp-b10327-cuda-sm89/
.local/opt/llama.cpp-b10327/bin/
```

## Build

Prerequisites are Git, a C++ compiler, CMake, Ninja, and a CUDA toolkit containing `nvcc`. No root-owned install path is used.

```bash
scripts/setup-llama-cpp.sh
```

The default parallelism is deliberately capped at six jobs so the CUDA build can coexist with model downloads. Override it only when enough RAM is free:

```bash
LOCALLLM_LLAMA_CPP_JOBS=12 scripts/setup-llama-cpp.sh
```

The default CUDA architecture can be overridden for a different NVIDIA GPU:

```bash
LOCALLLM_LLAMA_CPP_CUDA_ARCH=86 scripts/setup-llama-cpp.sh
```

## Run a loopback OpenAI-compatible server

The launcher expects a standalone GGUF file. It does not resolve Ollama tags, and relying on Ollama's internal content-addressed blob paths would couple the two runtimes. Supply a readable GGUF file:

```bash
scripts/start-llama-server.sh /absolute/path/to/model.Q4_K_M.gguf
```

The CUDA build omits the bundled upstream UI, HTTPS, and model-download client because this alternative serves local GGUF files on loopback. The launcher always binds to `127.0.0.1`, defaults to port `8010`, disables the mutable slot endpoint, restricts browser CORS to loopback origins, and does not enable built-in shell/file tools. Its OpenAI-style base URL is:

```text
http://127.0.0.1:8010/v1
```

Supported upstream routes include `/v1/models`, `/v1/chat/completions`, `/v1/responses`, and `/v1/embeddings`. This is common-shape compatibility, not full OpenAI parity: upstream translates Responses requests into Chat Completions, and embeddings require a model whose pooling type is not `none`. A minimal check after loading a model is:

```bash
curl http://127.0.0.1:8010/v1/models
```

Useful settings are environment variables rather than unrestricted command-line
pass-through, preserving the launcher's fixed option surface:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOCALLLM_LLAMA_CPP_PORT` | `8010` | local port |
| `LOCALLLM_LLAMA_CPP_CONTEXT` | `16384` | allocated context tokens |
| `LOCALLLM_LLAMA_CPP_GPU_LAYERS` | `all` | CUDA-offloaded layers |
| `LOCALLLM_LLAMA_CPP_SPLIT_MODE` | `layer` | multi-GPU placement |
| `LOCALLLM_LLAMA_CPP_TENSOR_SPLIT` | automatic | optional per-GPU ratio such as `1,1` |
| `LOCALLLM_LLAMA_CPP_FLASH_ATTN` | `auto` | Flash Attention selection |
| `LOCALLLM_LLAMA_CPP_ALIAS` | GGUF filename | model ID returned by the API |
| `LOCALLLM_LLAMA_CPP_MMPROJ` | unset | projector GGUF for a compatible vision model |
| `LOCALLLM_LLAMA_CPP_API_KEY` | unset | optional API bearer key, forwarded through the upstream environment variable rather than a process argument |

No API key is configured by default. Loopback binding and CORS do not isolate
other same-host processes. Set `LOCALLLM_LLAMA_CPP_API_KEY` when bearer
authentication is needed, and do not proxy or tunnel port 8010 without a
separate authenticated access-control layer.

`row` split mode is deprecated upstream. `tensor` is experimental, requires
Flash Attention, disables auto-fit, and is not implemented for every model
architecture. Keep the default `layer` mode unless a measured workload justifies
those constraints.

With both cards visible, default `layer` mode uses pipeline parallelism across
all visible GPUs; when `LOCALLLM_LLAMA_CPP_TENSOR_SPLIT` is unset, layer
placement is proportional to available memory. To confine a fitting model to
one card:

```bash
CUDA_VISIBLE_DEVICES=0 \
scripts/start-llama-server.sh /absolute/path/to/model.gguf
```

For a model that requires both cards, expose both and omit the tensor split for
automatic placement. Set an explicit `1,1` only when an equal split is
intentional:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LOCALLLM_LLAMA_CPP_TENSOR_SPLIT=1,1 \
scripts/start-llama-server.sh /absolute/path/to/model.gguf
```

Consumer RTX 4090 cards have no NVLink, so a two-card split primarily increases capacity rather than doubling generation speed. Start at 16K context and measure VRAM before increasing it.

The CUDA configure step probes for NCCL. Default `layer` splitting does not use
NCCL. In experimental `tensor` mode, a build with NCCL uses it for cross-GPU
reductions; without NCCL the build succeeds and falls back to an internal
AllReduce path with a lower-performance warning. NCCL remains an optional
operator-installed dependency.

For vision, use a `llama.cpp`-compatible model GGUF plus its matching multimodal projector. A projector from another model or revision is not interchangeable.

## Verification and limits

The installer verifies the exact Git commit, builds `llama-server`, `llama-cli`, and `llama-bench`, installs them locally, and checks that `llama-server --version` reports the pinned revision. Help/version checks do not load a model and can be run while Ollama is active:

```bash
.local/opt/llama.cpp-b10327/bin/llama-server --version
.local/opt/llama.cpp-b10327/bin/llama-server --help
```

A successful default `sm89` CUDA build proves the runtime contains RTX 4090
kernels; a build made with another architecture override does not. Neither case
proves GPU execution. Load a GGUF model and inspect startup logs only after
`nvidia-smi` works and both GPUs are visible; a driver/library mismatch must be
repaired first.

## Primary sources

- [Official `b10327` release](https://github.com/ggml-org/llama.cpp/releases/tag/b10327)
- [Official CUDA build documentation at the pinned revision](https://github.com/ggml-org/llama.cpp/blob/69bf6437914596fbbc4caf09a7ac16f2acdd1a94/docs/build.md)
- [Official multi-GPU documentation at the pinned revision](https://github.com/ggml-org/llama.cpp/blob/69bf6437914596fbbc4caf09a7ac16f2acdd1a94/docs/multi-gpu.md)
- [Official server documentation at the pinned revision](https://github.com/ggml-org/llama.cpp/blob/69bf6437914596fbbc4caf09a7ac16f2acdd1a94/tools/server/README.md)
