# LocalLLM Studio

> A bright, private control room for local language, vision, deep research, and AI-assisted reverse engineering.

[![React](https://img.shields.io/badge/React-19-20201e?style=flat-square&logo=react)](apps/web)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-20201e?style=flat-square&logo=fastapi)](apps/api)
[![OpenAI-compatible](https://img.shields.io/badge/API-OpenAI--compatible-685bc7?style=flat-square)](references/openai-api-compatibility.md)
[![Local-first](https://img.shields.io/badge/privacy-local--first-2aa98e?style=flat-square)](#privacy-and-safety)

LocalLLM Studio turns a capable Linux workstation into a coherent local AI system. It pairs a polished responsive web app with Ollama, explicit Qwen model presets, a local OpenAI-compatible gateway, source-grounded web research, and a Ghidra/MCP reverse-engineering workspace.

The application is useful before every model is downloaded: system diagnostics, model pulls, API docs, and safe binary inspection remain available independently.

## What is inside

| Workspace | Purpose |
| --- | --- |
| Playground | Streaming local chat, code, tool-capable prompts, and optional image attachments |
| Vision Lab | OCR, screenshot review, diagram reading, and visual question answering |
| Deep Research | Multi-query web search, page extraction, cited synthesis, and uncertainty tracking |
| Model Shelf | Exact Q4/Q8 artifacts, download progress, installed state, and stable aliases |
| Binary Studio | Hashing, static string inspection, local AI triage, Ghidra, OGhidra, and PyGhidra-MCP status |
| API Desk | Copy-ready examples for Chat Completions, Responses, vision inputs, embeddings, and models |

## Quick start

Requirements: Linux x86-64, Node.js 20.19+ or 22.12+, [`uv`](https://docs.astral.sh/uv/), and an NVIDIA driver appropriate for the installed GPU.

```bash
git clone https://github.com/lachlanchen/LocalLLM.git
cd LocalLLM

scripts/bootstrap.sh
scripts/run.sh
```

Open <http://127.0.0.1:8008>. The service binds to loopback by default.

Pull the three practical daily models (about 30 GB):

```bash
scripts/pull-models.sh core
```

Pull every requested Q4 and Q8 comparison model (88 GB):

```bash
scripts/pull-models.sh all
```

## Curated dual-4090 model set

| Role | Artifact | Disk | Stable alias |
| --- | --- | ---: | --- |
| Tiny and fast | `qwen3:4b-q4_K_M` / `qwen3:4b-q8_0` | 2.6 / 4.4 GB | `localllm-pocket` |
| Daily assistant | `qwen3:8b-q4_K_M` / `qwen3:8b-q8_0` | 5.2 / 8.9 GB | `localllm-fast`, `localllm-balanced` |
| Research and RE | `qwen3:30b-a3b-instruct-2507-q4_K_M` / `...q8_0` | 19 / 32 GB | `localllm-deep`, `localllm-max` |
| Vision and OCR | `qwen3-vl:8b-instruct-q4_K_M` / `...q8_0` | 6.1 / 9.8 GB | `localllm-vision`, `localllm-vision-max` |

Q4_K_M is the normal lane: lower VRAM, faster startup, and more room for KV cache. Q8_0 is deliberately kept as a fidelity comparison. See [the model and hardware guide](references/model-selection.md) for context and realistic expectations.

## OpenAI-compatible API

The local base URL is `http://127.0.0.1:8008/v1`. Chat Completions, Responses, Models, and Embeddings are forwarded to the local Ollama runtime with LocalLLM aliases resolved at the gateway.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8008/v1",
    api_key="local-dev-key",
)

response = client.responses.create(
    model="localllm-deep",
    input="Compare three architectures for a private research agent.",
)
print(response.output_text)
```

OpenAI’s Responses API is broader than the locally implemented subset. The exact contract and limitations are documented in [OpenAI API compatibility](references/openai-api-compatibility.md).

## Reverse-engineering toolchain

Install the reproducible user-local stack without `sudo`:

```bash
scripts/setup-re-toolchain.sh
```

This installs and verifies:

- Ghidra 12.0.3 from the official NSA release, including checksum validation;
- the LLNL OGhidra source and its Ghidra extension;
- PyGhidra-MCP in a dedicated Python 3.12 environment;
- a 20-tool headless MCP surface for decompilation, symbols, cross-references, imports, exports, types, comments, and project operations.

Launch the GUI or a target-scoped headless MCP project with:

```bash
scripts/start-re-workbench.sh gui
LOCALLLM_RE_PROJECT_NAME=device scripts/start-re-workbench.sh mcp ./driver.sys ./vendor.dll
```

The intended evidence loop is:

```text
Windows binary / firmware
      → static metadata and Ghidra decompilation
      → local LLM hypotheses and protocol specification
      → packet captures / descriptors / cross-references
      → clean implementation in libusb, Rust, C, or a kernel module
      → hardware tests and review
```

Read [the complete operator workflow](references/reverse-engineering-workflow.md) before analyzing drivers or untrusted binaries.

## Development and validation

```bash
# Frontend development server (proxies /api to port 8008)
npm run dev

# Backend
uv run --project apps/api uvicorn localllm.main:app --app-dir apps/api --reload --port 8008

# Quality gates
npm run build
uv run --project apps/api ruff check apps/api/localllm apps/api/tests
uv run --project apps/api pytest -q
python3 scripts/browser-smoke.py
```

For a persistent local service, run `scripts/install-user-services.sh`. It installs two user-level systemd units; it does not require root.

## Privacy and safety

- App, Ollama, noVNC, and MCP examples bind to `127.0.0.1` by default.
- Model weights, prompts, uploads, research reports, runtime logs, and reverse-engineering projects live under ignored local directories.
- Binary Studio never executes an uploaded binary. It enforces a 64 MB limit and performs static inspection only.
- Binary strings and fetched webpages are treated as untrusted data, not model instructions.
- AI reverse-engineering conclusions are hypotheses until supported by cross-references, captures, tests, or hardware behavior.
- Web access is explicit and isolated to Deep Research. Chat and Vision do not silently search the internet.

## Documentation map

- [Reference index](references/README.md)
- [Model selection and dual-4090 layout](references/model-selection.md)
- [Deep Research design](references/deep-research.md)
- [Reverse-engineering workflow](references/reverse-engineering-workflow.md)
- [OpenAI-compatible API contract](references/openai-api-compatibility.md)
- [NVIDIA driver recovery](references/gpu-driver-recovery.md)
- [Primary-source ledger](references/sources.md)

## Citation

GitHub exposes **Cite this repository** from [`CITATION.cff`](CITATION.cff).

```bibtex
@software{localllm_studio_2026,
  author = {Chen, Lachlan},
  title = {LocalLLM Studio},
  year = {2026},
  url = {https://github.com/lachlanchen/LocalLLM}
}
```

## License

[MIT](LICENSE). Model weights and third-party tools keep their own licenses.
