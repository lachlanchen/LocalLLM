# LocalLLM Studio

> A bright, private control room for local language, vision, deep research, and AI-assisted reverse engineering.

[![React](https://img.shields.io/badge/React-19-20201e?style=flat-square&logo=react)](apps/web)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-20201e?style=flat-square&logo=fastapi)](apps/api)
[![CI](https://github.com/lachlanchen/LocalLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/lachlanchen/LocalLLM/actions/workflows/ci.yml)
[![OpenAI-compatible](https://img.shields.io/badge/API-OpenAI--compatible-685bc7?style=flat-square)](references/openai-api-compatibility.md)
[![Local-first](https://img.shields.io/badge/privacy-local--first-2aa98e?style=flat-square)](#privacy-and-safety)

LocalLLM Studio turns a capable Linux workstation into a coherent local AI system. It pairs a polished responsive web app with Ollama, explicit Qwen model presets, a local OpenAI-compatible gateway, a bounded cited web-research pipeline, and a Ghidra/MCP reverse-engineering workspace.

The application is useful before every model is downloaded: system diagnostics, model pulls, the built-in API Desk, and bounded static binary inspection remain available independently.

## What is inside

| Workspace | Purpose |
| --- | --- |
| Playground | Streaming local chat, code help, and optional image attachments |
| Vision Lab | OCR, screenshot review, diagram reading, and visual question answering |
| Deep Research | Multi-query web search, page extraction, cited synthesis, and uncertainty tracking |
| Model Shelf | Curated Q4/Q8 tags, download progress, installed state, and stable aliases |
| Binary Studio | Static upload triage plus a read-only Ghidra/MCP investigator with mutation tools blocked |
| API Desk | Copy-ready examples for Chat Completions, Responses, vision inputs, embeddings, and models |

## Quick start

Requirements: Linux x86-64, Node.js 20.19+ or 22.12+, npm, [`uv`](https://docs.astral.sh/uv/), Git, curl, tar, zstd, `sha256sum`, and an NVIDIA driver appropriate for the installed GPU. Binary Studio's host-side static triage uses `file` and `strings`. The reverse-engineering installer additionally preflights Java/Javac 21 and unzip; packet tooling uses Docker without requiring host-level packet packages.

```bash
git clone https://github.com/lachlanchen/LocalLLM.git
cd LocalLLM

scripts/bootstrap.sh
scripts/run.sh
```

Open <http://127.0.0.1:8008>. The default launcher uses this IPv4 loopback endpoint; the app rejects non-loopback peers.

In a second terminal, pull the three practical daily models plus multilingual embeddings (about 32 GB):

```bash
scripts/pull-models.sh core
```

Pull every requested Q4/Q8 comparison model plus embeddings (about 89 GB):

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
| Retrieval and RAG | `bge-m3:latest` | 1.2 GB | `localllm-embed` |

Q4_K_M is the normal lane: lower VRAM, faster startup, and more room for KV cache. Q8_0 remains available as a fidelity comparison. See [the model and hardware guide](references/model-selection.md) for context, resolved-digest guidance, and realistic expectations.

These are named Ollama registry tags, not immutable weight pins; `bge-m3:latest` is explicitly mutable. Ollama stores verified content-addressed blobs after a pull, while a future registry pull may resolve a tag to newer content.

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

Install the revision-pinned user-local stack without `sudo`:

```bash
scripts/setup-re-toolchain.sh
```

This installs and verifies:

- Ghidra 12.0.3 from the official NSA release, including checksum validation;
- LLNL OGhidra pinned at `93a4380fc748a393690be9bfd2c2156fade82757`,
  with a repository-tracked loopback/browser-boundary patch applied before its Ghidra
  extension is built;
- PyGhidra-MCP pinned at `f29063b8636100b71e9c3aec61fe056827c556e4`
  in a dedicated Python 3.12 environment;
- a 20-tool headless MCP surface for decompilation, symbols, cross-references, imports, exports, types, comments, and project operations.

Launch the GUI or a target-scoped headless MCP project with:

```bash
scripts/start-re-workbench.sh gui
LOCALLLM_RE_PROJECT_NAME=device scripts/start-re-workbench.sh mcp ./driver.sys ./vendor.dll
```

Build the restricted offline packet-analysis lane and inspect a saved USBPcap
or usbmon capture:

```bash
scripts/setup-usb-evidence-tools.sh
scripts/analyze-usb-pcap.sh ./device-session.pcapng
```

The wrapper builds from a digest-pinned official Ubuntu base with TShark,
usbutils, and libusb headers. At runtime it has no network or Linux capabilities;
the selected capture is its only host evidence bind and is read-only. Live
usbmon capture remains an explicit operator-authorized host action; see
[USB packet evidence](references/usb-evidence-tooling.md).

The OGhidra and PyGhidra-MCP listeners are restricted to `127.0.0.1`, but they
do not implement API-key authentication. Any local process can invoke their
read and mutation operations. Do not expose or tunnel these ports; see the
[documented trust boundary](references/reverse-engineering-workflow.md#local-bridge-trust-boundary).

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
# Frontend development server (proxies /api, /v1, and /healthz to port 8008)
npm run dev

# Backend
uv run --project apps/api uvicorn localllm.main:app --app-dir apps/api --reload --port 8008

# Quality gates
npm test
npm run lint
npm run build
uv run --project apps/api ruff check apps/api/localllm apps/api/tests
uv run --project apps/api pytest -q
uv run --project apps/api --extra dev python scripts/verify-openai-api.py
scripts/verify-re-toolchain.sh
```

The optional visual gate requires a running app plus Google Chrome, Xvfb,
x11vnc, noVNC/websockify, xdotool, and `ss`; those host programs are not
installed by `scripts/bootstrap.sh`. The Python Playwright package is included
in the API development environment. On a prepared workstation, start the
persistent loopback desktop and run:

```bash
scripts/launch-novnc.sh start
uv run --project apps/api --extra dev python scripts/browser-smoke.py
scripts/launch-novnc.sh stop
```

For a persistent local service, run `scripts/install-user-services.sh`. It installs two user-level systemd units; it does not require root.

## Privacy and safety

- App, Ollama, noVNC, and MCP examples bind to loopback; the app also rejects non-loopback peers if it is accidentally started on a broader socket.
- `LOCALLLM_API_KEY` gates only `/v1/*`. The management `/api/*` routes rely on
  the loopback peer restriction; all browser requests to `/api/*` and `/v1/*`
  also receive origin/fetch-site checks. Native same-host processes can invoke
  them. The
  shipped `local-dev-key` is a development placeholder, not a secret; do not
  proxy or tunnel port 8008 without adding authentication and authorization.
- The optional browser harness is a same-host test fixture, not an authentication boundary: Xvfb disables X access control, x11vnc is passwordless, and Chrome DevTools is unauthenticated. Never forward, proxy, or tunnel ports `5930`, `6130`, or `9470`, and stop the harness after use.
- Browser API requests enforce an origin/fetch-site boundary, HTTP hosts are allowlisted, and the app emits a restrictive content-security policy plus anti-framing headers.
- Model weights, prompts, uploads, research reports, runtime logs, and reverse-engineering projects live under ignored local directories.
- Binary Studio never executes an uploaded binary. It enforces a 64 MB limit, bounds subprocess output and concurrency, stores artifacts privately, and provides an explicit local delete control.
- The web app exposes only 12 read-only PyGhidra-MCP tools to its investigator and blocks all eight discovered project/symbol mutation tools.
- Binary strings and fetched webpages are treated as untrusted data, not model instructions.
- AI reverse-engineering conclusions are hypotheses until supported by cross-references, captures, tests, or hardware behavior.
- Web access is explicit and isolated to Deep Research. Chat and Vision do not silently search the internet.

## Documentation map

- [Reference index](references/README.md)
- [Model selection and dual-4090 layout](references/model-selection.md)
- [Pinned llama.cpp CUDA alternative](references/llama-cpp.md)
- [Deep Research design](references/deep-research.md)
- [Reverse-engineering workflow](references/reverse-engineering-workflow.md)
- [USB packet-evidence tooling](references/usb-evidence-tooling.md)
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
