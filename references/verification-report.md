# LocalLLM verification report

The baseline below was verified on 2026-08-09 before the NVIDIA reboot and
before Vision XL expanded the curated catalog. Its observed measurements are
preserved as historical evidence, not silently rewritten as current results.
This report is not a claim of cloud API parity or theoretical model capability.

> **Historical pre-reboot condition:** every baseline GPU result below was collected before a
> reboot while Ollama could use only one GPU. The loaded kernel module is
> `595.71.05`, while the installed module and NVML library are `595.84`;
> `nvidia-smi` therefore fails with a driver/library mismatch. PCI enumeration
> shows two RTX 4090 D cards, but those conditions did not establish healthy
> dual-GPU inference. The original handoff therefore required a reboot before
> any result could be treated as a dual-GPU measurement.

The repository now declares a tenth underlying tag, Qwen3-VL 30B-A3B Q4, and a
ninth stable alias. The dated addendum at the end records the separate
post-reboot inventory and functional checks. Those observations do not rewrite
or retroactively upgrade any baseline measurement below.

## Source and quality gates

At this baseline snapshot, the checked application source was clean and
synchronized with `origin/main`.
Local gates passed:

- 69 backend tests plus Ruff;
- 12 frontend tests, TypeScript lint, and the Vite production build;
- shell syntax, Python script compilation, and `git diff --check`;
- relative documentation-link and metadata parsing checks.

The SHA-pinned GitHub Actions workflow independently passed its web, API, and
script jobs. The final report commit and run are recorded in the release
handoff rather than embedded here, so publishing this document does not create
a self-referential commit ID.

## Resolved nine-model baseline manifest

Ollama tags are mutable registry names. These full manifest digests and byte
sizes were observed locally on 2026-08-09 and should be recorded again after a
future pull.

| Ollama tag | Resolved manifest digest | Bytes | Observed result |
| --- | --- | ---: | --- |
| `qwen3:4b-q4_K_M` | `2bfd38a7daaf4b1037efe517ccb73d1a3bbd4822cf89f1a82be1569050a114e0` | 2,620,788,260 | 197.2 eval tok/s; 2.714 s load |
| `qwen3:4b-q8_0` | `6461746fd6b5a2327ba63d5cd1359af119852d82aa8c981efe948d1868a4dc20` | 4,368,891,938 | 149.9 eval tok/s; 5.553 s load |
| `qwen3:8b-q4_K_M` | `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` | 5,225,388,164 | 139.9 eval tok/s; 5.399 s load |
| `qwen3:8b-q8_0` | `e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130` | 8,851,089,538 | 93.6 eval tok/s; 4.365 s load; 100% GPU |
| `qwen3:30b-a3b-instruct-2507-q4_K_M` | `19e422b0231392335cfc49cfd172de7034bb1aeabb08aa307cce745c60b272fe` | 18,556,699,186 | 220.0 eval tok/s; 18.440 s load; 100% GPU |
| `qwen3:30b-a3b-instruct-2507-q8_0` | `528dfe43328ba6235a38e89f6e8ead082a70eda6b45bcae9bbdeba0f38ac3f9b` | 32,483,945,008 | 18.5 eval tok/s; 26.803 s load; 34%/66% CPU/GPU |
| `qwen3-vl:8b-instruct-q4_K_M` | `0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319` | 6,140,415,975 | exact OCR result; 133.2 eval tok/s |
| `qwen3-vl:8b-instruct-q8_0` | `eff3eb825b322d4ffb85695e5a15cfe00b4d994d3c336f13dc867d8743b00245` | 9,830,285,285 | exact OCR result; 84.0 eval tok/s; 100% GPU |
| `bge-m3:latest` | `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` | 1,157,672,605 | three 1,024-dimensional vectors verified |

At the time of this baseline snapshot, the catalog, pull script, and model guide
agreed on this six-text, two-vision, one-embedding set.

## Provisional single-visible-GPU measurements

These are single observations collected before the required reboot. They are
neither dual-GPU results nor general performance guarantees.

Text measurements used Ollama's native `/api/chat` with `think: false`,
temperature `0`, `num_ctx: 4096`, and `num_predict: 192`. The fixed prompt
requested integers 1 through 64 separated by commas. Each model produced 183
evaluation tokens and was unloaded afterward.

The 30B Q8 CPU/GPU split in the manifest table was observed after its
`localllm-max` alias smoke at the service default 32,768-token context; its
throughput and load figures came from the fixed 4,096-token benchmark above.

Vision measurements used the same 1440×913 LocalLLM interface screenshot and
an exact-headline prompt with 1,326 prompt tokens and seven evaluation tokens.
Q4 recorded 12.968 s wall time and 4.105 s load; Q8 recorded 6.265 s wall time
and 5.417 s load. Cache and warm state differed, so those wall times are not
comparable cold-start benchmarks. Both returned `Think locally. Build freely.`
exactly.

## OpenAI SDK, vision, and embeddings

The official `openai` Python package version `1.109.1` exercised the local
gateway at `http://127.0.0.1:8008/v1`:

- model listing and retrieval;
- Chat Completions and streaming;
- Responses;
- a forced function call with the expected arguments;
- JSON mode;
- three BGE-M3 embeddings of 1,024 dimensions;
- a data-URL image through `localllm-vision`.

The complete probe passed. The vision response also ran through the actual
browser Vision Lab, where it read the headline and all six sidebar workspace
names. BGE-M3 scored the two local-AI phrases at 0.7792 cosine similarity and
the unrelated fruit phrase at 0.3465. These are smoke observations, not an
embedding benchmark.

After all nine baseline tags were installed, that snapshot's final SDK rerun
returned 17 model IDs: nine raw Ollama tags and eight stable LocalLLM aliases.

Thinking-capable Qwen3 tags may consume a low completion-token limit with
hidden reasoning before visible text appears. The compatibility guide records
that behavior and the native no-thinking benchmark path.

## Reverse engineering and packet evidence

- **Ghidra:** release 12.0.3 passed the official archive SHA-256 check.
- **OGhidra:** pinned commit
  `93a4380fc748a393690be9bfd2c2156fade82757` was built with the tracked
  loopback/browser-boundary patch.
- **PyGhidra-MCP:** pinned commit
  `f29063b8636100b71e9c3aec61fe056827c556e4`, package version 0.2.5. The
  isolated `/bin/true` lane exposed exactly 20 expected tools and returned ten
  read-only symbol hits. Binary Studio exposes 12 read-only tools and blocks
  all eight discovered mutation tools.
- **USB evidence:** image
  `sha256:0f2a9441dbd18c733a92cca239a026c03b6cdbe63bc1fcff02c17cc655680ee9`
  uses the digest-pinned Ubuntu base. TShark 4.2.2, usbutils 017, and libusb
  1.0.27 passed. Its 144-package manifest SHA-256 is
  `b1b27d91e3a8e99c3d141508ce681da99c148238d0da44e033ecf3de8334004f`;
  the synthetic USBPcap frame parsed successfully.

Both MCP bridges are loopback-only but unauthenticated. The web app cannot call
their mutation tools, but another same-host process can; do not expose or
tunnel their ports.

## Alternative llama.cpp runtime

llama.cpp release `b10327`, commit
`69bf6437914596fbbc4caf09a7ac16f2acdd1a94`, was built locally for the
default `sm_89` target. A Qwen3 4B Q4 GGUF returned exact text through its
OpenAI-style endpoint at 197.71 eval tok/s over nine evaluation tokens. This
was also a one-visible-GPU check. NCCL was not available in this build; normal
layer splitting remains available, while experimental tensor mode uses an
internal AllReduce fallback.

## Browser, binding, and service handoff

Baseline browser validation passed with nine model cards, three API code cards, no
console errors, and a 390 px mobile document width matching the 390 px
viewport. Live Vision Lab and Binary Studio results were inspected and saved
under the ignored local evidence directory.

The active app, Ollama, PyGhidra-MCP, VNC, noVNC, and Chrome CDP listeners were
all observed on loopback. The browser harness is not an authentication
boundary and is stopped after the final service check.

At the baseline snapshot, the installer enabled `localllm-ollama.service` and
`localllm-api.service` in the user manager. Both units passed bounded readiness
checks, and a deliberate restart changed both main PIDs while returning Ollama
0.32.6 and a healthy API.
This host's user manager cannot apply `ProtectKernelModules`; the templates
therefore omit that redundant user-service directive while retaining
`NoNewPrivileges`, `PrivateTmp`, control-group and kernel-tunable protection,
and realtime/SUID restrictions.

## Post-reboot and Vision XL addendum — 2026-08-09

These are separate post-reboot observations. They establish inventory,
placement, and narrow functional smoke behavior; they are not throughput
benchmarks and do not alter the historical figures above.

### GPU and expanded API inventory

- `nvidia-smi` returned two NVIDIA GeForce RTX 4090 D cards, each with
  24,564 MiB total memory.
- Ollama's startup inventory reported both CUDA devices at PCI IDs
  `0000:01:00.0` and `0000:09:00.0`.
- The installed catalog contained ten raw Ollama tags and nine stable aliases.
  `GET /v1/models` consequently returned 19 IDs, and the curated catalog total
  was 109.2 GB before filesystem overhead.

### Qwen3 30B-A3B text Q8 across both cards

The tested tag was `qwen3:30b-a3b-instruct-2507-q8_0`, resolved digest
`528dfe43328ba6235a38e89f6e8ead082a70eda6b45bcae9bbdeba0f38ac3f9b`.
While it was loaded at context 4,096, Ollama `/api/ps` reported
`size_vram` 32,793,375,538 and 100% GPU placement. The Ollama process occupied
16,526 MiB on GPU 0 and 15,562 MiB on GPU 1. The controlled no-thinking smoke
request returned the exact visible text `DUAL-GPU-READY`.

### Qwen3-VL 30B-A3B Q4 modest-context placement

The Vision XL tag `qwen3-vl:30b-a3b-instruct-q4_K_M` resolved to digest
`c871fc73fabc5516500b70a298ea25fd44a6a23d5cffc46c63b50302543e3915`.
While loaded, `/api/ps` reported `size_vram` 19,256,175,491 and 100% GPU
placement on GPU 1; `nvidia-smi` showed approximately 20,246 MiB of process
memory there. In the screenshot smoke response, the model identified the page
as Playground, reproduced the visible headline, described the three workspace
cards, and identified the model selector.

### Final self-contained service checks

The installed user units passed their bounded restart and readiness checks.
The Ollama unit recorded both distinct CUDA PCI IDs and rendered these active
controls: `OLLAMA_CONTEXT_LENGTH=65536`, `OLLAMA_NO_CLOUD=1`, two loaded models
at most, a 32-request queue, and one parallel request per model. Both units are
enabled, and user lingering is enabled so they can run without an interactive
shell.

At the 65,536-token service context, Vision XL reported `size_vram`
28,248,742,296 and 100 percent GPU placement. Its runner occupied 15,860 MiB on
GPU 0 and 13,344 MiB on GPU 1, establishing automatic two-card placement for
this larger context. Through the integrated `/api/agent/chat` image path it
read the current screenshot as `CONTROL ROOM`, reproduced `Think locally.
Build freely.`, and counted three large suggestion cards.

The official OpenAI Python SDK 1.109.1 passed all final probes with 19 model
IDs: non-streaming and streaming Chat Completions, Responses, a forced function
call with arguments 7 and 6, JSON mode, three 1,024-dimensional BGE-M3 vectors,
and a Vision XL data-URL image. The keyless live federation returned eight web
results using structured APIs plus a bounded Yahoo/DDGS worker, and eight paper
results while exposing a Semantic Scholar HTTP 429 as a non-fatal diagnostic.
Grounded 4B Web chat returned a visible cited answer.

A live Quick Papers task driven by the 4B Q4 alias retained three validated
public sources. When both model synthesis attempts violated the strict citation
dialect, it completed at 100 percent with the explicit evidence-inventory-only
fallback and canonical source list instead of losing the evidence or inventing
a conclusion.

Final local source gates passed with 191 backend tests plus Ruff and lockfile
checks; 50 frontend tests, TypeScript lint, and the production build; shell
syntax, GPU-readiness fixtures, Python compilation, and `git diff --check`.
The final visible browser gate rendered ten installed model cards and three API
cards with no console or page errors; its 390 px document width exactly matched
the 390 px mobile viewport.
