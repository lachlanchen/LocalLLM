# LocalLLM references

These notes preserve the design decisions, source trail, operational boundaries, and recovery procedures behind LocalLLM Studio.

| Document | Use it when… |
| --- | --- |
| [Verification report](verification-report.md) | reviewing the historical nine-model/pre-reboot baseline and the completed 2026-08-09 post-reboot service, dual-GPU, and Vision XL addendum |
| [Model selection](model-selection.md) | choosing Q4/Q8, text/vision, one/two-GPU layouts, or context limits |
| [llama.cpp runtime](llama-cpp.md) | building the pinned CUDA alternative runtime or serving a GGUF model on a loopback OpenAI-compatible API |
| [Deep Research](deep-research.md) | understanding search, extraction, citations, limitations, and prompt-injection defenses |
| [Search and Research API](search-research-api.md) | configuring federated web/paper providers or understanding Auto/Local/Web/Papers/All grounded chat and deep-research tasks |
| [Conversation history](conversation-history.md) | using resumable Playground sessions, SQLite revision safety, context compaction, GFM/KaTeX rendering, or history quotas |
| [Agent capabilities](agent-capabilities.md) | using the mounted plan-preview panel or installing, enabling, and reviewing the confirmed offline Python sandbox |
| [Optional image generation](image-generation.md) | installing or operating the mounted default-off Z-Image-Turbo lane, selecting its GPU, releasing warm weights, or reviewing job/output safety limits |
| [Reverse engineering](reverse-engineering-workflow.md) | installing or operating Ghidra, OGhidra, PyGhidra-MCP, USB evidence, or clean-room workflows |
| [USB packet evidence](usb-evidence-tooling.md) | building the version-pinned offline TShark image from a digest-pinned base, inspecting captures, or planning an operator-authorized usbmon capture |
| [OpenAI API compatibility](openai-api-compatibility.md) | connecting OpenAI SDKs and determining which request fields are supported |
| [GPU driver recovery](gpu-driver-recovery.md) | `nvidia-smi` reports a driver/library mismatch or Ollama sees fewer GPUs than expected |
| [Source ledger](sources.md) | checking the primary sources and pinned release details used by this repository |

Machine-specific downloads, model blobs, Ghidra projects, uploads, and reports are intentionally excluded from Git.
