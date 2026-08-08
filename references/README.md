# LocalLLM references

These notes preserve the design decisions, source trail, operational boundaries, and recovery procedures behind LocalLLM Studio.

| Document | Use it when… |
| --- | --- |
| [Model selection](model-selection.md) | choosing Q4/Q8, text/vision, one/two-GPU layouts, or context limits |
| [Deep Research](deep-research.md) | understanding search, extraction, citations, limitations, and prompt-injection defenses |
| [Reverse engineering](reverse-engineering-workflow.md) | installing or operating Ghidra, OGhidra, PyGhidra-MCP, USB evidence, or clean-room workflows |
| [OpenAI API compatibility](openai-api-compatibility.md) | connecting OpenAI SDKs and determining which request fields are supported |
| [GPU driver recovery](gpu-driver-recovery.md) | `nvidia-smi` reports a driver/library mismatch or Ollama sees fewer GPUs than expected |
| [Source ledger](sources.md) | checking the primary sources and pinned release details used by this repository |

Machine-specific downloads, model blobs, Ghidra projects, uploads, and reports are intentionally excluded from Git.

