# AI-assisted reverse-engineering workflow

## What the article’s workflow actually is

The practical pipeline is a combination of established reverse-engineering tools and an LLM reasoning loop:

```text
Windows .sys / .dll / .exe / firmware
        ↓
Ghidra decompiler + call graph + types + cross-references
        ↓
MCP tools expose small, structured questions to a local model
        ↓
Names, hypotheses, structures, IOCTL/USB command candidates
        ↓
Protocol specification separated from implementation
        ↓
Packet captures, descriptors, tests, and hardware evidence
        ↓
New libusb / Rust / C / Linux implementation
```

The LLM does not replace Ghidra, packet capture, or validation. Its largest contribution is compressing the human pattern-recognition loop: identifying APIs, proposing names, connecting scattered call sites, and turning observations into a test plan.

## Installed components

### Ghidra 12.0.3

Installed from the official NSA GitHub release and validated against SHA-256:

```text
90d3fffb20b00030dcef8d2a24dd0f422d3a61e432b3ad43f77233ac6d667981
```

Launch:

```bash
scripts/start-re-workbench.sh gui
```

### LLNL OGhidra

OGhidra integrates Ollama, an agent loop, and a Ghidra extension. The extension is built against the pinned Ghidra release and installed under that local Ghidra tree.

The upstream August 2026 source includes two legacy `GHIDRA_MODULE_*=...` lines that Ghidra 12.0.3 reports as invalid. `scripts/setup-re-toolchain.sh` removes those metadata lines from the build input; identity and version remain in `extension.properties`.

### PyGhidra-MCP

Installed in `.venv-tools` with Python 3.12. A verified headless session exposes 20 tools, including:

```text
decompile_function
search_symbols_by_name
search_code
list_project_binaries
list_project_binary_metadata
rename_function
rename_variable
set_variable_type
set_function_prototype
set_comment
list_exports
list_imports
list_xrefs
save
```

Start a loopback-only project server:

```bash
LOCALLLM_RE_PROJECT_NAME=device \
  scripts/start-re-workbench.sh mcp ./driver.sys ./vendor.dll
```

MCP URL: `http://127.0.0.1:18765/mcp`.

## Recommended investigation sequence

### 1. Preserve evidence

- hash every vendor binary, installer, SDK, capture, and firmware image;
- record product name, revision, USB VID/PID or PCI ID, driver version, and OS build;
- keep originals read-only;
- work on a spare test machine or VM with passthrough where appropriate.

### 2. Establish the device boundary

Determine whether this is a user-space USB protocol, HID, serial, network protocol, kernel driver, PCIe DMA device, or a firmware-mediated system. The easiest successes are typically vendor-specific USB protocols that can be reproduced through libusb.

### 3. Enumerate before asking broad questions

Use structured evidence:

1. imports and exports;
2. strings, GUIDs, VID/PID, IOCTL constants, registry keys, and endpoint numbers;
3. cross-references to `DeviceIoControl`, USB transfer APIs, file I/O, memory mapping, sleeps, and checksums;
4. decompile only the relevant functions and their callers/callees;
5. apply names and types only after evidence repeats.

### 4. Capture real behavior

For USB devices, pair static work with Windows USBPcap/Wireshark or a hardware analyzer. Record control, bulk, interrupt, and isochronous transfers; include setup packets, payload lengths, timing, status, and device state.

On Linux, Wireshark/tshark capture requires system packages and usbmon permissions. LocalLLM does not silently add capture privileges. The operator must install and configure these with appropriate system authority.

### 5. Write a protocol specification

Separate observation from inference. Each command should record:

- direction, endpoint, request, value, index, length;
- payload layout, byte order, checksum, timing, and state prerequisites;
- observed examples and capture references;
- confidence and unresolved alternatives.

### 6. Implement independently

A clean implementation should consume the specification and public OS/device APIs rather than copying decompiler output. Start in user space when possible. Add timeouts, bounds checks, cancellation, device-version checks, and a hardware interlock for risky actuators.

### 7. Validate adversarially

- replay known-good transactions;
- vary lengths, state, timing, unplug/replug, and firmware versions;
- compare Windows and new implementation outputs;
- use sanitizers and fuzzers on parsers;
- never treat “compiles” or “model says correct” as hardware validation.

## Prompt-injection boundary

Binary strings can contain text crafted to manipulate an LLM agent. Webpages, symbols, debug strings, decompiler comments, and embedded resources are data. They must never alter the system prompt, available tools, tool arguments, network policy, or authorization boundary.

LocalLLM’s Binary Studio instructs the model to ignore embedded instructions and never executes uploads. MCP clients should also:

- scope analysis to an explicit project;
- keep listeners on loopback;
- review mutations such as rename, type change, comment, delete, or save;
- deny shell/network tools unless separately needed;
- log the evidence behind every conclusion.

## Legal boundary

Clean-room design can help separate factual interface understanding from implementation, but it is not a universal legal shield. Copyright, contracts/EULAs, anticircumvention rules, patents, trade secrets, export controls, and jurisdiction can differ. Preserve provenance and obtain qualified legal advice for commercial or redistributed work.

## Realistic scope

Most promising: USB instruments, HID devices, relays, small displays, CNC controllers, FPGA programmers, vendor control protocols, and older applications.

Still difficult: modern GPU stacks, Wi-Fi, power management, high-performance storage, complex DMA, security processors, signed firmware chains, and devices whose behavior lives primarily in undocumented firmware.

## Primary sources

- [Ghidra official repository](https://github.com/NationalSecurityAgency/ghidra)
- [Ghidra 12.0.3 official release](https://github.com/NationalSecurityAgency/ghidra/releases/tag/Ghidra_12.0.3_build)
- [LLNL OGhidra](https://github.com/LLNL/OGhidra)
- [PyGhidra-MCP](https://github.com/clearbluejar/pyghidra-mcp)
- [LaurieWired GhidraMCP](https://github.com/LaurieWired/GhidraMCP)
