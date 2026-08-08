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

The archive is downloaded from the official NSA GitHub release and validated against SHA-256 before extraction:

```text
90d3fffb20b00030dcef8d2a24dd0f422d3a61e432b3ad43f77233ac6d667981
```

Launch:

```bash
scripts/start-re-workbench.sh gui
```

### LLNL OGhidra

OGhidra integrates Ollama, an agent loop, and a Ghidra extension. The extension is built against the pinned Ghidra release and installed under that local Ghidra tree.

The installer checks out exact upstream commit
`93a4380fc748a393690be9bfd2c2156fade82757`. It then applies the repository-tracked
[`patches/oghidra-local-security.patch`](../patches/oghidra-local-security.patch)
before building. The patch removes two legacy `GHIDRA_MODULE_*=...` lines that
Ghidra 12.0.3 rejects; identity and version remain in `extension.properties`.
It also changes the embedded Java bridge from an all-interface listener to
`127.0.0.1` and puts every HTTP context behind one browser request filter.

### PyGhidra-MCP

Installed from exact upstream commit
`f29063b8636100b71e9c3aec61fe056827c556e4` in `.venv-tools` with Python
3.12. A verified headless session exposes 20 tools, including:

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

The installer finishes by running an isolated MCP verification against
`/bin/true`. It creates a temporary project, starts the server on an
OS-selected loopback port, initializes it with the official MCP Python client,
asserts the exact 20-tool surface, verifies the imported binary, and performs a
read-only function-symbol search. The server is terminated and its exact
temporary directory is removed after either success or failure. Rerun it, or
provide another explicitly benign binary, with:

```bash
scripts/verify-re-toolchain.sh
scripts/verify-re-toolchain.sh /path/to/benign-binary
```

For named working projects, `LOCALLLM_RE_PROJECT_NAME` is restricted to 1–64
ASCII letters, digits, dots, underscores, or hyphens, beginning with a letter
or digit. Path separators and traversal components are rejected before a
server or project is created.

## Local bridge trust boundary

Both reverse-engineering bridges are deliberately **local-only, not
authenticated services**:

- OGhidra binds specifically to `127.0.0.1`, not `0.0.0.0` or every network
  interface. Its centralized filter rejects `Sec-Fetch-Site: cross-site` and
  rejects every browser `Origin` except literal loopback origins
  (`localhost`, `127.0.0.1`, or `::1`). Native local clients that omit browser
  origin headers remain compatible.
- `scripts/start-re-workbench.sh mcp` explicitly binds PyGhidra-MCP to
  `127.0.0.1`.
- Neither bridge issues or verifies an API key. Any process or user session that
  can reach the workstation's loopback interface can call read and mutation
  tools, including rename, comment, type, prototype, and save operations.

The browser filter reduces cross-site request and DNS-rebinding risk; it is not
authentication or local process isolation. Use the bridges only on a trusted
single-user workstation, close them when analysis is finished, and do not
publish, port-forward, reverse-proxy, container-share, or tunnel their ports.
Remote or multi-user operation requires an authenticated authorization layer
that this project does not provide.

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

Saved USBPcap, usbmon, and PCAPNG evidence can be inspected without host
packet-package installation using the project's network-disabled container,
built from the digest-pinned Ubuntu base documented in
[USB packet evidence](usb-evidence-tooling.md). Live Linux capture
still requires system packages and usbmon permissions. LocalLLM does not
silently add capture privileges; the operator must explicitly install and
configure them with appropriate system authority.

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
- keep listeners on loopback and never treat loopback as authentication;
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
- [LLNL OGhidra at the installed commit](https://github.com/LLNL/OGhidra/tree/93a4380fc748a393690be9bfd2c2156fade82757)
- [PyGhidra-MCP at the installed commit](https://github.com/clearbluejar/pyghidra-mcp/tree/f29063b8636100b71e9c3aec61fe056827c556e4)
- [LaurieWired GhidraMCP](https://github.com/LaurieWired/GhidraMCP)
