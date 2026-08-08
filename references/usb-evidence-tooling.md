# USB packet-evidence tooling

LocalLLM includes a project-local, containerized lane for inspecting USBPcap,
usbmon, and ordinary PCAP/PCAPNG files without installing Wireshark packages on
the host. It is deliberately an **offline parser**, not a privilege shortcut for
live capture.

## Build and provenance

Build the image and run its smoke test:

```bash
scripts/setup-usb-evidence-tools.sh
```

The build starts from the official Ubuntu image. The base digest, local tag
name, and target architecture are fixed in source:

```text
base: docker.io/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea
tag:  localllm/usb-evidence:ubuntu24.04-20260808
arch: linux/amd64
```

The local tag is mutable, and the resulting image is not a bit-for-bit build
claim: Ubuntu dependency resolution is not snapshot-pinned. The Dockerfile pins
these requested top-level Debian package revisions:

```text
tshark, wireshark-common  4.2.2-1.1build3
usbutils                  1:017-3build1
libusb runtime/dev        2:1.0.27-1
pkg-config                1.8.1-2build1
```

It records the tool contract in
`/opt/localllm-usb-evidence/versions.txt` and every base and dependency package
in `/opt/localllm-usb-evidence/packages.tsv`. The build audited on 2026-08-08
contained 144 packages; that manifest's SHA-256 was
`b1b27d91e3a8e99c3d141508ce681da99c148238d0da44e033ecf3de8334004f`.
The verifier reports the current count and checksum, checks that the manifest
is populated, and asserts:

- TShark/Wireshark 4.2.2;
- usbutils 017;
- libusb 1.0.27 through `pkg-config`;
- availability of `text2pcap`;
- successful parsing of a locally generated, benign one-frame USBPcap PCAPNG
  fixture representing a bulk IN transfer.

Rebuilding needs network access to Ubuntu's package archive. Running the
offline analyzer does not.

The analyzer trusts whatever local image currently owns the tag above; it does
not revalidate the image ID or package contract on every invocation. Rerun
`scripts/verify-usb-evidence-tools.sh` after a rebuild or tag replacement and
record the setup script's printed image ID with important evidence.

## Analyze a saved capture

```bash
# Concise frame table, including USB fields when present
scripts/analyze-usb-pcap.sh evidence/device-session.pcapng

# Inspect bulk transfers in detail
scripts/analyze-usb-pcap.sh evidence/device-session.pcapng \
  -Y 'usb.transfer_type == 0x03' -V

# Emit USB packets as JSON for a local analysis script or model
scripts/analyze-usb-pcap.sh evidence/device-session.pcapng \
  -Y usb -T json
```

The wrapper resolves one readable regular file. That capture is its only host
evidence bind and is mounted at `/evidence/capture.pcap` read-only. The
container runs with:

- `--network none`;
- every Linux capability dropped;
- `no-new-privileges`;
- a read-only root filesystem;
- the caller's numeric UID/GID;
- bounded memory, CPU, and process count;
- an explicit 64 MiB, `noexec` `/tmp`; Docker's standard isolated `/dev` and
  `/dev/shm` mounts still exist, with no host device passed through.

The wrapper rejects live-interface options. It does not bind-mount the host USB
bus, debugfs, host `/dev`, or the Docker socket. Treat captures as untrusted
input anyway: Wireshark dissectors are a large parser surface, so keep the image
updated by a reviewed digest/package change and retain the original evidence
hash.

Membership in the host's `docker` group is itself a system-level trust decision:
a user who can issue arbitrary Docker commands can normally obtain broad host
access. These restrictions constrain this repository's wrapper; they do not
turn a Docker daemon into a security boundary against its authorized users.

## Live Linux usbmon capture is an operator action

Live usbmon access requires kernel and capture permissions that this setup does
not request. A system operator must explicitly decide how to provide them. A
typical host-managed path is:

```bash
sudo modprobe usbmon
sudo apt-get install tshark

# Identify the bus after connecting the device.
lsusb
sudo dumpcap -D

# Capture only the selected usbmon bus; stop with Ctrl-C.
sudo dumpcap -i usbmon1 -w "$PWD/device-session.pcapng"
sudo chown "$(id -u):$(id -g)" "$PWD/device-session.pcapng"
```

Use the narrowest appropriate bus, capture on a disposable analysis machine,
and avoid unrelated keyboards, storage devices, authentication tokens, or
other users' traffic. Some distributions instead authorize the dedicated
`dumpcap` binary and a capture group; that is also a deliberate system policy
change and should be handled by the machine operator.

The project does not run `sudo`, load `usbmon`, install host packages, change
groups, grant capture capabilities, bind host debugfs into a container, or use
`--privileged`. Once a capture exists, return to the offline wrapper above.

Windows USBPcap captures can be copied to this workstation and parsed by the
same wrapper. A hardware USB analyzer remains preferable when host-controller
or driver instrumentation would alter the behavior under investigation.

## Evidence workflow

1. Hash and preserve the original capture.
2. Record OS, driver, device revision, VID/PID, bus, endpoints, and device state.
3. Filter by endpoint and transfer type before inspecting payload patterns.
4. Correlate packets with Ghidra call sites and concrete UI/API actions.
5. Mark inferred command semantics as hypotheses until replayed or otherwise
   independently verified.

Primary references: [Wireshark USB capture setup](https://wiki.wireshark.org/CaptureSetup/USB),
[TShark manual](https://www.wireshark.org/docs/man-pages/tshark.html), and the
[Docker Official Image for Ubuntu](https://hub.docker.com/_/ubuntu).
