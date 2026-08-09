# Python is selected by immutable linux/amd64 manifest digest. The sandbox
# deliberately installs no compiler, package manager dependencies, or network
# tooling beyond what the official minimal runtime already contains.
FROM docker.io/library/python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49

LABEL org.opencontainers.image.title="LocalLLM Python agent sandbox" \
      org.opencontainers.image.description="Fixed, offline, non-root Python runtime for explicitly approved LocalLLM agent code" \
      org.opencontainers.image.source="https://github.com/lachlanchen/LocalLLM" \
      org.opencontainers.image.base.name="docker.io/library/python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49" \
      org.opencontainers.image.version="python3.12.11-20260809" \
      io.localllm.sandbox.profile="python-v1"

RUN set -eux; \
    mkdir -p /work; \
    chown 65532:65532 /work; \
    chmod 0700 /work

WORKDIR /work
USER 65532:65532

# The API wraps this executable with the image's fixed GNU timeout supervisor and
# passes source on stdin. Keeping the default constrained makes manual inspection
# harmless and unsurprising.
ENTRYPOINT ["/usr/local/bin/python3", "-I", "-S", "-B", "-u", "-"]
