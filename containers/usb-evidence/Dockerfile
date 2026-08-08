# Official Ubuntu 24.04.4 LTS amd64 image, selected by immutable content digest.
FROM docker.io/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea

ARG DEBIAN_FRONTEND=noninteractive
ARG TSHARK_VERSION=4.2.2-1.1build3
ARG USBUTILS_VERSION=1:017-3build1
ARG LIBUSB_DEV_VERSION=2:1.0.27-1
ARG PKG_CONFIG_VERSION=1.8.1-2build1

LABEL org.opencontainers.image.title="LocalLLM USB evidence tools" \
      org.opencontainers.image.description="Offline-only tshark, usbutils, and libusb analysis lane for LocalLLM" \
      org.opencontainers.image.source="https://github.com/lachlanchen/LocalLLM" \
      org.opencontainers.image.base.name="docker.io/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea" \
      org.opencontainers.image.version="ubuntu24.04-20260808"

RUN set -eux; \
    printf '%s\n' 'wireshark-common wireshark-common/install-setuid boolean false' \
      | debconf-set-selections; \
    apt-get update; \
    apt-get install --yes --no-install-recommends \
      "tshark=${TSHARK_VERSION}" \
      "wireshark-common=${TSHARK_VERSION}" \
      "usbutils=${USBUTILS_VERSION}" \
      "libusb-1.0-0-dev=${LIBUSB_DEV_VERSION}" \
      "pkg-config=${PKG_CONFIG_VERSION}"; \
    mkdir -p /opt/localllm-usb-evidence; \
    { \
      printf '%s\n' \
        'base=docker.io/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea' \
        'platform=linux/amd64'; \
      dpkg-query --show --showformat='${Package}\t${Version}\n' \
        tshark wireshark-common usbutils libusb-1.0-0 libusb-1.0-0-dev pkg-config; \
      tshark --version | sed -n '1p'; \
      lsusb --version; \
      printf 'libusb-pkg-config\t'; \
      pkg-config --modversion libusb-1.0; \
    } > /opt/localllm-usb-evidence/versions.txt; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN dpkg-query --show --showformat='${Package}\t${Version}\n' \
      | LC_ALL=C sort > /opt/localllm-usb-evidence/packages.tsv

WORKDIR /evidence
USER 65534:65534
ENTRYPOINT ["tshark"]
