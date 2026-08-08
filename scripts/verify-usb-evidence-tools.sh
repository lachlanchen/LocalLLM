#!/usr/bin/env bash
set -euo pipefail

image_ref="localllm/usb-evidence:ubuntu24.04-20260808"
expected_base="docker.io/library/ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea"

die() {
  echo "verify-usb-evidence-tools: $*" >&2
  exit 1
}

command -v docker >/dev/null || die "Docker is required"
docker image inspect "$image_ref" >/dev/null 2>&1 ||
  die "image is not installed; run scripts/setup-usb-evidence-tools.sh"

actual_base="$(
  docker image inspect "$image_ref" \
    --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}'
)"
[[ "$actual_base" == "$expected_base" ]] ||
  die "unexpected base label: ${actual_base:-none}"

runtime=(
  docker run
  --rm
  --network none
  --cap-drop ALL
  --security-opt no-new-privileges:true
  --read-only
  --pids-limit 128
  --memory 1g
  --cpus 2
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m
)

versions="$(
  "${runtime[@]}" --entrypoint /bin/sh "$image_ref" -ec '
    tshark --version | sed -n "1p"
    lsusb --version
    printf "libusb-pkg-config "
    pkg-config --modversion libusb-1.0
    test -x /usr/bin/text2pcap
    cat /opt/localllm-usb-evidence/versions.txt
    package_manifest=/opt/localllm-usb-evidence/packages.tsv
    test -s "$package_manifest"
    package_count="$(wc -l < "$package_manifest")"
    test "$package_count" -ge 100
    printf "package-manifest-count %s\n" "$package_count"
    package_checksum="$(sha256sum "$package_manifest")"
    printf "package-manifest-sha256 %s\n" "${package_checksum%% *}"
  '
)"
grep -Fq 'TShark (Wireshark) 4.2.2' <<<"$versions" || die "unexpected tshark version"
grep -Fq 'lsusb (usbutils) 017' <<<"$versions" || die "unexpected usbutils version"
grep -Fq 'libusb-pkg-config 1.0.27' <<<"$versions" || die "unexpected libusb version"
grep -Fq 'package-manifest-count ' <<<"$versions" || die "package manifest was not verified"
grep -Eq 'package-manifest-sha256 [0-9a-f]{64}$' <<<"$versions" ||
  die "package manifest checksum was not reported"

fixture="$(
  "${runtime[@]}" --entrypoint /bin/sh "$image_ref" -ec '
    # USBPcap pseudoheader: bulk IN transfer, bus 1, device 2, endpoint 0x81,
    # followed by the eight-byte benign payload "LocalLLM".
    printf "%s\n" \
      "000000 1b 00 01 00 00 00 00 00 00 00 00 00 00 00 09 00 01 01 00 02 00 81 03 08 00 00 00 4c 6f 63 61 6c 4c 4c 4d" \
      > /tmp/fixture.hex
    text2pcap -q -E usb-usbpcap /tmp/fixture.hex /tmp/fixture.pcapng
    tshark -n -r /tmp/fixture.pcapng \
      -T fields -E separator=, \
      -e frame.number -e usb.bus_id -e usb.device_address \
      -e usb.endpoint_address -e usb.transfer_type -e usb.capdata
  '
)"
[[ "$fixture" == '1,1,2,0x81,0x03,4c6f63616c4c4c4d' ]] ||
  die "synthetic PCAP parse mismatch: $fixture"

printf '%s\n' "$versions"
echo "Synthetic PCAP parse: $fixture"
echo "USB evidence verification PASS"
