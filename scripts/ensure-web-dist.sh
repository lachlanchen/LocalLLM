#!/usr/bin/env bash
set -euo pipefail

project_root_input="${LOCALLLM_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -d "$project_root_input" ]] || {
  printf '%s\n' "ensure-web-dist: project root does not exist: $project_root_input" >&2
  exit 1
}
project_root="$(cd "$project_root_input" && pwd -P)"
dist_dir="$project_root/apps/web/dist"
marker="$dist_dir/.localllm-source.sha256"
required_public_assets=(favicon.svg manifest.webmanifest sw.js)

die() {
  printf '%s\n' "ensure-web-dist: $*" >&2
  exit 1
}

for required_command in awk cmp find grep node sha256sum sort tr xargs; do
  command -v "$required_command" >/dev/null || die "missing prerequisite: $required_command"
done

manifest_contract_is_valid() {
  local manifest="$1"
  node - "$manifest" <<'NODE'
const fs = require('node:fs')

try {
  const manifest = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))
  const expected = {
    id: '/',
    start_url: '/',
    scope: '/',
    display: 'standalone',
  }
  for (const [key, value] of Object.entries(expected)) {
    if (manifest[key] !== value) process.exit(1)
  }
  if (
    !Array.isArray(manifest.icons) ||
    !manifest.icons.some(
      (icon) =>
        icon &&
        icon.src === '/favicon.svg' &&
        icon.type === 'image/svg+xml',
    )
  ) {
    process.exit(1)
  }
} catch {
  process.exit(1)
}
NODE
}

web_source_fingerprint() {
  local -a inputs=()
  local required

  for required in \
    package.json \
    package-lock.json \
    apps/web/package.json \
    apps/web/index.html \
    apps/web/tsconfig.json \
    apps/web/tsconfig.app.json \
    apps/web/tsconfig.node.json \
    apps/web/vite.config.ts; do
    [[ -f "$project_root/$required" ]] || die "missing web build input: $required"
    inputs+=("$required")
  done

  for required in "${required_public_assets[@]}"; do
    [[ -f "$project_root/apps/web/public/$required" ]] ||
      die "missing required web public asset: apps/web/public/$required"
    [[ ! -L "$project_root/apps/web/public/$required" ]] ||
      die "required web public asset must not be a symlink: apps/web/public/$required"
  done
  grep -Fq 'rel="manifest"' "$project_root/apps/web/index.html" &&
    grep -Fq 'href="/manifest.webmanifest"' "$project_root/apps/web/index.html" ||
    die "apps/web/index.html does not link /manifest.webmanifest"
  manifest_contract_is_valid "$project_root/apps/web/public/manifest.webmanifest" ||
    die "apps/web/public/manifest.webmanifest is invalid or incomplete"

  while IFS= read -r -d '' required; do
    inputs+=("$required")
  done < <(
    cd "$project_root"
    find apps/web/src apps/web/public -type f -print0 | sort -z
  )

  ((${#inputs[@]} > 8)) || die "no web source files found"
  (
    cd "$project_root"
    printf '%s\0' "${inputs[@]}" | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

dist_completeness_error() {
  local asset

  [[ -f "$dist_dir/index.html" && ! -L "$dist_dir/index.html" ]] || {
    printf '%s\n' "web build did not produce a regular apps/web/dist/index.html"
    return 1
  }
  grep -Fq 'rel="manifest"' "$dist_dir/index.html" &&
    grep -Fq 'href="/manifest.webmanifest"' "$dist_dir/index.html" || {
    printf '%s\n' "built index does not link /manifest.webmanifest"
    return 1
  }

  for asset in "${required_public_assets[@]}"; do
    [[ -f "$dist_dir/$asset" && ! -L "$dist_dir/$asset" ]] || {
      printf '%s\n' "web build did not produce a regular apps/web/dist/$asset"
      return 1
    }
    cmp -s "$project_root/apps/web/public/$asset" "$dist_dir/$asset" || {
      printf '%s\n' "built $asset does not match apps/web/public/$asset"
      return 1
    }
  done

  manifest_contract_is_valid "$dist_dir/manifest.webmanifest" || {
    printf '%s\n' "built manifest.webmanifest is invalid or incomplete"
    return 1
  }
}

assert_dist_complete() {
  local completeness_error

  completeness_error="$(dist_completeness_error)" || die "$completeness_error"
}

write_marker() {
  local fingerprint="$1"
  local temporary

  assert_dist_complete
  temporary="$(mktemp "$dist_dir/.localllm-source.sha256.XXXXXX")"
  trap 'rm -f "$temporary"' RETURN
  printf '%s\n' "$fingerprint" >"$temporary"
  chmod 0644 "$temporary"
  mv "$temporary" "$marker"
  trap - RETURN
}

is_current() {
  local expected="$1"
  dist_completeness_error >/dev/null || return 1
  [[ -f "$marker" && ! -L "$marker" ]] || return 1
  [[ "$(tr -d '\r\n' <"$marker")" == "$expected" ]]
}

mode="${1:---ensure}"
case "$mode" in
  --fingerprint)
    web_source_fingerprint
    ;;
  --check)
    fingerprint="$(web_source_fingerprint)"
    assert_dist_complete
    is_current "$fingerprint" ||
      die "web bundle is missing or stale; run scripts/ensure-web-dist.sh"
    ;;
  --stamp)
    fingerprint="$(web_source_fingerprint)"
    write_marker "$fingerprint"
    ;;
  --ensure)
    command -v npm >/dev/null || die "npm is required to build the web bundle"
    for attempt in 1 2; do
      fingerprint="$(web_source_fingerprint)"
      if is_current "$fingerprint"; then
        exit 0
      fi
      (
        cd "$project_root"
        npm run build
      )
      verified_fingerprint="$(web_source_fingerprint)"
      if [[ "$verified_fingerprint" == "$fingerprint" ]]; then
        write_marker "$verified_fingerprint"
        exit 0
      fi
      ((attempt == 1)) ||
        die "web sources changed during two consecutive builds"
    done
    ;;
  *)
    die "usage: $0 [--ensure|--check|--stamp|--fingerprint]"
    ;;
esac
