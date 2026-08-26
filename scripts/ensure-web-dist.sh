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

die() {
  printf '%s\n' "ensure-web-dist: $*" >&2
  exit 1
}

for required_command in awk find sha256sum sort tr xargs; do
  command -v "$required_command" >/dev/null || die "missing prerequisite: $required_command"
done

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

write_marker() {
  local fingerprint="$1"
  local temporary

  [[ -f "$dist_dir/index.html" ]] || die "web build did not produce apps/web/dist/index.html"
  temporary="$(mktemp "$dist_dir/.localllm-source.sha256.XXXXXX")"
  trap 'rm -f "$temporary"' RETURN
  printf '%s\n' "$fingerprint" >"$temporary"
  chmod 0644 "$temporary"
  mv "$temporary" "$marker"
  trap - RETURN
}

is_current() {
  local expected="$1"
  [[ -f "$dist_dir/index.html" && -f "$marker" ]] || return 1
  [[ "$(tr -d '\r\n' <"$marker")" == "$expected" ]]
}

mode="${1:---ensure}"
case "$mode" in
  --fingerprint)
    web_source_fingerprint
    ;;
  --check)
    fingerprint="$(web_source_fingerprint)"
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
