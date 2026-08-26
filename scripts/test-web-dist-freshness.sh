#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT

mkdir -p "$fixture/apps/web/src" "$fixture/apps/web/public" "$fixture/bin"
for file in \
  package.json \
  package-lock.json \
  apps/web/package.json \
  apps/web/index.html \
  apps/web/tsconfig.json \
  apps/web/tsconfig.app.json \
  apps/web/tsconfig.node.json \
  apps/web/vite.config.ts \
  apps/web/src/main.tsx \
  apps/web/public/favicon.svg; do
  printf '%s\n' "$file" >"$fixture/$file"
done
printf '%s\n' '<link rel="manifest" href="/manifest.webmanifest" />' \
  >"$fixture/apps/web/index.html"
printf '%s\n' \
  '{"name":"LocalLLM","id":"/","start_url":"/","scope":"/","display":"standalone","icons":[{"src":"/favicon.svg","type":"image/svg+xml","sizes":"any"}]}' \
  >"$fixture/apps/web/public/manifest.webmanifest"
printf '%s\n' 'self.addEventListener("fetch", (event) => event.respondWith(fetch(event.request)))' \
  >"$fixture/apps/web/public/sw.js"

printf '%s\n' '#!/usr/bin/env bash' >"$fixture/bin/npm"
printf '%s\n' 'set -euo pipefail' >>"$fixture/bin/npm"
printf '%s\n' 'rm -rf apps/web/dist' >>"$fixture/bin/npm"
printf '%s\n' 'mkdir -p apps/web/dist' >>"$fixture/bin/npm"
printf '%s\n' 'cp apps/web/index.html apps/web/dist/index.html' >>"$fixture/bin/npm"
printf '%s\n' 'cp apps/web/public/* apps/web/dist/' >>"$fixture/bin/npm"
printf '%s\n' 'printf "1\\n" >> .build-count' >>"$fixture/bin/npm"
chmod +x "$fixture/bin/npm"

run_helper() {
  LOCALLLM_PROJECT_ROOT="$fixture" PATH="$fixture/bin:$PATH" \
    "$project_root/scripts/ensure-web-dist.sh" "$@"
}

run_helper --ensure
[[ "$(wc -l <"$fixture/.build-count")" == 1 ]]
run_helper --check
run_helper --ensure
[[ "$(wc -l <"$fixture/.build-count")" == 1 ]]
for asset in favicon.svg manifest.webmanifest sw.js; do
  cmp -s "$fixture/apps/web/public/$asset" "$fixture/apps/web/dist/$asset"
done

mv "$fixture/apps/web/public/manifest.webmanifest" \
  "$fixture/apps/web/public/manifest.webmanifest.missing"
if run_helper --check 2>/dev/null; then
  printf '%s\n' 'missing required manifest source unexpectedly passed --check' >&2
  exit 1
fi
mv "$fixture/apps/web/public/manifest.webmanifest.missing" \
  "$fixture/apps/web/public/manifest.webmanifest"

rm "$fixture/apps/web/dist/sw.js"
if run_helper --check 2>/dev/null; then
  printf '%s\n' 'missing service worker output unexpectedly passed --check' >&2
  exit 1
fi
run_helper --ensure
run_helper --check
[[ "$(wc -l <"$fixture/.build-count")" == 2 ]]

printf '%s\n' 'stale output' >>"$fixture/apps/web/dist/manifest.webmanifest"
if run_helper --check 2>/dev/null; then
  printf '%s\n' 'stale manifest output unexpectedly passed --check' >&2
  exit 1
fi
run_helper --ensure
run_helper --check
[[ "$(wc -l <"$fixture/.build-count")" == 3 ]]

printf '%s\n' 'changed' >>"$fixture/apps/web/src/main.tsx"
if run_helper --check 2>/dev/null; then
  printf '%s\n' 'stale bundle unexpectedly passed --check' >&2
  exit 1
fi
run_helper --ensure
run_helper --check
[[ "$(wc -l <"$fixture/.build-count")" == 4 ]]

printf '%s\n' 'web dist freshness PASS'
