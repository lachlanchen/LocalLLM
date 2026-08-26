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

printf '%s\n' '#!/usr/bin/env bash' >"$fixture/bin/npm"
printf '%s\n' 'set -euo pipefail' >>"$fixture/bin/npm"
printf '%s\n' 'mkdir -p apps/web/dist' >>"$fixture/bin/npm"
printf '%s\n' 'printf "built\\n" > apps/web/dist/index.html' >>"$fixture/bin/npm"
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

printf '%s\n' 'changed' >>"$fixture/apps/web/src/main.tsx"
if run_helper --check 2>/dev/null; then
  printf '%s\n' 'stale bundle unexpectedly passed --check' >&2
  exit 1
fi
run_helper --ensure
run_helper --check
[[ "$(wc -l <"$fixture/.build-count")" == 2 ]]

printf '%s\n' 'web dist freshness PASS'
