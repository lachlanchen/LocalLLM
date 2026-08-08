#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

command -v node >/dev/null || { echo "Node.js 20.19+ or 22.12+ is required" >&2; exit 1; }
node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit((major === 20 && minor >= 19) || (major === 22 && minor >= 12) || major > 22 ? 0 : 1)' || {
  echo "Node.js 20.19+ or 22.12+ is required" >&2
  exit 1
}
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

npm ci
uv sync --project apps/api --extra dev --locked
npm run build
scripts/install-ollama-local.sh

if [[ ! -f .env ]]; then
  install -m 600 .env.example .env
else
  chmod 600 .env
fi

mkdir -p data
chmod 700 data

echo "Bootstrap complete. Start with scripts/run.sh"
