#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

command -v node >/dev/null || { echo "Node.js 20+ is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

npm install
uv sync --project apps/api --extra dev
npm run build
scripts/install-ollama-local.sh

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo "Bootstrap complete. Start with scripts/run.sh"

