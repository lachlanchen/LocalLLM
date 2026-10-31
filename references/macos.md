# Apple Silicon chat setup

LocalLLM's chat launcher can use native Ollama and an existing Python environment
on Apple Silicon. The Linux bootstrap, systemd services, NVIDIA tools, and optional
Linux workers remain separate; use the manual setup below on macOS.

Install native Ollama, Python 3.10 or newer, Node.js 22.12 or newer, npm and Git.
From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e apps/api
npm ci
npm run build
```

The launcher finds Ollama in `.local/ollama/ollama`,
`.local/ollama/bin/ollama`, `/Applications/Ollama.app/Contents/Resources/ollama`,
or `PATH`. Set `LOCALLLM_OLLAMA_BIN` to select another native executable.
It uses `.venv/bin/python` for the API; `LOCALLLM_API_PYTHON` overrides that path.

Reuse an existing model store by setting `OLLAMA_MODELS`. Without an override,
the launcher uses the repository's `models` directory when present on macOS,
otherwise `.local/models/ollama`. This does not copy weights or import a separate
Ollama installation's models automatically.

For a machine with `qwen3.6:35b-a3b` already installed in the selected store:

```bash
export LOCALLLM_REQUIRED_MODELS=qwen3.6:35b-a3b
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_NO_CLOUD=1
scripts/run.sh
```

Open <http://127.0.0.1:8008>. The app uses an 8K context for this preset.
The model shelf distinguishes that limit from the model's advertised maximum.
The Metal indicator reports detected hardware; `/readyz` separately verifies
Ollama and the configured required models. Hardware detection alone does not
qualify every model format or optional worker.

Stop the foreground launcher when finished. On a shared Mac, coordinate model
loading and other heavy work with the machine's current owner. Conversations
stay in the configured local data directory and must not be committed with
models, runtime logs or browser profiles.
