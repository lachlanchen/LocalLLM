# Node liveness, readiness, and capabilities

LocalLLM exposes separate process, admission, and discovery contracts so a
gateway or scheduler can replace an inference node without treating a running
Python process as a usable model worker.

| Route | Status contract | Meaning |
| --- | --- | --- |
| `GET /livez` | `200` while the API process can answer | Process liveness only; it never contacts Ollama. |
| `GET /readyz` | `200` ready, `503` not ready | Ollama's local model catalog is reachable and every configured required model resolves to an installed tag. |
| `GET /api/node/capabilities` | `200` even when unready | Versioned discovery document with runtime state, required-model state, supported protocol routes, and installed model metadata. |
| `GET /healthz` | compatibility `200` | Legacy aggregate document. Its top-level `ok` remains true even when its nested Ollama check fails, so it must not be used for admission. |

Dynamic responses send `Cache-Control: no-store`. The routes inherit LocalLLM's
loopback-only binding and browser-origin boundary. The capabilities document
does not expose credentials, prompts, host identity, filesystem paths, or raw
dependency errors, but it does list locally installed model tags just as the
existing model-catalog route does.

## Required models

`LOCALLLM_REQUIRED_MODELS` accepts a comma-separated list or JSON array. Its
default is:

```dotenv
LOCALLLM_REQUIRED_MODELS=localllm-fast,localllm-deep,localllm-vision,localllm-embed
```

That set exactly matches `scripts/pull-models.sh core`. Aliases are resolved
before comparison with Ollama's installed tags. A role-specific worker should
declare only the models assigned to it. An explicit `[]` makes readiness depend
on the Ollama catalog alone, which is useful for a model-management node but
does not claim that any inference model is installed.

The service installer now waits on `/readyz`; pull the required models first or
set the role-specific list before installing/restarting user services. Missing
models leave the API live and discoverable but fail admission with `503`.

## Capabilities schema v1

`/api/node/capabilities` returns `schema_version: 1`. Consumers should reject a
schema version they do not understand rather than guessing field semantics.
The v1 document contains:

- `service`: API name, software version, and generic node kind;
- `ready`: the same admission decision as `/readyz`;
- `runtime`: the Ollama dependency state and a stable, content-free error code;
- `required_models`: configured IDs, resolved IDs, and availability;
- `protocols`: the authenticated OpenAI-compatible routes and streaming support;
- `models`: installed tags, stable aliases, catalog status, advertised input
  modalities, and catalog context limit.

An installed tag is a catalog observation, not proof that weights are warm on a
GPU or that a fresh inference request will succeed. Ollama tags can also move
across future pulls. Schedulers needing stronger guarantees should pin model
digests outside this v1 contract and use a bounded inference canary separately;
they must not turn readiness into a heavyweight model-load request.
