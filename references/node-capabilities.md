# Node liveness, readiness, and capabilities

LocalLLM exposes separate process, admission, and discovery contracts so a
gateway or scheduler can replace an inference node without treating a running
Python process as a usable model worker.

| Route | Status contract | Meaning |
| --- | --- | --- |
| `GET /livez` | `200` while the API process can answer | Process liveness only; it never contacts Ollama. |
| `GET /readyz` | `200` ready, `503` not ready | Ollama's local model catalog is reachable and every configured required model resolves to an installed tag. |
| `GET /api/node/capabilities` | `200` even when unready | Versioned discovery document with runtime state, required-model state, supported protocol routes, installed model provenance, and separate functional-canary evidence. |
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

The 19 GB `localllm-code` specialist intentionally remains outside the practical
core download. General text models can answer coding questions, but that is not
the same contract as promising the dedicated `code` role. A full workstation
that promises all four inference roles should explicitly add `localllm-code` to
`LOCALLLM_REQUIRED_MODELS`, set
`LOCALLLM_NODE_CANARY_ROLES=text,code,vision,embedding`, and validate all four.
Smaller or role-specific nodes can truthfully configure a subset.

The service installer now waits on `/readyz`; pull the required models first or
set the role-specific list before installing/restarting user services. Missing
models leave the API live and discoverable but fail admission with `503`.

## Capabilities schema v2

`/api/node/capabilities` returns `schema_version: 2`. Consumers should reject a
schema version they do not understand rather than guessing field semantics.
Version 2 preserves the v1 fields while adding release-bound functional evidence
and installed model provenance. The document contains:

- `service`: API name, software version, strict non-secret release ID, and generic node kind;
- `ready`: the same admission decision as `/readyz`;
- `runtime`: the Ollama dependency state and a stable, content-free error code;
- `required_models`: configured IDs, resolved IDs, and availability;
- `protocols`: the authenticated OpenAI-compatible routes and streaming support;
- `models`: installed tags, stable aliases, catalog status, advertised input
  modalities, catalog context limit, resolved Ollama manifest digest, and size;
- `functional_readiness`: sanitized status from the configured canonical canary
  receipt, including freshness and per-role latency/provenance but no prompts,
  outputs, vectors, keys, paths, or raw errors.

An installed tag is a catalog observation, not proof that weights are warm on a
GPU or that a fresh inference request will succeed. `/readyz` deliberately
remains this lightweight catalog-admission contract. `functional_readiness` is
separate and is never required by `/readyz`; a scheduler may require a fresh
passing receipt before enrolling or switching a node.

## Bounded functional canary

The verifier validates every selected stable alias through `/v1/models`, checks
its resolved tag and digest against node capabilities, then runs actual Chat
Completions for text/code/vision and Embeddings for embedding. It uses a fixed
in-memory PNG and sequential requests. The OpenAI lane uses the
service-configured context together with a 32-token chat-output bound, bounded
response bytes, per-role deadlines, and OpenAI-compatible
`reasoning_effort: "none"`; it does not send Ollama-native lifetime or options
fields through `/v1`. Text, code, and vision must return exact normalized
answers that are not disclosed in their prompts. The code probe evaluates
`"".join(reversed("abc"))`, requests three unquoted lowercase letters, and
accepts only the normalized answer `CBA`; explanations, quotes, and code fences
fail closed. BGE-M3 must return two 1,024-value finite, nonzero vectors for two
distinct inputs, with at least one component differing between them. Chat and
embedding responses must identify the exact resolved model observed during
preflight. Its JSON contains only status, role, latency, alias, resolved model,
digest, release identity, and UTC timestamps.

After a role resolves its exact model and dispatches inference, the verifier
always makes one bounded native `POST /api/generate` to the fixed loopback
Ollama origin (default `http://127.0.0.1:11434`) with that exact tag and
`keep_alive: 0`. This cleanup runs in `finally`, including semantic, upstream,
postflight, timeout, and cancellation paths. It uses a separate client with no
API authorization header, ignores proxy environment variables, refuses
redirects and non-literal-loopback origins, and never enumerates or unloads
arbitrary `/api/ps` entries. A failed cleanup fails the role without adding
response content or errors to the receipt, stops further inference, and records
all remaining selected roles as failed. The configured per-role deadline covers
the probe. Shielded cleanup has its own 10-second bound, so cancellation at the
role deadline may add up to 10 seconds of cleanup grace before the verifier
returns.

Every receipt is bound to the `LOCALLLM_RELEASE_ID` returned by the running API.
A fresh receipt from an older release reports `release_mismatch` and cannot make
`functional_readiness.ready` true after deployment. Development defaults to the
safe non-secret ID `dev`; an immutable production activation must set the
reviewed lowercase `<commit8>-<archive8>` release ID. Configuring a receipt with
`dev`, `unknown`, or another reusable ID is rejected.

Functional readiness also compares every configured role's receipt tag and
digest with the current Ollama catalog probe. Repulling or retagging a model
after the canary reports `model_mismatch` and fails closed until that exact
model build passes a new canary.

The API key is accepted only from `LOCALLLM_API_KEY` or an owner-private regular
file; there is no command-line key-value option. A receipt is written only when
`--output` is explicit. Its only valid location is the existing owner-private
`data/node-canaries/<release-id>.json` path, tied to the release observed from
the API; this prevents the verifier from replacing other LocalLLM data. The
write is canonical, atomic, and no-symlink:

```bash
install -d -m 700 data/node-canaries
uv run --project apps/api --no-sync python scripts/verify-node-inference.py \
  --roles text,code,vision,embedding \
  --api-key-file .local/private/localllm-api-key \
  --output data/node-canaries/01234567-89abcdef.json
```

To expose that receipt, configure the matching roles and path, then restart only
the LocalLLM API through the normal reviewed service workflow:

```dotenv
LOCALLLM_NODE_CANARY_ROLES=text,code,vision,embedding
LOCALLLM_NODE_CANARY_RECEIPT_PATH=./data/node-canaries/01234567-89abcdef.json
LOCALLLM_NODE_CANARY_MAX_AGE_SECONDS=86400
LOCALLLM_RELEASE_ID=01234567-89abcdef
```

The script defaults to all four roles for this workstation's acceptance. A
role-specific node should select and configure only the roles it actually
promises. Ollama tags can move across future pulls, so the receipt records the
resolved manifest digest observed before inference and rechecks that provenance
immediately afterward. Freshness is measured from the oldest configured role's
timestamp, not merely from the aggregate receipt time.

The exact-tag cleanup is not proof that unrelated or earlier jobs left no model
resident. A production activation should additionally require Ollama `/api/ps`
to become empty before promotion, without asking the verifier to unload entries
it did not start.
