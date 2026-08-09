# Local agent capabilities and Python sandbox

LocalLLM's agent boundary is deliberately separate from ordinary chat. A normal
chat request cannot run Python, and plan validation never dispatches a tool. The
integrated service supplies two reusable pieces:

1. a model-independent JSON plan validator and staging coordinator; and
2. an explicitly enabled, two-step Python execution API backed by a fixed local
   Docker image.

This design borrows the useful separation found in AgInTiFlow—capability routing,
structured events, bounded tool loops, and explicit side-effect policy—without
copying its source code or changing its license boundary.

## Safety posture

Python execution is **off by default**. Building the image does not enable it.
An execution is accepted only when all of these conditions hold:

- the operator set `LOCALLLM_AGENT_CODE_EXECUTION_ENABLED=true` and restarted
  the API;
- the fixed `localllm/python-sandbox:3.12.11-20260809` image exists and carries
  the expected identity labels;
- the client explicitly acknowledged the sandbox risk and obtained a
  short-lived token bound to the SHA-256 hash of the exact code;
- the execution request repeats an explicit `confirmed: true` and consumes that
  token before it expires.

Tokens are in-memory, single-use, valid for 60 seconds, and consumed even when a
client tries them against mismatched code. API restart invalidates all pending
tokens.

These are management routes. They do not use `LOCALLLM_API_KEY`; they rely on
the application's loopback-peer boundary and, for browser traffic, its
origin/fetch-site checks. Any native process in the same host network namespace
can call them and, after the operator enables Python, can request its own
confirmation token. The confirmation flow is a deliberate-execution and code
integrity control, not per-user authentication. Do not proxy or tunnel port
8008 without adding authorization.

The Docker invocation is assembled entirely by the server. Requests cannot
select a Docker binary, image, flag, network, mount, user, or resource limit.
The fixed profile has:

| Boundary | Fixed value |
| --- | --- |
| Network | `none` |
| Host volumes | none |
| Root filesystem | read-only |
| Writable storage | 64 MiB ephemeral `tmpfs` at `/work`, `noexec,nosuid,nodev` |
| User | `65532:65532` |
| Linux capabilities | all dropped |
| Privilege escalation | `no-new-privileges` |
| Processes | 64 |
| Memory / swap | 512 MiB total / no additional swap |
| CPU | 1 CPU |
| File descriptors | 128 |
| Runtime | at most 20 seconds by the API, plus an independent 25-second container deadline |
| Captured output | at most 64 KiB combined |
| Concurrent runs | 2 |

Source is sent over Docker stdin to isolated Python (`-I -S -B -u -`). There are
no host mounts. Timeout, output overflow, and task cancellation trigger a fixed
`docker kill`/`docker rm --force` cleanup by the server-generated container name.
Every run also starts under GNU `timeout --signal=KILL 25s` inside the container.
That daemon-owned deadline remains effective if the API process is abruptly killed;
Docker's fixed `--rm` policy then removes the exited, labeled LocalLLM container.
This is a strong local containment boundary, not a guarantee against an unknown
Docker/kernel escape; do not expose the loopback service remotely.

## Build and verify

The Python base is pinned by its immutable Linux/amd64 manifest digest. Build the
local image and exercise its security invariants:

```bash
scripts/setup-agent-sandbox.sh
scripts/verify-agent-sandbox.sh
```

The verifier checks non-root identity, zero effective capabilities,
`NoNewPrivs`, a read-only root, writable ephemeral work space, and absent
external networking. It does not turn the feature on.

To opt in, set the environment variable and restart the API:

```dotenv
LOCALLLM_AGENT_CODE_EXECUTION_ENABLED=true
```

## Mounted routes and request bounds

`localllm.agent_runtime` is mounted by the main application under `/api/agent`.
The global request-size middleware caps `/api/agent/plans/validate`,
`/api/agent/plans/propose`, and `/api/agent/code/confirmations` at 20 KiB, and
`/api/agent/code/executions` at 40 KiB. These transport limits apply before JSON
parsing and are additional to the Pydantic field bounds.

## Capability contract

`GET /api/agent/capabilities` reports the fixed schema, whether the operator
enabled code, whether the sandbox image is ready, and every sandbox limit. A
default response before opt-in includes:

```json
{
  "schema_version": "1",
  "default_mode": "ordinary_chat",
  "ordinary_chat_auto_executes_tools": false,
  "operator_code_execution_enabled": false,
  "sandbox_ready": false
}
```

The two-stage execution exchange is:

1. `POST /api/agent/code/confirmations`

   ```json
   {
     "tool": "python",
     "code_sha256": "<64 lowercase hexadecimal characters>",
     "risk_acknowledgement": "RUN_IN_ISOLATED_SANDBOX"
   }
   ```

2. `POST /api/agent/code/executions`

   ```json
   {
     "tool": "python",
     "code": "print(6 * 7)",
     "timeout_seconds": 10,
     "confirmed": true,
     "confirmation_token": "<token from step 1>"
   }
   ```

Results contain a typed event sequence (`tool.input.accepted`, `tool.started`,
zero or more `tool.output`, and `tool.finished`) plus a compact result object.
The exact code and its hash remain visible as structured input evidence. Python
exceptions and non-zero exits are ordinary `failed` results; timeout and output
limits are distinct statuses.

## Untrusted model plans

### Propose without executing

`POST /api/agent/plans/propose` is the plan-only entry point used by the
collapsed, default-off-in-use frontend Agent panel:

```json
{
  "goal": "Compare the relevant research and explain the result",
  "model": "localllm-fast",
  "enabled_capabilities": ["respond", "web_search", "paper_search"]
}
```

`goal` is trimmed and limited to 4,000 characters. `model` must match the safe
local model identifier grammar. The capability list is required, cannot contain
duplicates, and must include `respond`; the other possible entries are
`web_search`, `paper_search`, `vision`, and `python`. No conversation transcript
or recent-context field is accepted. This goal-only contract deliberately avoids
sending unrelated private session history to the planner.

The route asks `app.state.ollama` through `proxy_json("/api/chat", ...)` with:

- `stream: false` and `think: false`;
- temperature `0.0`;
- an 8,192-token context and at most 2,048 output tokens;
- Ollama's native `format: "json"`, with the exact bounded plan shape and
  capability-specific argument shapes in the system instruction; and
- a 30-second outer timeout.

Ollama 0.32.x cannot compile the original Pydantic discriminated-union schema
(`oneOf` plus `$defs`) into its grammar and rejects it before generation. Native
JSON mode avoids that runtime incompatibility. It is only a generation hint:
the strict Pydantic `AgentPlan` schema, graph checks, URL rejection, capability
allowlist, byte/depth/node limits, and deterministic fallback remain the
authoritative post-generation boundary.

The goal is JSON-encoded under the `untrusted_goal` key and never interpolated
into the system instruction. Model output is then passed through the same
bounded `AgentPlanCoordinator`. A valid response returns:

```json
{
  "planner": "local-model",
  "warning": null,
  "plan": {
    "schema_version": "1",
    "goal": "Explain the result",
    "steps": [
      {
        "id": "step_1",
        "capability": "respond",
        "objective": "Answer directly",
        "depends_on": [],
        "arguments": {}
      }
    ]
  },
  "steps": [
    {
      "id": "step_1",
      "capability": "respond",
      "state": "ready",
      "objective": "Answer directly",
      "depends_on": []
    }
  ],
  "events": [
    {
      "type": "plan.staged",
      "schema_version": "1",
      "step_count": 1,
      "capabilities": ["respond"]
    }
  ],
  "executable": false
}
```

An unavailable, slow, malformed, prompt-injected, URL-bearing, or
capability-violating planner response is not exposed as an error and never gets
partially executed. The route instead returns `planner:
"deterministic-fallback"`, a visible sanitized warning, and one static
respond-only step. Request cancellation is propagated rather than swallowed.

One narrow completion rule improves small-model reliability without broadening
execution authority. If the decoded model object is otherwise a strict,
URL-free plan with one to seven sequential non-`respond` steps and valid
backward-only dependencies, LocalLLM appends a server-authored final `respond`
step depending on the previous last step. The response remains `executable:
false` and includes a visible warning. This rule does not touch plans that
already contain `respond`, contain eight steps, use malformed types, include
unknown wrapper fields, contain URLs, or violate a capability allowlist; those
still use the deterministic fallback. Python remains separately confirmation
gated after completion.

The proposal endpoint only stages a plan. Even a valid `python` step is marked
`awaiting_explicit_confirmation`; it cannot bypass the separate code-bound
confirmation and execution exchange.

In the bundled UI, **Plan** shows this staged graph without running it. Answer,
web-search, paper-search, and vision objectives remain previews for the user to
handle through the normal Auto chat composer, which retains its ordinary search
and image boundaries. Only a displayed Python step can enter the separate
review/confirm/**Run** flow. Closing the panel or using normal chat never opts
into execution. When an execution result is deliberately appended to the
resumable transcript, its rendered Markdown is capped at 30,000 characters so
it remains below the conversation store's 32,000-character message limit; the
result records whether either the API capture or UI formatting was truncated.

### Validate supplied plan JSON

`POST /api/agent/plans/validate` accepts a JSON string produced by any model and
an explicit capability allowlist. The inner JSON is limited to 16 KiB, eight
steps, eight levels, and 256 total nodes. It rejects Markdown fences, URLs,
unknown fields/tool wrappers, oversized integers, floats, non-finite values,
duplicate or forward dependencies, disabled capabilities, and plans without
exactly one final `respond` step.

Supported step schemas are discriminated by `capability`:

- `respond` with no arguments;
- `web_search` or `paper_search` with a bounded query and result count;
- `vision` with opaque image IDs and a bounded question;
- `python` with bounded code and timeout.

The response always has `"executable": false`. Search/vision steps are merely
staged, and Python steps are marked `awaiting_explicit_confirmation`; no model
output crosses into a command line. A higher-level agent can dispatch allowed
read-only capabilities and must use the separate two-stage confirmation flow for
each Python step. This keeps behavior consistent across small and large models:
model quality affects plan usefulness, not the enforcement boundary.
