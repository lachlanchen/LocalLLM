# Local conversation history

LocalLLM Studio stores resumable Playground conversations in a project-local
SQLite database at `data/conversations.sqlite3`. The database, write-ahead log,
and shared-memory sidecar are ignored by Git and are created with private
permissions. The API does not send saved conversations to a hosted service.

This is an app management API, not an implementation of OpenAI's hosted
`conversation` or `previous_response_id` behavior. Like the other ungated
management routes, it relies on LocalLLM's loopback peer and browser origin
boundary; it does not consult `LOCALLLM_API_KEY`. Do not expose port 8008
without adding authentication and per-user authorization. The separately
mounted image job/output routes require that key except for image status.

## SQLite durability

The store uses a versioned schema (`PRAGMA user_version`), WAL journaling, a
five-second busy timeout, full synchronous commits, explicit write
transactions, and an index ordered by most recently updated conversation. A
conversation update, summary cursor, and summary text commit atomically.
Every ordinary update, compaction, and delete uses the monotonic `revision` as
an optimistic compare-and-swap guard, so an old browser tab cannot overwrite or
delete a concurrently changed conversation.

The complete message history remains in the database until the local delete
route is called. Context compaction adds a summary and advances a cursor; it
does not remove, rewrite, or hide older messages from the full-history route.

## API

| Endpoint | Behavior |
| --- | --- |
| `GET /api/conversations` | lists newest-first metadata, usage, and configured limits without returning message bodies |
| `POST /api/conversations` | creates a conversation; accepts optional `title`, `model`, `mode`, and `messages` |
| `GET /api/conversations/{id}` | returns the full resumable history and context summary |
| `PATCH /api/conversations/{id}` | requires `expected_revision` and atomically replaces any supplied `title`, `model`, `mode`, or complete `messages` array |
| `DELETE /api/conversations/{id}` | requires an `expected_revision` JSON body and atomically removes that exact revision |
| `POST /api/conversations/{id}/compact` | summarizes older turns while retaining a requested recent tail |

IDs have the fixed opaque form `conv_` followed by 32 lowercase hexadecimal
characters. Unknown and malformed IDs return HTTP 404. A quota failure returns
HTTP 507 without silently deleting another conversation. A compaction race
returns HTTP 409 so the caller can reload, merge, or save its unsaved turn as a
new conversation. The bundled UI takes the last option automatically, keeping
both versions instead of silently losing either tab's messages.

A stale delete also returns HTTP 409. The bundled UI does not retry deletion
against the newer revision: it reloads the latest record and asks the user to
review it before deliberately confirming again.

Update example:

```json
{
  "expected_revision": 7,
  "title": "Resumable chat",
  "messages": [
    {"role": "user", "content": "Continue from our prior design."}
  ]
}
```

Delete example:

```http
DELETE /api/conversations/conv_0123456789abcdef0123456789abcdef
Content-Type: application/json

{"expected_revision": 7}
```

Create example:

```json
{
  "model": "localllm-fast",
  "mode": "web",
  "messages": [
    {
      "id": "b4682997-6bed-4573-a8c0-5b458158c394",
      "role": "user",
      "content": "Compare these two methods.",
      "mode": "web"
    }
  ]
}
```

A full conversation has this shape:

```json
{
  "id": "conv_0123456789abcdef0123456789abcdef",
  "title": "Compare these two methods.",
  "model": "localllm-fast",
  "mode": "web",
  "created_at": "2026-08-09T12:00:00.000Z",
  "updated_at": "2026-08-09T12:01:00.000Z",
  "revision": 1,
  "summary": "",
  "summarized_message_count": 0,
  "summary_method": null,
  "message_count": 1,
  "messages": []
}
```

The abbreviated `messages` value above indicates the field location; the real
response returns every persisted message.

## Persisted message shape

Each message contains `role` (`system`, `user`, or `assistant`) and a Markdown
`content` string. It may also contain:

- a caller-generated safe `id`;
- a validated PNG, JPEG, or WebP base64 data URL in `image`;
- the resolved `model` and grounding `mode` used for that turn;
- up to 20 normalized evidence `sources` with title, HTTP(S) URL, snippet,
  provider, author, DOI, publication, score, query, and provenance metadata;
- a bounded `warning` shown with the restored answer.

Transient renderer state such as `pending` and `activity` is rejected rather
than saved. This prevents a process restart from restoring an answer as if it
were still streaming. Message Markdown is preserved rather than converted to
HTML; the browser applies its separately hardened Markdown renderer on reload.

## Bundled Playground behavior

The conversation drawer loads newest-first metadata and automatically resumes
the most recently updated chat. A user can start a clean chat, reopen an older
one, rename it, compact its model context, or delete it through a two-stage
confirmation. The transcript scrolls inside the viewport so the session drawer
and composer remain available even in a very long desktop or mobile chat. The
selected model and Auto/Local/Web/Papers/All mode are restored with the
transcript. User turns are right-aligned and model turns are left-aligned; this
presentation does not change the stored role.

The UI saves the user turn before inference and the finalized or stopped model
turn afterward. The composer and an independent send-time check enforce the
32,000-character message limit. If that first durable write is rejected—for
example by validation, the 400-message limit, image/archive quota, or local
storage failure—the model request never starts. The UI rolls back the optimistic
turn and restores the exact untrimmed draft, attachment, and prior transcript,
so the failed write cannot poison later saves or silently discard an image.

A stale revision produces HTTP 409. Title-only conflicts are
reloaded and retried against the new revision; a transcript conflict is kept as
a separate continued conversation so neither tab silently destroys the other
branch. These behaviors reduce accidental loss, but they are not a multi-user
merge protocol.

`POST /api/agent/chat` streams at most 30,000 visible assistant characters. If
the local model produces more, the service stops forwarding at that boundary,
emits a visible truncation warning and a normal `done` event, and the UI saves
the bounded answer plus warning. The cap deliberately leaves room below the
store's 32,000-character per-message limit. It does not change direct `/v1/*`
proxy behavior.

Display history and inference history are deliberately different. Before a
turn, the client takes unsummarized recent messages newest-first within a
40-message and 30,000-UTF-8-byte envelope. It includes at most four newest images
and 15 MiB decoded image data, preserving older attachments in SQLite while
replacing omitted inference images with a visible note. A nonempty summary may
use at most 35 percent of the text envelope. After 20 unsummarized messages, the
bundled UI starts a background compaction that normally retains the latest 12;
the explicit **Compact context** control is also available.

## Markdown, tables, and math

Assistant messages, vision answers, research reports, and local analysis use the
same constrained renderer. It supports CommonMark plus GFM tables, task lists,
strikethrough, headings, nested lists, blockquotes, inline code, fenced code,
and indented code. Wide tables sit in a keyboard-focusable horizontal scroll
region instead of overflowing the chat column.

KaTeX renders inline and display math written as `$...$`, `$$...$$`,
`\(...\)`, or `\[...\]`, with HTML and MathML output. Delimiters inside inline,
fenced, or indented code remain literal. KaTeX runs with `trust: false`.

Model text is not a navigation surface: raw HTML is skipped, Markdown links and
autolinks render as inert text, and Markdown images render only their useful alt
text without issuing a request. KaTeX commands cannot restore links or images.
Grounded answers expose separately validated external destinations through the
service-owned source cards.

## Context compaction

The compact request accepts an optional model and a recent-tail size:

```json
{
  "model": "localllm-pocket",
  "keep_recent": 12
}
```

`keep_recent` is between 4 and 50. The service asks the selected local Ollama
model to merge the previous summary with newly eligible turns. Conversation
text is enclosed as untrusted transcript data, thinking is disabled, output is
bounded, and an empty response is rejected. If Ollama is unavailable or rejects
the request, a bounded deterministic extractive summary samples every eligible
turn so continuation remains available. The response reports
`summary_method` as `model` or `extractive`.

To build the next inference context, prepend the nonempty `summary` as a clearly
labelled assistant memory record—not a privileged system instruction—and then send messages beginning at
`summarized_message_count`. The full database history must remain the source of
truth for display and export.

## Limits and privacy

- 200 saved conversations;
- 512 MiB of logical encoded conversation content across the archive;
- 25 MiB encoded per create/update request and per stored record;
- 400 persisted messages per conversation;
- 32,000 characters per message and 8,000,000 message-content characters total;
- 100 validated image turns, 8 MiB decoded per image and 18 MiB decoded total;
- 12,000 summary characters and 20 evidence sources per message.

Images use the same signature, dimension, pixel-count, animation, MIME, and
remote-URL rejection checks as grounded chat. These limits bound local storage
and parser work; they are not a promise that every allowed history fits the
selected model context. Use compaction before inference when a conversation
outgrows the model's useful prompt window.
