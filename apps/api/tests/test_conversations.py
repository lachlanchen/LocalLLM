from __future__ import annotations

import base64
import os
import sqlite3
import struct
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import localllm.conversations as conversation_module
from localllm.conversations import (
    MAX_CONVERSATION_MESSAGES,
    ConversationCapacityError,
    ConversationConflictError,
    ConversationCreate,
    ConversationMessage,
    ConversationStore,
    ConversationUpdate,
    deterministic_summary,
)
from localllm.main import app


def png_data_url() -> str:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload).to_bytes(4, "big")
        return len(payload).to_bytes(4, "big") + kind + payload + checksum

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def messages(count: int = 2) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index in range(count):
        values.append(
            {
                "id": f"message_{index}",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"message {index}",
                "model": "localllm-fast" if index % 2 else None,
                "mode": "web" if index % 2 else "local",
            }
        )
    return [{key: value for key, value in item.items() if value is not None} for item in values]


def make_store(tmp_path: Path) -> ConversationStore:
    return ConversationStore(tmp_path / "data")


def test_sqlite_store_round_trips_ui_history_and_reopens(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    source = {
        "index": 1,
        "title": "Primary evidence",
        "url": "https://example.org/paper",
        "snippet": "Supported result",
        "provider": "crossref",
        "providers": ["crossref", "semantic_scholar"],
        "kind": "paper",
        "authors": ["Ada Researcher"],
        "year": 2026,
        "doi": "10.1234/example",
        "citation_count": 7,
        "score": 3.5,
        "query": "supported result",
        "provenance": [
            {
                "provider": "crossref",
                "query": "supported result",
                "record_id": "10.1234/example",
                "retrieved_at": "2026-08-09T00:00:00Z",
                "provider_rank": 1,
            }
        ],
    }
    created = store.create(
        ConversationCreate.model_validate(
            {
                "model": "localllm-fast",
                "mode": "web",
                "messages": [
                    {
                        "id": "user_1",
                        "role": "user",
                        "content": "  Preserve this Markdown:\n\n    x = 1  ",
                        "image": png_data_url(),
                        "mode": "web",
                    },
                    {
                        "id": "assistant_1",
                        "role": "assistant",
                        "content": "| A | B |\n|---|---|\n| 1 | 2 |",
                        "model": "localllm-fast",
                        "mode": "web",
                        "sources": [source],
                        "warning": "One provider was unavailable.",
                    },
                ],
            }
        )
    )

    assert created["id"].startswith("conv_")
    assert created["title"].startswith("Preserve this Markdown")
    assert created["messages"][0]["content"] == "  Preserve this Markdown:\n\n    x = 1  "
    assert created["messages"][1]["sources"][0]["doi"] == "10.1234/example"
    assert created["messages"][1]["sources"][0]["index"] == 1
    assert created["messages"][1]["sources"][0]["provenance"][0]["provider_rank"] == 1

    reopened = ConversationStore(store.data_dir)
    restored = reopened.get(created["id"])
    listing = reopened.list()

    assert restored == created
    assert listing["conversations"][0]["id"] == created["id"]
    assert "messages" not in listing["conversations"][0]
    assert listing["usage"]["conversation_count"] == 1
    assert listing["limits"]["max_messages"] == MAX_CONVERSATION_MESSAGES
    assert oct(os.stat(store.path).st_mode & 0o777) == "0o600"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        indexes = connection.execute("PRAGMA index_list(conversations)").fetchall()
    assert any(row[1] == "conversations_updated_at_idx" for row in indexes)


def test_schema_migration_adds_revision_transactionally(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "conversations.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                summarized_message_count INTEGER NOT NULL DEFAULT 0,
                summary_method TEXT,
                messages_json TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                encoded_bytes INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX conversations_updated_at_idx ON conversations(updated_at DESC, id DESC)"
        )
        connection.execute("PRAGMA user_version = 1")

    ConversationStore(data_dir)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        columns = {row[1] for row in connection.execute("PRAGMA table_info(conversations)")}
    assert "revision" in columns


def test_update_preserves_summary_for_append_and_resets_if_compacted_prefix_changes(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    created = store.create(ConversationCreate(messages=messages(8)))
    summarized = store.apply_summary(
        created["id"],
        expected_revision=created["revision"],
        summary="Earlier goal and decision.",
        summarized_message_count=4,
        method="extractive",
    )
    assert summarized is not None

    appended_messages = messages(10)
    appended = store.update(
        created["id"],
        ConversationUpdate(expected_revision=summarized["revision"], messages=appended_messages),
    )
    assert appended is not None
    assert appended["summary"] == "Earlier goal and decision."
    assert appended["summarized_message_count"] == 4

    changed_messages = messages(10)
    changed_messages[0]["content"] = "changed old turn"
    changed = store.update(
        created["id"],
        ConversationUpdate(expected_revision=appended["revision"], messages=changed_messages),
    )
    assert changed is not None
    assert changed["summary"] == ""
    assert changed["summarized_message_count"] == 0
    assert changed["summary_method"] is None


def test_summary_compare_and_swap_rejects_stale_compaction(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    created = store.create(ConversationCreate(messages=messages(8)))
    store.update(
        created["id"],
        ConversationUpdate(expected_revision=created["revision"], title="New title"),
    )

    with pytest.raises(ConversationConflictError, match="changed during compaction"):
        store.apply_summary(
            created["id"],
            expected_revision=created["revision"],
            summary="stale",
            summarized_message_count=4,
            method="model",
        )


def test_conversation_count_quota_fails_closed_without_deleting_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(tmp_path)
    first = store.create(ConversationCreate(messages=messages()))
    monkeypatch.setattr(conversation_module, "MAX_CONVERSATIONS", 1)

    with pytest.raises(ConversationCapacityError, match="item limit"):
        store.create(ConversationCreate(messages=messages()))

    assert store.get(first["id"]) is not None
    assert store.list()["usage"]["conversation_count"] == 1


def test_conversation_byte_quota_fails_closed_on_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(tmp_path)
    created = store.create(ConversationCreate(messages=messages()))
    original = store.get(created["id"])
    monkeypatch.setattr(conversation_module, "MAX_CONVERSATION_ARCHIVE_BYTES", 32)

    with pytest.raises(ConversationCapacityError, match="byte limit"):
        store.update(
            created["id"],
            ConversationUpdate(
                expected_revision=created["revision"],
                messages=[{"role": "user", "content": "larger update"}],
            ),
        )

    assert store.get(created["id"]) == original


def test_message_schema_rejects_transient_ui_state_and_unsafe_image() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ConversationMessage.model_validate(
            {"role": "assistant", "content": "pending", "pending": True}
        )
    with pytest.raises(ValidationError, match="image"):
        ConversationMessage.model_validate(
            {"role": "user", "content": "remote", "image": "https://example.org/image.png"}
        )


def test_long_conversation_allows_four_hundred_bounded_messages() -> None:
    accepted = ConversationCreate(messages=messages(MAX_CONVERSATION_MESSAGES))
    assert len(accepted.messages) == MAX_CONVERSATION_MESSAGES
    with pytest.raises(ValidationError, match="too_long"):
        ConversationCreate(messages=messages(MAX_CONVERSATION_MESSAGES + 1))


def test_durable_history_accepts_more_images_than_one_inference_request() -> None:
    history = ConversationCreate(
        messages=[
            {
                "id": f"image_{index}",
                "role": "user",
                "content": f"image turn {index}",
                "image": png_data_url(),
            }
            for index in range(5)
        ]
    )
    assert len(history.messages) == 5


def test_deterministic_summary_is_bounded_and_samples_every_turn() -> None:
    turns = messages(MAX_CONVERSATION_MESSAGES)
    summary = deterministic_summary("old summary", turns)

    assert summary.startswith("Conversation summary (extractive fallback):")
    assert len(summary) <= conversation_module.MAX_SUMMARY_CHARS
    assert summary.count("\n[") == MAX_CONVERSATION_MESSAGES


class SummaryOllama:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def proxy_json(self, endpoint: str, payload: dict[str, Any]) -> httpx.Response:
        self.calls.append((endpoint, payload))
        request = httpx.Request("POST", f"http://ollama.test{endpoint}")
        if self.fail:
            return httpx.Response(503, json={"error": "offline"}, request=request)
        return httpx.Response(
            200,
            json={"message": {"content": "Semantic summary with the active goal."}},
            request=request,
        )


@pytest.mark.parametrize(
    ("fail", "expected_method"),
    [(False, "model"), (True, "extractive")],
)
def test_conversation_api_crud_and_compaction(
    tmp_path: Path, fail: bool, expected_method: str
) -> None:
    store = make_store(tmp_path)
    fake_ollama = SummaryOllama(fail=fail)
    with TestClient(app) as client:
        client.app.state.conversations = store
        client.app.state.ollama = fake_ollama
        created = client.post(
            "/api/conversations",
            json={"model": "localllm-fast", "mode": "local", "messages": messages(20)},
        )
        assert created.status_code == 201
        conversation_id = created.json()["id"]

        listed = client.get("/api/conversations")
        fetched = client.get(f"/api/conversations/{conversation_id}")
        patched = client.patch(
            f"/api/conversations/{conversation_id}",
            json={
                "expected_revision": created.json()["revision"],
                "title": "Resumable chat",
            },
        )
        compacted = client.post(
            f"/api/conversations/{conversation_id}/compact",
            json={"model": "localllm-pocket", "keep_recent": 6},
        )

        assert listed.status_code == 200
        assert fetched.status_code == 200
        assert patched.status_code == 200
        assert patched.json()["title"] == "Resumable chat"
        assert compacted.status_code == 200
        assert compacted.json()["compacted"] is True
        assert compacted.json()["summary_method"] == expected_method
        assert compacted.json()["conversation"]["summarized_message_count"] == 14
        assert len(compacted.json()["conversation"]["messages"]) == 20
        if fail:
            assert compacted.json()["conversation"]["summary"].startswith(
                "Conversation summary (extractive fallback):"
            )
        else:
            assert compacted.json()["conversation"]["summary"].startswith("Semantic summary")
            assert fake_ollama.calls[0][0] == "/api/chat"
            assert fake_ollama.calls[0][1]["think"] is False

        deleted = client.request(
            "DELETE",
            f"/api/conversations/{conversation_id}",
            json={"expected_revision": compacted.json()["conversation"]["revision"]},
        )
        missing = client.get(f"/api/conversations/{conversation_id}")

    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "id": conversation_id}
    assert missing.status_code == 404


def test_conversation_api_rejects_invalid_ids_transient_fields_and_large_declared_body(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    with TestClient(app) as client:
        client.app.state.conversations = store
        invalid = client.get("/api/conversations/not-safe")
        transient = client.post(
            "/api/conversations",
            json={
                "messages": [
                    {"role": "assistant", "content": "still running", "activity": ["work"]}
                ]
            },
        )
        oversized = client.patch(
            "/api/conversations/conv_00000000000000000000000000000000",
            headers={"Content-Type": "application/json", "Content-Length": str(26 * 1024 * 1024)},
            content=b"{}",
        )

    assert invalid.status_code == 404
    assert transient.status_code == 422
    assert oversized.status_code == 413
    assert oversized.json()["detail"] == "Request body exceeds the endpoint size limit"


def test_empty_patch_is_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    created = store.create(ConversationCreate())
    with TestClient(app) as client:
        client.app.state.conversations = store
        response = client.patch(f"/api/conversations/{created['id']}", json={})

    assert response.status_code == 422


def test_stale_patch_is_rejected_without_losing_newer_turns(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    created = store.create(ConversationCreate(messages=messages(1)))

    first = store.update(
        created["id"],
        ConversationUpdate(
            expected_revision=created["revision"],
            messages=[*messages(1), {"role": "assistant", "content": "tab A answer"}],
        ),
    )
    assert first is not None

    with pytest.raises(ConversationConflictError, match="reload"):
        store.update(
            created["id"],
            ConversationUpdate(
                expected_revision=created["revision"],
                messages=[*messages(1), {"role": "assistant", "content": "tab B answer"}],
            ),
        )

    restored = store.get(created["id"])
    assert restored is not None
    assert [item["content"] for item in restored["messages"]] == [
        "message 0",
        "tab A answer",
    ]


def test_stale_tab_delete_is_rejected_without_losing_newer_turns(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    created = store.create(ConversationCreate(messages=messages(1)))
    newer_messages = [*messages(1), {"role": "assistant", "content": "newer tab answer"}]
    updated = store.update(
        created["id"],
        ConversationUpdate(
            expected_revision=created["revision"],
            messages=newer_messages,
        ),
    )
    assert updated is not None

    with TestClient(app) as client:
        client.app.state.conversations = store
        stale_delete = client.request(
            "DELETE",
            f"/api/conversations/{created['id']}",
            json={"expected_revision": created["revision"]},
        )
        restored = client.get(f"/api/conversations/{created['id']}")
        current_delete = client.request(
            "DELETE",
            f"/api/conversations/{created['id']}",
            json={"expected_revision": updated["revision"]},
        )

    assert stale_delete.status_code == 409
    assert "reload" in stale_delete.json()["detail"]
    assert restored.status_code == 200
    assert restored.json()["revision"] == updated["revision"]
    assert restored.json()["messages"] == updated["messages"]
    assert current_delete.status_code == 200


def test_delete_requires_exact_revision_body(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    created = store.create(ConversationCreate())
    with TestClient(app) as client:
        client.app.state.conversations = store
        missing_revision = client.request("DELETE", f"/api/conversations/{created['id']}", json={})
        extra_field = client.request(
            "DELETE",
            f"/api/conversations/{created['id']}",
            json={"expected_revision": created["revision"], "force": True},
        )

    assert missing_revision.status_code == 422
    assert extra_field.status_code == 422
    assert store.get(created["id"]) is not None
