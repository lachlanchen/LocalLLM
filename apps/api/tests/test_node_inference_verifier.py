from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

import localllm.node_canary as node_canary
from localllm.node_canary import CANARY_ROLES, ROLE_ALIASES, verify_node_inference

RELEASE_ID = "release-abcdef12"
RESOLVED = {
    "text": "qwen3:8b-q4_K_M",
    "code": "qwen3-coder:30b-a3b-q4_K_M",
    "vision": "qwen3-vl:8b-instruct-q4_K_M",
    "embedding": "bge-m3:latest",
}
DIGESTS = {
    "text": "1" * 64,
    "code": "2" * 64,
    "vision": "3" * 64,
    "embedding": "4" * 64,
}
ANSWERS = {
    "text": "42",
    "code": "cba",
    "vision": "RED,GREEN,BLUE",
}


def model_list() -> dict[str, Any]:
    ids = [*RESOLVED.values(), *ROLE_ALIASES.values()]
    return {"object": "list", "data": [{"id": model_id} for model_id in ids]}


def capabilities(release_id: str = RELEASE_ID) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "service": {"name": "localllm-api", "version": "0.1.0", "release_id": release_id},
        "models": [
            {
                "id": RESOLVED[role],
                "aliases": [ROLE_ALIASES[role]],
                "digest": DIGESTS[role],
                "size_bytes": index + 1,
            }
            for index, role in enumerate(CANARY_ROLES)
        ],
    }


def role_for_alias(alias: str) -> str:
    return next(role for role, candidate in ROLE_ALIASES.items() if candidate == alias)


def successful_ollama_transport(
    requests: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/generate"
        assert request.url.host == "127.0.0.1"
        assert request.url.port == 11434
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        assert set(payload) == {"model", "keep_alive", "stream"}
        assert payload["keep_alive"] == 0
        assert payload["stream"] is False
        if requests is not None:
            requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "done": True,
                "done_reason": "unload",
            },
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_all_roles_execute_with_bounded_unload_controls_and_content_free_receipt() -> None:
    requests: list[dict[str, Any]] = []
    unload_requests: list[httpx.Request] = []
    capability_requests = 0
    private_key = "private-canary-key-never-return"

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal capability_requests
        assert request.headers["authorization"] == f"Bearer {private_key}"
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.method == "GET" and request.url.path == "/api/node/capabilities":
            capability_requests += 1
            return httpx.Response(200, json=capabilities())
        payload = json.loads(request.content)
        requests.append(payload)
        role = role_for_alias(payload["model"])
        assert "keep_alive" not in payload
        assert "think" not in payload
        assert "options" not in payload
        if role == "embedding":
            assert request.url.path == "/v1/embeddings"
            assert payload["input"] == ["a red circle", "a blue ocean wave"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": RESOLVED[role],
                    "data": [
                        {"embedding": [0.25] * 1024},
                        {"embedding": [0.25] * 1023 + [0.5]},
                    ],
                },
            )
        assert request.url.path == "/v1/chat/completions"
        assert payload["reasoning_effort"] == "none"
        assert payload["max_tokens"] == 32
        normalized_prompt = json.dumps(payload["messages"]).replace(" ", "").upper()
        assert ANSWERS[role].upper() not in normalized_prompt
        if role == "code":
            prompt = payload["messages"][0]["content"]
            assert '"".join(reversed("abc"))' in prompt
            assert "three lowercase letters" in prompt
            assert "no quotes" in prompt
        if role == "vision":
            image = payload["messages"][0]["content"][1]["image_url"]["url"]
            assert image.startswith("data:image/png;base64,")
            assert len(image) < 2048
        return httpx.Response(
            200,
            json={
                "model": RESOLVED[role],
                "choices": [{"message": {"content": ANSWERS[role]}}],
            },
        )

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        private_key,
        CANARY_ROLES,
        {role: 5 for role in CANARY_ROLES},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(unload_requests),
    )

    assert receipt["status"] == "passed"
    assert receipt["release_id"] == RELEASE_ID
    assert [item["role"] for item in receipt["roles"]] == list(CANARY_ROLES)
    assert [item["resolved_model"] for item in receipt["roles"]] == [
        RESOLVED[role] for role in CANARY_ROLES
    ]
    assert len(requests) == 4
    assert [json.loads(request.content)["model"] for request in unload_requests] == [
        RESOLVED[role] for role in CANARY_ROLES
    ]
    assert capability_requests == 8
    serialized = json.dumps(receipt)
    for forbidden in (
        private_key,
        "a red circle",
        "private response",
    ):
        assert forbidden not in serialized
    assert "embedding" in serialized
    assert "0.25" not in serialized


@pytest.mark.asyncio
async def test_http_failure_is_content_free_and_does_not_claim_provenance() -> None:
    secret = "SECRET_UPSTREAM_RESPONSE_MUST_NOT_ESCAPE"
    unload_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        return httpx.Response(500, text=secret)

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("text",),
        {"text": 5},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(unload_requests),
    )

    assert receipt["status"] == "failed"
    assert receipt["release_id"] == RELEASE_ID
    assert receipt["roles"][0]["status"] == "failed"
    assert receipt["roles"][0]["resolved_model"] is None
    assert receipt["roles"][0]["digest"] is None
    assert len(unload_requests) == 1
    assert json.loads(unload_requests[0].content)["model"] == RESOLVED["text"]
    assert secret not in json.dumps(receipt)


@pytest.mark.asyncio
async def test_vision_canary_requires_the_exact_undisclosed_band_sequence() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        return httpx.Response(
            200,
            json={
                "model": RESOLVED["vision"],
                "choices": [{"message": {"content": "RED,GREEN"}}],
            },
        )

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("vision",),
        {"vision": 5},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(),
    )

    assert receipt["status"] == "failed"
    assert receipt["roles"][0]["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "invalid_answer"),
    [
        ("text", "The answer is 42"),
        ("code", "cba is the result"),
        ("code", '"cba"'),
        ("code", "`cba`"),
    ],
)
async def test_text_and_code_canaries_require_exact_semantic_answers(
    role: str, invalid_answer: str
) -> None:
    unload_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        return httpx.Response(
            200,
            json={
                "model": RESOLVED[role],
                "choices": [{"message": {"content": invalid_answer}}],
            },
        )

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        (role,),
        {role: 5},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(unload_requests),
    )

    assert receipt["status"] == "failed"
    assert len(unload_requests) == 1
    assert json.loads(unload_requests[0].content)["model"] == RESOLVED[role]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_vector", [[0.25] * 3, [0.0] * 1024])
async def test_embedding_canary_requires_1024_finite_nonzero_values(
    invalid_vector: list[float],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        return httpx.Response(
            200,
            json={
                "model": RESOLVED["embedding"],
                "data": [
                    {"embedding": invalid_vector},
                    {"embedding": [0.5] * 1024},
                ],
            },
        )

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("embedding",),
        {"embedding": 5},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(),
    )

    assert receipt["status"] == "failed"


@pytest.mark.asyncio
async def test_embedding_canary_rejects_constant_nonzero_stub() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        constant_vector = [0.25] * 1024
        return httpx.Response(
            200,
            json={
                "model": RESOLVED["embedding"],
                "data": [
                    {"embedding": constant_vector},
                    {"embedding": constant_vector},
                ],
            },
        )

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("embedding",),
        {"embedding": 5},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(),
    )

    assert receipt["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("response_model", [None, "localllm-fast", RESOLVED["code"]])
async def test_chat_response_model_must_equal_preflight_resolution(
    response_model: str | None,
) -> None:
    capability_requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal capability_requests
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            capability_requests += 1
            return httpx.Response(200, json=capabilities())
        response = {"choices": [{"message": {"content": ANSWERS["text"]}}]}
        if response_model is not None:
            response["model"] = response_model
        return httpx.Response(200, json=response)

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("text",),
        {"text": 5},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(),
    )

    assert capability_requests == 2
    assert receipt["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("response_model", [None, "localllm-embed", RESOLVED["text"]])
async def test_embedding_response_model_must_equal_preflight_resolution(
    response_model: str | None,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        response: dict[str, Any] = {
            "data": [
                {"embedding": [0.25] * 1024},
                {"embedding": [0.25] * 1023 + [0.5]},
            ]
        }
        if response_model is not None:
            response["model"] = response_model
        return httpx.Response(200, json=response)

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("embedding",),
        {"embedding": 5},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(),
    )

    assert receipt["status"] == "failed"


@pytest.mark.asyncio
async def test_per_role_timeout_cancels_the_request_and_reports_only_timeout_status() -> None:
    request_cancelled = asyncio.Event()
    unload_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("text",),
        {"text": 1},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(unload_requests),
    )

    assert request_cancelled.is_set()
    assert receipt["status"] == "failed"
    assert receipt["release_id"] == RELEASE_ID
    assert receipt["roles"][0]["status"] == "timed_out"
    assert len(unload_requests) == 1
    assert json.loads(unload_requests[0].content)["model"] == RESOLVED["text"]
    assert 900 <= receipt["roles"][0]["latency_ms"] <= 1500


@pytest.mark.asyncio
async def test_external_cancellation_propagates_and_closes_the_inflight_request() -> None:
    started = asyncio.Event()
    request_cancelled = asyncio.Event()
    unload_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            request_cancelled.set()
            raise

    task = asyncio.create_task(
        verify_node_inference(
            "http://127.0.0.1:18008/v1",
            "private-key",
            ("text",),
            {"text": 30},
            transport=httpx.MockTransport(handler),
            ollama_transport=successful_ollama_transport(unload_requests),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert request_cancelled.is_set()
    assert len(unload_requests) == 1
    assert json.loads(unload_requests[0].content)["model"] == RESOLVED["text"]


@pytest.mark.asyncio
async def test_unload_http_failure_fails_closed_once_without_exposing_content() -> None:
    unload_requests: list[httpx.Request] = []
    secret = "SECRET_OLLAMA_UNLOAD_RESPONSE_MUST_NOT_ESCAPE"

    async def api_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        return httpx.Response(
            200,
            json={
                "model": RESOLVED["text"],
                "choices": [{"message": {"content": ANSWERS["text"]}}],
            },
        )

    async def ollama_handler(request: httpx.Request) -> httpx.Response:
        unload_requests.append(request)
        assert "authorization" not in request.headers
        assert json.loads(request.content) == {
            "model": RESOLVED["text"],
            "keep_alive": 0,
            "stream": False,
        }
        return httpx.Response(500, text=secret)

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("text",),
        {"text": 5},
        transport=httpx.MockTransport(api_handler),
        ollama_transport=httpx.MockTransport(ollama_handler),
    )

    assert receipt["status"] == "failed"
    assert receipt["roles"][0]["status"] == "failed"
    assert receipt["roles"][0]["resolved_model"] is None
    assert len(unload_requests) == 1
    assert secret not in json.dumps(receipt)


@pytest.mark.asyncio
async def test_cleanup_failure_fails_later_roles_without_dispatching_them() -> None:
    inference_roles: list[str] = []
    unload_models: list[str] = []
    secret = "SECRET_FIRST_CLEANUP_FAILURE"

    async def api_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        payload = json.loads(request.content)
        role = role_for_alias(payload["model"])
        inference_roles.append(role)
        return httpx.Response(
            200,
            json={
                "model": RESOLVED[role],
                "choices": [{"message": {"content": ANSWERS[role]}}],
            },
        )

    async def ollama_handler(request: httpx.Request) -> httpx.Response:
        unload_models.append(json.loads(request.content)["model"])
        return httpx.Response(500, text=secret)

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("text", "code"),
        {"text": 5, "code": 5},
        transport=httpx.MockTransport(api_handler),
        ollama_transport=httpx.MockTransport(ollama_handler),
    )

    assert receipt["status"] == "failed"
    assert [item["status"] for item in receipt["roles"]] == ["failed", "failed"]
    assert receipt["roles"][1]["latency_ms"] == 0
    assert inference_roles == ["text"]
    assert unload_models == [RESOLVED["text"]]
    assert secret not in json.dumps(receipt)


@pytest.mark.asyncio
async def test_stalled_cleanup_has_a_bounded_grace_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_cancelled = asyncio.Event()
    unload_requests: list[httpx.Request] = []
    monkeypatch.setattr(node_canary, "OLLAMA_UNLOAD_TIMEOUT_SECONDS", 0.02)

    async def api_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        return httpx.Response(
            200,
            json={
                "model": RESOLVED["text"],
                "choices": [{"message": {"content": ANSWERS["text"]}}],
            },
        )

    async def ollama_handler(request: httpx.Request) -> httpx.Response:
        unload_requests.append(request)
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_cancelled.set()
            raise

    started = asyncio.get_running_loop().time()
    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("text",),
        {"text": 5},
        transport=httpx.MockTransport(api_handler),
        ollama_transport=httpx.MockTransport(ollama_handler),
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert receipt["status"] == "failed"
    assert receipt["roles"][0]["status"] == "failed"
    assert cleanup_cancelled.is_set()
    assert len(unload_requests) == 1
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_child_cleanup_cancellation_is_a_cleanup_failure_not_parent_cancellation() -> None:
    unload_requests: list[httpx.Request] = []

    async def api_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        return httpx.Response(
            200,
            json={
                "model": RESOLVED["text"],
                "choices": [{"message": {"content": ANSWERS["text"]}}],
            },
        )

    async def ollama_handler(request: httpx.Request) -> httpx.Response:
        unload_requests.append(request)
        raise asyncio.CancelledError()

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("text",),
        {"text": 5},
        transport=httpx.MockTransport(api_handler),
        ollama_transport=httpx.MockTransport(ollama_handler),
    )

    assert receipt["status"] == "failed"
    assert receipt["roles"][0]["status"] == "failed"
    assert len(unload_requests) == 1


@pytest.mark.asyncio
async def test_external_cancellation_during_unload_is_shielded_and_one_shot() -> None:
    cleanup_started = asyncio.Event()
    permit_cleanup = asyncio.Event()
    cleanup_completed = asyncio.Event()
    unload_requests: list[httpx.Request] = []

    async def api_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            return httpx.Response(200, json=capabilities())
        return httpx.Response(
            200,
            json={
                "model": RESOLVED["text"],
                "choices": [{"message": {"content": ANSWERS["text"]}}],
            },
        )

    async def ollama_handler(request: httpx.Request) -> httpx.Response:
        unload_requests.append(request)
        cleanup_started.set()
        await permit_cleanup.wait()
        cleanup_completed.set()
        return httpx.Response(
            200,
            json={
                "model": RESOLVED["text"],
                "done": True,
                "done_reason": "unload",
            },
        )

    task = asyncio.create_task(
        verify_node_inference(
            "http://127.0.0.1:18008/v1",
            "private-key",
            ("text",),
            {"text": 5},
            transport=httpx.MockTransport(api_handler),
            ollama_transport=httpx.MockTransport(ollama_handler),
        )
    )
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    permit_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleanup_completed.is_set()
    assert len(unload_requests) == 1


@pytest.mark.asyncio
async def test_alias_release_or_digest_mismatch_fails_closed() -> None:
    scenarios = ("missing-alias", "old-schema", "missing-digest")
    unload_requests: list[httpx.Request] = []
    for scenario in scenarios:

        async def handler(request: httpx.Request, current: str = scenario) -> httpx.Response:
            if request.url.path == "/v1/models":
                payload = model_list()
                if current == "missing-alias":
                    payload["data"] = [
                        item for item in payload["data"] if item["id"] != "localllm-fast"
                    ]
                return httpx.Response(200, json=payload)
            if request.url.path == "/api/node/capabilities":
                payload = capabilities()
                if current == "old-schema":
                    payload["schema_version"] = 1
                if current == "missing-digest":
                    payload["models"][0]["digest"] = None
                return httpx.Response(200, json=payload)
            raise AssertionError("inference must not run after failed inventory validation")

        receipt = await verify_node_inference(
            "http://127.0.0.1:18008/v1",
            "private-key",
            ("text",),
            {"text": 5},
            transport=httpx.MockTransport(handler),
            ollama_transport=successful_ollama_transport(unload_requests),
        )
        assert receipt["status"] == "failed"
        assert receipt["roles"][0]["status"] == "failed"
    assert unload_requests == []


@pytest.mark.asyncio
async def test_release_change_during_multi_role_canary_invalidates_all_results() -> None:
    capability_calls = 0
    unload_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal capability_calls
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            capability_calls += 1
            return httpx.Response(
                200,
                json=capabilities("release-one" if capability_calls <= 2 else "release-two"),
            )
        payload = json.loads(request.content)
        role = role_for_alias(payload["model"])
        return httpx.Response(
            200,
            json={
                "model": RESOLVED[role],
                "choices": [{"message": {"content": ANSWERS[role]}}],
            },
        )

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("text", "code"),
        {"text": 5, "code": 5},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(unload_requests),
    )

    assert receipt["status"] == "failed"
    assert receipt["release_id"] == "unknown"
    assert all(item["status"] == "failed" for item in receipt["roles"])
    assert [json.loads(request.content)["model"] for request in unload_requests] == [
        RESOLVED["text"],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["resolved_model", "digest", "release_id"])
async def test_postflight_provenance_change_fails_the_completed_role(
    changed_field: str,
) -> None:
    capability_calls = 0
    unload_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal capability_calls
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=model_list())
        if request.url.path == "/api/node/capabilities":
            capability_calls += 1
            payload = capabilities()
            if capability_calls == 2:
                if changed_field == "resolved_model":
                    payload["models"][0]["id"] = RESOLVED["code"]
                elif changed_field == "digest":
                    payload["models"][0]["digest"] = "9" * 64
                else:
                    payload["service"]["release_id"] = "release-changed"
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200,
            json={
                "model": RESOLVED["text"],
                "choices": [{"message": {"content": ANSWERS["text"]}}],
            },
        )

    receipt = await verify_node_inference(
        "http://127.0.0.1:18008/v1",
        "private-key",
        ("text",),
        {"text": 5},
        transport=httpx.MockTransport(handler),
        ollama_transport=successful_ollama_transport(unload_requests),
    )

    assert capability_calls == 2
    assert receipt["status"] == "failed"
    assert receipt["roles"][0]["status"] == "failed"
    assert receipt["roles"][0]["resolved_model"] is None
    assert len(unload_requests) == 1
    assert json.loads(unload_requests[0].content)["model"] == RESOLVED["text"]
