from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from localllm.main import app
from localllm.qwen_tool_repair import (
    chat_completion_sse,
    repair_qwen_chat_completion,
    should_buffer_qwen_tool_stream,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "integer"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    }
]


def request_payload(*, stream: bool = True) -> dict[str, Any]:
    return {
        "model": "localllm-code",
        "messages": [{"role": "user", "content": "Write the file"}],
        "tools": TOOLS,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }


def malformed_completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl_repair",
        "object": "chat.completion",
        "created": 123,
        "model": "qwen3-coder:30b-a3b-q4_K_M",
        "system_fingerprint": "fp_ollama",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_repairs_missing_tool_opener_and_preserves_intentional_newline() -> None:
    completion = malformed_completion(
        "I will write it.\n\n"
        "<function=write>\n"
        "<parameter=path>\nresult.txt\n</parameter>\n"
        "<parameter=content>\nPI_OK\n\n</parameter>\n"
        "<parameter=mode>\n600\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    repaired, applied = repair_qwen_chat_completion(request_payload(), completion)

    assert applied is True
    choice = repaired["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "I will write it."
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "write"
    assert json.loads(call["function"]["arguments"]) == {
        "path": "result.txt",
        "content": "PI_OK\n",
        "mode": 600,
    }


def test_repairs_standard_opener_and_multiple_calls() -> None:
    completion = malformed_completion(
        "<tool_call>\n<function=write>\n"
        "<parameter=path>\na.txt\n</parameter>\n"
        "<parameter=content>\na\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=write>\n"
        "<parameter=path>\nb.txt\n</parameter>\n"
        "<parameter=content>\nb\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    repaired, applied = repair_qwen_chat_completion(request_payload(), completion)

    assert applied is True
    calls = repaired["choices"][0]["message"]["tool_calls"]
    assert [json.loads(call["function"]["arguments"])["path"] for call in calls] == [
        "a.txt",
        "b.txt",
    ]


def test_repair_is_fail_closed_for_unknown_missing_malformed_or_suffix_content() -> None:
    cases = [
        "<function=delete><parameter=path>x</parameter></function></tool_call>",
        "<function=write><parameter=path>x</parameter></function></tool_call>",
        (
            "<function=write><parameter=path>x</parameter>"
            "<parameter=content>y</parameter><parameter=extra>z</parameter>"
            "</function></tool_call>"
        ),
        (
            "<function=write><parameter=path>x</parameter>"
            "<parameter=content>y</parameter></function></tool_call> trailing"
        ),
        "<function=write><parameter=path>x</parameter><parameter=content>y",
    ]

    for content in cases:
        completion = malformed_completion(content)
        repaired, applied = repair_qwen_chat_completion(request_payload(), completion)
        assert applied is False
        assert repaired == completion


def test_native_tool_calls_and_other_models_are_not_modified() -> None:
    native = malformed_completion("")
    native["choices"][0]["finish_reason"] = "tool_calls"
    native["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call_native",
            "type": "function",
            "function": {"name": "write", "arguments": "{}"},
        }
    ]
    repaired, applied = repair_qwen_chat_completion(request_payload(), native)
    assert applied is False
    assert repaired == native

    other = request_payload()
    other["model"] = "localllm-deep"
    repaired, applied = repair_qwen_chat_completion(other, malformed_completion("plain"))
    assert applied is False
    assert repaired == malformed_completion("plain")


def test_sse_projection_preserves_tool_call_terminal_reason_and_usage() -> None:
    repaired, applied = repair_qwen_chat_completion(
        request_payload(),
        malformed_completion(
            "<function=write><parameter=path>x</parameter>"
            "<parameter=content>y</parameter></function></tool_call>"
        ),
    )
    assert applied is True

    events = list(chat_completion_sse(repaired, include_usage=True))
    payloads = [
        json.loads(event.removeprefix(b"data: ").strip())
        for event in events[:-1]
    ]
    assert payloads[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "write"
    assert payloads[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[2]["choices"] == []
    assert payloads[2]["usage"]["total_tokens"] == 15
    assert events[-1] == b"data: [DONE]\n\n"


class RepairOllama:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def proxy_json(self, endpoint: str, payload: dict[str, Any]) -> httpx.Response:
        self.calls.append((endpoint, payload))
        request = httpx.Request("POST", f"http://ollama.test{endpoint}")
        return httpx.Response(
            200,
            json=malformed_completion(
                "<function=write>\n"
                "<parameter=path>\nendpoint.txt\n</parameter>\n"
                "<parameter=content>\nENDPOINT_OK\n\n</parameter>\n"
                "</function>\n</tool_call>"
            ),
            request=request,
        )

    async def proxy_stream(self, endpoint: str, payload: dict[str, Any]) -> Any:
        raise AssertionError("Qwen tool streams must use one buffered upstream generation")


def test_endpoint_buffers_one_generation_and_returns_repaired_sse() -> None:
    fake = RepairOllama()
    with TestClient(app) as client:
        client.app.state.ollama = fake
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer local-dev-key"},
            json=request_payload(),
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-localllm-tool-repair"] == "applied"
    assert len(fake.calls) == 1
    endpoint, upstream_payload = fake.calls[0]
    assert endpoint == "/v1/chat/completions"
    assert upstream_payload["stream"] is False
    assert "stream_options" not in upstream_payload

    data = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    call = data[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["function"]["name"] == "write"
    assert json.loads(call["function"]["arguments"])["content"] == "ENDPOINT_OK\n"
    assert data[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert data[2]["usage"]["total_tokens"] == 15
    assert response.text.endswith("data: [DONE]\n\n")


def test_buffer_lane_is_scoped_to_streamed_qwen_tool_requests() -> None:
    assert should_buffer_qwen_tool_stream("/v1/chat/completions", request_payload()) is True
    assert (
        should_buffer_qwen_tool_stream(
            "/v1/chat/completions", {**request_payload(), "model": "localllm-deep"}
        )
        is False
    )
    assert (
        should_buffer_qwen_tool_stream(
            "/v1/chat/completions", {**request_payload(), "tools": []}
        )
        is False
    )
    assert should_buffer_qwen_tool_stream("/v1/responses", request_payload()) is False
