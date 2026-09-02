#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from openai import OpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_fixture",
            "description": "Write exact text to an in-memory fixture",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_fixture",
            "description": "Read exact text from an in-memory fixture",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


def streamed_turn(
    client: OpenAI, model: str, messages: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        tools=TOOLS,  # type: ignore[arg-type]
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    text: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, int] = {}
    for chunk in stream:
        if chunk.usage is not None:
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            text.append(delta.content)
        for call in delta.tool_calls or []:
            current = calls.setdefault(
                call.index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if call.id:
                current["id"] = call.id
            if call.function is not None:
                if call.function.name:
                    current["function"]["name"] += call.function.name
                if call.function.arguments:
                    current["function"]["arguments"] += call.function.arguments
    return "".join(text), [calls[index] for index in sorted(calls)], usage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a streamed LocalLLM write/read/final tool loop without host writes"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8008/v1")
    parser.add_argument("--api-key", default=os.environ.get("LOCALLLM_API_KEY", "local-dev-key"))
    parser.add_argument("--model", default="localllm-code")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=120.0)
    models = {model.id for model in client.models.list().data}
    assert args.model in models, f"Missing coding model or alias: {args.model}"

    expected_path = "agent-loop.txt"
    expected_content = "LOCAL_AGENT_TOOL_LOOP_OK\n"
    expected_answer = "VERIFIED_LOCAL_AGENT_TOOL_LOOP_OK"
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Use write_fixture to store the line LOCAL_AGENT_TOOL_LOOP_OK followed by "
                f"exactly one newline at {expected_path!r}. "
                "Then use read_fixture to verify it. Only after the read matches, answer exactly "
                f"{expected_answer}."
            ),
        }
    ]
    fixture: dict[str, str] = {}
    calls_seen: list[str] = []
    token_usage: list[dict[str, int]] = []
    started = time.perf_counter()

    for _turn in range(6):
        content, calls, usage = streamed_turn(client, args.model, messages)
        if usage:
            token_usage.append(usage)
        if not calls:
            assert content.strip() == expected_answer, f"Unexpected final answer: {content!r}"
            assert fixture.get(expected_path) == expected_content
            assert calls_seen == ["write_fixture", "read_fixture"]
            print(
                json.dumps(
                    {
                        "ok": True,
                        "model": args.model,
                        "calls": calls_seen,
                        "turns": len(token_usage),
                        "elapsed_s": round(time.perf_counter() - started, 3),
                        "usage": token_usage,
                    },
                    indent=2,
                )
            )
            return 0

        messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"])
            if name == "write_fixture":
                assert set(arguments) == {"path", "content"}
                assert arguments["path"] == expected_path
                assert arguments["content"] == expected_content, repr(arguments["content"])
                fixture[arguments["path"]] = arguments["content"]
                result = {"ok": True}
            elif name == "read_fixture":
                assert set(arguments) == {"path"}
                assert arguments["path"] == expected_path
                assert fixture.get(arguments["path"]) == expected_content
                result = {"content": fixture[arguments["path"]]}
            else:
                raise AssertionError(f"Unexpected tool call: {name}")
            calls_seen.append(name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, separators=(",", ":")),
                }
            )

    raise AssertionError("Tool loop exceeded six turns")


if __name__ == "__main__":
    raise SystemExit(main())
