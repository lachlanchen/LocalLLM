from __future__ import annotations

import copy
import json
import math
import re
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

from .catalog import resolve_model

QWEN3_CODER_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
_FUNCTION_PREFIX = "<function="
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_FUNCTION_CLOSE = "</function>"
_PARAMETER_CLOSE = "</parameter>"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
_MAX_REPAIR_CONTENT_CHARS = 4 * 1024 * 1024
_MAX_CALLS = 16
_MAX_ARGUMENT_CHARS = 1024 * 1024


def should_buffer_qwen_tool_stream(endpoint: str, payload: Mapping[str, Any]) -> bool:
    """Return whether one streamed request needs the bounded Qwen repair lane.

    Ollama's Qwen3-Coder renderer asks the model to open calls with
    ``<tool_call>``, but the model can begin directly with ``<function=...>``.
    Ollama 0.32.6 then exposes the call as ordinary text. Buffering only this
    model/tool combination lets us repair that documented omission without
    changing normal chat streaming or interpreting text from other models.
    """

    if endpoint != "/v1/chat/completions" or not payload.get("stream"):
        return False
    return _is_qwen_tool_request(payload)


def repair_qwen_chat_completion(
    request_payload: Mapping[str, Any], completion: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Promote complete, declared Qwen3-Coder XML calls to OpenAI tool calls.

    The operation is deliberately fail-closed. Unknown tools or parameters,
    missing required values, malformed/truncated markup, schema mismatches, and
    suffix text all leave the upstream response untouched.
    """

    copied = copy.deepcopy(dict(completion))
    if not _is_qwen_tool_request(request_payload):
        return copied, False
    tool_schemas = _declared_tool_schemas(request_payload.get("tools"))
    if not tool_schemas:
        return copied, False
    choices = copied.get("choices")
    if not isinstance(choices, list):
        return copied, False

    repaired = False
    for choice in choices:
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            continue
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("tool_calls"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        parsed = _parse_complete_calls(content, tool_schemas)
        if parsed is None:
            continue
        prefix, calls = parsed
        message["content"] = prefix
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "index": index,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
            for index, (name, arguments) in enumerate(calls)
        ]
        choice["finish_reason"] = "tool_calls"
        repaired = True
    return copied, repaired


def chat_completion_sse(
    completion: Mapping[str, Any], *, include_usage: bool
) -> Iterator[bytes]:
    """Render one buffered Chat Completion as a compact OpenAI SSE stream."""

    common = {
        key: completion[key]
        for key in ("id", "created", "model", "system_fingerprint")
        if key in completion
    }
    common["object"] = "chat.completion.chunk"
    choices = completion.get("choices")
    if not isinstance(choices, list):
        choices = []

    opening_choices: list[dict[str, Any]] = []
    closing_choices: list[dict[str, Any]] = []
    for fallback_index, raw_choice in enumerate(choices):
        if not isinstance(raw_choice, Mapping):
            continue
        index = raw_choice.get("index", fallback_index)
        message = raw_choice.get("message")
        delta: dict[str, Any] = {}
        if isinstance(message, Mapping):
            role = message.get("role")
            if isinstance(role, str):
                delta["role"] = role
            content = message.get("content")
            if isinstance(content, str) and content:
                delta["content"] = content
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                delta["tool_calls"] = tool_calls
        opening_choices.append({"index": index, "delta": delta, "finish_reason": None})
        closing_choices.append(
            {
                "index": index,
                "delta": {},
                "finish_reason": raw_choice.get("finish_reason", "stop"),
            }
        )

    if opening_choices:
        yield _sse_json({**common, "choices": opening_choices})
        yield _sse_json({**common, "choices": closing_choices})
    if include_usage and isinstance(completion.get("usage"), Mapping):
        yield _sse_json({**common, "choices": [], "usage": completion["usage"]})
    yield b"data: [DONE]\n\n"


def _sse_json(payload: Mapping[str, Any]) -> bytes:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}\n\n".encode()


def _is_qwen_tool_request(payload: Mapping[str, Any]) -> bool:
    model = payload.get("model")
    tools = payload.get("tools")
    return (
        isinstance(model, str)
        and resolve_model(model) == QWEN3_CODER_MODEL
        and isinstance(tools, list)
        and bool(tools)
    )


def _declared_tool_schemas(raw_tools: object) -> dict[str, dict[str, Any]] | None:
    if not isinstance(raw_tools, list) or not raw_tools:
        return None
    schemas: dict[str, dict[str, Any]] = {}
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping) or raw_tool.get("type") != "function":
            return None
        function = raw_tool.get("function")
        if not isinstance(function, Mapping):
            return None
        name = function.get("name")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or name in schemas
            or not isinstance(parameters, dict)
        ):
            return None
        schemas[name] = parameters
    return schemas


def _parse_complete_calls(
    content: str, tool_schemas: Mapping[str, dict[str, Any]]
) -> tuple[str, list[tuple[str, dict[str, Any]]]] | None:
    if len(content) > _MAX_REPAIR_CONTENT_CHARS:
        return None
    marker_positions = [
        position
        for marker in (_TOOL_OPEN, _FUNCTION_PREFIX)
        if (position := content.find(marker)) >= 0
    ]
    if not marker_positions:
        return None
    first = min(marker_positions)
    prefix = content[:first].rstrip()
    cursor = first
    calls: list[tuple[str, dict[str, Any]]] = []

    while cursor < len(content):
        cursor = _skip_whitespace(content, cursor)
        if content.startswith(_TOOL_OPEN, cursor):
            cursor += len(_TOOL_OPEN)
            cursor = _skip_whitespace(content, cursor)
        if not content.startswith(_FUNCTION_PREFIX, cursor):
            return None
        name_end = content.find(">", cursor + len(_FUNCTION_PREFIX))
        if name_end < 0:
            return None
        name = content[cursor + len(_FUNCTION_PREFIX) : name_end]
        schema = tool_schemas.get(name)
        if schema is None:
            return None
        cursor = name_end + 1
        parsed = _parse_arguments(content, cursor, schema)
        if parsed is None:
            return None
        arguments, cursor = parsed
        calls.append((name, arguments))
        if len(calls) > _MAX_CALLS:
            return None

        cursor = _skip_whitespace(content, cursor)
        if content.startswith(_TOOL_CLOSE, cursor):
            cursor += len(_TOOL_CLOSE)
        cursor = _skip_whitespace(content, cursor)
        if cursor == len(content):
            break
        if not (
            content.startswith(_TOOL_OPEN, cursor)
            or content.startswith(_FUNCTION_PREFIX, cursor)
        ):
            return None

    return (prefix, calls) if calls else None


def _parse_arguments(
    content: str, cursor: int, schema: Mapping[str, Any]
) -> tuple[dict[str, Any], int] | None:
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return None
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        return None

    arguments: dict[str, Any] = {}
    consumed_chars = 0
    while True:
        cursor = _skip_whitespace(content, cursor)
        if content.startswith(_FUNCTION_CLOSE, cursor):
            cursor += len(_FUNCTION_CLOSE)
            break
        if not content.startswith("<parameter=", cursor):
            return None
        name_end = content.find(">", cursor + len("<parameter="))
        if name_end < 0:
            return None
        name = content[cursor + len("<parameter=") : name_end]
        property_schema = properties.get(name)
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or name in arguments
            or not isinstance(property_schema, Mapping)
        ):
            return None
        value_start = name_end + 1
        value_end = content.find(_PARAMETER_CLOSE, value_start)
        if value_end < 0:
            return None
        raw_value = content[value_start:value_end]
        consumed_chars += len(raw_value)
        if consumed_chars > _MAX_ARGUMENT_CHARS:
            return None
        value = _coerce_value(raw_value, property_schema)
        if value is _INVALID or not _matches_schema(value, property_schema):
            return None
        arguments[name] = value
        cursor = value_end + len(_PARAMETER_CLOSE)

    if any(name not in arguments for name in required):
        return None
    if not _matches_schema(arguments, schema):
        return None
    return arguments, cursor


class _Invalid:
    pass


_INVALID = _Invalid()


def _coerce_value(raw: str, schema: Mapping[str, Any]) -> Any:
    # Qwen's renderer surrounds each value with one formatting newline. Remove
    # only that pair so an intentional trailing newline in a file body survives.
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    types = _schema_types(schema)
    if raw.lower() == "null" and "null" in types:
        return None
    if "boolean" in types and raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if "integer" in types:
        try:
            return int(raw)
        except ValueError:
            pass
    if "number" in types:
        try:
            value = float(raw)
            if math.isfinite(value):
                return int(value) if value.is_integer() else value
        except ValueError:
            pass
    for expected_type in ("array", "object"):
        if expected_type not in types:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if _json_type(value) == expected_type:
            return value
    if "string" in types or not types:
        return raw
    return _INVALID


def _schema_types(schema: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        values.add(raw_type)
    elif isinstance(raw_type, list):
        values.update(item for item in raw_type if isinstance(item, str))
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, Mapping):
                    values.update(_schema_types(variant))
    return values


def _matches_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return False
    variants = schema.get("anyOf")
    if isinstance(variants, list) and variants:
        return any(
            isinstance(variant, Mapping) and _matches_schema(value, variant)
            for variant in variants
        )
    variants = schema.get("oneOf")
    if isinstance(variants, list) and variants:
        return (
            sum(
                1
                for variant in variants
                if isinstance(variant, Mapping) and _matches_schema(value, variant)
            )
            == 1
        )
    types = _schema_types(schema)
    if types:
        actual_type = _json_type(value)
        if actual_type not in types and not (actual_type == "integer" and "number" in types):
            return False
    if isinstance(value, dict):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            required = schema.get("required", [])
            if isinstance(required, list) and any(name not in value for name in required):
                return False
            if schema.get("additionalProperties") is False and any(
                name not in properties for name in value
            ):
                return False
            for name, child in value.items():
                child_schema = properties.get(name)
                if isinstance(child_schema, Mapping) and not _matches_schema(child, child_schema):
                    return False
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        return all(_matches_schema(item, schema["items"]) for item in value)
    return True


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _skip_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor
