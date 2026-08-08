#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    return call(), round(time.perf_counter() - started, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise LocalLLM through the official OpenAI SDK")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008/v1")
    parser.add_argument("--api-key", default="local-dev-key")
    parser.add_argument("--model", default="localllm-fast")
    parser.add_argument("--embedding-model", default="localllm-embed")
    parser.add_argument("--vision-model", default="localllm-vision")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600.0)
    report: dict[str, Any] = {
        "sdk": f"openai {importlib.metadata.version('openai')}",
        "base_url": args.base_url,
    }

    models, elapsed = timed(lambda: client.models.list())
    model_ids = {model.id for model in models.data}
    assert args.model in model_ids, f"Missing text model or alias: {args.model}"
    assert args.embedding_model in model_ids, f"Missing embedding alias: {args.embedding_model}"
    report["models"] = {"count": len(model_ids), "elapsed_s": elapsed}

    chat, elapsed = timed(
        lambda: client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply in one sentence: LocalLLM is ready."}],
            temperature=0,
        )
    )
    chat_text = chat.choices[0].message.content or ""
    assert chat_text.strip()
    report["chat"] = {"elapsed_s": elapsed, "characters": len(chat_text)}

    started = time.perf_counter()
    chunks = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Count from one to five, comma separated."}],
        temperature=0,
        stream=True,
    )
    streamed = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert streamed.strip()
    report["chat_stream"] = {
        "elapsed_s": round(time.perf_counter() - started, 3),
        "characters": len(streamed),
    }

    response, elapsed = timed(
        lambda: client.responses.create(
            model=args.model,
            input="Return one short sentence confirming the Responses API works.",
        )
    )
    assert response.output_text.strip()
    report["responses"] = {"elapsed_s": elapsed, "characters": len(response.output_text)}

    tool_result, elapsed = timed(
        lambda: client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Use multiply to calculate 7 times 6."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "multiply",
                        "description": "Multiply two integers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "multiply"}},
            temperature=0,
        )
    )
    calls = tool_result.choices[0].message.tool_calls or []
    assert calls and calls[0].function.name == "multiply"
    tool_arguments = json.loads(calls[0].function.arguments)
    assert {tool_arguments.get("a"), tool_arguments.get("b")} == {6, 7}
    report["tool_call"] = {"elapsed_s": elapsed, "arguments": tool_arguments}

    json_result, elapsed = timed(
        lambda: client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": 'Return JSON only with keys "ready" (true) and "runtime" ("local").',
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    )
    structured = json.loads(json_result.choices[0].message.content or "")
    assert structured.get("ready") is True
    report["json_mode"] = {"elapsed_s": elapsed, "value": structured}

    embeddings, elapsed = timed(
        lambda: client.embeddings.create(
            model=args.embedding_model,
            input=["local private language model", "private local AI", "tropical fruit salad"],
        )
    )
    dimensions = {len(item.embedding) for item in embeddings.data}
    assert len(embeddings.data) == 3 and dimensions == {1024}
    report["embeddings"] = {
        "elapsed_s": elapsed,
        "vectors": len(embeddings.data),
        "dimensions": 1024,
    }

    if args.image:
        assert args.image.is_file(), f"Image does not exist: {args.image}"
        assert args.vision_model in model_ids, f"Missing vision alias: {args.vision_model}"
        mime = mimetypes.guess_type(args.image.name)[0] or "image/png"
        data_url = f"data:{mime};base64,{base64.b64encode(args.image.read_bytes()).decode()}"
        vision, elapsed = timed(
            lambda: client.chat.completions.create(
                model=args.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this interface image concisely."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                temperature=0,
            )
        )
        vision_text = vision.choices[0].message.content or ""
        assert vision_text.strip()
        report["vision"] = {"elapsed_s": elapsed, "characters": len(vision_text)}

    report["status"] = "passed"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
