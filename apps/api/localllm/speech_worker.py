from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_TRANSCRIPT_CHARACTERS = 32_000
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
_LANGUAGE = re.compile(r"^[a-z]{2,3}$")


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _safe_audio_path(value: object, root: Path) -> Path:
    if not isinstance(value, str):
        raise ValueError("invalid_audio_path")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError("invalid_audio_path")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("invalid_audio_path")
    entry = resolved.stat()
    if (
        not stat.S_ISREG(entry.st_mode)
        or entry.st_nlink != 1
        or entry.st_size < 1
        or entry.st_size > MAX_AUDIO_BYTES
        or entry.st_uid != os.getuid()
    ):
        raise ValueError("invalid_audio_file")
    return resolved


def _request(value: object, root: Path) -> tuple[str, Path, str | None]:
    if not isinstance(value, dict) or set(value) != {"id", "path", "language"}:
        raise ValueError("invalid_request")
    request_id = value.get("id")
    if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
        raise ValueError("invalid_request")
    language = value.get("language")
    if language in (None, "", "auto"):
        normalized_language = None
    elif isinstance(language, str) and _LANGUAGE.fullmatch(language):
        normalized_language = language
    else:
        raise ValueError("invalid_language")
    return request_id, _safe_audio_path(value.get("path"), root), normalized_language


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument(
        "--compute-type", choices=("int8", "int8_float16", "float16", "float32"), required=True
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    input_root = Path(args.input_root)
    if not model_path.is_absolute() or not input_root.is_absolute():
        raise SystemExit(2)
    model_path = model_path.resolve(strict=True)
    input_root = input_root.resolve(strict=True)
    if not model_path.is_dir() or not input_root.is_dir():
        raise SystemExit(2)

    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(model_path),
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        local_files_only=True,
    )
    _emit({"schema": "localllm/speech-worker-ready/v1", "ready": True})

    for line in sys.stdin:
        request_id = ""
        try:
            request_id, audio_path, language = _request(json.loads(line), input_root)
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                best_of=5,
                vad_filter=True,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
            if not text:
                raise ValueError("no_speech_detected")
            if len(text) > MAX_TRANSCRIPT_CHARACTERS:
                raise ValueError("transcript_too_large")
            _emit(
                {
                    "schema": "localllm/speech-worker-result/v1",
                    "id": request_id,
                    "ok": True,
                    "text": text,
                    "language": str(info.language or "und")[:12],
                    "language_probability": round(float(info.language_probability or 0.0), 6),
                    "duration_seconds": round(float(info.duration or 0.0), 3),
                }
            )
        except Exception as exc:
            code = str(exc) if isinstance(exc, ValueError) else "transcription_failed"
            if not re.fullmatch(r"[a-z_]{3,64}", code):
                code = "transcription_failed"
            _emit(
                {
                    "schema": "localllm/speech-worker-result/v1",
                    "id": request_id,
                    "ok": False,
                    "error": code,
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
