"""Session-only, bounded local media processing transport."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from candidate_screener.config import PROJECT_ROOT

MAX_BYTES = 20_000_000
PREFIX = "CANDIDATE_MEDIA_RESULT:"


class MediaError(ValueError):
    """Safe user-facing media failure, never a raw backend exception."""


def validate_upload(name: str, content: bytes, language: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4a", ".mp3", ".wav", ".webm", ".ogg"}:
        raise MediaError("Upload an MP4, MOV, M4A, MP3, WAV, WebM or OGG file.")
    if not isinstance(content, bytes) or not 0 < len(content) <= MAX_BYTES:
        raise MediaError("Upload a non-empty audio or video file of at most 20 MB.")
    if language not in {"fr", "en"}:
        raise MediaError("Select French or English for this local prototype.")
    return suffix


def default_command() -> list[str]:
    return ["uv", "run", "--offline", "--project", str(PROJECT_ROOT / "runtime/transformer"),
            "--with", "openai-whisper", "--with", "imageio-ffmpeg", "--with", "sacremoses",
            "--with", "numba==0.60.0", "python", "-m", "candidate_screener.multimodal.media_worker"]


def validate_result(result: dict, language: str) -> dict:
    if result.get("source_language") != language or result.get("status") != "translated":
        raise MediaError("Media processing did not produce a complete transcript.")
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        raise MediaError("No speech was transcribed. Try a clearer recording.")
    import math
    for row in segments:
        try:
            start, end = float(row["start"]), float(row["end"])
            original, english = row["french"], row["english"]
            valid = (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= 120.1
                     and isinstance(original, str) and bool(original.strip())
                     and isinstance(english, str) and bool(english.strip()))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise MediaError("The transcript was incomplete. Please retry the recording.")
    result["review_required"] = True
    return result


def process_media(name: str, content: bytes, language: str = "fr", *, command=None,
                  runner=subprocess.run) -> dict:
    """Return unconfirmed text only; uploaded bytes disappear on success or failure.

    ``command``/``runner`` injection supports tests without models or subprocesses.
    The legacy segment key ``french`` holds original text also for English input;
    ``source_language`` identifies its actual language.
    """
    suffix = validate_upload(name, content, language)
    import os
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(PROJECT_ROOT / "src"),
                        "UV_CACHE_DIR": "/tmp/candidate-uv-cache",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    with tempfile.TemporaryDirectory(prefix="candidate-media-") as directory:
        media = Path(directory) / ("upload" + suffix)
        media.write_bytes(content)
        try:
            completed = runner(list(default_command() if command is None else command)
                               + ["--media", str(media), "--language", language],
                               capture_output=True, text=True, timeout=240,
                               check=False, env=environment, cwd=PROJECT_ROOT)
        except subprocess.TimeoutExpired:
            raise MediaError("Processing took too long. Upload a shorter recording.") from None
        except OSError:
            raise MediaError("Local media runtime is unavailable. Ask the project owner to set it up.") from None
        if completed.returncode:
            # Worker output carries only an allowlisted code; discard raw stderr.
            raise MediaError("Media could not be processed. Check the file and local model setup.")
        lines = [line[len(PREFIX):] for line in completed.stdout.splitlines() if line.startswith(PREFIX)]
        if len(lines) != 1:
            raise MediaError("Local processor returned an invalid response.")
        try:
            result = json.loads(lines[0])
        except (ValueError, TypeError):
            raise MediaError("Local processor returned an invalid response.") from None
        if not isinstance(result, dict):
            raise MediaError("Local processor returned an invalid response.")
        return validate_result(result, language)
