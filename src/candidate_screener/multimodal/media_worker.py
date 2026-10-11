"""Offline optional-runtime worker; never writes transcripts or applicant bytes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from candidate_screener.config import PROCESSED
from candidate_screener.multimodal.interview_demo import (
    MODEL, digest, token_chunks, translate_chunks, valid_segments,
)
from candidate_screener.multimodal.submission import PREFIX, validate_upload

MARIAN_REVISION = "c4aed37b318c763fd177aa449b44e3b783cc6c02"


def cached_models(language: str):
    root = PROCESSED / "multimodal" / "models"
    whisper = root / "whisper" / "tiny.pt"
    marian = root / "huggingface" / "models--Helsinki-NLP--opus-mt-fr-en" / "snapshots" / MARIAN_REVISION
    if not whisper.is_file():
        raise ValueError("missing_asr_model")
    if language == "fr" and not all((marian / f).is_file() for f in
                                    ("model.safetensors", "config.json", "source.spm", "target.spm", "vocab.json")):
        raise ValueError("missing_translation_model")
    return whisper, marian


def run(media: Path, language: str) -> dict:
    validate_upload(media.name, media.read_bytes(), language)
    whisper_path, marian_path = cached_models(language)
    import numpy as np
    import torch
    import whisper
    import imageio_ffmpeg
    torch.set_num_threads(4)
    decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(media),
                              "-vn", "-t", "121", "-ac", "1", "-ar", "16000",
                              "-f", "f32le", "pipe:1"], capture_output=True, timeout=30, check=False)
    if decoded.returncode or not decoded.stdout:
        raise ValueError("invalid_audio")
    audio = np.frombuffer(decoded.stdout, dtype=np.float32).copy()
    duration = len(audio) / 16000
    if not 0 < duration <= 120 or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
        raise ValueError("invalid_duration_or_silent_audio")
    start = time.perf_counter()
    asr = whisper.load_model(str(whisper_path), device="cpu")
    result = asr.transcribe(audio, language=language, task="transcribe", fp16=False,
                            verbose=False, temperature=0)
    segments = valid_segments(result["segments"], duration)
    asr_seconds = time.perf_counter() - start
    output = []
    mt_seconds = 0
    if language == "fr":
        from transformers import MarianTokenizer, MarianMTModel
        tokenizer = MarianTokenizer.from_pretrained(marian_path, local_files_only=True)
        model = MarianMTModel.from_pretrained(marian_path, local_files_only=True).to("cpu").eval()
        budget = min(480, int(model.config.max_position_embeddings) - 8)
        for segment in segments:
            chunks = token_chunks(segment["text"], tokenizer, budget)
            translated, elapsed = translate_chunks(chunks, tokenizer, model, beams=4,
                                                   torch=torch, max_tokens=budget)
            mt_seconds += elapsed
            output.append({"start": segment["start"], "end": segment["end"],
                           "french": segment["text"], "english": " ".join(translated)})
    else:
        output = [{"start": s["start"], "end": s["end"], "french": s["text"],
                   "english": s["text"]} for s in segments]
    return {"status": "translated", "review_required": True, "source_language": language,
            "segments": output, "provenance": {"media_sha256": digest(media),
                "duration_seconds": duration, "asr_model": "whisper-tiny",
                "asr_weights_sha256": digest(whisper_path), "asr_task": "transcribe",
                "asr_seconds_including_load": asr_seconds,
                "translation_model": MODEL if language == "fr" else None,
                "translation_revision": MARIAN_REVISION if language == "fr" else None,
                "translation_seconds": mt_seconds,
                "excluded_features": ["video_pixels", "voice_quality", "accent", "gender"]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--language", choices=["fr", "en"], required=True)
    args = parser.parse_args()
    try:
        result = run(args.media, args.language)
    except Exception:
        # Never emit raw decoder/model errors, filenames, transcripts or stderr.
        print("CANDIDATE_MEDIA_ERROR:processing_failed")
        raise SystemExit(1) from None
    print(PREFIX + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
