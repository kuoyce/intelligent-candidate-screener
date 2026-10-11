"""Small French ASR → Marian English prototype with private text artefacts.

Heavy dependencies are lazy: importing this module never downloads models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from collections import Counter
from pathlib import Path
import subprocess
import time

from candidate_screener.config import OUTPUT, PROCESSED, PROJECT_ROOT

MODEL = "Helsinki-NLP/opus-mt-fr-en"


def duration_preservation(source: str, translated: str) -> dict:
    """Numeric-duration sanity check, not semantic validation or a GLiNER score.

    Written-out numbers and changed units need human review. No text is returned.
    """
    french = Counter(re.findall(r"\b(\d+(?:[.,]\d+)?)\s*(?:ans?|années?)\b", source, re.I))
    english = Counter(re.findall(r"\b(\d+(?:[.,]\d+)?)\s*(?:years?|yrs?)\b", translated, re.I))
    return {"source_numeric_duration_mentions": sum(french.values()),
            "translated_numeric_duration_mentions": sum(english.values()),
            "same_numeric_duration_multiset": french == english,
            "human_review_required": True,
            "limitations": "Numeric year expressions only; not total experience or semantic equivalence"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_segments(segments: list[dict], duration: float) -> list[dict]:
    """Reject empty, non-finite and out-of-media evidence before translation."""
    result = []
    for segment in segments:
        text = segment.get("text", "").strip()
        start, end = float(segment["start"]), float(segment["end"])
        if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= duration + .1:
            raise ValueError("ASR timestamp outside media duration")
        if text:
            result.append({"start": start, "end": end, "text": text})
    if not result:
        raise ValueError("No speech transcript; nothing to translate")
    return result


def token_chunks(text: str, tokenizer, limit: int) -> list[str]:
    """Split on words, checking real tokenizer lengths without truncation."""
    if limit < 4 or not text.strip():
        raise ValueError("Non-empty text and usable token budget required")
    chunks, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(tokenizer(candidate, add_special_tokens=True)["input_ids"]) > limit:
            if not current:
                raise ValueError("Single word exceeds Marian token budget")
            chunks.append(current)
            current = word
            if len(tokenizer(word, add_special_tokens=True)["input_ids"]) > limit:
                raise ValueError("Single word exceeds Marian token budget")
        else:
            current = candidate
    if current:
        chunks.append(current)
    if " ".join(chunks) != " ".join(text.split()):
        raise AssertionError("Chunking lost source words")
    return chunks


def translate_chunks(chunks, tokenizer, model, *, beams: int, torch, max_tokens: int):
    outputs = []
    start = time.perf_counter()
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=False)
        with torch.inference_mode():
            generated = model.generate(**inputs, num_beams=beams, do_sample=False,
                                       max_new_tokens=max_tokens,
                                       **({"early_stopping": True} if beams > 1 else {}))
        # A hard generation limit must not silently turn a partial sentence into evidence.
        ids = generated[0].tolist()
        if tokenizer.eos_token_id not in ids[1:]:
            raise ValueError("Translation did not finish before output token limit")
        translated = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if not translated:
            raise ValueError("Empty Marian translation")
        outputs.append(translated)
    return outputs, time.perf_counter() - start


def run(media: Path, *, max_seconds: float = 60, max_segments: int = 5) -> dict:
    if not media.is_file() or media.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Existing media file at most 20 MiB required")
    if not 1 <= max_segments <= 5 or not 0 < max_seconds <= 60:
        raise ValueError("Prototype limited to five segments and sixty seconds")
    import numpy as np
    import torch
    import whisper
    import imageio_ffmpeg
    from huggingface_hub import snapshot_download
    from transformers import MarianTokenizer, MarianMTModel

    torch.set_num_threads(4)
    private = PROCESSED / "multimodal" / digest(media)
    private.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    decoded = subprocess.run([ffmpeg, "-v", "error", "-i", str(media), "-vn",
                              "-t", str(max_seconds + 1),
                              "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                             capture_output=True, timeout=90, check=False)
    if decoded.returncode:
        raise ValueError("Media audio decoding failed")
    audio = np.frombuffer(decoded.stdout, dtype=np.float32).copy()
    duration = len(audio) / 16000
    if not 0 < duration <= max_seconds or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
        raise ValueError("Silent, invalid or over-budget audio")
    start = time.perf_counter()
    asr_dir = PROCESSED / "multimodal" / "models" / "whisper"
    asr_dir.mkdir(parents=True, exist_ok=True)
    asr = whisper.load_model("tiny", device="cpu", download_root=str(asr_dir))
    asr_load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = asr.transcribe(audio, language="fr", task="transcribe", fp16=False,
                            verbose=False, temperature=0)
    asr_seconds = time.perf_counter() - start
    segments = valid_segments(result["segments"], duration)
    selected = segments[:max_segments]
    start = time.perf_counter()
    snapshot = Path(snapshot_download(MODEL,
        cache_dir=str(PROCESSED / "multimodal" / "models" / "huggingface"),
        allow_patterns=["config.json", "generation_config.json", "tokenizer_config.json",
                        "vocab.json", "source.spm", "target.spm", "special_tokens_map.json",
                        "model.safetensors"]))
    tokenizer = MarianTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = MarianMTModel.from_pretrained(snapshot, local_files_only=True).to("cpu").eval()
    marian_load_seconds = time.perf_counter() - start
    limit = min(480, int(model.config.max_position_embeddings) - 8)
    chunk_groups = [token_chunks(s["text"], tokenizer, limit) for s in segments]
    chunks = [c for group in chunk_groups for c in group]
    comparison_count = sum(len(group) for group in chunk_groups[:len(selected)])
    full_translation, full_seconds = translate_chunks(
        chunks, tokenizer, model, beams=4, torch=torch, max_tokens=limit)
    greedy, greedy_seconds = translate_chunks(
        chunks[:comparison_count], tokenizer, model, beams=1, torch=torch, max_tokens=limit)
    translations = {"1": greedy, "4": full_translation[:comparison_count]}
    # Every translated chunk retains the parent speech segment, even when split.
    linked_chunks = [{"segment_index": i, "start": segments[i]["start"],
                      "end": segments[i]["end"], "french": chunk}
                     for i, group in enumerate(chunk_groups) for chunk in group]
    raw = {"segments": segments, "translated_segment_count": len(segments),
           "linked_chunks": linked_chunks, "full_beam4_translation": full_translation,
           "comparison_segment_count": len(selected), "comparison_translations": translations}
    (private / "transcripts.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2))
    weights = {p.name: digest(p) for p in snapshot.iterdir()
               if p.suffix in {".bin", ".safetensors", ".spm", ".json"}}
    whisper_weights = asr_dir / "tiny.pt"
    record = {"status": "prototype_complete", "applicant_review_required": True,
              "screening_integration": "not_enabled", "media_sha256": digest(media),
              "excluded_features": ["video_pixels", "gender", "voice_quality", "accent"],
              "media_bytes": media.stat().st_size, "duration_seconds": duration,
              "audio": {"sample_rate": 16000, "channels": 1},
              "asr": {"model": "tiny", "language": "fr", "task": "transcribe",
                      "weights_sha256": digest(whisper_weights), "seconds": asr_seconds,
                      "segment_count": len(segments)},
              "translation": {"model": MODEL, "revision": snapshot.name,
                              "artifact_sha256": weights, "selected_segments": len(selected),
                              "complete": True, "translated_segments": len(segments),
                              "chunk_count": len(chunks), "comparison_chunk_count": comparison_count,
                              "beams": [1, 4], "full_beam4_seconds": full_seconds,
                              "comparison_greedy_seconds": greedy_seconds, "token_budget": limit,
                              "full_output_characters": sum(map(len, full_translation)),
                              "comparison_scope": "First at most five ASR segments; full transcript translated with beam 4",
                              "output_characters": {k: sum(map(len, v)) for k,v in translations.items()},
                              "differing_chunks": sum(a != b for a,b in zip(translations["1"], translations["4"]))},
              "setup_seconds": {"asr_load_download": asr_load_seconds,
                                "marian_load_download": marian_load_seconds},
              "duration_audit": duration_preservation(" ".join(s["text"] for s in segments),
                                                      " ".join(full_translation)),
              "entity_audit": {"status": "unavailable", "extractor": "GLiNER",
                               "reason": "Separate extraction stage has not run"},
              "versions": {n: importlib.metadata.version(n) for n in
                           ("torch", "transformers", "openai-whisper", "imageio-ffmpeg", "sentencepiece")},
              "evaluation": "One clip; no reference transcript or translation; no quality or fit claim"}
    target = OUTPUT / "multimodal" / "french-interview-demo.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, default=PROJECT_ROOT / "french_applicant_introduction_female_voice_french.mp4")
    args = parser.parse_args()
    record = run(args.media)
    print(json.dumps({"status": record["status"], "duration_seconds": record["duration_seconds"]}))


if __name__ == "__main__":
    main()
