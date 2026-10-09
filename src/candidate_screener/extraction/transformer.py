"""Local GLiNER inference with explicit original-text windows and coverage.

Model labels describe mentions, never proficiency. Context is the rules-v1
heuristic, not neural inference. Raw GLiNER scores are uncalibrated.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from .rules import _context
from .schema import validate_entity

TRANSFORMER_VERSION = "gliner-adapter-v1"
CONTEXT_VERSION = "rules-v1-context"
MODEL_LABELS = {
    "technology, software tool or programming language": "KNOWLEDGE",
    "professional skill or competency": "SKILL",
    "job title or occupation": "TITLE",
    "academic degree or qualification": "DEGREE",
}


def load_model(model_path: str | Path, device: str = "cpu"):
    """Load an already downloaded pinned checkpoint; never fetch weights here."""
    path = Path(model_path)
    if not path.is_dir() or not (path / "gliner_config.json").is_file():
        raise FileNotFoundError("Pinned local GLiNER checkpoint is missing")
    try:
        from gliner import GLiNER
    except ImportError:
        from .runtime_proxy import TransformerRuntimeProxy
        return TransformerRuntimeProxy(path, device=device)
    # GLiNER 0.2.21's from_pretrained(local_files_only=True) does not propagate
    # that flag to the backbone AutoConfig/AutoTokenizer calls. Construct with
    # the separately pinned local backbone assets to close that network path.
    from gliner.config import GLiNERConfig
    from transformers import AutoConfig, AutoTokenizer
    import torch
    encoder_assets = path / "encoder-assets"
    if not (encoder_assets / "config.json").is_file() or not (encoder_assets / "tokenizer_config.json").is_file():
        raise FileNotFoundError("Pinned local encoder config/tokenizer assets are missing")
    config_values = json.loads((path / "gliner_config.json").read_text())
    config_values["model_name"] = str(encoder_assets)
    config = GLiNERConfig(**config_values)
    config.encoder_config = AutoConfig.from_pretrained(str(encoder_assets), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_assets), local_files_only=True)
    model = GLiNER(config, tokenizer=tokenizer, encoder_from_pretrained=False)
    if (config.class_token_index == -1 or config.vocab_size == -1) and not config.labels_encoder:
        model.resize_token_embeddings(add_tokens=["[FLERT]", config.ent_token, config.sep_token])
    state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def chunk_parameters(model) -> dict:
    """Bound both GLiNER word spans and encoder subword inputs, including prompts."""
    if hasattr(model, "chunk_parameters"):
        return model.chunk_parameters()
    processor = model.data_processor
    tokenizer = processor.transformer_tokenizer
    limits = [int(model.config.max_len), 512]
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        limits.append(tokenizer_limit)
    encoder_config = getattr(model.config, "encoder_config", None)
    position_limit = getattr(encoder_config, "max_position_embeddings", None)
    if isinstance(position_limit, int) and position_limit > 0:
        limits.append(position_limit)
    return {"word_limit": int(model.config.max_len), "token_limit": min(limits),
            "boundary_policy": "GLiNER words_splitter; exact prepared prompt subword budget"}


def _fits(model, words: list[str], parameters: dict) -> bool:
    if len(words) > parameters["word_limit"]:
        return False
    processor = model.data_processor
    # GLiNER 0.2.21 treats a list as per-example labels; a dictionary is the
    # shared class mapping used by its actual prediction preprocessing.
    prepared, _ = processor.prepare_inputs([words], {label: index for index, label in enumerate(MODEL_LABELS, 1)})
    encoded = processor.transformer_tokenizer(prepared, is_split_into_words=True,
                                              truncation=False, add_special_tokens=True)
    return len(encoded["input_ids"][0]) <= parameters["token_limit"]


def _windows(text: str, model, overlap_tokens: int) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """Find maximum fitting overlapping word windows; quarantine oversized words."""
    words = list(model.data_processor.words_splitter(text))
    if not words:
        return ([(0, len(text))] if text else []), []
    parameters = chunk_parameters(model)
    windows, quarantine = [], []
    index = 0
    while index < len(words):
        lo, hi = index + 1, min(len(words), index + parameters["word_limit"])
        end_index = index
        while lo <= hi:
            middle = (lo + hi) // 2
            if _fits(model, [word[0] for word in words[index:middle]], parameters):
                end_index, lo = middle, middle + 1
            else:
                hi = middle - 1
        start = 0 if index == 0 else words[index][1]
        if end_index == index:
            end = words[index + 1][1] if index + 1 < len(words) else len(text)
            quarantine.append([start, end])
            index += 1
            continue
        end = words[end_index][1] if end_index < len(words) else len(text)
        windows.append((start, end))
        if end_index == len(words):
            break
        index = end_index - min(overlap_tokens, end_index - index - 1)
    return windows, quarantine


def extract_transformer_document(doc: dict, model, threshold: float = 0.5,
                                 overlap_tokens: int = 32,
                                 deadline: float | None = None) -> tuple[list[dict], dict]:
    """Return validated records and per-field submitted coverage.

    Exact repeated span/label predictions keep their maximum score. Other
    overlaps are retained explicitly; no ontology mapping is inferred.
    Deadline is checked before every chunk; an active model call cannot be
    preempted. Invalid neural predictions are counted and quarantined.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    if not isinstance(overlap_tokens, int) or overlap_tokens < 0:
        raise ValueError("overlap_tokens must be a nonnegative integer")
    if hasattr(model, "extract_document"):
        return model.extract_document(doc, threshold=threshold,
                                      overlap_tokens=overlap_tokens, deadline=deadline)
    records, coverage = [], {}
    for field, text in sorted(doc["fields"].items()):
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError("Document fields must contain strings or null")
        windows, quarantine = _windows(text, model, overlap_tokens)
        intervals, unique, invalid, interrupted = [], {}, 0, False
        for start, end in windows:
            if deadline is not None and time.monotonic() >= deadline:
                interrupted = True
                break
            chunk = text[start:end]
            if not chunk.strip():
                intervals.append([start, end])
                continue
            predictions = model.predict_entities(chunk, list(MODEL_LABELS), threshold=threshold,
                                                  flat_ner=False)
            intervals.append([start, end])
            for prediction in predictions:
                try:
                    local_start, local_end = prediction["start"], prediction["end"]
                    if isinstance(local_start, bool) or isinstance(local_end, bool) or not isinstance(local_start, int) or not isinstance(local_end, int) or not 0 <= local_start < local_end <= len(chunk):
                        raise ValueError("Invalid chunk offsets")
                    label = MODEL_LABELS[prediction["label"]]
                    if isinstance(prediction["score"], bool):
                        raise ValueError("Boolean model score")
                    score = float(prediction["score"])
                    if prediction["text"] != chunk[local_start:local_end]:
                        raise ValueError("Neural surface differs from source")
                    record = {"doc_id": str(doc["doc_id"]), "source": doc["source"],
                              "doc_type": doc["doc_type"], "field": field,
                              "text_version": doc.get("text_version", "raw-v1"),
                              "start": start + local_start, "end": start + local_end,
                              "label": label, "surface": prediction["text"], "canonical_id": None,
                              "method": "gliner-zero-shot", "extractor_version": TRANSFORMER_VERSION,
                              "confidence": score, "context_status": _context(text, start + local_start, start + local_end, field)}
                    validate_entity(record, text)
                    if score < threshold:
                        continue
                    key = (record["start"], record["end"], label)
                    if key not in unique or score > unique[key]["confidence"]:
                        unique[key] = record
                except (KeyError, TypeError, ValueError, OverflowError):
                    invalid += 1
        for key in sorted(unique):
            record = unique[key]
            record["entity_id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            records.append(record)
        coverage[field] = {"intervals": intervals, "characters": len(text), "chunks": len(intervals),
                           "planned_chunks": len(windows), "complete": not quarantine and not interrupted,
                           "quarantined_intervals": quarantine, "quarantined_predictions": invalid,
                           "interrupted": interrupted}
    return records, coverage
