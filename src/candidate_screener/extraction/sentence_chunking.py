"""Sentence-first windows over unchanged source; neural dependencies are optional."""
from __future__ import annotations

from functools import lru_cache
import re

STRATEGY_VERSION = "sentence-window-v1"
HEADROOM_TOKENS = 16


def chunk_parameters(model) -> dict:
    """Effective prepared-input limits for this versioned sentence strategy."""
    from .transformer import chunk_parameters as baseline_parameters
    parameters = dict(baseline_parameters(model))
    parameters["token_limit"] -= HEADROOM_TOKENS
    if parameters["token_limit"] <= 0 or parameters["word_limit"] <= 0:
        raise ValueError("Model budget is nonpositive after encoder headroom")
    parameters.update(strategy=STRATEGY_VERSION, headroom_tokens=HEADROOM_TOKENS)
    return parameters


@lru_cache(maxsize=1)
def _sentencizer():
    try:
        import spacy
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "unknown dependency"
        raise ImportError(
            f"Sentence chunking cannot import spaCy (missing: {missing}); "
            "synchronize the pinned medium runtime"
        ) from exc
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    # CVs can exceed spaCy's default limit; this rule-only pipeline has no parser.
    return nlp


def sentence_intervals(text: str) -> list[tuple[int, int]]:
    """Sentence offsets, with explicit resume line/paragraph boundaries.

    Whitespace is attached to the preceding span, covering the entire original
    source. Rule-based sentence detection is not a linguistic guarantee.
    """
    if not text:
        return []
    nlp = _sentencizer()
    nlp.max_length = max(nlp.max_length, len(text) + 1)
    boundaries = {0, len(text)}
    # Preserve headings and punctuation-free bullet/paragraph content as units.
    for match in re.finditer(r"\r?\n+", text):
        boundaries.add(match.end())
    for sentence in nlp(text).sents:
        boundaries.add(sentence.start_char)
    points = sorted(boundaries)
    return [(a, b) for a, b in zip(points, points[1:]) if a < b]


def sentence_windows(text: str, model, overlap_tokens: int = 32, *,
                     intervals: list[tuple[int, int]] | None = None
                     ) -> tuple[list[tuple[int, int]], list[list[int]], dict]:
    """Plan source windows, quarantine and auditable chunk diagnostics.

    ``intervals`` is an injectable sentence detector for synthetic tests. All
    fitting checks use GLiNER's real prompt preparation with no truncation.
    """
    from .transformer import _fits
    if not isinstance(text, str):
        raise ValueError("Source must be a string")
    if isinstance(overlap_tokens, bool) or not isinstance(overlap_tokens, int) or overlap_tokens < 0:
        raise ValueError("overlap_tokens must be a nonnegative integer")
    parameters = chunk_parameters(model)
    spans = sentence_intervals(text) if intervals is None else intervals
    cursor = 0
    for start, end in spans:
        if start != cursor or end <= start or end > len(text):
            raise ValueError("Sentence intervals must partition the unchanged source")
        cursor = end
    if cursor != len(text):
        raise ValueError("Sentence intervals must cover the unchanged source")
    splitter = model.data_processor.words_splitter

    def fits(start, end):
        words = [w[0] for w in splitter(text[start:end])]
        return not words or _fits(model, words, parameters)

    windows, quarantine = [], []
    fallback_windows = oversized_sentences = 0
    index = 0
    while index < len(spans):
        start, end = spans[index]
        if not fits(start, end):
            oversized_sentences += 1
            words = list(splitter(text[start:end]))
            word_index = 0
            while word_index < len(words):
                lo, hi, stop = word_index + 1, min(len(words), word_index + parameters["word_limit"]), word_index
                while lo <= hi:
                    middle = (lo + hi) // 2
                    if _fits(model, [w[0] for w in words[word_index:middle]], parameters):
                        stop, lo = middle, middle + 1
                    else:
                        hi = middle - 1
                a = start if word_index == 0 else start + words[word_index][1]
                next_index = max(stop, word_index + 1)
                b = start + words[next_index][1] if next_index < len(words) else end
                if stop == word_index:
                    quarantine.append([a, b])
                    word_index += 1
                    continue
                windows.append((a, b))
                fallback_windows += 1
                if stop == len(words):
                    break
                word_index = stop - min(overlap_tokens, 32, stop - word_index - 1)
            index += 1
            continue
        # One previous complete sentence is preferred; a suffix of up to 32
        # GLiNER words supplies context when the complete sentence cannot fit.
        if windows and not quarantine:
            previous_start = spans[index - 1][0] if index else start
            if fits(previous_start, end):
                start = previous_start
            else:
                previous_words = list(splitter(text[previous_start:start]))
                for count in range(min(overlap_tokens, 32, len(previous_words)), 0, -1):
                    candidate = previous_start + previous_words[-count][1]
                    if fits(candidate, end):
                        start = candidate
                        break
        stop = index + 1
        while stop < len(spans) and fits(start, spans[stop][1]):
            end = spans[stop][1]
            stop += 1
        windows.append((start, end))
        index = stop
    for start, end in windows:
        if not fits(start, end):
            raise ValueError("Planned chunk exceeds actual prepared-input budget")
    diagnostics = {"strategy": STRATEGY_VERSION, "sentence_count": len(spans),
                   "headroom_tokens": HEADROOM_TOKENS, "token_limit": parameters["token_limit"],
                   "word_limit": parameters["word_limit"], "fallback_windows": fallback_windows,
                   "oversized_sentences": oversized_sentences, "quarantine_count": len(quarantine),
                   "planned_intervals": [list(w) for w in windows]}
    return windows, quarantine, diagnostics


def plan_chunks(text: str, model, overlap_tokens: int = 32,
                headroom_tokens: int = HEADROOM_TOKENS) -> dict:
    """Public worker API with a fixed, identity-versioned headroom policy."""
    if headroom_tokens != HEADROOM_TOKENS:
        raise ValueError("sentence-window-v1 requires 16 encoder headroom tokens")
    windows, quarantine, diagnostics = sentence_windows(text, model, overlap_tokens)
    return {"windows": windows, "quarantined_intervals": quarantine,
            "diagnostics": diagnostics}
