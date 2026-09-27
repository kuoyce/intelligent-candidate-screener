"""`extraction.ner`: chunking (pure), alignment and the cached tagger (need torch)."""
from __future__ import annotations

import pytest

from candidate_screener.extraction import ner


def test_chunk_keeps_every_word_and_respects_the_limit():
    counts = [3, 4, 2, 9, 1, 1]
    out = ner.chunk(counts, 6)
    assert out == [(0, 1), (1, 3), (3, 4), (4, 6)]
    assert [i for s, e in out for i in range(s, e)] == list(range(len(counts)))


def test_chunk_of_nothing():
    assert ner.chunk([], 10) == []


def _cached_base():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        return ner.base_model_path()
    except Exception:
        pytest.skip("all-MiniLM-L6-v2 is not in the local Hugging Face cache")


def test_labels_sit_on_the_first_subword_only():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_cached_base(), local_files_only=True)
    words, tags = ["Node.js", "and", "C++"], ["B", "O", "B"]
    enc, labels, first = ner._encode(tok, [words], 32, [tags])
    kept = [int(x) for x in labels[0] if int(x) != -100]
    assert kept == [ner.LABELS.index(t) for t in tags]
    assert sorted(first[0]) == [0, 1, 2]


@pytest.mark.skipif(not (ner.NER_DIR / "ner-config.json").is_file(),
                    reason="tagger not trained (extraction.ner --train); weights are a cache")
def test_cached_tagger_emits_spans_inside_the_text():
    pytest.importorskip("torch")
    tagger = ner.load_ner()
    text = "Requirements:\nStrong experience with Python and PostgreSQL, (Docker)."
    spans = tagger.predict_spans(text)
    assert spans and all(0 <= s < e <= len(text) for s, e in spans)
    assert all(text[e - 1] not in ",.)" and text[s] != "(" for s, e in spans)
    assert tagger.predict_spans(text) == spans  # deterministic
