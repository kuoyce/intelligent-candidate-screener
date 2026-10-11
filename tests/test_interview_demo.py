import pytest

from candidate_screener.multimodal.interview_demo import (
    MODEL, token_chunks, valid_segments, duration_preservation, translate_chunks,
)


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [0] + list(range(len(text.split()))) + [1]}


def test_chunking_preserves_all_words_with_special_token_budget():
    text = "un deux trois quatre cinq six sept huit neuf"
    chunks = token_chunks(text, Tokenizer(), 5)
    assert " ".join(chunks) == text
    assert all(len(Tokenizer()(c)["input_ids"]) <= 5 for c in chunks)
    assert len(chunks) == 3


@pytest.mark.parametrize("start,end", [(0,46), (-1,3), (3,2),
                                       (0,float("nan")), (float("inf"),3)])
def test_timestamp_evidence_cannot_escape_clip(start,end):
    with pytest.raises(ValueError, match="timestamp"):
        valid_segments([{"start":start,"end":end,"text":"bonjour"}],45)


def test_empty_transcript_is_failure_not_successful_cv():
    with pytest.raises(ValueError, match="No speech"):
        valid_segments([{"start":0,"end":1,"text":" "}],45)


def test_original_timestamp_and_text_preserved():
    assert valid_segments([{"start":1,"end":2,"text":" bonjour "}],45) == [
        {"start":1.0,"end":2.0,"text":"bonjour"}]


def test_changed_duration_is_flagged_without_exporting_text():
    audit = duration_preservation("Marie travaille depuis 5 ans", "Marie worked for 3 years")
    assert not audit["same_numeric_duration_multiset"]
    assert "Marie" not in str(audit)
    assert audit["human_review_required"]


def test_checkpoint_direction_is_french_to_english():
    assert MODEL == "Helsinki-NLP/opus-mt-fr-en"


def test_unfinished_generation_is_not_submitted_as_complete_translation():
    from contextlib import nullcontext
    from types import SimpleNamespace

    class MarianTokenizer:
        eos_token_id = 0
        def __call__(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": [1,2]}
        def decode(self, ids, **kwargs):
            return "incomplete claim"

    class Model:
        def generate(self, **kwargs):
            return [SimpleNamespace(tolist=lambda: [0,1,2])]

    with pytest.raises(ValueError, match="did not finish"):
        translate_chunks(["bonjour"], MarianTokenizer(), Model(), beams=4,
                         torch=SimpleNamespace(inference_mode=nullcontext), max_tokens=3)
