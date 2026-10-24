"""Prompt-aware sentence planning without neural dependencies or source data."""
import re
from types import SimpleNamespace

import pytest

from candidate_screener.extraction.sentence_chunking import sentence_windows
from candidate_screener.extraction.transformer import _fits, chunk_parameters


class Tokenizer:
    model_max_length = 48

    def __call__(self, prepared, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [[0] * (10 + sum(len(w) for w in row)) for row in prepared]}


class Processor:
    transformer_tokenizer = Tokenizer()

    def words_splitter(self, text):
        return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def prepare_inputs(self, words, labels):
        assert isinstance(labels, dict)
        return words, None


class Model:
    config = SimpleNamespace(max_len=48)
    data_processor = Processor()


def covered(windows, quarantine, length):
    return all(any(a <= i < b for a, b in windows + [tuple(q) for q in quarantine]) for i in range(length))


def test_complete_sentences_pack_and_overlap_without_rewriting():
    text = "Python C++.\nC# .NET.\nNode.js SQL."
    spans = [(0, 12), (12, 21), (21, len(text))]
    windows, quarantine, info = sentence_windows(text, Model(), intervals=spans)
    assert quarantine == []
    assert covered(windows, quarantine, len(text))
    assert all(a in {0, 12, 21} and b in {12, 21, len(text)} for a, b in windows)
    assert windows[1][0] < windows[0][1]
    assert info["token_limit"] == 32
    assert info["fallback_windows"] == 0


def test_real_prepared_budget_reserves_headroom_and_never_truncates():
    text = "abcdefgh ijklmnop qrstuvwx yzabcdef"
    windows, quarantine, info = sentence_windows(text, Model(), intervals=[(0, len(text))])
    parameters = dict(chunk_parameters(Model()), token_limit=32)
    assert info["fallback_windows"] >= 2
    assert quarantine == []
    for a, b in windows:
        assert _fits(Model(), [w[0] for w in Processor().words_splitter(text[a:b])], parameters)
    assert covered(windows, quarantine, len(text))


def test_individually_unrepresentable_word_quarantined_without_slicing():
    text = "ok " + "x" * 30 + " end"
    windows, quarantine, info = sentence_windows(text, Model(), intervals=[(0, len(text))])
    assert quarantine == [[3, 34]]
    assert info["quarantine_count"] == 1
    assert covered(windows, quarantine, len(text))
    assert all(not (3 < a < 33 or 3 < b < 33) for a, b in windows)


@pytest.mark.parametrize("text,spans", [("", []), (" \n ", [(0, 3)])])
def test_empty_and_whitespace(text, spans):
    windows, quarantine, _ = sentence_windows(text, Model(), intervals=spans)
    assert quarantine == []
    assert covered(windows, quarantine, len(text))


def test_partition_gaps_rejected():
    with pytest.raises(ValueError, match="partition"):
        sentence_windows("abc", Model(), intervals=[(1, 3)])


def test_nonpositive_budget_rejected():
    model = Model()
    model.config = SimpleNamespace(max_len=16)
    with pytest.raises(ValueError, match="nonpositive"):
        sentence_windows("abc", model, intervals=[(0, 3)])


def test_resume_line_boundaries_preserve_source(monkeypatch):
    import candidate_screener.extraction.sentence_chunking as module

    class Detector:
        max_length = 100

        def __call__(self, text):
            return SimpleNamespace(sents=[SimpleNamespace(start_char=0)])

    monkeypatch.setattr(module, "_sentencizer", lambda: Detector())
    text = "SUMMARY\n- Python C++\n- C# .NET\n\nEducation"
    spans = module.sentence_intervals(text)
    assert "".join(text[a:b] for a, b in spans) == text
    assert [text[a:b] for a, b in spans] == ["SUMMARY\n", "- Python C++\n", "- C# .NET\n\n", "Education"]


def test_spacy_technical_mentions_keep_original_offsets():
    pytest.importorskip("spacy", reason="spaCy is installed only in the medium inference runtime")
    from candidate_screener.extraction.sentence_chunking import sentence_intervals
    text = "Dr. A. Smith uses C++, C#, .NET and Node.js 2.5.\nB.Sc. Computer Science, 2018–2022.\nhttps://example.org/cv"
    spans = sentence_intervals(text)
    assert "".join(text[a:b] for a, b in spans) == text
    for mention in ["C++", "C#", ".NET", "Node.js", "2.5", "B.Sc.", "2018–2022", "https://example.org/cv"]:
        start = text.index(mention)
        assert any(a <= start and start + len(mention) <= b for a, b in spans)


def test_all_oversized_words_terminate_and_explain_every_source_interval():
    text = " " + "x" * 30 + "\n" + "y" * 30 + " "
    windows, quarantine, info = sentence_windows(text, Model(), intervals=[(0, len(text))])
    assert windows == []
    assert len(quarantine) == info["quarantine_count"] == 2
    assert covered(windows, quarantine, len(text))


@pytest.mark.parametrize("overlap", [-1, True, 1.5])
def test_invalid_overlap_rejected(overlap):
    with pytest.raises(ValueError, match="nonnegative"):
        sentence_windows("abc", Model(), overlap, intervals=[(0, 3)])
