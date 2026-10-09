"""Neural adapter invariants use mock predictions, no network or weights."""
import math
import io
import json
import re
from types import SimpleNamespace

import pytest

from candidate_screener.extraction.schema import validate_entity
from candidate_screener.extraction.transformer import (
    MODEL_LABELS, extract_transformer_document, load_model,
)


class Tokenizer:
    model_max_length = 32

    def __call__(self, prepared, **kwargs):
        assert not kwargs["truncation"]
        return {"input_ids": [[0] + [1 for word in words for _ in word] + [2] for words in prepared]}


class Processor:
    transformer_tokenizer = Tokenizer()

    def words_splitter(self, text):
        return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def prepare_inputs(self, texts, labels):
        assert isinstance(labels, dict) and list(labels) == list(MODEL_LABELS)
        # Mock prompt still consumes actual tokenizer budget.
        return [["prompt"] + words for words in texts], [1] * len(texts)


class Model:
    config = SimpleNamespace(max_len=5)
    data_processor = Processor()

    def __init__(self):
        self.seen = []

    def predict_entities(self, text, labels, threshold, flat_ner):
        self.seen.append(text)
        assert labels == list(MODEL_LABELS)
        assert flat_ner is False
        label = next(iter(MODEL_LABELS))
        return [{"start": m.start(), "end": m.end(), "text": m.group(), "label": label, "score": 0.8}
                for m in re.finditer(r"Python|C\+\+|Kubernetes|TAIL", text)]


def doc(text):
    return {"doc_id": "synthetic", "source": "synthetic", "doc_type": "cv", "fields": {"Moreinfo": text}}


@pytest.mark.parametrize("kind,surface", [("cv", "5 years of experience"), ("jd", "3–5 years"), ("jd", "at least 2 years")])
def test_experience_years_preserve_source_evidence(kind, surface):
    class ExperienceModel(ModelWithLargeWords):
        data_processor = Processor()

        def predict_entities(self, text, labels, threshold, flat_ner):
            prompt = "years of experience"
            assert prompt in labels
            start = text.index(surface)
            return [{"start": start, "end": start + len(surface), "text": surface,
                     "label": prompt, "score": .83}]

    model = ExperienceModel()
    model.data_processor = Processor()
    model.data_processor.transformer_tokenizer = Tokenizer()
    model.data_processor.transformer_tokenizer.model_max_length = 512
    text = "🙂 Required: " + surface + "."
    document = {**doc(text), "doc_type": kind}
    rows, coverage = extract_transformer_document(document, model)
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "EXPERIENCE_YEARS"
    assert row["surface"] == text[row["start"]:row["end"]] == surface
    assert row["doc_type"] == kind and row["field"] == "Moreinfo"
    assert row["confidence"] == .83 and row["canonical_id"] is None
    assert coverage["Moreinfo"]["complete"]


def test_missing_checkpoint_fails_without_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_model(tmp_path)


def test_model_score_validation_and_rules_remain_null():
    model = Model()
    model.config = SimpleNamespace(max_len=100)
    rows, _ = extract_transformer_document(doc("Python"), model)
    assert len(rows) == 1
    for score in (None, math.nan, math.inf, -0.1, 1.01, True):
        with pytest.raises(ValueError, match="score"):
            validate_entity({**rows[0], "confidence": score}, "Python")
    with pytest.raises(ValueError, match="Rule"):
        validate_entity({**rows[0], "method": "seed-rule"}, "Python")


def test_overlapping_chunks_preserve_tail_and_offsets():
    model = Model()
    model.config = SimpleNamespace(max_len=100)
    text = "🙂 Python a b C++ c d Python e f TAIL  "
    original = doc(text)
    rows, coverage = extract_transformer_document(original, model, overlap_tokens=2)
    assert len(model.seen) > 1
    assert coverage["Moreinfo"]["complete"]
    assert coverage["Moreinfo"]["intervals"][0][0] == 0
    assert coverage["Moreinfo"]["intervals"][-1][1] == len(text)
    assert any(row["surface"] == "TAIL" for row in rows)
    assert [row["surface"] for row in rows].count("Python") == 2
    for row in rows:
        assert text[row["start"]:row["end"]] == row["surface"]
        assert row["canonical_id"] is None
        assert row["context_status"] == "unqualified"
    assert len({row["entity_id"] for row in rows}) == len(rows)
    assert original == doc(text)
    assert rows == extract_transformer_document(original, ModelWithLargeWords(), overlap_tokens=2)[0]


class ModelWithLargeWords(Model):
    config = SimpleNamespace(max_len=100)


def test_subword_budget_includes_prompt_and_oversize_quarantine():
    model = ModelWithLargeWords()
    text = "Python " + "x" * 40 + " TAIL"
    rows, coverage = extract_transformer_document(doc(text), model, overlap_tokens=2)
    assert not coverage["Moreinfo"]["complete"]
    assert coverage["Moreinfo"]["quarantined_intervals"]
    assert [row["surface"] for row in rows] == ["Python", "TAIL"]
    for chunk in model.seen:
        words = [w[0] for w in model.data_processor.words_splitter(chunk)]
        prepared, _ = model.data_processor.prepare_inputs([words], dict.fromkeys(MODEL_LABELS, 1))
        assert len(model.data_processor.transformer_tokenizer(prepared, truncation=False)["input_ids"][0]) <= 32


def test_invalid_prediction_offset_or_score_is_quarantined():
    model = ModelWithLargeWords()
    model.predict_entities = lambda *args, **kwargs: [
        {"start": 1, "end": 6, "text": "Python", "label": next(iter(MODEL_LABELS)), "score": 0.8},
        {"start": 0, "end": 6, "text": "Python", "label": next(iter(MODEL_LABELS)), "score": float("nan")},
    ]
    rows, coverage = extract_transformer_document(doc("Python"), model)
    assert rows == []
    assert coverage["Moreinfo"]["quarantined_predictions"] == 2


def test_deadline_marks_unsubmitted_tail_incomplete():
    model = ModelWithLargeWords()
    rows, coverage = extract_transformer_document(doc("Python TAIL"), model, deadline=0)
    assert rows == [] and not model.seen
    assert coverage["Moreinfo"]["interrupted"]
    assert not coverage["Moreinfo"]["complete"]
    assert coverage["Moreinfo"]["intervals"] == []


def test_context_is_explicitly_hybrid_and_empty_fields_safe():
    model = ModelWithLargeWords()
    document = doc("No Python experience; learning Kubernetes")
    document["fields"]["Highlights"] = None
    rows, coverage = extract_transformer_document(document, model)
    assert [(r["surface"], r["context_status"]) for r in rows] == [("Python", "negated"), ("Kubernetes", "learning")]
    assert coverage["Highlights"]["complete"] and not coverage["Highlights"]["intervals"]


def test_isolated_worker_pipe_forwards_deadline_and_hides_other_stdout(monkeypatch, tmp_path):
    from candidate_screener.extraction import runtime_proxy
    messages = [
        {"status": "ready", "chunk_parameters": {"word_limit": 10}, "runtime_versions": {"torch": "mock"}},
        {"status": "ok", "records": [], "coverage": {"Moreinfo": {"complete": True}}},
    ]
    class Process:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("third-party startup log\n" + "".join(runtime_proxy.PREFIX + json.dumps(row) + "\n" for row in messages))
            self.stopped = False
        def poll(self):
            return 0 if self.stopped else None
        def terminate(self):
            self.stopped = True
        def wait(self, timeout=None):
            return 0
    process = Process()
    def start(args, **kwargs):
        assert "--no-sync" in args
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["stderr"] == runtime_proxy.subprocess.DEVNULL
        return process
    monkeypatch.setattr(runtime_proxy.subprocess, "Popen", start)
    monkeypatch.setattr(runtime_proxy.Path, "is_file", lambda path: True)
    proxy = runtime_proxy.TransformerRuntimeProxy(tmp_path)
    rows, coverage = extract_transformer_document(doc("Python"), proxy, deadline=123.0)
    assert rows == [] and coverage["Moreinfo"]["complete"]
    sent = json.loads(process.stdin.getvalue())
    assert sent["parameters"]["deadline"] == 123.0
    assert sent["document"] == doc("Python")
    proxy.close()
    assert process.stopped
