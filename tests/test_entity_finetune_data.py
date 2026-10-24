import re
from types import SimpleNamespace
import pytest
from candidate_screener.entity_annotation.finetune_data import prepare_training, LABEL_PROMPTS
from candidate_screener.entity_annotation.text import digest


class Processor:
    transformer_tokenizer = SimpleNamespace(model_max_length=48)

    def words_splitter(self, text):
        return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def prepare_inputs(self, texts, labels):
        assert list(labels) == list(LABEL_PROMPTS.values())
        return texts, None


class Tokenizer:
    model_max_length = 48
    def __call__(self, prepared, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [[0] * (10 + sum(len(w) for w in row)) for row in prepared]}


class Model:
    config = SimpleNamespace(max_len=48, max_width=4)
    data_processor = Processor()


Model.data_processor.transformer_tokenizer = Tokenizer()


def row(text="neg Python SQL neg tail", start=4, end=14):
    return {"doc_id": "synthetic", "split": "train", "reviewed_all": True,
            "categories": list(LABEL_PROMPTS), "text": text, "source_sha256": digest(text),
            "spans": [{"start": start, "end": end, "surface": text[start:end], "label": "KNOWLEDGE"}]}


def prepare(rows, model=None):
    return prepare_training(rows, model or Model(), sentence_detector=lambda text: [(0, len(text))])


def test_gold_safe_nonoverlap_keeps_every_gold_and_negative_region():
    sample = row("neg Python SQL neg taillong")
    result = prepare([sample])
    assert not result["diagnostics"]["blocked"]
    assert sum(len(r["ner"]) for r in result["records"]) == 1
    assert any(not r["ner"] for r in result["records"])
    assert [word for r in result["records"] for word in r["tokenized_text"]] == sample["text"].split()
    windows = result["windows"]
    assert windows[0]["start"] == 0 and windows[-1]["end"] == len(sample["text"])
    assert all(a["end"] == b["start"] for a, b in zip(windows, windows[1:]))


def test_partial_word_gold_blocks_instead_of_rounding_or_dropping():
    result = prepare([row(start=5)])
    assert result["records"] == []
    assert result["diagnostics"]["unalignable_spans"] == 1
    assert result["diagnostics"]["blocked"]


def test_span_width_blocker_and_actual_prompt_budget_blocker():
    model = Model()
    model.config = SimpleNamespace(max_len=48, max_width=1)
    assert prepare([row()], model)["diagnostics"]["width_exceeded_spans"] == 1
    text = "x" * 23
    assert prepare([row(text, 0, 23)])["diagnostics"]["budget_exceeded_spans"] == 1


def test_heldout_and_unreviewed_never_train():
    for update in ({"split": "test"}, {"split": "dev"}, {"reviewed_all": False}):
        with pytest.raises(ValueError, match="train"):
            prepare([dict(row(), **update)])


def test_reviewed_empty_negative_document_retained():
    sample = row()
    sample["spans"] = []
    result = prepare([sample])
    assert result["records"] and all(not r["ner"] for r in result["records"])
    assert not result["diagnostics"]["blocked"]


def test_snapshot_refuses_any_heldout_latest_review(monkeypatch, tmp_path):
    import contextlib
    import candidate_screener.entity_annotation.finetune_data as module
    import candidate_screener.entity_annotation.service as service
    latest = [dict(row(), doc_id=str(i)) for i in range(10)]
    latest[-1]["split"] = "dev"
    fake = SimpleNamespace(_locked=lambda: contextlib.nullcontext(), campaign=lambda: {}, records=lambda: [])
    monkeypatch.setattr(service, "AnnotationService", lambda path: fake)
    monkeypatch.setattr(module, "build_exports", lambda *args: latest)
    with pytest.raises(ValueError, match="training documents"):
        module.freeze_snapshot(tmp_path / "manifest.json", tmp_path / "destination")
