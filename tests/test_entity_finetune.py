"""Dependency-free guards for frozen-backbone adaptation and snapshot identity."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from candidate_screener.entity_annotation.finetune import _snapshot, freeze_backbone, apply_tokenizer


def test_frozen_backbone_keeps_only_representation_trainable():
    class Parameter:
        def requires_grad_(self, value):
            self.requires_grad = value
    backbone, head = Parameter(), Parameter()
    model = SimpleNamespace(model=SimpleNamespace(named_parameters=lambda: [
        ("token_rep_layer.encoder", backbone), ("span_rep_layer.weight", head)]))
    names = freeze_backbone(model)
    assert backbone.requires_grad is False and head.requires_grad is True
    assert names["frozen"] == ["token_rep_layer.encoder"]


def synthetic_snapshot(tmp_path):
    rows = [{"doc_id": str(i), "text": "Python", "source_sha256": hashlib.sha256(b"Python").hexdigest(),
             "categories": ["KNOWLEDGE", "SKILL", "TITLE", "DEGREE"],
             "split": "train", "reviewed_all": True} for i in range(10)]
    def save():
        encoded = "\n".join(json.dumps(row) for row in rows)
        (tmp_path / "gold.jsonl").write_text(encoded)
        (tmp_path / "manifest.json").write_text(json.dumps({"jsonl_sha256": hashlib.sha256(encoded.encode()).hexdigest()}))
    save()
    return rows, save


def test_training_snapshot_excludes_held_out_documents(tmp_path):
    rows, save = synthetic_snapshot(tmp_path)
    assert len(_snapshot(tmp_path)[0]) == 10
    rows[3]["split"] = "test"
    save()
    with pytest.raises(ValueError, match="ten completely reviewed train"):
        _snapshot(tmp_path)


def test_training_snapshot_changed_bytes_cannot_resume(tmp_path):
    synthetic_snapshot(tmp_path)
    with (tmp_path / "gold.jsonl").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="snapshot digest"):
        _snapshot(tmp_path)


def test_character_policy_restores_all_span_geometry_without_parameters(monkeypatch):
    from candidate_screener.entity_annotation import character_tokenization
    monkeypatch.setattr(character_tokenization, "BoundaryTokenizer", lambda policy: lambda text: [(text, 0, len(text))])
    config = SimpleNamespace(span_mode="markerV0", max_width=12)
    processor_config = SimpleNamespace(max_width=12)
    head = SimpleNamespace(max_width=12)
    model = SimpleNamespace(config=config, data_processor=SimpleNamespace(config=processor_config),
                            model=SimpleNamespace(span_rep_layer=SimpleNamespace(span_rep_layer=head),
                                                  state_dict=lambda: {"head": SimpleNamespace(shape=(3, 3))}))
    policy = {"max_width_units": 24}
    apply_tokenizer(model, policy)
    assert config.max_width == processor_config.max_width == head.max_width == 24
    assert model.tokenizer_policy is policy
    assert list(model.data_processor.words_splitter("x")) == [("x", 0, 1)]


def test_character_policy_refuses_unverified_head_geometry(monkeypatch):
    from candidate_screener.entity_annotation import character_tokenization
    monkeypatch.setattr(character_tokenization, "BoundaryTokenizer", lambda policy: None)
    model = SimpleNamespace(config=SimpleNamespace(span_mode="marker"))
    with pytest.raises(ValueError, match="markerV0"):
        apply_tokenizer(model, {"max_width_units": 24})
