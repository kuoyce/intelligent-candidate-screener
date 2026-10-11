"""Selected-weight restoration must fail closed before publishing a cache."""
import json

import pytest

from candidate_screener.extraction import role_family as rf


def test_restore_rejects_a_new_selection_before_loading_data(tmp_path, monkeypatch):
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"selection": {"chosen": "minilm"}, "config": rf.CONFIG}))
    monkeypatch.setattr(rf, "RECORD", record)
    monkeypatch.setattr(rf, "load_train_documents", lambda: pytest.fail("must not load data"))
    with pytest.raises(ValueError, match="selected classical"):
        rf.restore_selected()


def test_restore_rejects_partition_mismatch_before_fitting(tmp_path, monkeypatch):
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"selection": {"chosen": "tfidf"}, "config": rf.CONFIG,
                                 "fingerprint": "original", "label_space": {"labels": ["A"]}}))
    monkeypatch.setattr(rf, "RECORD", record)
    monkeypatch.setattr(rf, "load_train_documents", lambda: {})
    monkeypatch.setattr(rf, "prepare", lambda docs: {"labels": ["A"]})
    monkeypatch.setattr(rf, "config_fingerprint", lambda prep: "changed")
    monkeypatch.setattr(rf, "fit_tfidf", lambda *args: pytest.fail("must not fit"))
    monkeypatch.setattr(rf, "save_model", lambda *args: pytest.fail("must not publish"))
    with pytest.raises(ValueError, match="partition/config/label"):
        rf.restore_selected()


def test_restore_cli_does_not_call_final_measurement(monkeypatch, capsys):
    monkeypatch.setattr(rf, "restore_selected", lambda: {"chosen": "tfidf"})
    monkeypatch.setattr(rf, "build", lambda **kwargs: pytest.fail("must not measure final"))
    assert rf.main(["--restore-selected"]) == 0
    assert '"tfidf"' in capsys.readouterr().out
