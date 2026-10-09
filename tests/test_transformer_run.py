"""Transformer artifact guards run without neural dependencies or weights."""
import json
import sys
import types

import pytest

from candidate_screener.extraction import transformer_run as tr


def test_experience_years_summary_keeps_span_and_document_coverage():
    entities = [{"doc_id": "cv-a", "doc_type": "cv", "label": "EXPERIENCE_YEARS", "confidence": .8},
                {"doc_id": "cv-a", "doc_type": "cv", "label": "EXPERIENCE_YEARS", "confidence": .9}]
    documents = [{"doc_id": "cv-a", "doc_type": "cv", "fields": {"Moreinfo": "5 years; 2 years"}},
                 {"doc_id": "jd-a", "doc_type": "jd", "fields": {"Description": "No duration stated"}}]
    fields = [{"doc_id": doc["doc_id"], "doc_type": doc["doc_type"], "complete": True} for doc in documents]
    summary = tr._coverage_summary(entities, documents, fields)
    assert summary["cv"]["by_label"]["EXPERIENCE_YEARS"] == 2
    assert summary["cv"]["documents_with_label"]["EXPERIENCE_YEARS"] == 1
    assert summary["jd"]["by_label"]["EXPERIENCE_YEARS"] == 0
    assert summary["jd"]["documents_with_label"]["EXPERIENCE_YEARS"] == 0


@pytest.fixture
def mocked_run(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    manifest = {"revision": "a" * 40, "model_id": "synthetic",
                "artifact_sha256": {"config.json": tr.sha256(model_dir / "config.json")}}
    (model_dir / "model-manifest.json").write_text(json.dumps(manifest))
    doc = {"doc_id": "synthetic", "source": "djinni-cv", "doc_type": "cv", "fields": {"Position": "Python"}}
    payload = {"documents": [doc], "availability": {}, "summary": {},
               "provenance": {"sample_size": 1, "seed": 0, "region": "train"}}
    entity = {"entity_id": "synthetic-entity", "doc_id": "synthetic", "source": "djinni-cv", "doc_type": "cv",
              "field": "Position", "text_version": "raw-v1", "start": 0, "end": 6,
              "label": "KNOWLEDGE", "surface": "Python", "canonical_id": None,
              "method": "gliner-zero-shot", "extractor_version": "gliner-v1", "confidence": .8,
              "context_status": "unqualified"}
    calls = []
    adapter = types.ModuleType("candidate_screener.extraction.transformer")
    adapter.CONTEXT_VERSION = "synthetic-context"
    adapter.MODEL_LABELS = {"tool": "KNOWLEDGE"}
    adapter.TRANSFORMER_VERSION = "synthetic-transformer"
    adapter.load_model = lambda *args, **kwargs: object()
    adapter.chunk_parameters = lambda model: {"token_limit": 32, "word_limit": 16}
    def extract(*args, **kwargs):
        calls.append(True)
        return [entity.copy()], {"Position": {"intervals": [[0, 6]], "chunks": 1, "complete": True,
                                              "quarantined_predictions": 0}}
    adapter.extract_transformer_document = extract
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(set_num_threads=lambda _: None))
    monkeypatch.setattr(tr, "prepare_inputs", lambda *args: payload)
    monkeypatch.setattr(tr, "_code", lambda: {"synthetic": "fixed"})
    monkeypatch.setattr(tr, "_dependencies", lambda: {"synthetic": "fixed"})
    # Adapter confidence validation is covered independently; use real span guard here.
    monkeypatch.setattr(tr, "_validate", lambda records, docs: {"entities_checked": len(records),
        "exact_field_offsets": all(doc["fields"][e["field"]][e["start"]:e["end"]] == e["surface"] for e in records)})
    return model_dir, tmp_path / "runs", calls


def test_cached_transformer_artifact_does_not_repeat_inference(mocked_run):
    model_dir, root, calls = mocked_run
    result = tr.run_transformer(1, model_dir=model_dir, artifact_root=root)
    assert result["summary"]["status"] == "complete"
    assert tr.validate_transformer_run(result)["checkpoint_hashes"]
    assert tr.run_transformer(1, model_dir=model_dir, artifact_root=root) == result
    assert len(calls) == 1


def test_checkpoint_corruption_refused(mocked_run):
    model_dir, root, _ = mocked_run
    tr.run_transformer(1, model_dir=model_dir, artifact_root=root)
    (model_dir / "config.json").write_text("corrupt")
    with pytest.raises(ValueError, match="digest"):
        tr.run_transformer(1, model_dir=model_dir, artifact_root=root)


def test_saved_transformer_entities_corruption_refused(mocked_run):
    model_dir, root, _ = mocked_run
    result = tr.run_transformer(1, model_dir=model_dir, artifact_root=root)
    (root / result["run_id"] / "entities.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        tr.validate_transformer_run(result)


def test_zero_budget_refused_before_inference(mocked_run):
    model_dir, root, calls = mocked_run
    with pytest.raises(ValueError):
        tr.run_transformer(1, max_seconds=0, model_dir=model_dir, artifact_root=root)
    assert not calls


@pytest.mark.parametrize("configuration", [{"max_seconds": float("nan")}, {"max_seconds": float("inf")},
                                            {"threshold": float("nan")}, {"threshold": float("inf")}])
def test_nonfinite_inference_configuration_refused(mocked_run, configuration):
    model_dir, root, calls = mocked_run
    with pytest.raises(ValueError, match="Invalid"):
        tr.run_transformer(1, model_dir=model_dir, artifact_root=root, **configuration)
    assert not calls


def test_live_runtime_version_mismatch_refused(mocked_run):
    model_dir, root, calls = mocked_run
    adapter = sys.modules["candidate_screener.extraction.transformer"]
    adapter.load_model = lambda *args, **kwargs: types.SimpleNamespace(runtime_versions={"synthetic": "changed"})
    with pytest.raises(ValueError, match="Live transformer runtime"):
        tr.run_transformer(1, model_dir=model_dir, artifact_root=root)
    assert not calls


def test_missing_model_has_clear_gate(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pinned local transformer manifest missing"):
        tr._model_manifest(tmp_path)


def test_unknown_model_path_is_refused(tmp_path):
    (tmp_path / "model-manifest.json").write_text(json.dumps({"revision": "abc", "artifact_sha256": {"../bad": "hash"}}))
    with pytest.raises(ValueError, match="unsafe"):
        tr._model_manifest(tmp_path)


def test_partial_field_never_claims_document_completion(mocked_run, monkeypatch):
    model_dir, root, _ = mocked_run
    adapter = sys.modules["candidate_screener.extraction.transformer"]
    adapter.extract_transformer_document = lambda *args, **kwargs: ([], {
        "Position": {"intervals": [], "chunks": 0, "complete": False,
                     "quarantined_predictions": 0, "interrupted": True}})
    result = tr.run_transformer(1, model_dir=model_dir, artifact_root=root)
    assert result["summary"]["status"] == "partial"
    assert result["summary"]["completed_documents"] == 0
    assert result["summary"]["rule_comparison"]["completed_documents_compared"] == 0
    assert tr.validate_transformer_run(result)["status"] == "partial"


def test_complete_field_with_uncovered_tail_is_rejected(mocked_run):
    model_dir, root, _ = mocked_run
    result = tr.run_transformer(1, model_dir=model_dir, artifact_root=root)
    path = root / result["run_id"] / "field-coverage.json"
    records = json.loads(path.read_text())
    records[0]["intervals"] = [[0, 4]]
    path.write_text(json.dumps(records))
    result["artifact_sha256"]["field-coverage.json"] = tr.sha256(path)
    metadata = root / result["run_id"] / "metadata.json"
    metadata.write_text(tr._json({k: v for k, v in result.items() if k != "run_dir"}))
    with pytest.raises(ValueError, match="uncovered tail"):
        tr.validate_transformer_run(result)
