import hashlib
import pytest

from candidate_screener.entity_annotation.export import build_exports
from candidate_screener.entity_annotation.evaluation import score_predictions


def score(gold, predictions, **kwargs):
    metadata = {row["doc_id"]: {"source_sha256": row.get("source_sha256"), "status": "complete"} for row in gold}
    return score_predictions(gold, predictions, prediction_documents=metadata, **kwargs)


def fixture():
    campaign = {"campaign_id": "synthetic", "labels": ["SKILL"], "documents": [
        {"doc_id": "one", "side": "cv", "split": "test", "source_text": "Python"}]}
    records = [{"doc_id": "one", "annotator": "human", "revision": 1,
                "campaign_id": "synthetic", "guideline_version": "entity-guide-v1",
                "source_sha256": hashlib.sha256(b"Python").hexdigest(),
                "status": "reviewed", "reviewed_all": True,
                "spans": [{"start": 0, "end": 6, "label": "SKILL", "surface": "Python"}]}]
    return campaign, records


def test_gold_rejects_reviewed_but_incomplete():
    campaign, records = fixture()
    records[0]["reviewed_all"] = False
    with pytest.raises(ValueError, match="whole-document"):
        build_exports(campaign, records, "human")


def test_assisted_annotations_preserve_provenance_and_cannot_be_blind_gold():
    campaign, records = fixture()
    campaign["documents"][0]["split"] = "train"
    campaign["annotators"] = ["human"]
    records[0].update(assistance={"suggestion_set_id": "set123", "model_id": "model123"},
                      annotation_mode="model_assisted")
    rows = build_exports(campaign, records, "human")
    assert rows[0]["assistance"] == records[0]["assistance"]
    assert rows[0]["annotation_mode"] == "model_assisted"
    with pytest.raises(ValueError, match="model-assisted"):
        score(rows, [], bootstrap_samples=20)


def test_latest_inprogress_never_exports_stale_gold():
    campaign, records = fixture()
    records.append(dict(records[0], revision=2, status="in_progress"))
    assert build_exports(campaign, records, "human") == []


def test_export_rejects_source_or_guideline_change():
    campaign, records = fixture()
    records[0]["source_sha256"] = "forged"
    with pytest.raises(ValueError, match="digest"):
        build_exports(campaign, records, "human")


def test_exact_scoring_filters_unselected_labels_and_preserves_undefined():
    campaign, records = fixture()
    gold = build_exports(campaign, records, "human")
    predictions = [{"doc_id": "one", "start": 0, "end": 6, "label": "SKILL"},
                   {"doc_id": "one", "start": 0, "end": 6, "label": "TITLE"}]
    result = score(gold, predictions, bootstrap_samples=20)
    assert result["total"]["f1"] == 1
    predictions[0]["end"] = 5
    assert score(gold, predictions, bootstrap_samples=20)["total"]["f1"] == 0
    gold[0]["spans"] = []
    empty = score(gold, [], bootstrap_samples=20)["total"]
    assert empty["f1"] is None and empty["ci95"]["f1"] is None


def test_incomplete_gold_cannot_be_scored():
    with pytest.raises(ValueError, match="complete reviewed"):
        score_predictions([{"reviewed_all": False}], [], prediction_documents={})


def test_missing_predictions_are_not_completed_zero():
    campaign, records = fixture()
    gold = build_exports(campaign, records, "human")
    with pytest.raises(ValueError, match="complete document coverage"):
        score_predictions(gold, [], prediction_documents={})


def test_scoring_rejects_forged_gold_and_duplicate_spans():
    campaign, records = fixture()
    gold = build_exports(campaign, records, "human")
    gold[0]["spans"] *= 2
    with pytest.raises(ValueError, match="Duplicate gold"):
        score(gold, [])
    gold[0]["source_sha256"] = "forged"
    with pytest.raises(ValueError, match="Gold source digest"):
        score(gold, [])


def test_export_rejects_unassigned_annotator_document():
    campaign, records = fixture()
    campaign["assignments"] = {"other": ["one"]}
    with pytest.raises(ValueError, match="not assigned"):
        build_exports(campaign, records, "human")


def test_export_revision_publication_is_immutable_and_complete(tmp_path, monkeypatch):
    from candidate_screener.entity_annotation import export
    monkeypatch.setattr(export, "PROJECT_ROOT", tmp_path)
    campaign, records = fixture()
    rows = build_exports(campaign, records, "human")
    destination = tmp_path / "data/gold/revision-1"
    result = export.write_exports(rows, destination)
    assert result["parquet_sha256"]
    assert result["source_revisions"][0]["revision"] == 1
    assert {path.name for path in destination.iterdir()} == {"gold.jsonl", "gold.parquet", "manifest.json"}
    with pytest.raises(FileExistsError):
        export.write_exports(rows, destination)
    with pytest.raises(ValueError, match="No fully reviewed"):
        export.write_exports([], tmp_path / "data/gold/empty")


def test_duplicate_source_cannot_cross_splits():
    campaign, records = fixture()
    campaign["documents"].append(dict(campaign["documents"][0], doc_id="two", split="train"))
    with pytest.raises(ValueError, match="crosses entity splits"):
        build_exports(campaign, records, "human")


def test_flat_gold_rejects_overlapping_categories():
    campaign, records = fixture()
    records[0]["spans"].append({"start": 0, "end": 3, "label": "SKILL", "surface": "Pyt"})
    with pytest.raises(ValueError, match="overlapping"):
        build_exports(campaign, records, "human")


@pytest.mark.parametrize("key", ["campaign_id", "source_sha256", "guideline_version"])
def test_export_requires_provenance(key):
    campaign, records = fixture()
    del records[0][key]
    with pytest.raises(ValueError):
        build_exports(campaign, records, "human")
