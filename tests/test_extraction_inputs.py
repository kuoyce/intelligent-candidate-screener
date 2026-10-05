"""Input guard invariants with no downloaded data."""
import pandas as pd
import pytest

from candidate_screener.extraction.inputs import audit_sources, select_training


def fixture_frames():
    frame = pd.DataFrame({"id": ["a", "b", "c"], "Position": ["Engineer", None, "Developer"]})
    partition = pd.DataFrame({"doc_id": ["a", "b", "c"], "doc_type": ["cv"] * 3,
                              "region": ["train", "eval", "train"]})
    return frame, partition


def test_eval_documents_never_enter_extraction_sample():
    frame, partition = fixture_frames()
    selected = select_training(frame, partition, "cv", 1000, 0)
    assert set(selected.id) == {"a", "c"}


def test_stable_sample_ignores_raw_and_partition_order():
    frame, partition = fixture_frames()
    first = select_training(frame, partition, "cv", 1, 1)
    second = select_training(frame.iloc[::-1], partition.iloc[::-1], "cv", 1, 1)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("defect", ["duplicate_raw", "duplicate_partition", "unknown", "region"])
def test_refuses_invalid_full_partition_before_sampling(defect):
    frame, partition = fixture_frames()
    if defect == "duplicate_raw":
        frame.loc[2, "id"] = "a"
    elif defect == "duplicate_partition":
        partition = pd.concat([partition, partition.iloc[:1]], ignore_index=True)
    elif defect == "unknown":
        partition.loc[2, "doc_id"] = "unknown"
    else:
        partition.loc[2, "region"] = "unexpected"
    with pytest.raises(ValueError):
        select_training(frame, partition, "cv", 1, 0)


def test_missing_optional_sources_visible(tmp_path):
    audit = audit_sources(tmp_path)
    assert audit["skillspan"]["status"] == "missing"
    assert audit["dataturks"]["missing_artifacts"] == ["traindata.json", "testdata.json"]


def test_selected_original_fields_are_not_modified():
    frame, partition = fixture_frames()
    frame.loc[0, "Position"] = "  C++ Developer\n"
    before = frame.copy(deep=True)
    selected = select_training(frame, partition, "cv", 10, 0)
    assert selected.loc[selected.id == "a", "Position"].item() == "  C++ Developer\n"
    pd.testing.assert_frame_equal(frame, before)


def test_run_refuses_corrupted_manifest(tmp_path, monkeypatch):
    import json
    from candidate_screener.extraction import run

    payload = {"documents": [{"doc_id": "synthetic", "source": "djinni-cv", "doc_type": "cv",
                               "fields": {"Position": "Python developer"}}],
               "summary": {"cv": {"sampled_documents": 1}}, "availability": {},
               "provenance": {"sample_size": 1, "seed": 0, "region": "train"}}
    monkeypatch.setattr(run, "prepare_inputs", lambda *args: payload)
    result = run.run_extraction(1, artifact_root=tmp_path)
    assert run.validate_run(result)["reproduces"]
    manifest = tmp_path / result["run_id"] / "sample-manifest.json"
    manifest.write_text(json.dumps({"corrupted": True}))
    with pytest.raises(ValueError, match="manifest"):
        run.validate_run(result)
    with pytest.raises(ValueError, match="digest"):
        run.run_extraction(1, artifact_root=tmp_path)
