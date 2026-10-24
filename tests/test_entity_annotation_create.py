"""Corpus adapters preserve original splits and never use pair labels as gold."""
import pandas as pd
import pytest

from candidate_screener.entity_annotation.create import a1_documents


def write(root, split, rows):
    pd.DataFrame(rows, columns=["resume_id", "jd_id", "resume_text", "job_description_text", "label"]).to_parquet(root / f"{split}.parquet", index=False)


def test_a1_unique_documents_and_cross_split_overlap_excluded(tmp_path):
    write(tmp_path, "train", [["r1", "j1", "Python", "Developer", "Good Fit"],
                               ["r1", "j2", "Python", "Scientist", "No Fit"]])
    write(tmp_path, "test", [["r1", "j3", "Python", "Accountant", "No Fit"],
                              ["r2", "j3", "Java", "Accountant", "Good Fit"]])
    docs = a1_documents(tmp_path)
    assert not any(doc["doc_id"] == "r1" for doc in docs)
    assert len(docs) == 4
    assert {doc["source_split"] for doc in docs} == {"train", "test"}
    assert all("label" not in doc for doc in docs)


def test_a1_conflicting_identity_refused(tmp_path):
    write(tmp_path, "train", [["r1", "j1", "Python", "Developer", "Fit"]])
    write(tmp_path, "test", [["r1", "j2", "Changed", "Scientist", "Fit"]])
    with pytest.raises(ValueError, match="conflicting"):
        a1_documents(tmp_path)
