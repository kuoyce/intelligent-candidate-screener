"""Saved medium comparison must not present incomplete evidence as absence."""
import ast
import json

import pandas as pd
import pytest

from candidate_screener.config import NOTEBOOKS
from candidate_screener.extraction.medium_report import comparison_tables, save_comparison


def sample():
    return {"pairs": pd.DataFrame([{"split": "train", "resume_id": "c", "jd_id": "j"}]),
            "statuses": pd.DataFrame([
                {"doc_type": "cv", "doc_id": "c", "status": "complete_zero", "runtime_seconds": 1.0},
                {"doc_type": "jd", "doc_id": "j", "status": "incomplete", "runtime_seconds": 2.0}]),
            "entities": pd.DataFrame([{"doc_type": "jd", "doc_id": "j", "label": "SKILL",
                                       "surface": "SENSITIVE SOURCE", "confidence": .9}]),
            "metadata": {"campaign_id": "a" * 24, "provenance": {
                "source_sha256": {"train": "b" * 64, "test": "c" * 64}}}}


def test_comparison_excludes_incomplete_predictions_retains_requested_denominator():
    tables = comparison_tables(sample(), sample())
    jd = tables["labels"].query("split == 'train' and doc_type == 'jd'")
    assert jd.requested_documents.eq(1).all()
    assert jd.completed_documents.eq(0).all()
    assert jd.spans.eq(0).all()
    assert jd.score_median.isna().all()
    cv = tables["operations"].query("split == 'train' and doc_type == 'cv'")
    assert cv.complete_zero.eq(1).all()
    assert cv.completed_documents.eq(1).all()
    assert cv.chunks.isna().all()


def test_comparison_refuses_different_pair_populations():
    medium = sample()
    medium["pairs"].loc[0, "jd_id"] = "different"
    with pytest.raises(ValueError, match="populations"):
        comparison_tables(sample(), medium)


def test_comparison_refuses_changed_source_text_with_identical_pair_ids():
    medium = sample()
    medium["metadata"]["provenance"]["source_sha256"]["train"] = "d" * 64
    with pytest.raises(ValueError, match="source digests"):
        comparison_tables(sample(), medium)


def test_saved_aggregates_never_contain_document_ids_or_mentions(tmp_path):
    save_comparison(sample(), sample(), tmp_path)
    for path in tmp_path.iterdir():
        text = path.read_text()
        assert "SENSITIVE SOURCE" not in text
        assert "doc_id" not in text
        assert "surface" not in text


def test_medium_notebook_is_thin_and_display_is_bounded():
    notebook = json.loads((NOTEBOOKS / "29-gliner-medium-sentence-comparison.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "load_baseline(BASELINE_CAMPAIGN_ID)" in code
    assert "load_medium()" in code
    for node in ast.walk(ast.parse(code)):
        assert not isinstance(node, (ast.FunctionDef, ast.ClassDef))
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            assert name not in {"run_campaign", "from_pretrained", "fit", "extract_medium_document"}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if "print(view" in source:
            assert ".head(5)" in source and "[:1999]" in source
        outputs = cell.get("outputs", [])
        assert all(o["output_type"] != "error" for o in outputs)
        assert all("text/html" not in o.get("data", {}) for o in outputs)
        text = "".join("".join(o.get("text", [])) + "".join(o.get("data", {}).get("text/plain", [])) for o in outputs)
        assert len(text) <= 2000
