"""Saved fit reporting keeps completed zeros separate from unknown partial results."""
import ast
import json

import pandas as pd
import pytest

from candidate_screener.config import NOTEBOOKS
from candidate_screener.extraction.fit_plots import LABELS, label_document_counts, pair_mentions, save_fit_charts, split_views


def campaign():
    pairs = pd.DataFrame([
        {"split":"train", "resume_id":"c0", "jd_id":"j0"},
        {"split":"test", "resume_id":"c1", "jd_id":"j1"},
    ])
    statuses = pd.DataFrame([
        {"doc_type":"cv","doc_id":"c0","status":"complete_zero"},
        {"doc_type":"jd","doc_id":"j0","status":"pending"},
        {"doc_type":"cv","doc_id":"c1","status":"complete"},
        {"doc_type":"jd","doc_id":"j1","status":"incomplete"},
    ])
    entities = pd.DataFrame([
        {"doc_type":"cv","doc_id":"c1","label":"KNOWLEDGE","surface":"Python","confidence":.8},
        {"doc_type":"jd","doc_id":"j1","label":"KNOWLEDGE","surface":"Python","confidence":.9},
    ])
    return {"pairs":pairs,"statuses":statuses,"entities":entities}


def test_completed_zero_retains_all_five_categories():
    counts = label_document_counts(campaign(), "train", "cv")
    assert list(counts.columns) == list(LABELS)
    assert list(counts.index) == ["c0"]
    assert counts.iloc[0].tolist() == [0] * 5


def test_incomplete_predictions_never_enter_completed_distributions():
    c = campaign()
    assert label_document_counts(c, "test", "jd").empty
    assert split_views(c, "test", "jd")[1].empty
    assert len(split_views(c, "test", "jd")[0]) == 1


def test_pair_overlap_unknown_state_is_not_zero_or_a_match():
    table = pair_mentions(campaign(), "c1", "j1")
    assert len(table) == 5
    assert table.shared_strings.isna().all()
    assert table.cv_unique_strings.isna().all()


def test_missing_status_refuses_population_denominator():
    c = campaign()
    c["statuses"] = c["statuses"].loc[c["statuses"].doc_id != "c0"]
    with pytest.raises(ValueError, match="population"):
        split_views(c, "train", "cv")


def test_saved_fit_charts_handle_complete_zero_and_unfinished_inputs(tmp_path):
    paths = save_fit_charts(campaign(), tmp_path)
    assert set(paths) == {"counts","states","coverage","distribution","scores"}
    from pathlib import Path
    assert all(Path(path).read_bytes().startswith(b"\x89PNG") for path in paths.values())


def notebook():
    return json.loads((NOTEBOOKS / "28-fit-gliner-comparison.ipynb").read_text())


def test_notebook_reports_saved_campaign_without_inference():
    code = "\n".join("".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "code")
    definitions = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append(node.name)
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func,"attr", "")
            assert name not in {"run_campaign","load_model","from_pretrained","fit","extract_transformer_document"}
    assert definitions == ["show_summary"]
    assert "load_campaign()" in code
    assert "pair_mentions(" in code
    assert "assert_clean(preview[\"context\"]" in code


def test_fit_notebook_saved_output_is_bounded():
    for i, cell in enumerate(notebook()["cells"]):
        text = []
        for output in cell.get("outputs", []):
            assert output["output_type"] != "error", i
            text.extend(output.get("text", []))
            data = output.get("data", {})
            assert "text/html" not in data
            text.extend(data.get("text/plain", []))
        assert len("".join(text)) <= 2000, i
