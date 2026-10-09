"""Saved previews must not expose unreviewed neural spans or inline models."""
import ast
import json

import pytest

from candidate_screener.config import NOTEBOOKS
from candidate_screener.extraction.display import preview_entities


def test_real_neural_name_is_suppressed_even_with_technology_label():
    records = [{"doc_id": "real-1", "source": "djinni-cv", "doc_type": "cv",
                "surface": "Jane Candidate", "label": "KNOWLEDGE", "method": "gliner"}]
    assert preview_entities(records) == []


def test_only_seed_aliases_can_enter_real_rule_preview():
    common = {"doc_id": "real-1", "source": "djinni-cv", "doc_type": "cv",
              "method": "seed-rule", "label": "KNOWLEDGE", "field": "Moreinfo",
              "context_status": "unqualified"}
    records = [{**common, "surface": "Jane Candidate"}, {**common, "surface": "Python"}]
    assert preview_entities(records) == [{"surface": "Python", "label": "KNOWLEDGE",
                                         "field": "Moreinfo", "context_status": "unqualified", "doc_type": "cv"}]


def test_synthetic_preview_redacts_contacts_and_does_not_change_offsets():
    entity = {"doc_id": "synthetic-test", "source": "synthetic", "doc_type": "cv",
              "surface": "demo@example.com", "label": "TITLE", "field": "Moreinfo",
              "method": "gliner", "start": 3, "end": 19, "context_status": "unqualified"}
    assert preview_entities([entity])[0]["surface"] == "[EMAIL]"
    assert entity["surface"] == "demo@example.com"
    assert (entity["start"], entity["end"]) == (3, 19)


@pytest.mark.parametrize("limit", [-1, 6, 1.5])
def test_preview_refuses_unbounded_record_counts(limit):
    with pytest.raises(ValueError, match="five rows"):
        preview_entities([], limit=limit)


def test_preview_limits_rows_and_surface_length():
    row = {"doc_id": "synthetic-test", "source": "synthetic", "surface": "X" * 500}
    assert len(preview_entities([row] * 10)) == 5
    assert len(preview_entities([row])[0]["surface"]) == 100
    assert preview_entities([row], limit=0) == []


def test_transformer_notebook_uses_package_inference_and_synthetic_previews():
    notebook = json.loads((NOTEBOOKS / "13-transformer-entity-extraction.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    sources = "\n".join(cells)
    assert "run_transformer(" in sources
    assert "candidate_screener.extraction.transformer_run" in sources
    assert "assert_clean(" in sources
    assert "synthetic" in sources
    assert "pd.read_parquet" not in sources
    assert '["provenance"]' not in sources
    definitions = [node.name for src in cells for node in ast.walk(ast.parse(src))
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    assert definitions == ["show_summary"]
    assert "GLiNER.from_pretrained" not in sources
    assert "AutoModel" not in sources


def test_transformer_notebook_saved_output_limits():
    notebook = json.loads((NOTEBOOKS / "13-transformer-entity-extraction.ipynb").read_text())
    for i, cell in enumerate(notebook["cells"]):
        text = []
        for output in cell.get("outputs", []):
            assert output["output_type"] != "error", i
            assert "data" not in output, "Use bounded text summaries"
            text.extend(output.get("text", []))
        assert len("".join(text)) <= 2000, i
