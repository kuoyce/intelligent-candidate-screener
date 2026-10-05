"""Guard the reporting boundary and saved-output privacy for extraction notebooks."""
from __future__ import annotations

import ast
import json

import pytest

from candidate_screener.config import NOTEBOOKS

NAMES = (
    "10-entity-extraction-inputs.ipynb",
    "11-rule-based-entity-extraction.ipynb",
    "12-entity-extraction-evaluation.ipynb",
)


def load(name):
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", NAMES)
def test_extraction_notebooks_keep_rules_in_package(name):
    """Pasting an extractor or model into a notebook breaks the thin boundary."""
    notebook = load(name)
    trees = [ast.parse("".join(cell["source"])) for cell in notebook["cells"]
             if cell["cell_type"] == "code"]
    definitions = [node.name for tree in trees for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert definitions == ["show_summary"], "Only the bounded display helper belongs inline"
    sources = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "candidate_screener.extraction" in sources
    assert 'prepared["documents"]' not in sources
    assert 'result["provenance"]' not in sources
    assert 'prepared["provenance"]' not in sources


@pytest.mark.parametrize("name", NAMES)
def test_extraction_notebook_saved_outputs_are_bounded(name):
    """Enforce D15 on the committed execution output, including multiple streams."""
    for index, cell in enumerate(load(name)["cells"]):
        if cell["cell_type"] != "code":
            continue
        rendered = []
        for output in cell.get("outputs", []):
            assert output["output_type"] != "error", (name, index)
            if "text" in output:
                rendered.append("".join(output["text"]))
            else:
                data = output.get("data", {})
                assert not any(key.startswith("image/") for key in data), (name, index)
                rendered.append("".join(data.get("text/plain", [])))
                assert "text/html" not in data, "Use capped aggregate JSON, not unbounded tables"
        assert len("".join(rendered)) <= 2000, (name, index)


@pytest.mark.parametrize("name", NAMES[1:])
def test_synthetic_evidence_is_redacted_inside_notebook(name):
    sources = "\n".join("".join(c["source"]) for c in load(name)["cells"] if c["cell_type"] == "code")
    assert "redact_text(" in sources or ".map(redact_text)" in sources
    assert "assert_clean(" in sources
    assert "synthetic_text[e[\"start\"]:e[\"end\"]] == e[\"surface\"]" in sources
