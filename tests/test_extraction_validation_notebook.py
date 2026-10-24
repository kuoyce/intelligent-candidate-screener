"""The source-inspection notebook reports saved evidence without running a model."""
import ast
import json

from candidate_screener.config import NOTEBOOKS

NAME = "27-entity-extraction-source-validation.ipynb"


def load():
    return json.loads((NOTEBOOKS / NAME).read_text())


def test_validation_notebook_uses_saved_run_without_fitting_or_inference():
    source = "\n".join("".join(c["source"]) for c in load()["cells"] if c["cell_type"] == "code")
    assert "RUN_ID = None" in source
    assert "audit_saved_run(run_id=RUN_ID)" in source
    assert "inventory_runs()" in source and "inventory_gold()" in source
    definitions = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append(node.name)
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            assert name not in {"load_model", "run_transformer", "extract_transformer_document", "from_pretrained", "fit"}
    assert definitions == ["show_summary"]


def test_each_label_and_source_has_bounded_redacted_examples():
    code = "\n".join("".join(c["source"]) for c in load()["cells"] if c["cell_type"] == "code")
    for label in ("KNOWLEDGE", "SKILL", "TITLE", "DEGREE"):
        for kind in ("cv", "jd"):
            assert f'label="{label}", doc_type="{kind}", limit=3, context_chars=40' in code
    assert code.count('assert_clean(preview["surface"]') >= 8
    assert code.count('assert_clean(preview["context"]') >= 8


def test_saved_validation_output_is_bounded_and_error_free():
    for index, cell in enumerate(load()["cells"]):
        rendered = []
        for output in cell.get("outputs", []):
            assert output["output_type"] != "error", index
            rendered.extend(output.get("text", []))
            assert "text/html" not in output.get("data", {})
            rendered.extend(output.get("data", {}).get("text/plain", []))
        assert len("".join(rendered)) <= 2000, index
