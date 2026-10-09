"""EDA stays on saved evidence and bounded inspection outputs, without a model."""
import ast
import json

from candidate_screener.config import NOTEBOOKS

NAME = "14-entity-extraction-eda.ipynb"


def notebook():
    return json.loads((NOTEBOOKS / NAME).read_text(encoding="utf-8"))


def test_eda_notebook_loads_saved_run_without_neural_inference():
    code = "\n".join("".join(cell["source"]) for cell in notebook()["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(code)
    assert "load_analysis(METRICS_PATH)" in code
    assert "save_tables(tables, EDA_OUTPUT)" in code
    assert "save_charts(tables," in code
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            assert name not in {"run_transformer", "load_model", "extract_transformer_document"}


def test_eda_real_example_categories_have_bounded_redacted_context():
    code = "\n".join("".join(cell["source"]) for cell in notebook()["cells"] if cell["cell_type"] == "code")
    for category in ("shared", "rule_only", "neural_only"):
        assert f'category="{category}", limit=5, context_chars=60' in code
    assert code.count('assert_clean(examples["surface"]') == 3
    assert code.count('assert_clean(examples["context"]') == 3
    assert code.count('len(examples) <= 5 and len(rendered) <= 1990') == 3


def test_eda_saved_text_bounded_and_figures_are_images():
    for index, cell in enumerate(notebook()["cells"]):
        text = []
        for output in cell.get("outputs", []):
            assert output["output_type"] != "error", index
            if "text" in output:
                text.extend(output["text"])
            else:
                data = output.get("data", {})
                assert "text/html" not in data, index
                text.extend(data.get("text/plain", []))
                assert not any(key.startswith("image/") and key != "image/png" for key in data)
        assert len("".join(text)) <= 2000, index
