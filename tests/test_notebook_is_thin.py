"""The notebook may plot, tabulate and explain. It may not build a model.

This is the check that answers "a future edit to the notebook can't silently change
what the baseline means". Before this phase the baseline existed *only* as notebook
cells: skill extraction, the shared vectoriser, the cosine scorer, the BM25 tokeniser
and the evaluation were all inline, so "regenerate the baseline" meant "re-run a
notebook and trust whoever last edited it" — and three bugs shipped that way.

It fails on exactly the regression it is aimed at: someone pasting model code back
inline instead of importing it *(design spec §5.3)*.
"""
from __future__ import annotations

import json

import pytest

from candidate_screener.config import NOTEBOOKS

NOTEBOOK = NOTEBOOKS / "07-baseline-tfidf-keyword.ipynb"

#: Constructing a model, or restating the keyword list, is what must not reappear.
FORBIDDEN = {
    "TfidfVectorizer(": "fit the vectoriser via baselines.tfidf.fit_shared_vectorizer",
    "BM25Okapi(": "fit BM25 via baselines.bm25.fit_bm25",
    "SKILL_PATTERNS =": "import SKILL_PATTERNS from baselines.skills; do not restate it",
    "LogisticRegression(": "fit via baselines.classifier.fit_single_feature, which "
                           "pins the solver and raises on non-convergence",
}


def code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_exists():
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing"


@pytest.mark.parametrize("fragment,remedy", sorted(FORBIDDEN.items()))
def test_notebook_does_not_build_the_model(fragment, remedy):
    offenders = [i for i, src in enumerate(code_cells()) if fragment in src]
    assert not offenders, (
        f"{NOTEBOOK.name} code cell(s) {offenders} contain {fragment!r}. The notebook "
        f"reports the baseline, it does not define it — {remedy}.")


def test_notebook_reads_the_committed_metrics():
    """It must *cite* the golden record, not restate figures in prose.

    `AGENTS.md` §Notebooks already requires this for `profile-metrics.json`; the
    same rule is what stops re-running the notebook from *becoming* the definition
    of the baseline.
    """
    sources = "\n".join(code_cells())
    assert "BASELINE_METRICS" in sources or "baseline-metrics.json" in sources
