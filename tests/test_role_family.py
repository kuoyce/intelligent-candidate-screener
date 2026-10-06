"""E2 role family (plan 2026-10-06, P3): synthetic data only, no `data/` needed."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from candidate_screener.extraction import role_family as rf


# --- the label -------------------------------------------------------------------------

def test_generic_keywords_split_by_title_with_residual():
    assert rf.role_family("Data Science", "Senior Data Scientist") == "Data Science/data scientist"
    assert rf.role_family("Lead", "CTO") == "Lead/cto"
    assert rf.role_family("Lead", "Chief Happiness Officer") == "Lead/other"
    assert rf.role_family("Other", "Team Lead") == "Other/other"      # empty after normalising


def test_specific_keyword_is_its_own_label_whatever_the_title():
    assert rf.role_family("Python", "Data Scientist") == "Python"
    assert rf.role_family("Java", "") == "Java"


def test_merged_family_takes_the_specific_keyword_label():
    """`Data Science` + `data engineer` is the `Data Engineer` class, not a near-duplicate."""
    assert rf.role_family("Data Science", "Middle Data Engineer") == "Data Engineer"
    assert rf.role_family("Other", "Frontend developer") == "JavaScript"


def test_jd_only_singleton_keyword_folds_into_other():
    """`SAP` exists on the JD side only: it is split like `Other`, so both sides share labels."""
    assert "SAP" not in rf.label_space()
    assert rf.role_family("SAP", "SAP consultant") == "Other/other"
    assert rf.role_family("SAP", "Lawyer") == "Other/lawyer"


def test_missing_keyword_has_no_label():
    assert rf.role_family(None, "Java Developer") is None
    assert rf.role_family(float("nan"), "Java Developer") is None


def test_every_label_is_in_the_shared_space():
    space = set(rf.label_space())
    for kw in [*rf.KEYWORD_CLASSES, *rf.GENERIC_KEYWORDS, "SAP"]:
        for pos in ["Data Scientist", "CTO", "Lawyer", "Nothing At All", ""]:
            assert rf.role_family(kw, pos) in space
    assert len(space) == len(rf.label_space())
    assert {f"{g}/other" for g in rf.GENERIC_KEYWORDS} <= space


def _docs(rows):
    return pd.DataFrame(rows, columns=["Primary Keyword", "Position"])


def test_derive_needs_m_on_both_sides():
    jd = _docs([("Lead", "CTO")] * 3 + [("Lead", "Architect")] * 3 + [("Java", "Java dev")] * 3)
    cv = _docs([("Lead", "CTO")] * 3 + [("Lead", "Architect")] * 2 + [("Java", "Java dev")] * 3)
    space = rf.derive_label_space(jd, cv, m=3)
    assert space["split_families"]["Lead"] == {"cto": "Lead/cto"}
    assert space["keyword_classes"] == ["Java"]


def test_merge_only_into_a_keywords_own_title():
    """Regression: `data scientist` is mostly tagged `Python` outside `Data Science`, but it
    is not Python's own title, so it must stay a `Data Science` family (a modal-keyword-only
    merge rule folded the whole of Data Science into Python)."""
    rows = ([("Data Science", "Data Scientist")] * 5 + [("Data Science", "Data Engineer")] * 5
            + [("Python", "Python Developer")] * 9 + [("Python", "Data Scientist")] * 5
            + [("Data Engineer", "Data Engineer")] * 9)
    space = rf.derive_label_space(_docs(rows), _docs(rows), m=3)
    fams = space["split_families"]["Data Science"]
    assert fams["data scientist"] == "Data Science/data scientist"
    assert fams["data engineer"] == "Data Engineer"


def test_under_supported_keyword_is_folded_in_derivation():
    jd = _docs([("Java", "Java dev")] * 3 + [("SAP", "SAP dev")])
    cv = _docs([("Java", "Java dev")] * 3)
    space = rf.derive_label_space(jd, cv, m=3)
    assert space["folded"] == ["SAP"]


# --- text_v2 ---------------------------------------------------------------------------

def test_cv_text_v2_never_contains_position():
    cv = pd.DataFrame({"Position": ["Zyxwv Quantum Herder"], "CV": ["Built things."],
                       "Highlights": [None], "Moreinfo": [""], "Looking For": ["Remote."]})
    text = rf.cv_text_v2(cv).iloc[0]
    assert "Zyxwv" not in text
    assert text == "Built things.\n\nRemote."


# --- spans -----------------------------------------------------------------------------

ALIASES = {"python developer": "Python", "cto": "Lead/cto", "frontend developer": "JavaScript",
           "dotnet developer": ".NET", "java developer": "Java", "developer": "Other/other"}


def test_span_offsets_index_the_original_text():
    """Regression: `.net` -> ` dotnet` shifts the normalised text, so offsets must be mapped
    back; `text[start:end]` is the surface string."""
    text = "We need a .NET developer and a Front End Developer, reporting to the CTO."
    spans = rf.title_spans(text, ALIASES)
    assert [s.family for s in spans] == [".NET", "JavaScript", "Lead/cto"]
    for s in spans:
        assert text[s.start:s.end] == s.text
    assert [s.text for s in spans] == [".NET developer", "Front End Developer", "CTO"]


def test_spans_are_token_bounded_and_longest_first():
    assert [s.title for s in rf.title_spans("javascript developer", ALIASES)] == ["developer"]
    spans = rf.title_spans("Senior Python Engineer", ALIASES)
    assert [(s.title, s.text) for s in spans] == [("python developer", "Python Engineer")]


def test_spans_empty_text():
    assert rf.title_spans(None, ALIASES) == []
    assert rf.title_spans("", ALIASES) == []


def test_alias_table_purity_and_support():
    titles = pd.Series(["python developer"] * 4 + ["developer"] * 4 + ["cto"] * 2)
    labels = pd.Series(["Python"] * 4 + ["Python", "Java", "Java", "PHP"] + ["Lead/cto"] * 2)
    table = rf.build_alias_table(titles, labels, m=3)
    assert table == {"python developer": "Python"}          # `developer` impure, `cto` too rare


# --- split and sampling ----------------------------------------------------------------

def test_identical_texts_share_a_side():
    texts = ["Same  JD text"] * 20 + ["same jd TEXT"] * 5 + [f"doc {i}" for i in range(200)]
    for seed in range(10):                    # case and whitespace variants never split
        assert len(set(rf.split_by_document(texts, seed=seed, val_share=0.5)[:25])) == 1
    val = rf.split_by_document(texts, seed=0, val_share=0.3)
    assert (rf.split_by_document(texts, seed=0, val_share=0.3) == val).all()
    assert 0.2 < val[25:].mean() < 0.4


def test_floor_sample_keeps_every_class():
    y = np.array([0] * 1000 + [1] * 3)
    idx = rf.floor_sample(y, np.arange(len(y)), n=10, floor=2, seed=0)
    assert (y[idx] == 1).sum() >= 2


# --- metrics ---------------------------------------------------------------------------

def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 6, 300), rng.integers(0, 7, 300)
    want = f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)
    assert rf.macro_f1(y, p, 7) == pytest.approx(want)


def test_temperature_keeps_argmax_and_lowers_nll():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    logits = rng.normal(size=(500, 4))
    logits[np.arange(500), y] += 1.0
    logits *= 6                                       # over-confident for a 1.0 margin
    t = rf.fit_temperature(logits, y)
    assert t > 1
    assert (rf.softmax(logits, t).argmax(1) == logits.argmax(1)).all()

    def nll(tt):
        return -np.log(rf.softmax(logits, tt)[np.arange(500), y]).mean()
    assert nll(t) < nll(1.0)


def test_topk_hits():
    proba = np.array([[0.5, 0.3, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]])
    assert rf.topk_hits(proba, np.array([2, 0]), k=3).tolist() == [True, False]


def test_classification_block_ci_contains_value():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    proba = rng.dirichlet(np.ones(3), 200)
    blk = rf.classification_block(proba, y, groups=np.repeat(np.arange(20), 10), n_boot=200)
    assert blk["n"] == 200 and blk["n_groups"] == 20
    for k in ("accuracy", "macro_f1", "top3_accuracy"):
        lo, hi = blk[k]["ci95"]
        assert lo <= blk[k]["value"] <= hi


# --- the public API on a tiny fitted model ---------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    labels = ("Java", "Lead/cto", "Python")
    texts = (["java spring hibernate backend services"] * 6
             + ["python django pandas services"] * 6
             + ["cto leadership strategy roadmap budget"] * 6)
    y = ["Java"] * 6 + ["Python"] * 6 + ["Lead/cto"] * 6
    aliases = {"cto": "Lead/cto"}
    sides = {s: rf.fit_tfidf(texts, y, aliases, labels) for s in rf.SIDES}
    # a class that no training document has: its probability must be exactly zero
    sides["cv"] = rf.fit_tfidf(texts[:12], y[:12], aliases, labels)
    return rf.RoleModel("tfidf", labels, aliases, sides, "test")


def test_predict_proba_rows_are_distributions_over_the_label_space(tiny_model):
    p = rf.predict_proba(["spring java backend", "django python", "spring java backend"],
                         "jd", tiny_model)
    assert p.shape == (3, 3)
    assert np.allclose(p.sum(1), 1)
    assert p[0].argmax() == 0 and p[1].argmax() == 2
    assert np.allclose(p[0], p[2])
    q = rf.predict_proba(["cto roadmap"], "cv", tiny_model)
    assert q[0, 1] == 0.0                              # unseen in the CV model's training


def test_role_match_is_soft_agreement_in_unit_interval(tiny_model):
    jd = ["spring java backend", "django python"]
    cv = ["java hibernate", "java hibernate"]
    s = rf.role_match(jd, cv, tiny_model)
    assert s.shape == (2,) and ((0 <= s) & (s <= 1)).all()
    assert s[0] > s[1]
    want = rf.soft_agreement(rf.predict_proba(jd, "jd", tiny_model),
                             rf.predict_proba(cv, "cv", tiny_model))
    assert np.allclose(s, want)
    with pytest.raises(ValueError):
        rf.role_match(jd, cv[:1], tiny_model)


def test_inference_api_takes_text_only():
    """D53: the inference functions accept texts, never a frame with metadata columns."""
    for fn in (rf.predict_proba, rf.role_match, rf.title_spans):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"keyword", "position", "frame", "df", "pairs"}


def test_module_never_reads_final_labels_or_judgements():
    """D46: code that fits a model never reads `a2-final-labels.csv` or `judgements.csv`.

    Docstrings may name the files (to say they are not read); code may not."""
    import ast
    tree = ast.parse(inspect.getsource(rf))
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
               and isinstance(n.value, str) and id(n) not in docstrings]
    assert not names & {"FINAL_CSV", "JUDGEMENTS_CSV", "load_inputs", "load_humans", "load_llm"}
    assert not any(b in s for s in strings for b in ("a2-final-labels", "judgements.csv"))


def test_save_model_refuses_classes_defined_in_main(tmp_path, monkeypatch):
    """Regression: `python -m ... --train` pickled `__main__.SideModel`, which no importer
    could load. The save refuses such classes, and `__main__` delegates to the module."""
    monkeypatch.setattr(rf, "CACHE", tmp_path)
    Main = type("SideModel", (rf.SideModel,), {"__module__": "__main__"})
    side = Main("tfidf", ("a",), None)
    with pytest.raises(TypeError):
        rf.save_model(rf.RoleModel("tfidf", ("a",), {}, {"jd": side, "cv": side}, "x"))
    ok = rf.SideModel("tfidf", ("a",), None)
    rf.save_model(rf.RoleModel("tfidf", ("a",), {}, {"jd": ok, "cv": ok}, "x"))
    rf._LOADED.clear()
    assert rf.load_role_model("x").sides["jd"].labels == ("a",)
    assert "import_module(MODULE).main()" in inspect.getsource(rf)
