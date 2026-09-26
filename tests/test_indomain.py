"""The in-domain protocol (D41, D47): figures carry scope, n and CI; totals travel with strata."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from candidate_screener.evaluation import indomain


def _lab():
    """Two generic JDs and one targeted JD, five candidates each."""
    rows = []
    for jd, stratum, ys in (("j1", "generic", [1, 1, 0, 0, 1]), ("j2", "generic", [0, 1, 0, 0, 1]),
                            ("j3", "targeted", [1, 0, 1, 0, 0])):
        for i, y in enumerate(ys):
            rows.append((f"{jd}_{i}", stratum, jd, "humans_agree" if i % 2 else "llm_breaks_tie", y))
    return pd.DataFrame(rows, columns=["pair_id", "stratum", "jd_id", "tier", "y"]).set_index("pair_id")


def _pairs(lab):
    return pd.DataFrame({"pair_id": lab.index, "keyword_match": [1, 1, 1, 0, 0] * 3})


def test_a_figure_refuses_to_exist_without_its_n_scope_or_rule_version():
    ok = dict(system="s", metric="auc", scope="total", subset="all", value=0.5, ci_low=0.4,
              ci_high=0.6, n_jd=3, n_pairs=15, rule_version=1)
    indomain.IndomainFigure(**ok)
    for bad in ({"n_jd": 0}, {"scope": "both"}, {"subset": "some"}, {"rule_version": None}):
        with pytest.raises(ValueError):
            indomain.IndomainFigure(**{**ok, **bad})


def test_per_jd_auc_skips_single_class_jds():
    y = pd.Series([1, 0, 1, 1]); s = pd.Series([0.9, 0.1, 0.5, 0.4]); jd = pd.Series(["a", "a", "b", "b"])
    auc = indomain.per_jd_auc(y, s, jd)
    assert list(auc.index) == ["a"] and auc["a"] == 1.0


def test_rank_figures_emit_generic_targeted_and_total_for_every_metric():
    lab = _lab()
    perfect = lab.y.astype(float)
    figs = [f.as_dict() for f in indomain.rank_figures("oracle", perfect, lab, _pairs(lab), 1)]
    indomain.assert_total_has_strata(figs)
    total = {(f["metric"], f["subset"]): f for f in figs if f["scope"] == "total"}
    assert ("auc", "all") in total
    assert total[("auc", "all")]["value"] == 1.0 and total[("auc", "all")]["n_jd"] == 3
    assert {f["scope"] for f in figs} == {"generic", "targeted", "total"}


def test_a_group_with_an_unscoreable_scope_is_skipped_whole_and_listed():
    lab = _lab()                                     # targeted llm_free: one class only
    skipped = []
    figs = indomain.rank_figures("oracle", lab.y.astype(float), lab, _pairs(lab), 1,
                                 unscoreable=skipped)
    assert not [f for f in figs if f.subset == "llm_free"]
    assert {s["subset"] for s in skipped} == {"llm_free"}


def test_a_total_without_its_strata_is_refused():
    with pytest.raises(ValueError, match="without both strata"):
        indomain.assert_total_has_strata([{"system": "s", "metric": "auc", "subset": "all",
                                           "scope": "total"}])


def test_llm_free_drops_tie_broken_pairs():
    lab = _lab()
    sub = indomain.restrict(lab, "total", "llm_free")
    assert set(sub.tier) == {"humans_agree"}


def test_a_system_compared_with_itself_has_zero_delta():
    lab = _lab(); s = pd.Series(np.linspace(0, 1, len(lab)), index=lab.index)
    d = indomain.paired_delta(s, s, lab, _pairs(lab))
    assert d["delta"] == 0 and d["ci"] == [0, 0]


def test_binary_stats_match_sklearn():
    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    s = indomain._cls_stats(y, p)
    assert s["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert s["f1_fit"] == pytest.approx(f1_score(y, p))
    assert s["kappa"] == pytest.approx(cohen_kappa_score(y, p))
    assert s["accuracy"] == pytest.approx(np.mean(y == p))


def test_classifier_figures_cover_every_scope_and_metric():
    lab = _lab()
    preds = pd.Series(np.where(lab.y == 1, "Fit", "No Fit"), index=lab.index)
    figs = [f.as_dict() for f in indomain.cls_figures("perfect", preds, lab, 1, n_boot=50)]
    indomain.assert_total_has_strata(figs)
    assert all(f["value"] == 1.0 for f in figs if f["metric"] == "accuracy")
    # macro-F1 is 1.0 where both classes occur; a one-class subset caps it at 0.5
    assert all(f["value"] == 1.0 for f in figs if f["metric"] == "macro_f1" and f["subset"] == "all")
