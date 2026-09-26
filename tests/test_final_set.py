"""The final annotation set (D41, D47): the rule, its invariants, and its inputs."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from candidate_screener.evaluation import final_set as fs
from candidate_screener.evaluation import resample

AGENT, MODEL = fs.LLM_AGENT_SHA256, fs.LLM_MODEL


# --- the rule ---------------------------------------------------------------------

@pytest.mark.parametrize("humans, llm, expected", [
    (("Fit", "Fit"), "No Fit", ("Fit", "humans_agree")),           # rule 1: LLM ignored
    (("No Fit", "No Fit"), "Fit", ("No Fit", "humans_agree")),
    (("Fit", "No Fit"), "No Fit", ("No Fit", "llm_breaks_tie")),   # rule 2
    (("No Fit", "Fit"), "Fit", ("Fit", "llm_breaks_tie")),
    (("Fit", None), "No Fit", ("Fit", "single_human")),            # rule 3: LLM ignored
    ((None, "No Fit"), "Fit", ("No Fit", "single_human")),
    ((None, None), "Fit", ("Fit", "llm_only")),                     # rule 4
])
def test_rule_v1(humans, llm, expected):
    assert fs.resolve(humans, llm) == expected


def test_a_missing_human_as_nan_is_not_a_second_human():
    """Regression: pandas fills an absent human with NaN, and NaN is not None.

    Counting it as a present label read all 274 single-human pairs as splits, and the 55
    where the LLM disagreed silently took the LLM's label. The build's invariant check
    caught it; this pins it at the rule.
    """
    assert fs.resolve(("No Fit", np.nan), "Fit") == ("No Fit", "single_human")
    assert fs.resolve((np.nan, np.nan), "Fit") == ("Fit", "llm_only")
    with pytest.raises(ValueError, match="no LLM label"):
        fs.resolve(("Fit", "No Fit"), np.nan)


def test_the_rule_refuses_what_it_does_not_define():
    with pytest.raises(ValueError, match="at most two humans"):
        fs.resolve(("Fit", "Fit", "Fit"), "Fit")
    with pytest.raises(ValueError, match="cannot be labelled"):
        fs.resolve((None, None), None)


def test_fit_is_good_or_potential():
    assert fs.to_fit("Good Fit") == fs.to_fit("Potential Fit") == "Fit"
    assert fs.to_fit("No Fit") == "No Fit"
    assert fs.to_fit(np.nan) is None and fs.to_fit(None) is None
    with pytest.raises(ValueError):
        fs.to_fit("Great Fit")


# --- the build, on synthetic inputs ---------------------------------------------------

def _inputs():
    pairs = pd.DataFrame({
        "pair_id": [f"p{i}" for i in range(6)], "stratum": ["generic"] * 3 + ["targeted"] * 3,
        "batch": [1, 1, 1, 2, 2, 2], "jd_id": ["j1", "j1", "j2", "j3", "j3", "j3"],
        "cv_id": [f"c{i}" for i in range(6)]})
    judgements = pd.DataFrame([
        ("p0", "yc", "Good Fit"), ("p0", "van", "Potential Fit"),   # agree (both Fit)
        ("p1", "yc", "No Fit"), ("p1", "van", "Good Fit"),          # split
        ("p2", "yc", "No Fit"),                                     # single
        ("p3", "yc", "Potential Fit"), ("p3", "van", "No Fit"),     # split
        ("p4", "yc", "No Fit"), ("p4", "van", "No Fit"),            # agree
        # p5: no human
    ], columns=["pair_id", "annotator", "label"]).assign(corpus="a2")
    llm_rows = pd.DataFrame([
        (p, lab, AGENT, MODEL, True, 9) for p, lab in
        [("p0", "No Fit"), ("p1", "Good Fit"), ("p2", "Good Fit"), ("p3", "No Fit"),
         ("p4", "Good Fit"), ("p5", "Potential Fit")]
    ] + [("p1", "No Fit", "b" * 64, MODEL, True, 5),                 # another instrument
         ("p1", "No Fit", AGENT, "claude-sonnet-5", True, 7)],       # another model
        columns=["pair_id", "label", "agent_sha256", "model", "parsed_ok", "run"])
    return pairs, fs.load_humans(judgements), fs.load_llm([llm_rows])


def test_build_applies_the_rule_and_passes_its_invariants():
    frame = fs.build_frame(*_inputs())
    got = frame.set_index("pair_id")[["final_label", "tier"]].to_dict("index")
    assert got == {
        "p0": {"final_label": "Fit", "tier": "humans_agree"},
        "p1": {"final_label": "Fit", "tier": "llm_breaks_tie"},
        "p2": {"final_label": "No Fit", "tier": "single_human"},
        "p3": {"final_label": "No Fit", "tier": "llm_breaks_tie"},
        "p4": {"final_label": "No Fit", "tier": "humans_agree"},
        "p5": {"final_label": "Fit", "tier": "llm_only"},
    }
    assert all(ok for ok, _ in fs.check_invariants(frame))
    assert list(frame.columns) == fs.COLUMNS


def test_the_invariants_fail_when_the_llm_overrides_agreeing_humans():
    """The check must fail against a broken rule, or it guards nothing."""
    frame = fs.build_frame(*_inputs())
    frame.loc[frame.pair_id == "p4", "final_label"] = "Fit"          # humans said No Fit
    failed = [m for ok, m in fs.check_invariants(frame) if not ok]
    assert failed and failed[0].startswith("rule 1")


def test_only_the_committed_series_is_read_and_a_second_vote_is_refused():
    _, _, llm = _inputs()
    assert llm.loc["p1", "llm_raw"] == "Good Fit"                    # decoys dropped
    rows = pd.DataFrame([("p0", "Fit", AGENT, MODEL, True, 9)] * 2,
                        columns=["pair_id", "label", "agent_sha256", "model", "parsed_ok", "run"])
    with pytest.raises(ValueError, match="two votes"):
        fs.load_llm([rows])


def test_an_unknown_annotator_is_a_new_rule_version():
    judgements = pd.DataFrame([("p0", "zed", "No Fit")],
                              columns=["pair_id", "annotator", "label"]).assign(corpus="a2")
    with pytest.raises(ValueError, match="new rule version"):
        fs.load_humans(judgements)


def test_the_report_carries_a_total_beside_both_strata():
    rep = fs.report(fs.build_frame(*_inputs()))
    assert list(rep["scopes"]) == ["generic", "targeted", "total"]
    total = rep["scopes"]["total"]
    assert sum(total["tiers"].values()) == 6
    assert total["inter_annotator"]["yc_vs_van"]["n_pairs"] == 4


def test_llm_free_keeps_only_pairs_no_tie_break_touched():
    frame = fs.build_frame(*_inputs())
    assert set(fs.llm_free(frame).pair_id) == {"p0", "p2", "p4"}


# --- resampling -------------------------------------------------------------------

def test_binary_kappa_matches_sklearn():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    assert resample.binary_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


def test_the_bootstrap_resamples_jds_not_pairs():
    """Two JDs, each internally constant: a pair bootstrap would give a tight interval,
    a JD bootstrap must be able to land on either JD alone."""
    frame = pd.DataFrame({"jd_id": ["a"] * 50 + ["b"] * 50, "x": [1] * 50 + [0] * 50})
    low, high = resample.pair_stat_ci(frame, np.mean, ["x"], n_boot=2000)
    assert low == 0.0 and high == 1.0
