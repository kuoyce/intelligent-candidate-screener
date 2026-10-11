"""Stage 2 on the final set (plan 2026-10-06 P8.4/8.5, amendment A3): verdicts, O5, the §5
selection and the end-to-end helpers, on inline synthetic data."""
import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import ordering, stage2

CE, R3, WEAK = stage2.CE, "R3", stage2.SBERT_WEAK
G, P, N = "Good Fit", "Potential Fit", "No Fit"


# --- invariant 4: selection ---------------------------------------------------------------

def _v(ce_weak="not resolved", r3_weak="not resolved", ce_r3="not resolved"):
    return {(CE, WEAK): ce_weak, (R3, WEAK): r3_weak, (CE, R3): ce_r3}


def test_selection_defaults_to_sbert_weak_when_nothing_beats_it():
    out = stage2.select({WEAK: True, R3: True, CE: True}, _v())
    assert out["selected"] == WEAK


def test_a_justified_dearer_reranker_displaces_a_cheaper_one():
    """Fails if `justified` does not displace, or if the walk is not cheapest first."""
    out = stage2.select({WEAK: True, R3: True, CE: True}, _v(ce_weak="justified"))
    assert out["selected"] == R3           # R3 is not beaten by CE (O4 not justified)
    out = stage2.select({WEAK: True, R3: True, CE: True},
                        _v(ce_weak="justified", ce_r3="justified"))
    assert out["selected"] == CE


@pytest.mark.parametrize("weaker", ["justified, one stratum", "justified, LLM-dependent",
                                    "not resolved", "harmful"])
def test_only_a_plain_justified_verdict_displaces(weaker):
    out = stage2.select({WEAK: True, R3: True, CE: True}, _v(ce_weak=weaker, r3_weak=weaker))
    assert out["selected"] == WEAK


def test_a_candidate_failing_o5_is_skipped_and_cannot_displace():
    """Fails if an O5 failure is adopted, or still counts as a beater."""
    out = stage2.select({WEAK: False, R3: True, CE: True}, _v())
    assert out["selected"] == R3
    out = stage2.select({WEAK: True, R3: True, CE: False}, _v(ce_weak="justified"))
    assert out["selected"] == WEAK
    out = stage2.select({WEAK: False, R3: False, CE: False}, _v())
    assert out["selected"] is None


# --- verdict and O5 on synthetic grades ---------------------------------------------------

def _grades(n_jd=24):
    rows = []
    for j in range(n_jd):
        stratum = "generic" if j % 2 else "targeted"
        for k, lab in enumerate([G, P, N, N, P]):
            van = lab if k < 3 else None
            rows.append({"pair_id": f"j{j}__c{k}", "stratum": stratum, "jd_id": f"j{j}",
                         "yc_raw": lab, "van_raw": van,
                         "tier": "humans_agree" if van else "single_human",
                         "final_label": "No Fit" if lab == N else "Fit"})
    return ordering.grade_frame(pd.DataFrame(rows))


def test_verdict_and_o5_follow_the_registered_rule():
    """A perfect scorer beats a reversed one (justified) and random (O5); the reverse is
    harmful and fails O5."""
    grades = _grades()
    perfect = grades.yc.astype(float)
    reverse = -perfect
    assert stage2.verdict(perfect, reverse, grades, "yc")["verdict"] == "justified"
    assert stage2.verdict(reverse, perfect, grades, "yc")["verdict"] == "harmful"
    assert stage2.beats_random(perfect, grades, "yc")["passes"]
    assert not stage2.beats_random(reverse, grades, "yc")["passes"]
    cells = stage2.verdict(perfect, reverse, grades, "mean")["cells"]
    assert {(c["scope"], c["subset"]) for c in cells} == set(stage2.CELLS.values())


# --- 8.5 helpers --------------------------------------------------------------------------

def test_top_k_and_unjudged_counts():
    grades = _grades(2)
    pools = pd.DataFrame({"jd_id": ["j0"] * 7, "stratum": "targeted",
                          "pair_id": [f"j0__c{k}" for k in range(5)] + ["j0__x1", "j0__x2"]})
    scores = pd.Series([5, 4, 3, 2, 1, 10, 0], index=pools.pair_id, dtype=float)
    top = stage2.top_k(pools, scores, k=3)
    assert top.pair_id.tolist() == ["j0__x1", "j0__c0", "j0__c1"]
    out = stage2.shortlist_ordering(top, grades)
    assert out["unjudged_in_top"] == 1 and out["judged_in_top"] == 2
