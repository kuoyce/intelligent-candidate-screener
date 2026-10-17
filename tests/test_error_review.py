"""Manual error review (plan 2026-10-13-error-analysis-2): synthetic rows only."""
import pandas as pd
import pytest

from candidate_screener.evaluation import error_review as er


def row(**kw):
    base = {"yc_raw": "Good Fit", "van_raw": None, "tier": "single_human", "final_label": "Fit",
            "llm_label": "Fit", "llm_score": 70, "llm_note": "ok"}
    return base | kw


def test_a2_source_names_the_tier_and_only_the_humans_who_labelled():
    info = er.a2_source(row())
    assert info["source"] == "one human only" and info["humans"] == {"yc": "Good Fit"}
    tie = er.a2_source(row(van_raw="No Fit", tier="llm_breaks_tie"))
    assert "LLM broke the tie" in tie["source"] and set(tie["humans"]) == {"yc", "van"}


def test_d50_source_is_llm_only():
    info = er.d50_source("Fit", 55)
    assert "LLM only" in info["source"] and info["humans"] == {}
    assert "Humans:** none" in er.describe(info)


def test_save_appends_and_latest_answer_wins(tmp_path):
    p = tmp_path / "r.csv"
    er.save("c1", "role_match", "A", "", "me", path=p)
    er.save("c1", "role_match", "D", "bad label", "me", path=p)
    er.save("c2", "tfidf_scaled", "C", "", "me", path=p)
    got = er.latest(p)
    assert got.loc["c1", "reason"] == "D" and len(got) == 2 and len(pd.read_csv(p)) == 3


def test_save_rejects_a_reason_outside_the_fixed_list(tmp_path):
    with pytest.raises(ValueError):
        er.save("c1", "role_match", "Z", "", "me", path=tmp_path / "r.csv")


def test_shuffled_is_reproducible_and_keeps_every_case():
    c = pd.DataFrame({"case_id": list("abcdefgh")})
    assert list(er.shuffled(c).case_id) == list(er.shuffled(c).case_id) and sorted(er.shuffled(c).case_id) == list("abcdefgh")


def test_override_is_a_flag_that_needs_reason_d_and_a_note(tmp_path):
    p = tmp_path / "r.csv"
    for reason, note in (("A", "why"), ("D", "")):
        with pytest.raises(ValueError):
            er.save("c1", "role_match", reason, note, "me", "fit_side", path=p)
    with pytest.raises(ValueError):
        er.save("c1", "role_match", "D", "x", "me", "sideways", path=p)
    er.save("c1", "role_match", "D", "clearly a backend role", "me", "fit_side", path=p)
    assert er.latest(p).loc["c1", "override"] == "fit_side"


def test_summary_uses_the_latest_answer_and_counts_flags_per_label_source():
    import json
    det = lambda f, n: json.dumps({"fit": {"source": f}, "nofit": {"source": n}})
    cases = pd.DataFrame({"case_id": ["c1", "c2"], "set": ["A2 final", "D50 selection"],
                          "details": [det("humans split, the LLM broke the tie", "one human only"),
                                      det("LLM only (D50 scoring batch, run 11, no human)", "LLM only (x)")]})
    reasons = pd.DataFrame({"case_id": ["c1", "c1", "c2"], "bucket": ["role_match"] * 3, "reason": ["A", "D", "D"],
                            "override": ["none", "both", "fit_side"], "reviewer": ["me"] * 3})
    s = er.summarise(reasons, cases)
    assert s["n_cases"] == 2 and s["reason_by_bucket"]["role_match"] == {"D": 2}
    assert s["fit_side_flagged_by_source"]["humans split, the LLM broke the tie"] == {"n": 1, "flagged": 1}
    assert s["nofit_side_flagged_by_source"]["LLM only"] == {"n": 1, "flagged": 0}
