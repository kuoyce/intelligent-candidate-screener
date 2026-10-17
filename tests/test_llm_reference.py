"""LLM as reference (exploratory): synthetic scores only."""
import pandas as pd

from candidate_screener.evaluation import llm_reference as lr


def test_per_jd_skips_jds_with_one_reference_value_and_uses_shared_pairs():
    ref = pd.Series([90, 10, 50, 50], index=list("abcd"))
    jd = pd.Series(["j1", "j1", "j2", "j2"], index=list("abcd"))
    s = pd.Series([0.2, 0.9, 0.1, 0.3], index=list("abcd"))
    v = lr.per_jd(ref, s, jd, "pairwise")
    assert list(v.index) == ["j1"] and v["j1"] == 0.0


def test_only_the_rerankers_are_tagged_as_trained_on_llm_scores():
    tagged = {k for k, v in lr.LEARNED_FROM.items() if v == lr.LLM_TRAINED}
    assert tagged == {"R1", "R2", "R3"}
    assert lr.LEARNED_FROM.get("extracted rule + TF-IDF", lr.NO_LLM) == lr.NO_LLM
