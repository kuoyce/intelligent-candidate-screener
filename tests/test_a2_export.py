import pandas as pd

from candidate_screener.evaluation.a2_export import rank_within_jd


def test_rank_is_within_jd_and_ties_break_on_cv_id():
    pairs = pd.DataFrame({"pair_id": list("abcd"), "stratum": "generic",
                          "jd_id": ["j1", "j1", "j1", "j2"], "cv_id": ["c3", "c1", "c2", "c1"]})
    out = rank_within_jd(pairs, pd.Series([0.5, 0.5, 0.9, 0.1])).set_index("pair_id")
    assert out.x2_rank.to_dict() == {"c": 1, "b": 2, "a": 3, "d": 1}
