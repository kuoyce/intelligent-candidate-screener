"""Stage-1 recall curve: synthetic pools only."""
import pandas as pd

from candidate_screener.evaluation import recall_curve as rc


def test_recall_rises_with_k_and_reaches_one_at_the_pool_size():
    frame = pd.DataFrame({"jd_id": ["j"] * 4, "pair_id": list("abcd")})
    scores = pd.Series([0.9, 0.1, 0.5, 0.3])
    rel = pd.Series([0, 1, 1, 0])
    out = {c["k"]: c["recall"] for c in rc.curve(frame, scores, rel, ks=(1, 2, 4))}
    assert out == {1: 0.0, 2: 0.5, 4: 1.0}
