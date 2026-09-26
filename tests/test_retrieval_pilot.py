"""Pilot-scale retrieval helpers: per-JD recall and retrieve-then-rank."""
import pandas as pd

from candidate_screener.evaluation import retrieval_pilot as rp
from candidate_screener.matching.base import BaseSystem


def _frame():
    return pd.DataFrame({"jd_id": ["j"] * 6, "pair_id": [f"j__{i}" for i in range(6)],
                         "cv_id": [str(i) for i in range(6)], "stratum": "generic",
                         "jd_text": "x", "cv_text": [f"c{i}" for i in range(6)]})


def test_recall_at_k_per_jd():
    f = _frame()
    scores = pd.Series([6, 5, 4, 3, 2, 1], index=f.pair_id, dtype=float)
    rel = pd.Series([0, 1, 0, 0, 1, 0], index=f.pair_id)
    assert rp.recall_at(f, scores, rel, 2)["j"] == 0.5
    assert rp.recall_at(f, scores, rel, 5)["j"] == 1.0


class _Reverse(BaseSystem):
    name = "reverse"
    def _score(self, pairs):
        return [-int(c[1:]) for c in pairs.cv_text]


def test_rerank_reorders_only_the_shortlist():
    f = _frame()
    retriever = pd.Series([6, 5, 4, 3, 2, 1], index=f.pair_id, dtype=float)
    out = rp.rerank_top(f, retriever, _Reverse(), k=3)
    order = out.sort_values(ascending=False).index.tolist()
    top3 = set(order[:3])
    assert top3 == {"j__0", "j__1", "j__2"}                     # same shortlist
    assert order[0] == "j__0"                                   # reverse prefers low index
    assert order[3:] == ["j__3", "j__4", "j__5"]                # tail keeps retriever order
