"""W3 (plan 2026-10-08): the per-run `per_jd_auc` memo in `evaluation.indomain`.

Synthetic data only. The memo must return exactly the uncached values, and two systems
(or scopes, or metrics) must never share an entry."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from candidate_screener.evaluation import indomain


def _synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(8):
        for c in range(10):
            rows.append({"pair_id": f"p{j}-{c}", "jd_id": f"jd{j}",
                         "stratum": "generic" if j < 5 else "targeted",
                         "tier": "humans_agree" if c % 3 else "llm_tiebreak",
                         "y": int(rng.integers(0, 2)), "keyword_match": int(c % 2 == 0)})
    frame = pd.DataFrame(rows)
    lab = frame.set_index("pair_id")[["stratum", "jd_id", "tier", "y"]]
    pairs = frame[["pair_id", "keyword_match"]]
    a = pd.Series(rng.random(len(frame)), index=frame.pair_id)
    b = pd.Series(rng.random(len(frame)), index=frame.pair_id)
    return lab, pairs, a, b


def _uncached(scores, lab, pairs, metric):
    assert indomain._AUC_MEMO is None
    return indomain._auc_values(scores, lab, pairs, metric)


def test_memo_returns_the_uncached_values_and_keys_on_content():
    lab, pairs, a, b = _synthetic()
    expected = {(name, m, scope): _uncached(s, indomain.restrict(lab, scope, "all"), pairs, m)
                for name, s in (("a", a), ("b", b)) for m in indomain.RANK_METRICS
                for scope in ("generic", "targeted", "total")}
    with indomain.auc_memo() as memo:
        for _ in range(2):                                   # cold, then from the memo
            for (name, m, scope), (values, n) in expected.items():
                s = a if name == "a" else b
                got, got_n = indomain._auc_values(s, indomain.restrict(lab, scope, "all"),
                                                  pairs, m)
                pdt.assert_series_equal(got, values, check_exact=True)
                assert got_n == n
        # one entry per (system, scope, metric): nothing shared, nothing duplicated
        assert len(memo) == len(expected)
    assert indomain._AUC_MEMO is None                        # scoped to the run


def test_memo_never_shares_an_entry_between_systems():
    lab, pairs, a, b = _synthetic(1)
    with indomain.auc_memo():
        va, _ = indomain._auc_values(a, lab, pairs, "auc")
        vb, _ = indomain._auc_values(b, lab, pairs, "auc")
        va.iloc[:] = -1.0                                    # a caller mutating its copy
        va2, _ = indomain._auc_values(a, lab, pairs, "auc")
    assert not vb.equals(va2)
    pdt.assert_series_equal(va2, indomain.per_jd_auc(lab.y, a.reindex(lab.index), lab.jd_id),
                            check_exact=True)


def test_paired_delta_is_identical_with_and_without_the_memo():
    lab, pairs, a, b = _synthetic(2)
    cells = [(m, sc, sub) for m in indomain.RANK_METRICS
             for sc in ("generic", "targeted", "total") for sub in indomain.SUBSETS]
    plain = [indomain.paired_delta(b, a, lab, pairs, *cell) for cell in cells]
    with indomain.auc_memo():
        memo = [indomain.paired_delta(b, a, lab, pairs, *cell) for cell in cells]
        swapped = [indomain.paired_delta(a, b, lab, pairs, *cell) for cell in cells]
    assert memo == plain
    assert [d["delta"] for d in swapped if d["delta"]] == \
        [-d["delta"] for d in plain if d["delta"]]
