"""The pool text-join guard *(deviation X1)*.

Task 3.3 built the pools and committed them; nothing scored them for six days. The
first module that did — `evaluation.retrieval`, the Q24 pass deferred twice — failed
immediately on `np.nan is an invalid document`, because **1,755 of 10,000 N100 rows**
(17.6%) carried a null `resume_text`.

The cause is a real asymmetry, not a typo: the candidate universe is every resume
*assigned* to test (193), but only 158 appear in a judged test pair. The other 35 are
test-assigned resumes whose every pair was discarded by the doubly-disjoint rule, so
their text is in the pooled A1 corpus and not in `test.parquet`.

It stayed invisible in the shape this repo keeps meeting — a plausible number, no
exception. The null count *was* computed, but assigned to `report` after the manifest
had already been written, so it never reached `pools-yield.json`.

The fix is the wider text source; this test is the guard, and it fails against the
`test.parquet` source it replaced.
"""
from __future__ import annotations

import pandas as pd
import pytest

from candidate_screener.data import pools


def synthetic_pools() -> pd.DataFrame:
    return pd.DataFrame(
        [("j_0", "r_judged", "good", "labelled", "N20"),
         ("j_0", "r_universe_only", "none", "distractor", "N20")],
        columns=["query_jd_id", "candidate_resume_id", "relevance", "source",
                 "pool_variant"])


def corpus(resume_ids: list[str]) -> pd.DataFrame:
    """Stand-in for `load_pooled()`'s first return value."""
    return pd.DataFrame({
        "resume_id": resume_ids,
        "resume_text": [f"text of {r}" for r in resume_ids],
        "jd_id": ["j_0"] * len(resume_ids),
        "job_description_text": ["text of j_0"] * len(resume_ids)})


def test_attach_text_covers_a_distractor_absent_from_the_judged_pairs(monkeypatch):
    """The 35-resume case: in the universe, never in a judged pair."""
    monkeypatch.setattr(pools, "load_pooled",
                        lambda: (corpus(["r_judged", "r_universe_only"]), {}))
    joined = pools.attach_text(synthetic_pools())
    assert joined.resume_text.notna().all()
    assert joined.job_description_text.notna().all()


def test_attach_text_raises_rather_than_shipping_a_textless_distractor(monkeypatch):
    """What the old `test.parquet` source did silently, 1,755 times.

    A distractor with no text is worse than a missing row: a scorer either raises or
    ranks it arbitrarily, and either way the realised pool depth is not the depth
    `pools-yield.json` records.
    """
    monkeypatch.setattr(pools, "load_pooled", lambda: (corpus(["r_judged"]), {}))
    with pytest.raises(AssertionError, match="no document text"):
        pools.attach_text(synthetic_pools())
