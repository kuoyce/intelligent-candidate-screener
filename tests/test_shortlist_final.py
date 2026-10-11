"""Q8 (plan 2026-10-07): the freeze matches the dev record; the selection rule."""
import json

import pandas as pd

from candidate_screener.evaluation import shortlist_dev, shortlist_final as sf


def test_frozen_options_are_the_dev_records_choices():
    """03 §2: what Q8 scores is exactly what the committed dev record chose."""
    dev = json.loads(shortlist_dev.VARIANTS_RECORD.read_text(encoding="utf-8"))["choices"]
    for name in (sf.Z1, sf.Z1L):
        assert dev[name]["options"] == {"weight": sf.FROZEN[name]["weight"]}
    assert dev[sf.Z1]["name"].startswith("Z1 soft role, JD text,")
    assert dev[sf.Z1L]["name"].startswith("Z1L soft role, JD lookup_or_text,")
    assert sf.FROZEN[sf.Z1]["jd_mode"] == "text"
    assert sf.FROZEN[sf.Z1L]["jd_mode"] == "lookup_or_text"


def test_selection_prefers_the_cheaper_unless_the_dearer_is_justified():
    assert sf.select([sf.Z1, sf.Z1L], "not resolved") == sf.Z1
    assert sf.select([sf.Z1, sf.Z1L], "justified") == sf.Z1L
    assert sf.select([sf.Z1L], "harmful") == sf.Z1L
    assert sf.select([sf.Z1], "justified") == sf.Z1
    assert sf.select([], "justified") is None


def test_llm_free_pool_drops_only_llm_labelled_judged_pairs():
    pools = pd.DataFrame({"pair_id": ["a", "b", "c", "d"]})
    lab = pd.DataFrame({"tier": ["humans_agree", "llm_only"]}, index=["a", "b"])
    keep = sf.llm_free_pool(pools, lab).pair_id.tolist()
    assert "b" not in keep and {"c", "d"} <= set(keep)
