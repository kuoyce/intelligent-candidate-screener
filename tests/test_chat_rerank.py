"""Chat reranking (plan 2026-10-17-llm-rerank): synthetic pools and replies only."""
import json

import numpy as np
import pandas as pd
import pytest

from candidate_screener.rerank import chat_rerank as cr


def pools(n_jd=2, n=30):
    rows = [(f"j{j}", f"c{i}", "generic") for j in range(n_jd) for i in range(n)]
    d = pd.DataFrame(rows, columns=["jd_id", "cv_id", "stratum"])
    return d.assign(pair_id=d.jd_id + "__" + d.cv_id)


def test_plan_keeps_x2_top_k_shuffled_with_short_ids_and_batches():
    p = pools()
    scores = np.tile(np.arange(30, 0, -1), 2).astype(float)       # c0 best ... c29 worst
    plan = cr.plan_from(p, scores, k=25, batch=10)
    g = plan[plan.jd_id == "j0"]
    assert set(g.cv_id) == {f"c{i}" for i in range(25)}            # the top 25 only
    assert list(g.short_id) == [f"C{i:03d}" for i in range(1, 26)]
    assert list(g.batch.value_counts().sort_index()) == [10, 10, 5]
    assert list(g.x2_rank) != sorted(g.x2_rank)                    # shuffled
    assert plan.equals(cr.plan_from(p, scores, k=25, batch=10))     # reproducible


def reply(ids, **over):
    return json.dumps([{"cv": i, "label": "Potential Fit", "score": 50, "reason": "x"} | over for i in ids])


def test_parse_reply_accepts_exactly_the_batch_ids_and_a_fence():
    ids = ["C001", "C002"]
    out = cr.parse_reply("```json\n" + reply(ids) + "\n```", ids)
    assert list(out.short_id) == ids and list(out.score) == [50, 50]


@pytest.mark.parametrize("text,why", [
    (reply(["C001"]), "missing"),
    (reply(["C001", "C002", "C003"]), "unexpected"),
    (reply(["C001", "C001"]), "duplicate"),
    (reply(["C001", "C002"], score=101), "integer 0-100"),
    (reply(["C001", "C002"], score=55.5), "integer 0-100"),
    (reply(["C001", "C002"], label="Fit"), "not one of"),
    ("Here you go: []", "not valid JSON"),
])
def test_parse_reply_refuses_with_the_reason(text, why):
    with pytest.raises(ValueError, match=why):
        cr.parse_reply(text, ["C001", "C002"])


def test_the_prompt_carries_the_guide_and_hides_x2():
    text = cr.prompt("Python developer, mail me at a@b.com", [("C001", "I write Python")], 1, 5, first=True)
    assert "# Annotation guide" in text and "[EMAIL]" in text and "x2" not in text.lower()
    assert cr.prompt("jd", [("C021", "cv")], 2, 5, first=False).startswith("Batch 2 of 5")


def test_parse_reply_accepts_curly_quotes_from_autocorrect():
    text = "[{\u201ccv\u201d:\u201dC001\",\u201dlabel\u201d:\u201dNo Fit\u201d,\u201dscore\u201d:5,\u201dreason\u201d:\u201dsales\u201d}]"
    assert cr.parse_reply(text, ["C001"]).score.tolist() == [5]
