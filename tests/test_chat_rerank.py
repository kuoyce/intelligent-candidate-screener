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


def test_save_reply_accepts_numpy_batch_numbers_and_logs_the_raw_paste(tmp_path):
    plan = cr.plan_from(pools(1, 25), np.arange(25, 0, -1).astype(float), k=20, batch=20)
    raw = reply([f"C{i:03d}" for i in range(1, 21)])
    path = tmp_path / "r.csv"
    cr.save_reply("j0", np.int64(1), raw, cr.parse_reply(raw, [f"C{i:03d}" for i in range(1, 21)]), plan, path=path)
    assert len(cr.latest(path)) == 20
    assert json.loads(path.with_suffix(".raw.jsonl").read_text())["raw"] == raw


def test_extend_adds_one_batch_per_jd_after_the_last_short_id_and_skips_planned_pairs():
    """An extension must not reuse a short id (the reply would overwrite a scored CV), must not
    re-plan a pair already scored, and must not invent a JD."""
    plan = pd.DataFrame({"jd_id": ["a", "a", "b"], "cv_id": ["x", "y", "z"],
                         "pair_id": ["a__x", "a__y", "b__z"], "stratum": ["generic", "generic", "targeted"],
                         "short_id": ["C001", "C002", "C001"], "batch": [1, 2, 1], "jd_order": [1, 1, 2]})
    extra = pd.DataFrame({"jd_id": ["a", "a", "b"], "cv_id": ["w", "x", "z"],
                          "pair_id": ["a__w", "a__x", "b__z"]})
    out = cr.extend(plan, extra, "s1_passed")
    new = out[out.source == "s1_passed"]
    assert new.pair_id.tolist() == ["a__w"]
    assert new.short_id.tolist() == ["C003"] and new.batch.tolist() == [3]
    assert new.jd_order.tolist() == [1] and new.stratum.tolist() == ["generic"]
    assert (out[out.source == "x2_top100"].pair_id.tolist() == plan.pair_id.tolist())
    assert cr.extend(out, extra, "s1_passed").equals(out)            # idempotent
    with pytest.raises(ValueError, match="never a JD"):
        cr.extend(plan, pd.DataFrame({"jd_id": ["c"], "cv_id": ["q"], "pair_id": ["c__q"]}), "s1_passed")


@pytest.mark.parametrize("n_new,expected", [(5, 15), (20, 20), (18, 22), (1, 19), (40, 20), (62, 18), (16, 24)])
def test_anchors_fill_to_a_multiple_of_20_with_at_least_5(n_new, expected):
    """Owner, 18 Oct 2026: 5 new -> 15 anchors; never fewer than 5 (18 new -> 22, not 2)."""
    k = cr.n_anchors(n_new)
    assert k == expected and (n_new + k) % 20 == 0 and k >= 5


def _scored_plan():
    plan = pd.DataFrame({"jd_id": ["a"] * 30, "cv_id": [f"x{i:02d}" for i in range(30)],
                         "pair_id": [f"a__x{i:02d}" for i in range(30)], "stratum": "generic",
                         "short_id": [f"C{i + 1:03d}" for i in range(30)], "batch": [1 + i // 20 for i in range(30)],
                         "jd_order": 1, "source": "x2_top100"})
    scored = plan[["pair_id", "jd_id", "cv_id"]].assign(score=np.arange(30) * 3)
    return plan, scored


def test_fresh_chat_mixes_new_cvs_with_spread_anchors_in_its_own_ids_and_batches():
    """Anchors must be already-scored pairs spread over the score range (not the top few), the
    fresh chat must not reuse a C id or a chat-1 batch number (a reply would overwrite a chat-1
    score), and a second build must be refused (it would reshuffle a collection under way)."""
    plan, scored = _scored_plan()
    new = pd.DataFrame({"jd_id": "a", "cv_id": ["n1", "n2", "n3"], "pair_id": ["a__n1", "a__n2", "a__n3"]})
    out = cr.fresh(plan, scored, new)
    f = out[out.run == cr.FRESH_RUN]
    assert len(f) == 20 and (f.source == "s1_t030").sum() == 3 and (f.source == "anchor").sum() == 17
    assert f.short_id.str.startswith("F").all() and set(f.batch) == {101}
    a = f[f.source == "anchor"].pair_id
    assert a.isin(scored.pair_id).all() and a.is_unique
    s = scored.set_index("pair_id").score.loc[a]
    assert s.min() <= 6 and s.max() >= 81                       # spread, not one end of the range
    assert (out[out.run == "chat-1"].drop(columns="run").reset_index(drop=True)
            .equals(plan.reset_index(drop=True)))
    with pytest.raises(ValueError, match="never rebuild"):
        cr.fresh(out, scored, new)


def test_a_fresh_chat_starts_at_its_own_batch_1_with_the_full_instrument():
    plan, scored = _scored_plan()
    new = pd.DataFrame({"jd_id": "a", "cv_id": [f"n{i}" for i in range(25)], "pair_id": [f"a__n{i}" for i in range(25)]})
    out = cr.fresh(plan, scored, new)
    assert cr.chat_batches(out, "a", 2) == (2, 2)               # chat-1: batch 2 of 2
    assert cr.chat_batches(out, "a", 101) == (1, 2) and cr.chat_batches(out, "a", 102) == (2, 2)


def test_relabelled_retest_never_replaces_chat_1(tmp_path):
    """The 18 Oct 2026 mistake: a chat-1 batch re-scored later must not win over the committed
    chat-1 score once it is listed as `chat-1r`; an unlisted save still replaces (latest wins)."""
    rows = pd.DataFrame({"jd_id": "j", "batch": [1, 1, 2, 2], "short_id": ["C001", "C001", "C021", "C021"],
                         "pair_id": ["p1", "p1", "p2", "p2"], "label": "No Fit", "score": [10, 40, 20, 50],
                         "run": "chat-1", "saved_at": ["t1", "t9", "t2", "t8"]})
    rows.to_csv(tmp_path / "r.csv", index=False)
    pd.DataFrame({"jd_id": ["j"], "batch": [1], "saved_at": ["t9"], "run": ["chat-1r"]}).to_csv(tmp_path / "m.csv", index=False)
    d = cr.latest(tmp_path / "r.csv", tmp_path / "m.csv")
    c1 = d[d.run == "chat-1"].set_index("pair_id").score
    assert c1.to_dict() == {"p1": 10, "p2": 50}
    assert d[d.run == "chat-1r"].score.tolist() == [40]
