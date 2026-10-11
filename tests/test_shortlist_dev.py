"""Shortlist recall dev scoring and its fixed selection rules (plan 2026-10-07, Q3)."""
import numpy as np
import pandas as pd

from candidate_screener.evaluation import shortlist_dev as sd
from conftest import reads_final_labels


def test_recall_per_jd_counts_top_k_and_skips_jds_without_relevant():
    frame = pd.DataFrame({"jd_id": ["a"] * 4 + ["b"] * 2,
                          "pair_id": ["a1", "a2", "a3", "a4", "b1", "b2"]})
    s = np.array([0.9, 0.8, 0.8, 0.1, 0.5, 0.4])
    r = np.array([0, 0, 1, 1, 0, 0])
    v = sd.recall_per_jd(frame, s, r, 2)
    assert list(v.index) == ["a"]
    assert v["a"] == 0.0                         # tie at 0.8 broken by pair_id: a2 before a3
    assert sd.recall_per_jd(frame, s, r, 3)["a"] == 0.5


def _rung(rung, k, ctx, sel, train=0.5):
    fig = lambda v: {"recall@10": {"value": 0.0}, "recall@50": {"value": v}}
    return {"rung": rung, "options": None if k is None else {"k": k, "context": ctx},
            "proxy": {"dev-select": fig(sel), "dev-train": fig(train)}}


def _rungs(y2t_best=0.70, x2=0.60):
    out = {"X2": _rung("X2", None, None, x2)}
    for rung, base in (("Y1", 0.50), ("Y2", 0.65), ("Y4", 0.66)):
        for k, ctx in sd.OPTIONS:
            out[f"{rung} {k} {ctx}"] = _rung(rung, k, ctx, base)
    for k, ctx in sd.OPTIONS:
        out[f"Y2t {k} {ctx}"] = _rung("Y2t", k, ctx, 0.6)
    out["Y2 2 True"]["proxy"]["dev-select"]["recall@50"]["value"] = 0.68
    out["Y2t 1 False"]["proxy"]["dev-select"]["recall@50"]["value"] = y2t_best
    return out


def test_option_ties_go_to_the_cheaper_option_and_y2t_must_win_strictly():
    c = sd.choose(_rungs())
    assert c["Y1"]["options"] == {"k": 0, "context": False}          # all tied: cheapest
    assert c["Y2"]["options"] == {"k": 2, "context": True}
    assert c["Y2t"]["options"] == {"k": 1, "context": False}
    assert c["Y2_vs_Y2t"] == "Y2t"
    assert sd.choose(_rungs(y2t_best=0.68))["Y2_vs_Y2t"] == "Y2"      # tie -> Y2


def test_early_stop_when_the_best_classical_y_does_not_beat_x2():
    assert sd.choose(_rungs(x2=0.60))["early_stop"]["decision"] == "continue"
    stop = sd.choose(_rungs(x2=0.70))["early_stop"]                   # equal is not a beat
    assert stop["dev_best_classical"] == "Y2t" and not stop["beats_x2"]
    assert stop["decision"].startswith("stop")


def test_dev_scoring_never_reads_final_labels_or_judgements():
    assert reads_final_labels(sd) == []
