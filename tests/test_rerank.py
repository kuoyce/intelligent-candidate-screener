"""Stage-2 rerankers (plan 2026-10-06 P8, amendment A3): the batch, the grid, selection and
the D46 guard, on inline synthetic data. No `data/` and no model download needed."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from candidate_screener.matching import neural
from candidate_screener.matching.base import BaseSystem
from candidate_screener.rerank import base, batch, models

RERANK_DIR = Path(batch.__file__).resolve().parent


def _scores_frame():
    """Two JDs; j0's scores 90, 75, 70, 20; j1 has a tie at 50."""
    rows = [("j0", "a", 90), ("j0", "b", 75), ("j0", "c", 70), ("j0", "d", 20),
            ("j1", "e", 50), ("j1", "f", 50), ("j1", "g", 10)]
    return pd.DataFrame({"pair_id": [f"{j}__{c}" for j, c, _ in rows],
                         "jd_id": [j for j, _, _ in rows], "cv_id": [c for _, c, _ in rows],
                         "split": "train", "score": [float(s) for *_, s in rows],
                         "jd_text": [f"jd {j}" for j, _, _ in rows],
                         "cv_text": [f"cv {c}" for _, c, _ in rows]})


# --- invariant 1: preference pairs ---------------------------------------------------------

@pytest.mark.parametrize("delta", [10, 20])
def test_preference_pairs_stay_within_a_jd_and_respect_delta(delta):
    """Fails if pairs cross JDs, if a gap below δ (or a tie) slips in, or if hi/lo flip."""
    frame = _scores_frame()
    prefs = batch.preference_pairs(frame, delta)
    score = frame.set_index("pair_id").score
    jd_of = frame.set_index("pair_id").jd_id
    assert len(prefs)
    assert (prefs.hi_pair.map(jd_of) == prefs.jd_id).all()
    assert (prefs.lo_pair.map(jd_of) == prefs.jd_id).all()
    gaps = prefs.hi_pair.map(score) - prefs.lo_pair.map(score)
    assert (gaps >= delta).all() and (gaps == prefs.gap).all()
    # exhaustive: every within-JD pair with gap >= delta is present
    expected = sum(1 for _, g in frame.groupby("jd_id") for x in g.score for y in g.score
                   if x - y >= delta)
    assert len(prefs) == expected
    # δ=20 drops (90, 75) and (75, 70) but keeps (90, 70)
    if delta == 20:
        assert ("j0__a", "j0__b") not in set(zip(prefs.hi_pair, prefs.lo_pair))
        assert ("j0__a", "j0__c") in set(zip(prefs.hi_pair, prefs.lo_pair))


def test_preference_pairs_refuse_a_nonpositive_delta():
    with pytest.raises(ValueError):
        batch.preference_pairs(_scores_frame(), 0)


# --- invariant 2: the batch frame ----------------------------------------------------------

def _raw_docs():
    jd = pd.DataFrame({"id": ["j0", "j1"], "Long Description": ["  JD zero ", "JD one"]})
    cv = pd.DataFrame({"id": ["a", "b"], "Position": ["SECRET_TITLE", "SECRET_TITLE"],
                       "CV": ["cv a", "cv b"], "Highlights": ["", "h"],
                       "Moreinfo": [None, ""], "Looking For": ["", ""]})
    return jd, cv


def test_batch_text_is_v2_without_position():
    """Fails if the CV text keeps `Position` (D53, A3.1) or the JD is not stripped."""
    jd, cv = _raw_docs()
    scores = pd.DataFrame({"pair_id": ["j0__a", "j1__b"], "jd_id": ["j0", "j1"],
                           "cv_id": ["a", "b"], "split": ["train", "validation"],
                           "score": [80.0, 10.0]})
    out = batch.attach_text(scores, jd, cv)
    assert not out.cv_text.str.contains("SECRET_TITLE").any()
    assert out.jd_text.tolist() == ["JD zero", "JD one"]
    assert list(out.columns) == list(batch.FRAME_COLUMNS)


@pytest.mark.parametrize("shared", ["jd", "cv"])
def test_batch_refuses_a_document_in_both_splits(shared):
    """Fails if train and validation may share a JD or a CV."""
    jd, cv = _raw_docs()
    jd_ids = ["j0", "j0"] if shared == "jd" else ["j0", "j1"]
    cv_ids = ["a", "a"] if shared == "cv" else ["a", "b"]
    scores = pd.DataFrame({"pair_id": ["p0", "p1"], "jd_id": jd_ids, "cv_id": cv_ids,
                           "split": ["train", "validation"], "score": [80.0, 10.0]})
    with pytest.raises(ValueError, match="both train and validation"):
        batch.attach_text(scores, jd, cv)


def test_load_scores_keeps_run_11_and_refuses_bad_rows(tmp_path):
    cols = dict(agent_sha256="5d986c82ffff", model="claude-haiku-4-5", draw="a2_train_batch",
                parsed_ok=True)
    rows = [dict(cols, run=11, pair_id="p0", jd_id="j0", cv_id="a", split="train", score=80),
            dict(cols, run=11, pair_id="p1", jd_id="j1", cv_id="b", split="validation",
                 score=10),
            dict(cols, run=12, pair_id="p0", jd_id="j0", cv_id="a", split="train", score=5)]
    path = tmp_path / "s.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = batch.load_scores(path)
    assert out.pair_id.tolist() == ["p0", "p1"] and out.score.tolist() == [80, 10]
    rows[1]["score"] = 101
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="0-100"):
        batch.load_scores(path)


# --- invariant 3: selection on validation --------------------------------------------------

def test_validation_metric_is_ordering_pairwise_against_the_llm_score():
    """Fails if the metric is not per-JD pairwise vs the run-11 score (reference ties are
    no pair, a score tie counts 0.5), or if an all-tied JD is not skipped."""
    frame = _scores_frame()
    perfect = models.per_jd_pairwise(frame, frame.score.to_numpy())
    assert perfect.to_dict() == {"j0": 1.0, "j1": 1.0}
    # j0 reversed -> 0; j1: e/f tie in the reference is no pair; g lowest is right
    s = np.array([1, 2, 3, 4, 9, -2, -1], float)
    v = models.per_jd_pairwise(frame, s)
    assert v["j0"] == 0.0
    assert v["j1"] == 0.5          # (e, g) right, (f, g) wrong; (e, f) tied: no pair
    flat = frame.assign(score=50.0)
    assert models.per_jd_pairwise(flat, s).empty


def test_pick_prefers_the_earlier_point_on_a_tie():
    """Fails if a tie goes to the later grid point (A3.6)."""
    pts = [{"validation_pairwise": 0.7}, {"validation_pairwise": 0.8},
           {"validation_pairwise": 0.8}]
    assert models.pick(pts) == 1
    assert models.pick([{"validation_pairwise": 0.6}] * 3) == 0


def test_grid_is_the_registered_one():
    """A3.2/A3.3: base MiniLM everywhere; CE_GRID x δ {10, 20}; R3 pointwise first."""
    r1, r2, r3 = (models.grid(r) for r in (models.R1, models.R2, models.R3))
    assert len(r1) == 2 and len(r2) == 4 and len(r3) == 6
    assert all(c["base"] == neural.MODEL_ID for c in r1 + r2 + r3)
    assert [(c["lr"], c["epochs"]) for c in r1] == [(2e-5, 1), (5e-5, 3)]
    assert [(c["lr"], c["delta"]) for c in r2] == [(2e-5, 10), (2e-5, 20), (5e-5, 10),
                                                    (5e-5, 20)]
    assert [c["form"] for c in r3] == ["pointwise"] * 2 + ["pairwise"] * 4
    assert {c["max_length"] for c in r1 + r2} == {384} and {c["max_length"] for c in r3} == {256}
    assert models.DELTAS == (10, 20)
    assert all(c.get("micro_batch") == models.MICRO_BATCH for c in r2 + r3[2:])
    assert all("micro_batch" not in c for c in r1 + r3[:2])


def test_vs_random_is_the_paired_delta_against_one_half():
    v = pd.Series([0.9, 0.8, 0.7, 0.85], index=list("abcd"))
    out = models.vs_random(v)
    assert out["delta"] == pytest.approx(0.3125) and out["beats_random"]
    assert not models.vs_random(pd.Series([0.4, 0.6, 0.5], index=list("abc")))["beats_random"]


# --- 8.1: the interface -------------------------------------------------------------------

class _Length(BaseSystem):
    name, rung = "length", "t"

    def _score(self, pairs):
        return pairs.cv_text.str.len().to_numpy(float)


def test_system_reranker_scores_each_cv_in_input_order():
    r = base.SystemReranker(_Length())
    assert r.score("jd", ["aaa", "a", "aa"]) == [3.0, 1.0, 2.0]
    assert r.score("jd", []) == []


# --- the pairwise loop --------------------------------------------------------------------

def test_ranknet_loop_learns_the_preference_direction():
    """Fails if the loss sign is flipped (the model would learn lo > hi)."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.zeros(1))
    prefs = pd.DataFrame({"x_hi": np.linspace(1, 2, 32), "x_lo": np.linspace(0, 1, 32)})

    def score_hi_lo(rows):
        return (w * torch.tensor(rows.x_hi.to_numpy(), dtype=torch.float32),
                w * torch.tensor(rows.x_lo.to_numpy(), dtype=torch.float32))

    config = {"seed": 0, "batch": 8, "epochs": 3, "lr": 0.1, "warmup_ratio": 0.0}
    losses = models.ranknet_loop(score_hi_lo, [w], prefs, config)
    assert float(w.detach()) > 0 and losses[-1] < losses[0]


def test_micro_batches_do_not_change_the_step():
    """Accumulating over micro-batches is the same optimiser step as one full batch."""
    torch = pytest.importorskip("torch")
    prefs = pd.DataFrame({"x_hi": np.linspace(1, 2, 32), "x_lo": np.linspace(0, 1.5, 32)})
    finals = []
    for micro in (16, 4, 1):
        w = torch.nn.Parameter(torch.zeros(1))

        def score_hi_lo(rows, w=w):
            return (w * torch.tensor(rows.x_hi.to_numpy(), dtype=torch.float32),
                    w * torch.tensor(rows.x_lo.to_numpy(), dtype=torch.float32))

        config = {"seed": 0, "batch": 16, "epochs": 2, "lr": 0.1, "warmup_ratio": 0.0,
                  "micro_batch": micro}
        losses = models.ranknet_loop(score_hi_lo, [w], prefs, config)
        finals.append((float(w.detach()), losses))
    assert finals[1][0] == pytest.approx(finals[0][0], rel=1e-5)
    assert finals[2][0] == pytest.approx(finals[0][0], rel=1e-5)
    assert finals[1][1] == pytest.approx(finals[0][1], rel=1e-5)


# --- invariant 5: the D46 guard -----------------------------------------------------------

FORBIDDEN = re.compile(r"a2-final-labels|judgements|final_set|FINAL_CSV|FINAL_LABELS")


def test_no_training_module_reads_the_final_set():
    """D46 / A3.11: nothing under `rerank/` but the batch dispatcher names the final set's
    labels or the human judgements. Fails if a training module learns to read them."""
    offenders = {p.name: FORBIDDEN.findall(p.read_text(encoding="utf-8"))
                 for p in RERANK_DIR.glob("*.py") if p.name != "scoring_batch.py"}
    assert {"base.py", "batch.py", "models.py"} <= set(offenders)
    assert not {k: v for k, v in offenders.items() if v}
