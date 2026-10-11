"""The run-11 training data for the stage-2 rerankers (amendment A3, 06 §8).

Targets: the 0-100 score of run 11 (`llm-train-scores.csv`, D50), one per pair, split by
JD into `train` (80 JDs) and `validation` (20), CV-disjoint by construction (15 §3).
Inputs: `text_v2` (D53), JD = `Long Description` stripped, CV = `corpus.build_cv_text(…,
"v2")`, from the A2 train region. The judge saw `Position`; the inputs do not (A3.1).

This module reads the batch only, never a final-set or human label (D46). `tests/test_rerank.py` holds it to that.

Frames are cached under `data/processed/rerank/`, keyed by the scores file's digest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from candidate_screener.config import DOCS_DATA, PROCESSED
from candidate_screener.matching import corpus, neural

SCORES_CSV = DOCS_DATA / "manifests" / "llm-train-scores.csv"
CACHE = PROCESSED / "rerank"
RUN = 11
AGENT_PREFIX = "5d986c82"
MODEL = "claude-haiku-4-5"
TEXT_VARIANT = "v2"
SPLITS = ("train", "validation")
FRAME_COLUMNS = ("pair_id", "jd_id", "cv_id", "split", "score", "jd_text", "cv_text")


def load_scores(path=SCORES_CSV) -> pd.DataFrame:
    """Run 11, every row parsed, one row per pair, scores in 0-100; or it raises."""
    s = pd.read_csv(path)
    s = s[(s.run.astype(int) == RUN) & s.agent_sha256.str.startswith(AGENT_PREFIX)
          & (s.model == MODEL)]
    if s.empty or not s.pair_id.is_unique:
        raise ValueError("run 11 must hold one row per pair")
    if not s.parsed_ok.astype(bool).all():
        raise ValueError("run 11 holds unparsed rows; the plan has no rule for them (15 §9)")
    if not s.score.between(0, 100).all():
        raise ValueError("a run-11 score is outside 0-100")
    if set(s.split) != set(SPLITS):
        raise ValueError(f"run 11 splits must be {SPLITS}")
    return s[["pair_id", "jd_id", "cv_id", "split", "score"]].reset_index(drop=True)


def check_disjoint(frame: pd.DataFrame) -> None:
    """Train and validation share no JD and no CV (the batch's own guarantee, re-checked)."""
    tr, va = frame[frame.split == "train"], frame[frame.split == "validation"]
    if set(tr.jd_id) & set(va.jd_id):
        raise ValueError("a JD is in both train and validation")
    if set(tr.cv_id) & set(va.cv_id):
        raise ValueError("a CV is in both train and validation")


def attach_text(scores: pd.DataFrame, jd: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    """`jd` / `cv`: raw Djinni rows (`id` plus the text fields). CV text is `text_v2`."""
    jd_text = jd.set_index("id")["Long Description"].fillna("").astype(str).str.strip()
    cv_text = pd.Series(corpus.build_cv_text(cv, TEXT_VARIANT).to_numpy(), index=cv["id"])
    out = scores.assign(jd_text=scores.jd_id.map(jd_text), cv_text=scores.cv_id.map(cv_text))
    if out[["jd_text", "cv_text"]].isna().any().any():
        raise ValueError("a run-11 pair has no document text")
    check_disjoint(out)
    return out[list(FRAME_COLUMNS)]


def load() -> pd.DataFrame:
    """The 1,000 run-11 pairs with `text_v2` and their score (cached)."""
    path = CACHE / f"batch-{TEXT_VARIANT}-{neural.file_digest(SCORES_CSV)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from candidate_screener.data.profile import load_split
    scores = load_scores()
    jd = load_split("djinni-jd", "train")
    cv = load_split("djinni-cv", "train")
    jd = jd[jd["id"].isin(set(scores.jd_id))]
    cv = cv[cv["id"].isin(set(scores.cv_id))].copy()
    frame = attach_text(scores, jd, cv)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def preference_pairs(frame: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Every within-JD pair whose score gap is at least `delta`, as (hi, lo) rows.

    Pairs never cross JDs (A3.5). Order is deterministic: JD, then hi, then lo position.
    """
    if delta <= 0:
        raise ValueError("delta must be positive: a tie is no preference")
    rows = []
    for jd_id, g in frame.groupby("jd_id", sort=True):
        g = g.sort_values("pair_id")
        s = g.score.to_numpy(float)
        for i in range(len(g)):
            for j in range(len(g)):
                if s[i] - s[j] >= delta:
                    hi, lo = g.iloc[i], g.iloc[j]
                    rows.append({"jd_id": jd_id, "jd_text": hi.jd_text,
                                 "hi_pair": hi.pair_id, "lo_pair": lo.pair_id,
                                 "hi_text": hi.cv_text, "lo_text": lo.cv_text,
                                 "gap": float(s[i] - s[j])})
    return pd.DataFrame(rows, columns=["jd_id", "jd_text", "hi_pair", "lo_pair", "hi_text",
                                       "lo_text", "gap"])


def gap_counts(frame: pd.DataFrame, deltas) -> dict:
    """How many preference pairs each δ yields (for the record)."""
    return {str(d): int(len(preference_pairs(frame, d))) for d in deltas}


def score_summary(frame: pd.DataFrame) -> dict:
    out = {}
    for split in SPLITS:
        s = frame[frame.split == split].score.to_numpy(float)
        out[split] = {"pairs": int(len(s)), "jds": int(frame[frame.split == split].jd_id.nunique()),
                      "score_mean": round(float(np.mean(s)), 3),
                      "score_sd": round(float(np.std(s)), 3),
                      "distinct_scores": int(len(np.unique(s)))}
    return out
