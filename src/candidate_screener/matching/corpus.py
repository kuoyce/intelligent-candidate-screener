"""Pair frames for the ladder: A2 (primary, D40) and A1 (secondary), plus A2 train text.

Text comes from the raw sources through the same builders the annotation campaign used,
so a rung scores exactly what the annotators read. Frames are cached under
`data/processed/matching/` (git-ignored), keyed by the digest of their manifest.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.data.a2_finetune import (JD_EXP_TO_BAND, cv_experience_band,
                                                load_train_region)
from candidate_screener.data.profile import load_split
from candidate_screener.evaluation import final_set

CACHE = PROCESSED / "matching"
EXP_ORDER = {"0-1": 0, "2-3": 1, "4-6": 2, "7+": 3}


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _djinni(region_ids: tuple[set[str], set[str]] | None = None
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """English, non-empty Djinni JDs and CVs, optionally restricted to a region's ids."""
    from candidate_screener.annotation.sample import cv_text
    jd, cv = load_split("djinni-jd", "train"), load_split("djinni-cv", "train")
    if region_ids is not None:
        jd = jd[jd["id"].isin(region_ids[0])]
        cv = cv[cv["id"].isin(region_ids[1])]
    jd = jd[jd["Long Description_lang"] == "en"].copy()
    jd["jd_text"] = jd["Long Description"].fillna("").astype(str).str.strip()
    jd = jd[jd.jd_text.str.len() > 0]
    jd["exp_band"] = jd["Exp Years"].map(JD_EXP_TO_BAND)
    cv = cv[cv["CV_lang"] == "en"].copy()
    cv["cv_text"] = cv_text(cv)
    cv = cv[cv.cv_text.str.len() > 0]
    cv["exp_band"] = cv["Experience Years"].map(cv_experience_band)
    return jd, cv


@lru_cache(maxsize=1)
def a2_pairs() -> pd.DataFrame:
    """The 600 labelled A2 pairs with text and metadata — no labels (D46)."""
    manifest = final_set.PAIRS_CSV
    path = CACHE / f"a2-pairs-{_digest(manifest)}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from candidate_screener.annotation.sample import load_english
    pairs = pd.read_csv(manifest)
    jd, cv = load_english()
    jd, cv = jd.set_index("id"), cv.set_index("id")
    frame = pd.DataFrame({
        "pair_id": pairs.pair_id, "stratum": pairs.stratum,
        "jd_id": pairs.jd_id, "cv_id": pairs.cv_id,
        "jd_text": pairs.jd_id.map(jd.jd_text), "cv_text": pairs.cv_id.map(cv.cv_text),
        "jd_keyword": pairs.jd_id.map(jd["Primary Keyword"]),
        "cv_keyword": pairs.cv_id.map(cv["Primary Keyword"]),
        "jd_exp_band": pairs.jd_id.map(jd.exp_band), "cv_exp_band": pairs.cv_id.map(cv.exp_band),
        "jd_position": pairs.jd_id.map(jd.Position), "cv_position": pairs.cv_id.map(cv.Position),
    })
    if frame[["jd_text", "cv_text"]].isna().any().any():
        raise ValueError("an A2 pair has no text — the eval-region loader and the pair "
                         "manifest disagree")
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def keyword_match(pairs: pd.DataFrame) -> pd.Series:
    return (pairs.jd_keyword.notna() & (pairs.jd_keyword == pairs.cv_keyword)).astype(int)


def experience_compatible(pairs: pd.DataFrame) -> pd.Series:
    """1 when the CV's experience band meets the JD's; 0 when it does not or is unknown."""
    cv, jd = pairs.cv_exp_band.map(EXP_ORDER), pairs.jd_exp_band.map(EXP_ORDER)
    return (cv.notna() & jd.notna() & (cv >= jd)).astype(int)


@lru_cache(maxsize=4)
def a2_train_documents(n_jd: int = 20_000, n_cv: int = 20_000, seed: int = 0
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A seeded sample of A2 **train-region** JDs and CVs (D19): for IDF and weak pairs."""
    path_jd = CACHE / f"a2-train-jd-{n_jd}-{seed}.parquet"
    path_cv = CACHE / f"a2-train-cv-{n_cv}-{seed}.parquet"
    if path_jd.exists() and path_cv.exists():
        return pd.read_parquet(path_jd), pd.read_parquet(path_cv)
    jd, cv = _djinni(load_train_region())
    keep_jd = ["id", "jd_text", "Primary Keyword", "exp_band", "Position"]
    keep_cv = ["id", "cv_text", "Primary Keyword", "exp_band", "Position"]
    jd = jd.sort_values("id").sample(min(n_jd, len(jd)), random_state=seed)[keep_jd]
    cv = cv.sort_values("id").sample(min(n_cv, len(cv)), random_state=seed)[keep_cv]
    CACHE.mkdir(parents=True, exist_ok=True)
    jd.reset_index(drop=True).to_parquet(path_jd, index=False)
    cv.reset_index(drop=True).to_parquet(path_cv, index=False)
    return pd.read_parquet(path_jd), pd.read_parquet(path_cv)


def a1_pairs(split: str) -> pd.DataFrame:
    """A1 `train` or `test` from the leak-free split, as a pair frame with a Fit label."""
    from candidate_screener.baselines.run import load_split as load_fit_split
    df = load_fit_split(split)
    return pd.DataFrame({
        "pair_id": df.jd_id + "__" + df.resume_id, "jd_id": df.jd_id, "cv_id": df.resume_id,
        "jd_text": df.job_description_text, "cv_text": df.resume_text,
        "jd_keyword": np.nan, "cv_keyword": np.nan, "jd_exp_band": np.nan, "cv_exp_band": np.nan,
        "label": df.label, "fit": df.label.isin(final_set.FIT_CLASSES).astype(int),
    })
