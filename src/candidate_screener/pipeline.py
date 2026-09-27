"""End-to-end screening: rank a pool of CVs against one JD and show the evidence.

Extraction follows D49: the learned tagger (optional, `tagger=`) adds free-text JD
requirements, and the rules matcher does every exact / alias match on the CV side. The
semantic `supported` step runs when an `encoder` is passed.

    screen(jd_text, cvs) -> [Result, ...] in rank order

The ranker is the rung the registered rule selected (D48, `SELECTED`). The Fit / No Fit
call comes from a rung with a threshold fitted off the final set (D46): the metadata rule
when both sides carry a `Primary Keyword`, otherwise none. "Confidence" is the CV's
percentile within this pool, labelled as relative, because no rung here is calibrated to
a probability of fit. Every quoted span is verbatim (evidence.assert_faithful).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from candidate_screener.evidence import evidence as ev
from candidate_screener.evidence import requirements as req

#: The selected pipeline (D48, `07-selection.md`): the registered rule's choice per context.
SELECTED = "metadata + TF-IDF"
SELECTED_TEXT_ONLY = "SBERT (weak fine-tuned)"


@dataclass
class Result:
    cv_id: str
    rank: int
    score: float
    fit_call: str | None
    confidence: float
    coverage: float | None
    years_required: int | None
    years_stated: int | None
    evidence: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def default_ranker(has_metadata: bool):
    """D48: metadata + TF-IDF where both sides carry a keyword; else the weakly
    fine-tuned SBERT, falling back to TF-IDF when the `ml` group is not installed."""
    from candidate_screener.matching import lexical
    if has_metadata:
        return lexical.MetadataPlus(lexical.Tfidf(), "1a+2")
    try:
        from candidate_screener.matching.sbert import SbertWeak
        return SbertWeak()
    except ImportError:
        return lexical.Tfidf()


def _frame(jd_text: str, cvs: pd.DataFrame, jd_meta: dict | None) -> pd.DataFrame:
    jd_meta = jd_meta or {}
    frame = pd.DataFrame({
        "cv_id": cvs.cv_id.astype(str).to_numpy(), "cv_text": cvs.cv_text.astype(str).to_numpy(),
        "jd_id": jd_meta.get("jd_id", "jd"), "jd_text": jd_text,
        "jd_keyword": jd_meta.get("keyword", np.nan), "jd_exp_band": jd_meta.get("exp_band", np.nan),
        "cv_keyword": cvs.get("cv_keyword", pd.Series([np.nan] * len(cvs))).to_numpy(),
        "cv_exp_band": cvs.get("cv_exp_band", pd.Series([np.nan] * len(cvs))).to_numpy()})
    frame["pair_id"] = frame.jd_id + "__" + frame.cv_id
    return frame


def screen(jd_text: str, cvs: pd.DataFrame, jd_meta: dict | None = None, ranker=None,
           matcher=None, encoder=None, tagger=None) -> list[Result]:
    """Rank `cvs` (columns `cv_id`, `cv_text`, optional `cv_keyword`, `cv_exp_band`)."""
    frame = _frame(jd_text, cvs, jd_meta)
    has_meta = frame.jd_keyword.notna().all() and frame.cv_keyword.notna().any()
    if ranker is None:
        ranker = default_ranker(has_meta)
        ranker.fit(None)
    scores = ranker.score(frame).to_numpy()
    if matcher is None:
        from candidate_screener.extraction.rules import default_matcher
        matcher = default_matcher()
    wanted = req.requirements(jd_text, matcher, tagger)
    need_years = req.years_required(jd_text)
    order = np.lexsort((frame.cv_id.to_numpy(), -scores))
    pct = pd.Series(scores).rank(pct=True).to_numpy()
    results = []
    for rank, i in enumerate(order, start=1):
        row = frame.iloc[i]
        found = ev.gather(wanted, row.cv_text, matcher, encoder)
        call = None
        if has_meta and pd.notna(row.cv_keyword):
            call = "Fit" if row.cv_keyword == row.jd_keyword else "No Fit"
        results.append(Result(
            cv_id=row.cv_id, rank=rank, score=round(float(scores[i]), 6), fit_call=call,
            confidence=round(float(pct[i]), 4), coverage=ev.coverage(found),
            years_required=need_years, years_stated=req.years_stated(row.cv_text),
            evidence=ev.as_dicts(found)))
    return results
