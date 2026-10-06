"""Residual title leak under `text_v2` (D53, plan 2026-10-06 02 §2, task P1.2).

`text_v2` removes the CV's `Position` **field**; it does not scrub the string. A CV can
still say "Senior Python Developer" in its own words. This module measures how often the
removed title survives in the text a system reads, so every `text_v2` figure can sit beside
it. It measures and changes nothing.

A title is normalised before matching: lower-cased, seniority tokens removed, whitespace
collapsed, punctuation trimmed from the ends. A title that normalises to nothing (e.g.
"Senior") is skipped and counted. A match is the normalised title as a whole-word
substring of the lower-cased, whitespace-collapsed text.

Two shares, each for `generic`, `targeted` and `total`:

- **CV:** of the distinct final-set CVs, how many have their own (removed) `Position` in
  their `v2` text. CI resamples CVs. A pair-level share (per-JD mean over its CVs) is also
  given, with the CI resampling JDs.
- **JD:** of the final-set JDs, how many have their own `Position` in `Long Description`
  (the JD `Position` was never part of the JD text). CI resamples JDs.

Usage:
    uv run python -m candidate_screener.evaluation.text_leak
"""
from __future__ import annotations

import json
import re

import pandas as pd

from candidate_screener.evaluation import resample

#: Removed before matching. The plan names junior / middle / senior / lead / head /
#: trainee; jr, sr, mid and intern are their common short or equivalent forms.
SENIORITY = ("junior", "jr", "middle", "mid", "senior", "sr", "lead", "head", "trainee",
             "intern")
_SENIORITY_RE = re.compile(r"(?<!\w)(?:" + "|".join(SENIORITY) + r")\.?(?!\w)")
_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")
SCOPES = ("generic", "targeted", "total")
DECIMALS = 6


def normalise_text(text) -> str:
    return re.sub(r"\s+", " ", str(text if isinstance(text, str) else "")).strip().lower()


def normalise_title(title) -> str:
    """Lower-case, drop seniority tokens, collapse whitespace, trim edge punctuation."""
    out = _SENIORITY_RE.sub(" ", normalise_text(title))
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r" ([,;:/)])", r"\1", out)    # "team lead, java" -> "team, java"
    previous = None
    while previous != out:                     # "- / qa ," -> "qa"
        previous, out = out, _EDGE_RE.sub("", out).strip()
    return out


def contains_title(text, title) -> bool | None:
    """True / False when the normalised title is / is not a whole-word span of the text;
    None when the title normalises to nothing (skipped, not counted as either)."""
    norm = normalise_title(title)
    if not norm:
        return None
    return re.search(r"(?<!\w)" + re.escape(norm) + r"(?!\w)", normalise_text(text)) is not None


def _share(flags: pd.Series, seed: int, resampled: str, binary: bool = True) -> dict:
    flags = flags.dropna().astype(float)
    if flags.empty:
        return {"share": None, "ci": None, "n": 0, "resampled": resampled}
    low, high = resample.mean_ci(flags.reset_index(drop=True), seed=seed)
    out = {"share": round(float(flags.mean()), DECIMALS),
           "ci": [round(low, DECIMALS), round(high, DECIMALS)],
           "n": int(len(flags)), "resampled": resampled}
    if binary:
        out["positives"] = int(flags.sum())
    return out


def report(pairs: pd.DataFrame, seed: int = 0) -> dict:
    """`pairs`: an `a2_pairs(text_variant="v2")` frame (stratum, jd/cv ids, texts,
    `jd_position`, `cv_position`)."""
    pairs = pairs.assign(
        cv_leak=[contains_title(t, p) for t, p in zip(pairs.cv_text, pairs.cv_position)],
        jd_leak=[contains_title(t, p) for t, p in zip(pairs.jd_text, pairs.jd_position)])
    out: dict = {"cv": {}, "cv_pairs_per_jd": {}, "jd": {}}
    for scope in SCOPES:
        sub = pairs if scope == "total" else pairs[pairs.stratum == scope]
        cvs = sub.drop_duplicates("cv_id").set_index("cv_id")
        jds = sub.drop_duplicates("jd_id").set_index("jd_id")
        per_jd = sub.dropna(subset=["cv_leak"]).assign(
            cv_leak=lambda d: d.cv_leak.astype(float)).groupby("jd_id").cv_leak.mean()
        out["cv"][scope] = {**_share(cvs.cv_leak, seed, "CVs"),
                            "skipped_empty_title": int(cvs.cv_leak.isna().sum()),
                            "empty_v2_text": int((cvs.cv_text.str.strip() == "").sum())}
        out["cv_pairs_per_jd"][scope] = {**_share(per_jd, seed, "JDs", binary=False),
                                         "n_pairs": int(sub.cv_leak.notna().sum())}
        out["jd"][scope] = {**_share(jds.jd_leak, seed, "JDs"),
                            "skipped_empty_title": int(jds.jd_leak.isna().sum())}
    out["method"] = {
        "seniority_tokens_removed": list(SENIORITY),
        "match": "normalised title as a whole-word substring of the lower-cased, "
                 "whitespace-collapsed text; no scrubbing applied (D53)",
        "ci": "95% percentile bootstrap, 10,000 resamples of the unit named in `resampled`",
    }
    return out


def main() -> int:
    from candidate_screener.matching import corpus
    print(json.dumps(report(corpus.a2_pairs(text_variant="v2")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
