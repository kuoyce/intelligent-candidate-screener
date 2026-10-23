"""Hand-off table for stage 3: the 600 A2 final-set pairs with their X2 scores and ranks.

Ids, stratum, final label and the X2 score only. No JD or CV text, so the record is safe to
commit (the text is rebuilt with `corpus.a2_pairs(text_variant="v2")`). The labels are
for checking results, never inputs (D46). X2 is fitted on nothing (`fit(None)`) and scored
on `text_v2` (D53).

Records: `output/matching/a2-x2-scores.csv` and `output/matching/a2-features.csv` (the 15
`learned.F12` + `learned.EXTRA` features per pair, numbers only; `--features` writes it),
and `a2-jd-extraction.csv` / `a2-cv-extraction.csv` (role family, vocabulary skills, years and
seniority per document; `--extract`). Skills are canonical vocabulary names, never CV quotes.

    uv run python -m candidate_screener.evaluation.a2_export
    uv run python -m candidate_screener.evaluation.a2_export --features
    uv run python -m candidate_screener.evaluation.a2_export --extract
"""
from __future__ import annotations

import argparse

import pandas as pd

from candidate_screener.config import OUTPUT

RECORD = OUTPUT / "matching" / "a2-x2-scores.csv"
FEATURES_RECORD = OUTPUT / "matching" / "a2-features.csv"
JD_EXTRACT_RECORD = OUTPUT / "matching" / "a2-jd-extraction.csv"
CV_EXTRACT_RECORD = OUTPUT / "matching" / "a2-cv-extraction.csv"
TEXT_VARIANT = "v2"
DECIMALS = 6
COLUMNS = ["pair_id", "stratum", "jd_id", "cv_id", "x2_score", "x2_rank", "final_label"]


def rank_within_jd(pairs: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """Add `x2_score` and `x2_rank` (1 = best; ties broken by cv_id, as `pipeline.screen` does)."""
    out = pairs[["pair_id", "stratum", "jd_id", "cv_id"]].copy()
    out["x2_score"] = score.to_numpy().round(DECIMALS)
    out = out.sort_values(["jd_id", "x2_score", "cv_id"], ascending=[True, False, True])
    out["x2_rank"] = out.groupby("jd_id").cumcount() + 1
    return out


def build() -> pd.DataFrame:
    from candidate_screener.evaluation import final_set
    from candidate_screener.matching import corpus, extracted, lexical
    pairs = corpus.a2_pairs(text_variant=TEXT_VARIANT)
    system = extracted.ExtractedPlus(lexical.Tfidf(TEXT_VARIANT), "X2")
    system.fit(None)
    out = rank_within_jd(pairs, system.score(pairs))
    labels = final_set.load().set_index("pair_id").final_label
    out["final_label"] = out.pair_id.map(labels)
    if len(out) != 600 or out.jd_id.nunique() != 60 or out.final_label.isna().any():
        raise SystemExit("expected 600 labelled pairs over 60 JDs")
    return out[COLUMNS].reset_index(drop=True)


def build_features() -> pd.DataFrame:
    """One row per pair: ids, stratum, then every `learned.F12` + `learned.EXTRA` feature."""
    from candidate_screener.matching import corpus, learned
    pairs = corpus.a2_pairs(text_variant=TEXT_VARIANT)
    pairs = pairs.assign(jd_title=pairs.jd_position)   # the JD title is the one allowed field (D54)
    x = learned.Features()(pairs, list(learned.F12) + list(learned.EXTRA)).round(DECIMALS)
    ids = pairs.set_index("pair_id")[["stratum", "jd_id", "cv_id"]]
    out = ids.join(x, how="left").reset_index()
    if len(out) != 600 or out.isna().any().any():
        raise SystemExit("expected 600 pairs with every feature present")
    return out


def build_extraction() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(JD table, CV table): role family, vocabulary skills, years and seniority per document.

    One definition with inference: `inference.profiles` (A2 documents, so "present" in CV date
    ranges is `experience.REFERENCE_DATE`)."""
    from candidate_screener.extraction import role_family, rules
    from candidate_screener.inference import profiles
    from candidate_screener.matching import corpus
    pairs = corpus.a2_pairs(text_variant=TEXT_VARIANT)
    model, matcher = role_family.load_role_model(), rules.default_matcher()
    jds = pairs.drop_duplicates("jd_id").reset_index(drop=True)
    cvs = pairs.drop_duplicates("cv_id").reset_index(drop=True)

    jp = profiles.extract_jds(jds.jd_text, jds.jd_position, model, matcher, ids=jds.jd_id)
    jd = pd.DataFrame({
        "jd_id": jds.jd_id, "stratum": jds.stratum, "jd_title": jds.jd_position,
        "role": [p.role for p in jp], "role_prob": [p.role_prob for p in jp],
        "top_roles": [profiles.roles_text(p.top_roles) for p in jp],
        "skills": [";".join(p.skills) for p in jp],
        "skills_qualified": [";".join(p.skills_qualified) for p in jp],
        "min_years": [p.min_years for p in jp], "max_years": [p.max_years for p in jp],
        "seniority": [p.seniority for p in jp]})

    cp = profiles.extract_cvs(cvs.cv_text, model, matcher, ids=cvs.cv_id)
    cv = pd.DataFrame({
        "cv_id": cvs.cv_id, "role": [p.role for p in cp], "role_prob": [p.role_prob for p in cp],
        "top_roles": [profiles.roles_text(p.top_roles) for p in cp],
        "skills": [";".join(p.skills) for p in cp],
        "skills_qualified": [";".join(p.skills_qualified) for p in cp],
        "years": [p.years for p in cp], "years_source": [p.years_source for p in cp],
        "seniority": [p.seniority for p in cp], "seniority_source": [p.seniority_source for p in cp]})
    if len(jd) != 60 or jd.role.isna().any() or cv.role.isna().any():
        raise SystemExit("expected 60 JDs with a role, and a role for every CV")
    return jd, cv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", action="store_true", help="write the feature table instead")
    ap.add_argument("--extract", action="store_true", help="write the JD and CV extraction tables")
    args = ap.parse_args(argv)
    if args.extract:
        jd, cv = build_extraction()
        for frame, path in ((jd, JD_EXTRACT_RECORD), (cv, CV_EXTRACT_RECORD)):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            print(f"{len(frame)} rows -> {path}")
        return 0
    frame, path = (build_features(), FEATURES_RECORD) if args.features else (build(), RECORD)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"{len(frame)} pairs, {frame.jd_id.nunique()} JDs -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
