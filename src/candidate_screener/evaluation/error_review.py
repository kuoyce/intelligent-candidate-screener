"""Manual error review (plan 2026-10-13-error-analysis-2, step 4): cases, provenance, saving.

A case is one mistake: a (Fit, No Fit) pair in a JD that the hand-set score orders the wrong
way. Texts are redacted and cached under `data/interim` (git-ignored). Answers are error-reason
tags, appended to `output/extraction/error-review-reasons.csv`; they are not relevance labels (an override is a flag on a disputed label, not a replacement)
and never enter `judgements.csv`, the final set or a training set (D42, D46).

    uv run python -m candidate_screener.evaluation.error_review --build
    uv run streamlit run src/candidate_screener/evaluation/error_review_app.py
    uv run python -m candidate_screener.evaluation.error_review --summary
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from candidate_screener.annotation.redact import redact_text
from candidate_screener.config import DATA, OUTPUT

CASES = DATA / "interim" / "error-review-cases.csv"
REASONS = OUTPUT / "extraction" / "error-review-reasons.csv"
SUMMARY = OUTPUT / "extraction" / "error-review-summary.json"
CHOICES = {"A": "role mismatch, but the Fit CV has transferable skills",
           "B": "seniority or years are wrong",
           "C": "the No Fit CV matches on words but lacks the core requirement",
           "D": "the label looks wrong",
           "E": "other (add a note)"}
TIER_TEXT = {"humans_agree": "both humans agree", "single_human": "one human only",
             "llm_breaks_tie": "humans split, the LLM broke the tie", "llm_only": "no human, LLM only"}
SEED = 0
#: Which side's label the reviewer is very sure is wrong (a flag, never a label: D42, D46).
OVERRIDES = ("none", "fit_side", "nofit_side", "both")
FIELDS = ["case_id", "bucket", "reason", "note", "override", "reviewer", "timestamp", "seed", "scorer"]


def a2_source(row) -> dict:
    """Who labelled one A2 final-set pair: each human's grade, the LLM's, and the tier."""
    humans = {h: row[f"{h}_raw"] for h in ("yc", "van") if pd.notna(row[f"{h}_raw"])}
    return {"source": TIER_TEXT[row["tier"]], "final_label": row["final_label"], "humans": humans,
            "llm": row["llm_label"], "llm_score": row["llm_score"], "llm_note": row["llm_note"]}


def d50_source(label, score) -> dict:
    return {"source": "LLM only (D50 scoring batch, run 11, no human)", "final_label": label,
            "humans": {}, "llm": label, "llm_score": score, "llm_note": ""}


def describe(info: dict) -> str:
    humans = ", ".join(f"{h}: {g}" for h, g in info["humans"].items()) or "none"
    out = (f"**Labelled by:** {info['source']}  \n**Final label:** {info['final_label']}  \n"
           f"**Humans:** {humans}  \n**LLM:** {info['llm']} (score {info['llm_score']})")
    return out + (f"  \n**LLM note:** {info['llm_note']}" if info["llm_note"] else "")


def build(per_bucket: int = 20, seed: int = SEED) -> pd.DataFrame:
    """Sample up to `per_bucket` mistakes per (set, culprit) with the hand-set score, redacted."""
    from candidate_screener.evaluation import a1_weights as aw, coverage_variants as cv
    from candidate_screener.evaluation import error_analysis as ea, final_set, indomain
    from candidate_screener.matching import corpus
    from candidate_screener.rerank import scoring_batch as sb
    frame = final_set.load().set_index("pair_id")
    lab = indomain.labels(frame.reset_index())
    a2 = corpus.a2_pairs(text_variant="v2").set_index("pair_id").loc[lab.index].reset_index()
    d50, y50 = aw.d50_pairs()
    sel, _ = cv.split_jds(d50.jd_id)
    d50 = d50[d50.jd_id.isin(sel)]
    model = aw.build_scorer()
    notes = aw.llm_notes(frame)
    scores = pd.read_csv(sb.SCORES_CSV).query("run == @sb.RUN and parsed_ok").set_index("pair_id")
    parts = []
    for name, pairs, y in (("A2 final", a2, lab.y), ("D50 selection", d50, y50.loc[d50.pair_id])):
        X = aw.features(model, pairs).join(aw.extra_features(model, pairs)).set_axis(pairs.pair_id)
        s = aw.linear_score(X, ea.HAND)
        inv = ea.sample_mistakes(ea.mistakes(s, y, pairs.set_index("pair_id").jd_id, X, ea.HAND), per_bucket, seed)
        p = pairs.set_index("pair_id")
        for r in inv.itertuples():
            def info(pid):
                if name == "A2 final":
                    f = frame.loc[pid]
                    return a2_source(f.to_dict() | {"llm_label": notes.llm_label[pid], "llm_score": notes.llm_score[pid],
                                                     "llm_note": notes.notes[pid]})
                return d50_source(scores.label[pid], scores.score[pid])
            parts.append({"case_id": f"{name}|{r.fit_id}|{r.nofit_id}", "set": name, "bucket": r.culprit, "kind": r.kind,
                          "jd_text": redact_text(p.jd_text[r.fit_id]), "fit_cv": redact_text(p.cv_text[r.fit_id]),
                          "nofit_cv": redact_text(p.cv_text[r.nofit_id]),
                          "details": json.dumps({"fit": info(r.fit_id), "nofit": info(r.nofit_id),
                                                 "score_fit": float(s[r.fit_id]), "score_nofit": float(s[r.nofit_id]),
                                                 "features_fit": X.loc[r.fit_id].round(3).to_dict(),
                                                 "features_nofit": X.loc[r.nofit_id].round(3).to_dict()}, default=str)})
    return pd.DataFrame(parts)


def shuffled(cases: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Random order, so the bucket cannot be read off the sequence."""
    return cases.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def latest(path=REASONS) -> pd.DataFrame:
    """The latest answer per case (append-only file; the last row wins)."""
    if not path.exists():
        return pd.DataFrame(columns=FIELDS).set_index("case_id")
    return pd.read_csv(path).drop_duplicates("case_id", keep="last").set_index("case_id")


def save(case_id: str, bucket: str, reason: str, note: str, reviewer: str, override: str = "none",
         path=REASONS) -> None:
    if reason not in CHOICES:
        raise ValueError(f"reason {reason!r} is not one of {sorted(CHOICES)}")
    if override not in OVERRIDES:
        raise ValueError(f"override {override!r} is not one of {OVERRIDES}")
    if override != "none" and (reason != "D" or not note):
        raise ValueError("an override needs reason D and a note saying why")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"case_id": case_id, "bucket": bucket, "reason": reason, "note": note, "override": override, "reviewer": reviewer,
                         "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED,
                         "scorer": "hand-set X2"}], columns=FIELDS)
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def label_source(details: str, side: str) -> str:
    """The tier name of one side's label, e.g. `humans split, the LLM broke the tie`."""
    return json.loads(details)[side]["source"].split(" (")[0]


def summarise(reasons: pd.DataFrame, cases: pd.DataFrame) -> dict:
    """Counts only (no text): reason by bucket and set, override by the label source of each side.

    The cases are model mistakes, so these are counts on a biased sample, not label-noise rates."""
    r = reasons.drop_duplicates("case_id", keep="last").merge(cases[["case_id", "set", "details"]], on="case_id")
    r["fit_source"] = [label_source(d, "fit") for d in r.details]
    r["nofit_source"] = [label_source(d, "nofit") for d in r.details]
    r["fit_flagged"] = r.override.isin(["fit_side", "both"])
    r["nofit_flagged"] = r.override.isin(["nofit_side", "both"])

    def table(col, by):
        return {str(k): {str(c): int(v) for c, v in g.items()} for k, g in
                r.groupby(by)[col].value_counts().unstack(fill_value=0).iterrows()}

    flag = lambda side: {str(k): {"n": int(len(g)), "flagged": int(g[f"{side}_flagged"].sum())}
                         for k, g in r.groupby(f"{side}_source")}
    return {"n_cases": int(len(r)), "n_sampled": int(len(cases)), "reviewers": sorted(r.reviewer.astype(str).unique()),
            "reason_by_bucket": table("reason", "bucket"), "reason_by_set": table("reason", "set"),
            "override_by_set": table("override", "set"),
            "fit_side_flagged_by_source": flag("fit"), "nofit_side_flagged_by_source": flag("nofit")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="sample the cases and cache them (redacted, git-ignored)")
    ap.add_argument("--per-bucket", type=int, default=20)
    ap.add_argument("--summary", action="store_true", help="write the counts-only record of the answers so far")
    args = ap.parse_args(argv)
    if args.build:
        cases = build(args.per_bucket)
        CASES.parent.mkdir(parents=True, exist_ok=True)
        cases.to_csv(CASES, index=False)
        print(CASES, cases.groupby(["set", "bucket"]).size().to_dict())
    if args.summary:
        SUMMARY.write_text(json.dumps(summarise(pd.read_csv(REASONS), pd.read_csv(CASES)), indent=2, sort_keys=True) + "\n")
        print(SUMMARY)


if __name__ == "__main__":
    main()
