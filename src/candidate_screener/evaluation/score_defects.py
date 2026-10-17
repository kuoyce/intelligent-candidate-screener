"""Score defects named by the manual review (plan 2026-10-17-score-defects), exploratory.

Three defects in the hand-set X2 score (`2·role + experience + 0.999·TF-IDF`):

- **F1 negation**: a skill inside a negated clause ("No Angular/Vue projects") still counts. The
  fix drops such clauses from the CV text before every term is computed.
- **F2 no experience**: a CV with no work history has an unknown experience band, and X2 counts
  an unknown band as compatible. F2 sets the term to 0 when the CV says it has no professional
  experience and the JD asks for years. F2b is the blunter variant: every unknown band is 0
  (the rung-1a rule).
- **F3 role misses**: diagnostic only, from the reviewed cases (no fix here; a role-model change is
  training and gets its own plan).

The owner deferred the prevalence floor and the bar until after reading the results, so every
figure here is `exploratory` (rule 8), not a registered test. Nothing is fitted; A2 final is scored
on, never tuned on (D46). Counts and AUCs only, no document text.

    uv run python -m candidate_screener.evaluation.score_defects
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import resample

RECORD = OUTPUT / "extraction" / "score-defects.json"
HAND = pd.Series({"role_match": 2.0, "experience_compatible": 1.0, "tfidf_scaled": 0.999})

#: A clause (split on . ; newline and bullets) that says the candidate lacks something:
#: a negation cue, then within the same clause a word about having used it.
_CLAUSE = re.compile(r"[^.;\n•]+")
_NEG = re.compile(r"\b(?:no|not|never|without|lack(?:ing|s)?|zero|none)\b", re.I)
_USE = re.compile(r"\b(?:experience[ds]?|projects?|familiar|worked|work(?:ing)? with|knowledge|used|"
                  r"skills?|exposure|background)\b", re.I)
#: The CV states it has no professional experience.
NO_EXPERIENCE = re.compile(
    r"\b(?:no|without|zero|lack(?:ing)? (?:of )?)\s*(?:any\s+)?(?:prior\s+)?"
    r"(?:commercial|professional|work(?:ing)?|real|industry|production|practical)\s+experience\b"
    r"|\b(?:do not|don't|didn't|did not)\s+have\s+(?:any\s+)?(?:commercial|professional|work(?:ing)?|real)\s+"
    r"experience\b", re.I)


def negated_clauses(text: str) -> list[str]:
    """Clauses that carry a negation cue and a usage word, in order."""
    return [c for c in _CLAUSE.findall(str(text)) if _NEG.search(c) and _USE.search(c)]


def strip_negated(text: str) -> str:
    """F1: the text with every negated clause removed (punctuation kept)."""
    return _CLAUSE.sub(lambda m: "" if _NEG.search(m.group()) and _USE.search(m.group()) else m.group(), str(text))


def states_no_experience(text: str) -> bool:
    return bool(NO_EXPERIENCE.search(str(text)))


def f2_experience(base: pd.Series, jd_text: pd.Series, cv_text: pd.Series) -> pd.Series:
    """F2: 0 where the CV states no professional experience and the JD asks for years."""
    from candidate_screener.extraction import experience
    asks = jd_text.map(lambda t: (experience.jd_experience(str(t)).min_years or 0) > 0)
    hit = cv_text.map(states_no_experience)
    return base.where(~(asks & hit).to_numpy(), 0.0)


def paired(new: pd.Series, base: pd.Series, y: pd.Series, jd: pd.Series, seed: int = 0) -> dict:
    """Mean per-JD AUC difference new - base with a JD-resampled CI."""
    from candidate_screener.evaluation import indomain
    d = (indomain.per_jd_auc(y, new, jd) - indomain.per_jd_auc(y, base, jd)).dropna()
    lo, hi = resample.mean_ci(d, seed=seed)
    return {"delta": float(d.mean()), "ci": [lo, hi], "n_jd": int(len(d)),
            "jds_up": int((d > 0).sum()), "jds_down": int((d < 0).sum())}


def run() -> dict:
    from candidate_screener.evaluation import a1_weights as aw, coverage_variants as cv, final_set, indomain
    from candidate_screener.extraction import experience
    from candidate_screener.matching import corpus
    frame = final_set.load()
    lab = indomain.labels(frame)
    a2 = corpus.a2_pairs(text_variant="v2").set_index("pair_id").loc[lab.index].reset_index()
    d50, y50 = aw.d50_pairs()
    d50 = d50.reset_index(drop=True)
    sel, held = cv.split_jds(d50.jd_id)
    model = aw.build_scorer()
    sets = {"D50 selection (70 JDs)": (d50[d50.jd_id.isin(sel)], y50),
            "D50 held-out (30 JDs)": (d50[d50.jd_id.isin(held)], y50),
            "A2 final (59 JDs)": (a2, lab.y)}
    out = {"sets": {}}
    for name, (pairs, y) in sets.items():
        pairs = pairs.reset_index(drop=True)
        ids = pairs.pair_id.to_numpy()
        X = aw.features(model, pairs)
        X1 = aw.features(model, pairs.assign(cv_text=pairs.cv_text.map(strip_negated)))
        jd_t, cv_t = pairs.set_index("pair_id").jd_text, pairs.set_index("pair_id").cv_text
        f2 = f2_experience(X.experience_compatible, jd_t, cv_t)
        f2b = experience.experience_compatible_extracted(list(jd_t), list(cv_t), unknown_compatible=False).set_axis(ids)
        arms = {"X2": X,
                "F1 negation": X1,
                "F2 no experience": X.assign(experience_compatible=f2),
                "F2b unknown = 0": X.assign(experience_compatible=f2b.astype(float)),
                "F1 + F2": X1.assign(experience_compatible=f2_experience(X1.experience_compatible, jd_t, cv_t))}
        scores = {a: aw.linear_score(x, HAND) for a, x in arms.items()}
        jd = pairs.set_index("pair_id").jd_id
        yy = y.loc[ids]
        scopes = {"total": ids}
        if "stratum" in pairs:
            scopes |= {s: pairs.pair_id[pairs.stratum == s].to_numpy() for s in ("generic", "targeted")}
        n_neg = cv_t.map(lambda t: len(negated_clauses(t)))
        rec = {"n_pairs": int(len(ids)), "n_jds": int(jd.nunique()), "fit_rate": float(yy.mean()),
               "prevalence": {
                   "CVs with a negated clause": float((n_neg > 0).mean()),
                   "pairs whose F1 score changes": float((scores["F1 negation"] != scores["X2"]).mean()),
                   "CVs stating no professional experience": float(cv_t.map(states_no_experience).mean()),
                   "pairs whose F2 score changes": float((scores["F2 no experience"] != scores["X2"]).mean()),
                   "pairs whose F2b score changes": float((scores["F2b unknown = 0"] != scores["X2"]).mean()),
                   "pairs with an unknown experience band": float(
                       (X.experience_compatible.astype(float) != f2b.astype(float)).mean())},
               "auc": {}, "delta_vs_x2": {}}
        for a, s in scores.items():
            rec["auc"][a] = {sc: float(indomain.per_jd_auc(yy.loc[k], s.loc[k], jd.loc[k]).mean()) for sc, k in scopes.items()}
            if a != "X2":
                rec["delta_vs_x2"][a] = {sc: paired(s.loc[k], scores["X2"].loc[k], yy.loc[k], jd.loc[k])
                                         for sc, k in scopes.items()}
        out["sets"][name] = rec
    out["f3_role_misses"] = f3_from_review()
    return out


def f3_from_review() -> dict:
    """Reviewed cases tagged A (role mismatch, transferable) or B: the role model's values there."""
    from candidate_screener.evaluation import error_review as er
    if not (er.REASONS.exists() and er.CASES.exists()):
        return {}
    r = er.latest().reset_index()
    c = pd.read_csv(er.CASES).set_index("case_id")
    r = r[r.case_id.isin(c.index)]
    det = r.case_id.map(lambda k: json.loads(c.details[k]))
    fit_role = det.map(lambda d: d["features_fit"]["role_match"])
    nofit_role = det.map(lambda d: d["features_nofit"]["role_match"])
    a = r.reason == "A"
    return {"n_reviewed": int(len(r)), "n_tagged_A": int(a.sum()),
            "A: Fit CV out of role (role < 0.5)": int((a & (fit_role < 0.5)).sum()),
            "A: median role_match, Fit side": float(fit_role[a].median()) if a.any() else None,
            "all reviewed: Fit side out of role": int((fit_role < 0.5).sum()),
            "all reviewed: No Fit side out of role": int((nofit_role < 0.5).sum())}


def main(argv=None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = run()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    print(RECORD)


if __name__ == "__main__":
    main()
