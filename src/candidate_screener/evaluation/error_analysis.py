"""Steps 1-3 of plan 2026-10-13-error-analysis-2: who causes each mistake, headroom, slices.

A *mistake* is a (Fit, No Fit) pair inside one JD that the score orders the wrong way
(`a1_weights.pair_inversions`; ties count half). Step 1 names the feature term that pushes the
wrong way hardest. Step 2 sizes headroom with oracle ablations (diagnostic only, never a system).
Step 3 slices by data-quality attributes of the Fit-side row. Aggregates only: no document text.

    uv run python -m candidate_screener.evaluation.error_analysis
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import a1_weights as aw

RECORD = OUTPUT / "extraction" / "error-analysis-2.json"
HAND = pd.Series({"role_match": 2.0, "experience_compatible": 1.0, "tfidf_scaled": 0.999})
TERMS = ("role_match", "experience_compatible", "tfidf_scaled", "coverage")
KINDS = ("both out of role", "both in role", "Fit out of role, No Fit in", "Fit in role, No Fit out")


def kind_of(fit_role, nofit_role) -> np.ndarray:
    """Where the mistake sits relative to the JD's role family (role_match >= 0.5 is in role)."""
    fi, ni = np.asarray(fit_role) >= 0.5, np.asarray(nofit_role) >= 0.5
    return np.select([fi & ni, ~fi & ni, fi & ~ni], [KINDS[1], KINDS[2], KINDS[3]], KINDS[0])


def culprit_of(weights: pd.Series, X: pd.DataFrame, fit_id, nofit_id) -> np.ndarray:
    """The term with the largest `weight * (x_NoFit - x_Fit)`: the one pushing hardest the wrong way.

    A pair whose terms all push no further than 0 (an exact tie) is labelled `tie`."""
    cols = [c for c in weights.index if c in X]
    gap = (X.loc[nofit_id, cols].to_numpy() - X.loc[fit_id, cols].to_numpy()) * weights[cols].to_numpy()
    best = gap.argmax(axis=1)
    return np.where(gap.max(axis=1) > 0, np.array(cols)[best], "tie")


def mistakes(scores: pd.Series, y: pd.Series, jd: pd.Series, X: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Every wrongly ordered pair with its kind and culprit term."""
    inv = aw.pair_inversions(scores, y, jd)
    inv = inv[inv.inv > 0].reset_index(drop=True)
    inv["kind"] = kind_of(X.role_match.loc[inv.fit_id], X.role_match.loc[inv.nofit_id])
    inv["culprit"] = culprit_of(weights, X, inv.fit_id.to_numpy(), inv.nofit_id.to_numpy())
    return inv


def shares(inv: pd.DataFrame, col: str, order=None) -> dict:
    s = inv.groupby(col).inv.sum() / inv.inv.sum()
    s = s.reindex(order, fill_value=0.0) if order is not None else s
    return {k: float(v) for k, v in s.items()}


def sample_mistakes(inv: pd.DataFrame, per_bucket: int = 20, seed: int = 0) -> pd.DataFrame:
    """Up to `per_bucket` mistakes per culprit bucket (inv == 1 only), reproducible by seed."""
    hard = inv[inv.inv == 1.0]
    parts = [g.sample(min(per_bucket, len(g)), random_state=seed) for _, g in hard.groupby("culprit")]
    return pd.concat(parts, ignore_index=True) if parts else hard.iloc[0:0]


def inversion_rate(scores: pd.Series, y: pd.Series, jd: pd.Series) -> float:
    d = aw.pair_inversions(scores, y, jd)
    return float(d.inv.mean()) if len(d) else float("nan")


def slice_table(inv_all: pd.DataFrame, X: pd.DataFrame, extra: dict) -> dict:
    """Inversion rate over all (Fit, No Fit) pairs, sliced by an attribute of the Fit-side row.

    `inv_all` is the full pair table (`pair_inversions`), `extra` maps a slice name to a Series
    over pair ids (e.g. CV length tercile)."""
    out = {}
    fit = inv_all.fit_id.to_numpy()
    attrs = {"experience years known": X.exp_known.loc[fit].map({1.0: "known", 0.0: "unknown"}),
             "any concept overlap": (X.coverage.loc[fit] > 0).map({True: "overlap", False: "none"}),
             "JD concepts <= 3": (np.expm1(X.log_jd_concepts.loc[fit]).round() <= 3).map({True: "<= 3", False: "> 3"})}
    for name, s in extra.items():
        attrs[name] = s.loc[fit]
    for name, s in attrs.items():
        g = inv_all.assign(_a=np.asarray(s)).groupby("_a").inv
        out[name] = {str(k): {"n_pairs": int(c), "inversion_rate": float(m)} for k, (c, m) in
                     zip(g.mean().index, zip(g.size(), g.mean()))}
    return out


def tercile(lengths: pd.Series) -> pd.Series:
    return pd.qcut(lengths, 3, labels=["short", "mid", "long"], duplicates="drop").astype(str)


def run() -> dict:
    from candidate_screener.evaluation import coverage_variants as cv, final_set, indomain
    from candidate_screener.matching import corpus
    frame = final_set.load().set_index("pair_id"); lab = indomain.labels(frame.reset_index())
    a2 = (corpus.a2_pairs(text_variant="v2").assign(keyword_match=lambda d: corpus.keyword_match(d))
          .set_index("pair_id").loc[lab.index].reset_index())
    d50, y50 = aw.d50_pairs()
    model = aw.build_scorer()
    X_a2 = aw.features(model, a2).join(aw.extra_features(model, a2))
    X50 = aw.features(model, d50).join(aw.extra_features(model, d50))
    sel, held = cv.split_jds(d50.jd_id)
    jd50 = d50.set_index("pair_id").jd_id
    ids_sel = d50.pair_id[d50.jd_id.isin(sel)].to_numpy(); ids_held = d50.pair_id[d50.jd_id.isin(held)].to_numpy()
    w_log = aw.fit_weights(X50.loc[ids_sel], y50.loc[ids_sel])                  # base-4 logistic, D50 selection only
    scorers = {"hand-set X2": HAND, "logistic base (4)": w_log}
    sets = {"D50 selection (70 JDs)": (X50.loc[ids_sel], y50, jd50, d50.set_index("pair_id").cv_text.loc[ids_sel].str.len()),
            "D50 held-out (30 JDs)": (X50.loc[ids_held], y50, jd50, d50.set_index("pair_id").cv_text.loc[ids_held].str.len()),
            "A2 final (59 JDs)": (X_a2.set_axis(a2.pair_id), lab.y, lab.jd_id, a2.set_index("pair_id").cv_text.str.len())}
    out = {"logistic_weights": {k: float(v) for k, v in w_log.items()}, "sets": {}}
    for sname, (X, y, jd, length) in sets.items():
        rec = {}
        for wname, w in scorers.items():
            s = aw.linear_score(X, w)
            inv = mistakes(s, y, jd, X, w)
            allp = aw.pair_inversions(s, y, jd)
            n_pairs = len(allp)
            rec[wname] = {"auc": float(aw.mean_jd_auc(s, y, jd)[0]), "pairs": n_pairs,
                          "inversion_rate": float(allp.inv.mean()),
                          "share_by_kind": shares(inv, "kind", KINDS),
                          "rate_by_kind": {k: float(allp.assign(kind=kind_of(X.role_match.loc[allp.fit_id], X.role_match.loc[allp.nofit_id]))
                                                    .query("kind == @k").inv.mean()) for k in KINDS},
                          "share_by_culprit": shares(inv, "culprit"),
                          "slices": slice_table(allp, X, {"CV length": tercile(length)})}
        out["sets"][sname] = rec
    # step 2: oracle ablations and label ceilings, A2 final only (needs keyword metadata and tiers)
    X, y, jd, _ = sets["A2 final (59 JDs)"]
    kw = a2.set_index("pair_id").keyword_match.astype(float)
    ab = {}
    for wname, w in scorers.items():
        base = aw.linear_score(X, w).set_axis(a2.pair_id)
        Xo = X.copy(); Xo["role_match"] = kw.reindex(Xo.index).to_numpy()
        oracle = aw.linear_score(Xo, w).set_axis(a2.pair_id)
        d = {sc: indomain.paired_delta(oracle, base, lab, a2, "auc", sc, "all", 0) for sc in final_set.SCOPES}
        tiers = {}
        for t in ("humans_agree", "single_human", "llm_breaks_tie"):
            keep = lab.index[lab.tier == t]
            v = indomain.per_jd_auc(y.loc[keep], base.loc[keep], jd.loc[keep])
            tiers[t] = {"auc": float(v.mean()), "n_pairs": int(len(keep)), "n_jds": int(len(v))}
        ab[wname] = {"role_oracle_total": d["total"], "role_oracle_generic": d["generic"], "role_oracle_targeted": d["targeted"],
                     "by_tier": tiers}
    out["ablations_a2"] = ab
    return out


def main(argv=None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = run()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    print(RECORD)


if __name__ == "__main__":
    main()
