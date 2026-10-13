"""Stage 2 of plan 2026-10-13-gliner-concepts-in-learned-weights (exploratory, D56).

Repeats the notebook 18 Part 2 fits and the notebook 19 Parts 3-4 learning curves with one
concept source, scored on the A2 final set (scored on, never fitted on, D46). Registered bar,
unchanged from notebook 18: a learner beats hand-set X2 only if its paired total dAUC has a
JD-bootstrap CI above 0 and `generic` and `targeted` both have a positive delta. A feature arm
counts as a pass for a learner only if the bar is met by that learner's final fit on the
A1-LLM labels **and** on the D50 labels. The label verdict also needs the A1-LLM curve to rise
by more than 0.01 from 25 to 70 JDs.

    uv run python -m candidate_screener.evaluation.concept_stage2 --arm R+N
    uv run python -m candidate_screener.evaluation.concept_stage2 --arm R     # reproduction
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import a1_weights as aw, final_set, indomain
from candidate_screener.extraction import concept_sources as cs
from candidate_screener.matching import corpus

SEED = 0
RISE = 0.01
SOURCE_OF = {"R+G": ("gliner",), "R+N": ("minilm",), "G": ("gliner",), "N": ("minilm",), "R": ()}


def record_path(arm: str):
    return OUTPUT / "extraction" / f"concept-stage2-{arm.replace('+', '_')}.json"


def bar(d: dict) -> bool:
    return bool(d["total"]["ci"][0] > 0 and d["generic"]["delta"] > 0 and d["targeted"]["delta"] > 0)


def deltas(s, ref, lab, a2) -> dict:
    return {sc: indomain.paired_delta(s, ref, lab, a2, "auc", sc, "all", SEED) for sc in final_set.SCOPES}


def row(name: str, d: dict) -> dict:
    return {"name": name, "delta_generic": d["generic"]["delta"], "delta_targeted": d["targeted"]["delta"],
            "delta_total": d["total"]["delta"], "ci_total": d["total"]["ci"], "bar_met": bar(d)}


def evaluate(Xs: dict, ctx: dict, reps: int) -> tuple[dict, dict]:
    """Notebook 18 Part 2 grid and notebook 19 Part 3-4 curves for one set of feature frames.

    `Xs` has keys a2, train, match, d50; `ctx` has the labels and pair frames. Returns the record
    and the A2 score series of every fit (for paired comparisons between feature versions)."""
    lab, a2, orig, llm, y50 = ctx["lab"], ctx["a2"], ctx["orig"], ctx["llm"], ctx["y50"]
    X_a2, X_train, X_match, X50 = Xs["a2"], Xs["train"], Xs["match"], Xs["d50"]
    hand = aw.linear_score(X_a2, aw.HAND_WEIGHTS).set_axis(a2.pair_id)
    out, scores = {"reps": reps, "fits": [], "curves": {}, "final": []}, {}
    srcs = {"A original all": (X_train, orig), "B original matched": (X_match, orig),
            "C LLM matched": (X_match, llm)}
    for fs, cols in aw.FEATURE_SETS.items():
        for src, (X, y) in srcs.items():
            w = aw.fit_weights(X, y.loc[X.index], cols)
            s = aw.linear_score(X_a2, w).set_axis(a2.pair_id)
            scores[("fit", fs, src)] = s
            out["fits"].append({"features": fs, "labels": src, **row(f"{fs} | {src}", deltas(s, hand, lab, a2))})
    jd_m, jd50 = ctx["matched"].jd_id, ctx["d50"].jd_id
    curves = {"A1-LLM": aw.learning_curve(X_match, llm, jd_m, X_a2, lab.y, lab.jd_id, reps=reps, seed=SEED),
              "D50": aw.learning_curve(X50, y50, jd50, X_a2, lab.y, lab.jd_id, reps=reps, seed=SEED)}
    train_sets = {"A1-LLM": (X_match, llm), "D50": (X50, y50)}
    for name, c in curves.items():
        g = c.groupby(["learner", "train JDs"])["A2 AUC"].mean()
        out["curves"][name] = {f"{l}|{n}": float(v) for (l, n), v in g.items()}
        for learner, (kind, cols) in aw.LEARNERS.items():
            X, y = train_sets[name]
            s = aw.fit_learner(kind, cols, X, y)(X_a2).set_axis(a2.pair_id)
            scores[("final", name, learner)] = s
            rise = float(g[(learner, 70)] - g[(learner, 25)])
            out["final"].append({"labels": name, "learner": learner, "rise_25_70": rise,
                                 "still_rising": rise > RISE, **row(learner, deltas(s, hand, lab, a2))})
    out["verdict"] = verdict(out["final"])
    return out, scores


def context(model_unused=None) -> dict:
    frame = final_set.load(); lab = indomain.labels(frame)
    a2 = corpus.a2_pairs(text_variant="v2").set_index("pair_id").loc[lab.index].reset_index()
    llm = aw.llm_labels()
    a1_train = corpus.a1_pairs("train")
    a1_all = pd.concat([a1_train, corpus.a1_pairs("test")]).drop_duplicates("pair_id").set_index("pair_id", drop=False)
    d50, y50 = aw.d50_pairs()
    return {"lab": lab, "a2": a2, "llm": llm, "a1_train": a1_train, "matched": a1_all.loc[llm.index],
            "orig": a1_all.fit.astype(int), "d50": d50, "y50": y50}


def exact_features(model, ctx) -> dict:
    def feats(p):
        return aw.features(model, p).join(aw.extra_features(model, p))
    return {"a2": feats(ctx["a2"]), "train": feats(ctx["a1_train"]),
            "match": feats(ctx["matched"]), "d50": feats(ctx["d50"])}


def run_arm(arm: str, reps: int = 20) -> dict:
    sources = {n: cs.CachedEntities(n, None) for n in SOURCE_OF[arm]}
    extractor = cs.make_extractors(sources)[arm]
    model = aw.build_scorer(SEED, extractor=extractor)
    ctx = context()
    out, _ = evaluate(exact_features(model, ctx), ctx, reps)
    return {"arm": arm, **out}


def soft_sensitivity(Xs_by_t: dict, exact: dict, ctx: dict) -> dict:
    """Base-4 and all-11 logistic fitted on the 70 D50 selection JDs; per-JD AUC on the 30 held-out
    D50 JDs and on A1-LLM, for exact coverage and each soft threshold, with the paired delta of
    each threshold against exact. A2 is not used."""
    from candidate_screener.evaluation import coverage_variants as cv
    d50, y50, llm, matched = ctx["d50"], ctx["y50"], ctx["llm"], ctx["matched"]
    sel, held = cv.split_jds(d50.jd_id)
    ids_sel = d50.pair_id[d50.jd_id.isin(sel)].to_numpy(); ids_held = d50.pair_id[d50.jd_id.isin(held)].to_numpy()
    jd50 = d50.set_index("pair_id").jd_id

    def aucs(Xs, cols):
        w = aw.fit_weights(Xs["d50"].loc[ids_sel], y50.loc[ids_sel], cols)
        return {"D50 held-out": cv.per_jd_auc(aw.linear_score(Xs["d50"].loc[ids_held], w), y50, jd50),
                "A1-LLM": cv.per_jd_auc(aw.linear_score(Xs["match"], w), llm, matched.jd_id)}
    out = {}
    for name, cols in (("logistic base (4)", aw.FEATURES), ("logistic all (11)", aw.ALL_FEATURES)):
        base = aucs(exact, cols)
        out[name] = {"exact": {k: {"auc": float(v.mean())} for k, v in base.items()}}
        for t, Xs in Xs_by_t.items():
            cur = aucs(Xs, cols)
            out[name][str(t)] = {k: {"auc": float(cur[k].mean()), **cv.paired(cur[k], base[k])} for k in cur}
    return out


def run_soft(thresholds=(0.55, 0.60, 0.65), main: float = 0.60, reps: int = 20) -> dict:
    """Plan 2026-10-13-soft-matching-in-the-score: R+N exact versus soft coverage at `main`."""
    from candidate_screener.evaluation import match_keys as mk, soft_coverage as sc
    extractor = cs.make_extractors({"minilm": cs.CachedEntities("minilm", None)})["R+N"]
    model = aw.build_scorer(SEED, extractor=extractor)
    ctx = context()
    exact = exact_features(model, ctx)
    frames = {"a2": ctx["a2"], "train": ctx["a1_train"], "match": ctx["matched"], "d50": ctx["d50"]}
    phrases = set()
    for p in frames.values():
        for t in p.jd_text.unique(): phrases |= set(extractor.concepts(t, "jd"))
        for t in p.cv_text.unique(): phrases |= set(extractor.concepts(t, "cv"))
    vec = mk.embed(phrases)
    sims = {k: sc.SoftSims(p, extractor, model.frequencies, vec) for k, p in frames.items()}
    soft = {t: {k: sc.replace_columns(exact[k], sims[k].columns(t)) for k in frames} for t in thresholds}
    out_exact, s_exact = evaluate(exact, ctx, reps)
    out_soft, s_soft = evaluate(soft[main], ctx, reps)
    lab, a2 = ctx["lab"], ctx["a2"]
    paired_vs_exact = []
    for key, s in s_soft.items():
        d = deltas(s, s_exact[key], lab, a2)
        paired_vs_exact.append({"fit": list(key), **row("soft - exact", d)})
    return {"arm": "R+N", "main_threshold": main, "distinct_phrases": len(phrases),
            "exact": out_exact, "soft": out_soft, "soft_minus_exact": paired_vs_exact,
            "sensitivity": soft_sensitivity(soft, exact, ctx)}


def verdict(final: list[dict]) -> dict:
    """Per learner: bar met on the A1-LLM final fit AND the D50 final fit; label verdict adds the rise."""
    by = {(r["labels"], r["learner"]): r for r in final}
    res = {}
    for learner in aw.LEARNERS:
        a1, d50 = by[("A1-LLM", learner)], by[("D50", learner)]
        passed = a1["bar_met"] and d50["bar_met"]
        res[learner] = {"bar_a1": a1["bar_met"], "bar_d50": d50["bar_met"], "pass": passed,
                        "still_rising_a1": a1["still_rising"],
                        "more_labels_worth_scoping": bool(passed and a1["still_rising"])}
    return res


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=sorted(SOURCE_OF), default="R+N")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--soft", action="store_true", help="R+N exact vs soft coverage at 0.60 (plan soft-matching)")
    a = p.parse_args(argv)
    if a.soft:
        result = run_soft(reps=a.reps)
        path = OUTPUT / "extraction" / "concept-stage2-R_N-soft0.60.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
        print(json.dumps({"exact": result["exact"]["verdict"], "soft": result["soft"]["verdict"]}, indent=1), path)
        return
    result = run_arm(a.arm, a.reps)
    path = record_path(a.arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps(result["verdict"], indent=1), path)


if __name__ == "__main__":
    main()
