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


def run_arm(arm: str, reps: int = 20) -> dict:
    sources = {n: cs.CachedEntities(n, None) for n in SOURCE_OF[arm]}
    extractor = cs.make_extractors(sources)[arm]
    model = aw.build_scorer(SEED, extractor=extractor)

    frame = final_set.load(); lab = indomain.labels(frame)
    a2 = corpus.a2_pairs(text_variant="v2").set_index("pair_id").loc[lab.index].reset_index()
    llm = aw.llm_labels()
    a1_train = corpus.a1_pairs("train")
    a1_all = pd.concat([a1_train, corpus.a1_pairs("test")]).drop_duplicates("pair_id").set_index("pair_id", drop=False)
    matched, orig = a1_all.loc[llm.index], a1_all.fit.astype(int)

    def feats(p):
        return aw.features(model, p).join(aw.extra_features(model, p))
    X_a2, X_train, X_match = feats(a2), feats(a1_train), feats(matched)
    hand = aw.linear_score(X_a2, aw.HAND_WEIGHTS).set_axis(a2.pair_id)
    out = {"arm": arm, "reps": reps, "fits": [], "curves": {}, "final": []}

    # notebook 18 Part 2: label source x feature set
    srcs = {"A original all": (X_train, orig), "B original matched": (X_match, orig),
            "C LLM matched": (X_match, llm)}
    for fs, cols in aw.FEATURE_SETS.items():
        for src, (X, y) in srcs.items():
            w = aw.fit_weights(X, y.loc[X.index], cols)
            s = aw.linear_score(X_a2, w).set_axis(a2.pair_id)
            out["fits"].append({"features": fs, "labels": src, **row(f"{fs} | {src}", deltas(s, hand, lab, a2))})

    # notebook 19 Parts 3-4: learning curves and final fits
    d50, y50 = aw.d50_pairs()
    X50 = feats(d50)
    jd_m, jd50 = matched.jd_id, d50.jd_id
    curves = {"A1-LLM": aw.learning_curve(X_match, llm, jd_m, X_a2, lab.y, lab.jd_id, reps=reps, seed=SEED),
              "D50": aw.learning_curve(X50, y50, jd50, X_a2, lab.y, lab.jd_id, reps=reps, seed=SEED)}
    train_sets = {"A1-LLM": (X_match, llm), "D50": (X50, y50)}
    for name, c in curves.items():
        g = c.groupby(["learner", "train JDs"])["A2 AUC"].mean()
        out["curves"][name] = {f"{l}|{n}": float(v) for (l, n), v in g.items()}
        for learner, (kind, cols) in aw.LEARNERS.items():
            X, y = train_sets[name]
            s = aw.fit_learner(kind, cols, X, y)(X_a2).set_axis(a2.pair_id)
            rise = float(g[(learner, 70)] - g[(learner, 25)])
            out["final"].append({"labels": name, "learner": learner, "rise_25_70": rise,
                                 "still_rising": rise > RISE, **row(learner, deltas(s, hand, lab, a2))})
    out["verdict"] = verdict(out["final"])
    return out


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
    p.add_argument("--arm", required=True, choices=sorted(SOURCE_OF))
    p.add_argument("--reps", type=int, default=20)
    a = p.parse_args(argv)
    result = run_arm(a.arm, a.reps)
    path = record_path(a.arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps(result["verdict"], indent=1), path)


if __name__ == "__main__":
    main()
