"""Pilot-scale retrieval (plan 02 §5.2): does cheap retrieval keep the fits in the shortlist?

For each labelled A2 JD, a 500-candidate pool: its 10 labelled CVs plus 490 CVs drawn at
random (seeded per JD) from `indomain-holdout.csv`, the eval-region CVs reserved around
the 600 pairs. Relevant = the JD's candidates the final set calls `Fit`. The 490
distractors are unjudged and assumed not to be fits, the same exposure Q18 recorded for
the A1 pools, so recall here is an upper bound on what a fully judged pool would show.

Reported per retriever: Recall@10, @50 and @100 over JDs with at least one fit, with
JD-resampled CIs, for generic, targeted and total; plus retrieve-then-rank (4c): the
retriever's top 50 re-ordered by a cross-encoder, scored on Recall@10.

Usage:
    uv run python -m candidate_screener.evaluation.retrieval_pilot --seed 0 [--check]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from candidate_screener.config import DOCS_DATA, OUTPUT, PROCESSED
from candidate_screener.evaluation import final_set, resample

POOL_SIZE = 500
KS = (10, 50, 100)
SHORTLIST = 50
HOLDOUT = DOCS_DATA / "manifests" / "indomain-holdout.csv"
RECORD = OUTPUT / "matching" / "retrieval-pilot.json"
COSTS = OUTPUT / "matching" / "retrieval-pilot-costs.json"
CACHE = PROCESSED / "matching"


def build_pools(seed: int = 0, text_variant: str = "v1") -> pd.DataFrame:
    """Pair frame for every (JD, candidate) in every pool, with text and metadata.

    `text_variant` (D53) changes the CV text only: the pools hold the same candidates
    under every variant (the draw never reads text), and the `v1` cache name is unchanged."""
    from candidate_screener.matching import corpus
    path = CACHE / f"pilot-pools{corpus.variant_tag(text_variant)}-{POOL_SIZE}-{seed}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from candidate_screener.annotation.sample import load_english
    labelled = corpus.a2_pairs()
    jd, cv = load_english()
    if text_variant != "v1":
        cv["cv_text"] = corpus.build_cv_text(cv, text_variant)
    jd, cv = jd.set_index("id"), cv.set_index("id")
    holdout = pd.read_csv(HOLDOUT)
    distractors = np.array(sorted(set(holdout[holdout.doc_type == "cv"].doc_id)
                                  - set(labelled.cv_id)
                                  & set(cv.index)))
    rows = []
    for k, (jd_id, group) in enumerate(labelled.groupby("jd_id", sort=True)):
        rng = np.random.default_rng([seed, k])
        pool = rng.choice(distractors, POOL_SIZE - len(group), replace=False)
        for cv_id in [*group.cv_id, *pool]:
            rows.append((jd_id, cv_id, group.stratum.iloc[0]))
    frame = pd.DataFrame(rows, columns=["jd_id", "cv_id", "stratum"])
    frame["pair_id"] = frame.jd_id + "__" + frame.cv_id
    frame = frame.assign(
        jd_text=frame.jd_id.map(jd.jd_text), cv_text=frame.cv_id.map(cv.cv_text),
        jd_keyword=frame.jd_id.map(jd["Primary Keyword"]),
        cv_keyword=frame.cv_id.map(cv["Primary Keyword"]),
        jd_exp_band=frame.jd_id.map(jd.exp_band), cv_exp_band=frame.cv_id.map(cv.exp_band))
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def relevant(frame: pd.DataFrame) -> pd.Series:
    """1 for a candidate the final set calls Fit; 0 for everything else (incl. distractors)."""
    fs = final_set.load().set_index("pair_id")
    fit = fs.final_label.reindex(frame.pair_id.to_numpy()).eq(final_set.FIT)
    return pd.Series(fit.to_numpy().astype(int), index=frame.pair_id.to_numpy())


def recall_at(frame: pd.DataFrame, scores: pd.Series, rel: pd.Series, k: int) -> pd.Series:
    """Per-JD recall@k over JDs with at least one relevant candidate."""
    df = pd.DataFrame({"jd": frame.jd_id.to_numpy(), "s": scores.to_numpy(),
                       "r": rel.to_numpy(), "pid": frame.pair_id.to_numpy()})
    out = {}
    for jd, g in df.groupby("jd", sort=True):
        n = g.r.sum()
        if n:
            top = g.sort_values(["s", "pid"], ascending=[False, True]).head(k)
            out[jd] = top.r.sum() / n
    return pd.Series(out, dtype=float)


def rerank_top(frame: pd.DataFrame, retriever: pd.Series, reranker, k: int = SHORTLIST
               ) -> pd.Series:
    """Retriever order, with its top-k per JD re-ordered by `reranker` (scored only there)."""
    df = frame.assign(s=retriever.to_numpy())
    df["rank"] = df.groupby("jd_id").s.rank(ascending=False, method="first")
    top = df[df["rank"] <= k]
    rerank = reranker.score(top)
    out = pd.Series(df.s.to_numpy() - 1e6, index=df.pair_id.to_numpy())   # below any shortlist
    order = pd.DataFrame({"pid": top.pair_id.to_numpy(), "jd": top.jd_id.to_numpy(),
                          "r": rerank.to_numpy()})
    order["new"] = order.groupby("jd").r.rank(ascending=True, method="first")
    out.loc[order.pid.to_numpy()] = order.new.to_numpy()
    return out


def figures(name: str, frame: pd.DataFrame, scores: pd.Series, rel: pd.Series, ks=KS,
            seed: int = 0) -> list[dict]:
    out = []
    for scope in final_set.SCOPES:
        sub = frame if scope == "total" else frame[frame.stratum == scope]
        idx = sub.pair_id.to_numpy()
        for k in ks:
            values = recall_at(sub, scores.loc[idx], rel.loc[idx], k)
            low, high = resample.mean_ci(values, seed=seed)
            out.append({"system": name, "metric": f"recall@{k}", "scope": scope,
                        "value": round(float(values.mean()), 6),
                        "ci_low": round(low, 6), "ci_high": round(high, 6),
                        "n_jd": int(len(values))})
    return out


def evaluate(retrievers: list, rerankers: list, seed: int = 0) -> tuple[dict, dict]:
    frame = build_pools(seed)
    rel = relevant(frame)
    record = {"seed": seed, "pool_size": POOL_SIZE, "shortlist": SHORTLIST,
              "reference": {"file": final_set.FINAL_CSV.name, "rule_version": final_set.RULE_VERSION},
              "note": ("Relevant = final-set Fit among the JD's 10 labelled candidates; the 490 "
                       "distractors are unjudged and assumed not fits (upper-bound recall)."),
              "retrievers": {}, "retrieve_then_rank": {}}
    costs, scores = {}, {}
    for factory in retrievers:
        system = factory()
        system.fit(None)
        s = system.score(frame)
        scores[system.name] = s
        record["retrievers"][system.name] = figures(system.name, frame, s, rel, seed=seed)
        card = system.card()
        costs[system.name] = {"ms_per_500_candidate_role": card["ms_per_500_candidate_role"]}
    for retriever_name, factory in rerankers:
        if retriever_name not in scores:
            continue
        reranker = factory()
        reranker.fit(None)
        name = f"{retriever_name} → {reranker.name} (top {SHORTLIST})"
        s = rerank_top(frame, scores[retriever_name], reranker)
        record["retrieve_then_rank"][name] = figures(name, frame, s, rel, ks=(10,), seed=seed)
        a = recall_at(frame, scores[retriever_name], rel, 10)
        b = recall_at(frame, s, rel, 10)
        diff = (b - a).dropna()
        low, high = resample.mean_ci(diff, seed=seed)
        record["retrieve_then_rank"][name].append(
            {"comparison": f"recall@10 delta vs {retriever_name} alone", "scope": "total",
             "delta": round(float(diff.mean()), 6), "ci": [round(low, 6), round(high, 6)],
             "n_jd": int(len(diff))})
    return record, costs


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    from candidate_screener.matching import ladder
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--warm", action="store_true",
                    help="with --check: replay cross-encoder scores from the persistent "
                         "store (D55 warm tier); without it the check re-infers")
    args = ap.parse_args(argv)
    if args.warm and not args.check:
        ap.error("--warm is only valid with --check")
    if args.warm:
        from candidate_screener.matching import cross_encoder
        cross_encoder.set_persistent(True)
    record, costs = evaluate(ladder.retrievers(), ladder.rerankers(), args.seed)
    if args.check:
        same = RECORD.exists() and RECORD.read_text(encoding="utf-8") == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {RECORD.name}")
        return 0 if same else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    COSTS.write_text(render(costs), encoding="utf-8")
    for name, figs in {**record["retrievers"], **record["retrieve_then_rank"]}.items():
        cells = [f"{f['metric']} {f['value']:.3f}" for f in figs
                 if f.get("scope") == "total" and "metric" in f]
        print(f"  {name:55s} {'  '.join(cells)}")
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
