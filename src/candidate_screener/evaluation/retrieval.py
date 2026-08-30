"""The Stage 1 baseline scored as a **ranker** over the task 3.3 pools *(Q24, D27)*.

`baselines.run` measures the baseline as a *classifier*: given one (resume, JD) pair,
Good / Potential / No Fit. That is not the question the pools were built to ask. This
module asks the retrieval question — given one JD and a pool of candidates, how good is
the ordering — and produces the three adopted figures (**Recall@10, Precision@5,
nDCG@10**, deviation W8) with their *n* and bootstrap CI.

It also answers **Q18b**, which had been deferred twice: of the five slots a system puts
at the top, how many are filled by *unjudged distractors*? That number is what decides
whether the downward precision bias `pools.ASSUMPTION` describes is large or negligible,
and it needs no annotator — only this scoring pass.

Nothing here re-fits anything. `baselines.run.build` is called for the vectoriser and the
BM25 statistics, so the ranker scored here is byte-for-byte the model
`baselines.run --check` guards. Fitting a second vectoriser "just for retrieval" is
exactly the bug Phase 4 found and removed *(W-series)*; there is no second fit path.

Both scorers are safe on pool documents even though the pools are test-side and the fit
corpus is train-only: `tfidf` transforms, and `bm25.score_pair` reads only `idf` and
`avgdl` from the fitted model *(D24)*. Under the leak-free split there is zero resume
overlap by construction, so a corpus-lookup scorer would return 0.0 for every pool
document and look plausible doing it.

Usage:
    uv run python -m candidate_screener.evaluation.retrieval --seed 0
    uv run python -m candidate_screener.evaluation.retrieval --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.baselines import bm25, run, skills, tfidf
from candidate_screener.config import RETRIEVAL_METRICS
from candidate_screener.data.pools import ASSUMPTION, POOLS_OUT, VARIANTS
from candidate_screener.evaluation import metrics

#: The adopted figures *(deviation W8)*. Recall@10 leads: it is the metric the project's
#: success measures actually name, and — unlike Precision@5 — it carries no judged-supply
#: ceiling at a median of 6 relevant resumes per query *(D26)*.
FIGURES = (("Recall", 10), ("Precision", 5), ("nDCG", 10))

#: Deepest variant is scored once; N20 and N100 are nested inside it, so one pass over
#: the unique pairs serves all three and no variant can drift from another.
DEEPEST = "Nfull"


def load_pools() -> pd.DataFrame:
    path = POOLS_OUT / "pools.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run python -m candidate_screener.data.pools "
            "--build --seed 0` first")
    return pd.read_parquet(path)


def score_pools(pools: pd.DataFrame, built: dict) -> dict[str, pd.DataFrame]:
    """Long-form `query_jd_id, candidate_resume_id, score` per model.

    Scored over the **unique** pairs of the deepest variant. The nested-variant
    guarantee (`N20 ⊂ N100 ⊂ Nfull`) is what makes that sufficient; scoring each
    variant separately would let a sensitivity run differ from the primary run by
    something other than depth, which is the one thing a sensitivity run must not do.
    """
    pairs = pools[pools.pool_variant == DEEPEST][
        ["query_jd_id", "candidate_resume_id", "job_description_text", "resume_text"]
    ].drop_duplicates(["query_jd_id", "candidate_resume_id"]).reset_index(drop=True)

    jd_repr = skills.augment(pairs.job_description_text)
    resume_repr = skills.augment(pairs.resume_text)

    return {
        "tfidf": pairs[["query_jd_id", "candidate_resume_id"]].assign(
            score=tfidf.pair_cosine_scores(built["vectorizer"], resume_repr, jd_repr)),
        "bm25": pairs[["query_jd_id", "candidate_resume_id"]].assign(
            score=bm25.pair_bm25_scores(built["bm25_model"], resume_repr, jd_repr)),
    }


def top_k_exposure(pools: pd.DataFrame, scores: pd.DataFrame, variant: str = "N100",
                   k: int = 5) -> dict:
    """**Q18b** — what a system actually surfaces, by judged status.

    For each query, rank the pool and look at the top *k*. Every slot is one of:

    - `judged_relevant_strict` / `judged_relevant_graded` — a document A1 marked Good,
      and Good-or-Potential respectively. Both are reported because a slot that is a
      miss under `strict` and a hit under `graded` is neither an error nor a rounding
      difference — it is the two definitions answering their two different questions;
    - `judged_none` — a document A1 explicitly marked No Fit. Precision counts this as
      a miss and is *right* to;
    - `unjudged_distractor` — nobody ever looked. Precision counts it as a miss and
      **may be wrong**. This is the entire content of Q18.

    The distractor share is the upper bound on the bias: if every surfaced distractor
    were secretly relevant, precision would rise by exactly this fraction. It is an
    upper bound and not an estimate — deciding what share is genuinely relevant needs a
    human, which is what the deferred judging wave would have measured.
    """
    group = pools[pools.pool_variant == variant]
    counts = {"judged_relevant_strict": 0, "judged_relevant_graded": 0,
              "judged_none": 0, "unjudged_distractor": 0}
    per_query = []
    for query, pool in group.groupby("query_jd_id"):
        ordered = pool.assign(
            score=pool.candidate_resume_id.map(
                scores[scores.query_jd_id == query]
                .set_index("candidate_resume_id").score)
        ).sort_values(["score", "candidate_resume_id"], ascending=[False, True],
                      kind="stable").head(k)
        distractors = int((ordered.source == "distractor").sum())
        counts["unjudged_distractor"] += distractors
        counts["judged_relevant_strict"] += int((ordered.relevance == "good").sum())
        counts["judged_relevant_graded"] += int((ordered.relevance != "none").sum())
        counts["judged_none"] += int(
            ((ordered.source == "labelled") & (ordered.relevance == "none")).sum())
        per_query.append(distractors / k)

    slots = k * group.query_jd_id.nunique()
    share = np.array(per_query, dtype=float)
    low, high = metrics.bootstrap_ci(share)
    return {"variant": variant, "k": k, "queries": int(group.query_jd_id.nunique()),
            "total_slots": slots, "slots": counts,
            "distractor_share_mean": round(float(share.mean()), 4),
            "distractor_share_ci": [round(low, 4), round(high, 4)],
            "note": "Upper bound on the precision bias, not an estimate — see D26."}


def figure_row(figure: metrics.Figure, ceiling: float | None = None) -> dict:
    row = {"metric": figure.metric, "definition": figure.definition,
           "variant": figure.variant, "value": round(figure.value, 6),
           "n_queries": figure.n_queries,
           "ci": [round(figure.ci_low, 6), round(figure.ci_high, 6)]}
    if ceiling is not None:
        row["ceiling"] = round(ceiling, 6)
        row["pct_of_attainable"] = round(100 * figure.value / ceiling, 1) if ceiling else None
    return row


def evaluate(seed: int = 0) -> dict:
    pools = load_pools()
    built = run.build(seed)
    scored = score_pools(pools, built)
    scored["random"] = metrics.random_scores(pools, seed)

    models: dict = {}
    for model, frame in scored.items():
        rows = []
        for metric, k in FIGURES:
            for definition in metrics.DEFINITIONS:
                for variant in VARIANTS:
                    figure = metrics.score(pools, frame, variant, metric, k, definition, seed)
                    ceiling = (metrics.attainable_ceiling(
                        pools, metric, k, definition, variant, seed).value
                        if metric != "nDCG" else None)
                    rows.append(figure_row(figure, ceiling))
        models[model] = {"figures": rows,
                         "top5_exposure": top_k_exposure(pools, frame, "N100", 5)}

    return {"seed": seed, "models": models,
            "precision_ceiling": metrics.ceilings(pools, "N100", seed),
            "note_not_comparable": run.NOT_COMPARABLE,
            "note_distractors": ASSUMPTION,
            "note_ceiling": metrics.CEILING_NOTE}


def print_report(doc: dict) -> None:
    for model, block in doc["models"].items():
        label = "random floor" if model == "random" else model
        print(f"\n=== {label} on the task 3.3 pools")
        print(f"  {'metric':<12} {'defn':<7} {'pool':<6} {'value':<7} {'95% CI':<18} "
              f"{'n':<4} {'ceiling':<8} % attainable")
        for row in block["figures"]:
            if row["variant"] != "N100":
                continue
            ceiling = f"{row['ceiling']:.3f}" if row.get("ceiling") else "—"
            pct = f"{row['pct_of_attainable']:.0f}%" if row.get("pct_of_attainable") else "—"
            print(f"  {row['metric']:<12} {row['definition']:<7} {row['variant']:<6} "
                  f"{row['value']:.3f}  [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}]  "
                  f"{row['n_queries']:<4} {ceiling:<8} {pct}")
        e = block["top5_exposure"]
        print(f"  --- Q18b: of {e['total_slots']} top-5 slots over {e['queries']} queries — "
              f"judged relevant {e['slots']['judged_relevant_graded']} graded "
              f"({e['slots']['judged_relevant_strict']} strict), "
              f"judged No Fit {e['slots']['judged_none']}, "
              f"unjudged {e['slots']['unjudged_distractor']} "
              f"({100 * e['distractor_share_mean']:.1f}% "
              f"[{100 * e['distractor_share_ci'][0]:.1f}, "
              f"{100 * e['distractor_share_ci'][1]:.1f}])")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RETRIEVAL_METRICS)
    ap.add_argument("--check", action="store_true",
                    help="re-evaluate and diff against the committed record")
    args = ap.parse_args(argv)

    document = evaluate(args.seed)
    if args.check:
        if not args.out.exists():
            print(f"{args.out} missing — nothing to check against", file=sys.stderr)
            return 1
        committed = json.loads(args.out.read_text(encoding="utf-8"))
        differences = run._differences(committed, document)
        for line in differences[:20]:
            print(f"  {line}")
        print(f"\n{'FAIL' if differences else 'OK'} — "
              f"{len(differences)} difference(s) against {args.out}")
        return 1 if differences else 0

    print_report(document)
    run.write_metrics(document, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
