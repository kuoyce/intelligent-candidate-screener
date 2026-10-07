"""Q8: the frozen post-hoc rungs scored once on the final set (plan 2026-10-07, 03 §2).

Z1 and Z1L (soft role term with a larger weight; Z1L takes the JD title's family on a
lookup hit) were chosen on the development set and frozen in 03 §2 before this module
first ran. X2, TF-IDF and the oracle 1a+2 are recomputed as references and must reproduce
P7's `stage1-metrics.json` scores. Status of Z1 / Z1L: **post-hoc** (designed after the
P7 S2 miss); reported beside the registered P7 result, never in its place.

Record: `output/matching/shortlist-recall.json`.

    uv run python -m candidate_screener.evaluation.shortlist_final
    uv run python -m candidate_screener.evaluation.shortlist_final --check
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import final_set, indomain, resample, retrieval_pilot, stage1
from candidate_screener.matching.selection import CELLS, verdict_of

RECORD = OUTPUT / "matching" / "shortlist-recall.json"
TEXT_VARIANT = "v2"
DECIMALS = 6
Z1, Z1L = "Z1", "Z1L"
#: 03 §2, frozen 8 Oct 2026 from the dev records.
FROZEN = {Z1: {"jd_mode": "text", "weight": 64}, Z1L: {"jd_mode": "lookup_or_text", "weight": 32}}
X2, TFIDF, ORACLE = stage1.X2, stage1.TFIDF, stage1.ORACLE_1A2
#: (id, B, A, decides).
COMPARISONS = (("M1", Z1L, X2, "reported"), ("M2", Z1, X2, "reported"),
               ("M3", Z1L, Z1, "selection"), ("M4", Z1L, TFIDF, "reported"),
               ("M5", Z1L, ORACLE, "nothing"))
S2_BAR = 0.80
S2_METRIC = "recall@50"
P7_REGISTERED = "P7 registered: S2 miss (X2 Recall@50 0.779 < 0.80; X4 0.761 < 0.90)"


def select(passing: list[str], m3_verdict_auc: str) -> str | None:
    """03 §2: Z1 (cheaper) among passing rungs unless Z1L is justified over it on M3."""
    if Z1 in passing and not (Z1L in passing and m3_verdict_auc == "justified"):
        return Z1
    return Z1L if Z1L in passing else None


def _sha(s: pd.Series) -> str:
    return hashlib.sha256(np.round(s.to_numpy(), 9).tobytes()).hexdigest()


def systems() -> dict[str, tuple[str, object]]:
    """name -> (status, fitted system)."""
    from candidate_screener.matching import extracted, lexical, role_filter as yf
    v = TEXT_VARIANT
    out = {TFIDF: ("exploratory", lexical.Tfidf(v)),
           X2: ("held-out (P7)", extracted.ExtractedPlus(lexical.Tfidf(v), "X2")),
           ORACLE: ("oracle", lexical.MetadataPlus(lexical.Tfidf(v), "1a+2"))}
    for name, cfg in FROZEN.items():
        rule = yf.SoftRoleWeight(cfg["jd_mode"], cfg["weight"])
        out[name] = ("post-hoc", yf.RoleFilterPlus(lexical.Tfidf(v), name, rule))
    for _, s in out.values():
        s.fit(None)
    return out


def llm_free_pool(pools: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """The pools without judged pairs whose final label needed the LLM (03 §2)."""
    tier = lab.tier.reindex(pools.pair_id.to_numpy())
    drop = tier.notna().to_numpy() & ~tier.isin(final_set.LLM_FREE_TIERS).to_numpy()
    return pools[~drop]


def recall_delta(b: pd.Series, a: pd.Series, pools: pd.DataFrame, rel: pd.Series,
                 scope: str, seed: int, k: int = 50) -> dict:
    sub = pools if scope == "total" else pools[pools.stratum == scope]
    idx = sub.pair_id.to_numpy()
    rb = retrieval_pilot.recall_at(sub, b.loc[idx], rel.loc[idx], k)
    ra = retrieval_pilot.recall_at(sub, a.loc[idx], rel.loc[idx], k)
    d = (rb - ra).dropna()
    lo, hi = resample.mean_ci(d, seed=seed)
    return {"metric": f"recall@{k}", "scope": scope, "delta": round(float(d.mean()), DECIMALS),
            "ci": [round(lo, DECIMALS), round(hi, DECIMALS)],
            "half_width": round((hi - lo) / 2, DECIMALS), "n_jd": int(len(d))}


def evaluate(seed: int = 0) -> dict:
    from candidate_screener.matching import corpus
    frame = final_set.load()
    rule_version = int(frame.rule_version.iloc[0])
    lab = indomain.labels(frame)
    pairs = corpus.a2_pairs(text_variant=TEXT_VARIANT).assign(
        keyword_match=lambda d: corpus.keyword_match(d))
    pairs = pairs.set_index("pair_id").loc[lab.index].reset_index()
    title_of = dict(zip(pairs.jd_id, pairs.jd_position))
    pairs = pairs.assign(jd_title=pairs.jd_position)
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    pools = pools.assign(jd_title=pools.jd_id.map(title_of))
    rel = retrieval_pilot.relevant(pools)
    free = llm_free_pool(pools, lab)
    p7 = json.loads(stage1.RECORD.read_text(encoding="utf-8"))["systems"]

    scores, pool_scores, out_systems, unscoreable = {}, {}, {}, []
    for name, (status, system) in systems().items():
        s, ps = system.score(pairs), system.score(pools)
        scores[name], pool_scores[name] = s, ps
        figs = [f.as_dict() for f in indomain.rank_figures(name, s, lab, pairs, rule_version,
                                                            seed, unscoreable)]
        indomain.assert_total_has_strata(figs)
        block = {"status": status, "scores_sha256": _sha(s), "pool_scores_sha256": _sha(ps),
                 "auc": figs,
                 "retrieval": retrieval_pilot.figures(name, pools, ps, rel, seed=seed),
                 "retrieval_llm_free": retrieval_pilot.figures(
                     name, free, ps.loc[free.pair_id.to_numpy()],
                     rel.loc[free.pair_id.to_numpy()], seed=seed)}
        if name in FROZEN:
            block["frozen"] = FROZEN[name]
        else:
            block["reproduces_p7"] = (p7[name]["scores_sha256"] == block["scores_sha256"]
                                      and p7[name]["pool_scores_sha256"]
                                      == block["pool_scores_sha256"])
        out_systems[name] = block

    comparisons = []
    for cid, b, a, decides in COMPARISONS:
        auc = {cell: indomain.paired_delta(scores[b], scores[a], lab, pairs, "auc", *cell,
                                           seed=seed) for cell in CELLS}
        rec = {(sc, "all"): recall_delta(pool_scores[b], pool_scores[a], pools, rel, sc, seed)
               for sc in final_set.SCOPES}
        rec[("total", "llm_free")] = recall_delta(pool_scores[b], pool_scores[a], free, rel,
                                                  "total", seed)
        comparisons.append({
            "id": cid, "b": b, "a": a, "decides": decides,
            "verdict_auc": verdict_of(auc), "verdict_recall@50": verdict_of(rec),
            "auc_deltas": [indomain.paired_delta(scores[b], scores[a], lab, pairs, m, sc, sub,
                                                 seed)
                           for sub in indomain.SUBSETS for sc in final_set.SCOPES
                           for m in indomain.RANK_METRICS],
            "recall@50_deltas": [{**v, "subset": cell[1]} for cell, v in rec.items()]})

    s2 = {}
    for name in FROZEN:
        tot = next(f for f in out_systems[name]["retrieval"]
                   if f["metric"] == S2_METRIC and f["scope"] == "total")
        s2[name] = {"bar": S2_BAR, "value": tot["value"], "ci": [tot["ci_low"], tot["ci_high"]],
                    "verdict": "pass" if tot["value"] >= S2_BAR else "miss",
                    "ci_above_bar": tot["ci_low"] > S2_BAR}
    m3 = next(c for c in comparisons if c["id"] == "M3")
    passing = [n for n in (Z1, Z1L) if s2[n]["verdict"] == "pass"]
    selected = select(passing, m3["verdict_auc"])
    return {
        "plan": "plan/2026-10-07-shortlist-recall (Q8; 03 §2 frozen before this ran)",
        "registered_result": P7_REGISTERED,
        "status": "Z1 and Z1L are post-hoc: designed after the P7 S2 miss, options chosen on "
                  "the A2-train dev set only (D46)",
        "text_variant": TEXT_VARIANT, "jd_title": "A2 JD Position as given (D54)",
        "reference": {"file": final_set.FINAL_CSV.name, "rule_version": rule_version,
                      "sha256": hashlib.sha256(final_set.FINAL_CSV.read_bytes()).hexdigest()},
        "seed": seed,
        "note": ("AUC per JD over JDs with both classes; Recall over JDs with a fit, on P7's "
                 "500-candidate pools (490 unjudged distractors assumed not fits: upper-bound "
                 "recall). CIs resample JDs. Recall llm_free drops the pool rows of judged "
                 "pairs whose label needed the LLM."),
        "systems": out_systems, "unscoreable": unscoreable, "comparisons": comparisons,
        "s2_classical": s2,
        "selection": {"rule": "03 §2: among passing Z rungs, Z1 (cheaper) unless Z1L is "
                              "justified over it on M3's AUC verdict; adopted post-hoc (Q-R1)",
                      "passing": passing, "selected": selected},
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="rebuild and diff, write nothing")
    a = ap.parse_args(argv)
    record = evaluate(a.seed)
    if a.check:
        same = RECORD.exists() and RECORD.read_text(encoding="utf-8") == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {RECORD.name}")
        return 0 if same else 1
    RECORD.write_text(render(record), encoding="utf-8")
    for name, b in record["systems"].items():
        auc = {(f["scope"], f["subset"]): f["value"] for f in b["auc"] if f["metric"] == "auc"}
        r = {(f["metric"], f["scope"]): f["value"] for f in b["retrieval"]}
        print(f"  {name:40s} AUC {auc[('generic', 'all')]:.3f}/{auc[('targeted', 'all')]:.3f}/"
              f"{auc[('total', 'all')]:.3f}  R@10 {r[('recall@10', 'total')]:.3f}  R@50 "
              f"{r[('recall@50', 'generic')]:.3f}/{r[('recall@50', 'targeted')]:.3f}/"
              f"{r[('recall@50', 'total')]:.3f}  {b.get('reproduces_p7', '')}")
    for c in record["comparisons"]:
        print(f"  {c['id']} {c['b']} vs {c['a']}: AUC {c['verdict_auc']}; "
              f"R@50 {c['verdict_recall@50']}")
    print(json.dumps({"s2": record["s2_classical"], "selection": record["selection"]}, indent=1))
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
