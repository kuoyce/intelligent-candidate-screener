"""Stage 1, text-only: the X rungs, K1–K5, and S2 (plan 2026-10-06 P7; 06-registration).

Everything is scored on `text_v2` (D53). The X rungs are `held-out` (registered 7 Oct,
before this module first ran); the text rungs they are compared with are `exploratory`
(P1 re-scored them on `text_v2`); the 1a / 1a+2 rows on the **true** fields are an
`oracle` reference, an upper bound, never a result (02 §3.4).

Two halves, one record (`output/matching/stage1-metrics.json`):

1. **AUC** against the final set, per JD (`evaluation.indomain`), with within-family AUC
   beside it, `generic` / `targeted` / `total`, `all` and `llm_free`; the registered
   comparisons K1–K5 as paired deltas with the registration §3 verdict.
2. **Retrieval** on the pilot's 500-candidate pools (`evaluation.retrieval_pilot`, same
   pools, CV text cut to `v2`): Recall@10/50/100. S2 is judged on Recall@50, point
   estimate against the bar, with "CI above the bar" stated separately; Recall@10 has no bar.

Selection (06 §5) runs mechanically over the S2 candidates; `SELECTION_NOTE` states how
the registration's wording was read, decided before any X rung was scored.

Usage:
    uv run python -m candidate_screener.evaluation.stage1 --seed 0
    uv run python -m candidate_screener.evaluation.stage1 --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import final_set, indomain, resample, retrieval_pilot
from candidate_screener.matching.selection import CELLS, verdict_of

TEXT_VARIANT = "v2"
RECORD = OUTPUT / "matching" / "stage1-metrics.json"
COSTS = OUTPUT / "matching" / "stage1-costs.json"
DECIMALS = 6

RANDOM, TFIDF = "random", "TF-IDF"
SBERT_ZS, SBERT_WEAK = "SBERT (zero-shot)", "SBERT (weak fine-tuned)"
X1, X2, X3, X4 = ("extracted rule", "extracted rule + TF-IDF",
                  "extracted rule + skill overlap", "extracted rule + SBERT (weak fine-tuned)")
ORACLE_1A, ORACLE_1A2 = "metadata rules", "metadata + TF-IDF"

#: Registration §4 (stage 1): (id, B, A, decides selection?).
COMPARISONS = (
    ("K1", X2, TFIDF, True),
    ("K2", X2, ORACLE_1A2, False),
    ("K3", X4, SBERT_WEAK, True),
    ("K4", X2, SBERT_WEAK, True),
    ("K5", X3, X1, False),
)
#: S2 bars on Recall@50 (02 §4, registration §4): the group passes on its best member.
S2_METRIC = "recall@50"
S2_GROUPS = {"classical": ((TFIDF, X2), 0.80), "advanced": ((SBERT_WEAK, X4), 0.90)}
#: Selection candidates, cheapest first. Within a registered cost tier ("TF-IDF / X2",
#: "SBERT weak / X4") the extracted rung counts as dearer: it is the text rung plus X1.
CANDIDATES = (TFIDF, X2, SBERT_WEAK, X4)
#: (more expensive B, cheaper A, the registered comparison it comes from).
BEATS = ((X2, TFIDF, "K1"), (SBERT_WEAK, X2, "K4, B and A swapped"), (X4, SBERT_WEAK, "K3"))
SELECTION_NOTE = (
    "Read before scoring (7 Oct 2026): candidates are the four rungs S2 names (TF-IDF, X2: "
    "bar 0.80; SBERT weak, X4: bar 0.90); a candidate qualifies if its own Recall@50 point "
    "estimate meets its group's bar. A qualifying candidate is beaten when a dearer "
    "qualifying candidate is 'justified' over it on a registered comparison (K1, K3, or K4 "
    "with B and A swapped, since SBERT weak is the dearer side). K2 and K5 decide nothing. "
    "The cheapest qualifying candidate not beaten is adopted.")


@dataclass(frozen=True)
class Spec:
    factory: Callable
    status: str          # held-out | exploratory | oracle


def specs() -> list[Spec]:
    from candidate_screener.matching import extracted, lexical, rules, sbert
    v = TEXT_VARIANT
    return [
        Spec(rules.Random, "exploratory"),
        Spec(extracted.ExtractedRule, "held-out"),
        Spec(lambda: extracted.ExtractedPlus(rules.SkillOverlap(), "X3"), "held-out"),
        Spec(lambda: lexical.Tfidf(v), "exploratory"),
        Spec(lambda: extracted.ExtractedPlus(lexical.Tfidf(v), "X2"), "held-out"),
        Spec(lambda: sbert.SbertZeroShot(v), "exploratory"),
        Spec(lambda: sbert.SbertWeak(v), "exploratory"),
        Spec(lambda: extracted.ExtractedPlus(sbert.SbertWeak(v), "X4"), "held-out"),
        Spec(rules.Metadata, "oracle"),
        Spec(lambda: lexical.MetadataPlus(lexical.Tfidf(v), "1a+2"), "oracle"),
    ]


# --- S2 and selection ---------------------------------------------------------------------

def s2_verdicts(retrieval: dict[str, list[dict]]) -> dict:
    """Per S2 group: its best member on total Recall@50, pass / miss, CI above the bar."""
    def total(name):
        return next(f for f in retrieval[name] if f["metric"] == S2_METRIC
                    and f["scope"] == "total")
    out = {}
    for group, (members, bar) in S2_GROUPS.items():
        rows = {m: total(m) for m in members}
        best = max(members, key=lambda m: (rows[m]["value"], -members.index(m)))
        out[group] = {
            "bar": bar, "members": {m: {"value": rows[m]["value"],
                                        "ci": [rows[m]["ci_low"], rows[m]["ci_high"]],
                                        "meets_bar": rows[m]["value"] >= bar} for m in members},
            "best": best, "verdict": "pass" if rows[best]["value"] >= bar else "miss",
            "ci_above_bar": rows[best]["ci_low"] > bar}
    return out


def select(s2: dict, verdicts: dict[tuple[str, str], str]) -> dict:
    """Registration §5, as `SELECTION_NOTE` reads it. `verdicts[(b, a)]` is §3's verdict
    of the dearer B over the cheaper A."""
    qualifies = {m: info["meets_bar"] for g in s2.values() for m, info in g["members"].items()}
    trail = []
    for name in CANDIDATES:
        if not qualifies[name]:
            trail.append({"rung": name, "qualifies_s2": False, "beaten_by": []})
            continue
        checks = [{"b": b, "via": via, "verdict": verdicts[(b, a)]}
                  for b, a, via in BEATS if a == name and qualifies[b]]
        beaten = [c["b"] for c in checks if c["verdict"] == "justified"]
        trail.append({"rung": name, "qualifies_s2": True, "checks": checks, "beaten_by": beaten})
        if not beaten:
            return {"selected": name, "trail": trail, "note": SELECTION_NOTE}
    return {"selected": None, "trail": trail, "note": SELECTION_NOTE}


# --- the record ---------------------------------------------------------------------------

def _sha(s: pd.Series) -> str:
    return hashlib.sha256(np.round(s.to_numpy(), 9).tobytes()).hexdigest()


def _deltas(b: pd.Series, a: pd.Series, lab, pairs, seed: int) -> list[dict]:
    return [indomain.paired_delta(b, a, lab, pairs, metric, scope, subset, seed)
            for subset in indomain.SUBSETS for scope in final_set.SCOPES
            for metric in indomain.RANK_METRICS]


def _verdict(b: pd.Series, a: pd.Series, lab, pairs, seed: int) -> str:
    return verdict_of({cell: indomain.paired_delta(b, a, lab, pairs, "auc", *cell, seed=seed)
                       for cell in CELLS})


def x1_profile(rule, pairs: pd.DataFrame) -> dict:
    """What X1's two terms look like on the 600 pairs: descriptive, decides nothing."""
    from candidate_screener.extraction import experience
    c = rule.components(pairs)
    jd_known = pd.Series([experience._jd_band_of(str(t)) for t in pairs.jd_text]).notna()
    cv_known = pd.Series([experience._cv_band_of(str(t)) for t in pairs.cv_text]).notna()
    q = np.quantile(c.role_match, [0.1, 0.5, 0.9])
    return {"n_pairs": int(len(pairs)),
            "role_match_quantiles_10_50_90": [round(float(v), DECIMALS) for v in q],
            "experience_compatible_rate": round(float(c.experience_compatible.mean()), DECIMALS),
            "experience_unknown_either_side_rate": round(float((~(jd_known & cv_known)).mean()),
                                                         DECIMALS)}


@indomain.auc_memo()
def evaluate(seed: int = 0, spec_list: list[Spec] | None = None) -> tuple[dict, dict]:
    from candidate_screener.evaluation import text_leak
    from candidate_screener.matching import corpus
    frame = final_set.load()
    rule_version = int(frame.rule_version.iloc[0])
    lab = indomain.labels(frame)
    pairs = corpus.a2_pairs(text_variant=TEXT_VARIANT).assign(
        keyword_match=lambda d: corpus.keyword_match(d))
    pairs = pairs.set_index("pair_id").loc[lab.index].reset_index()
    pools = retrieval_pilot.build_pools(seed, TEXT_VARIANT)
    rel = retrieval_pilot.relevant(pools)

    systems, costs, scores, retrieval, unscoreable, profile = {}, {}, {}, {}, [], None
    for spec in spec_list or specs():
        system = spec.factory()
        system.fit(None)
        s = system.score(pairs)
        scores[system.name] = s
        figures = [f.as_dict() for f in indomain.rank_figures(
            system.name, s, lab, pairs, rule_version, seed, unscoreable)]
        indomain.assert_total_has_strata(figures)
        if system.rung == "X1":
            profile = x1_profile(system, pairs)
        pool_scores = system.score(pools)
        retrieval[system.name] = retrieval_pilot.figures(system.name, pools, pool_scores, rel,
                                                         seed=seed)
        card = system.card()
        costs[system.name] = {k: card[k] for k in ("fit_seconds", "ms_per_pair",
                                                   "ms_per_500_candidate_role")}
        systems[system.name] = {
            "rung": system.rung, "stage": system.stage, "status": spec.status,
            "trained_on": system.trained_on, "params": card["params"],
            "can_explain": card["can_explain"], "deterministic": card["deterministic"],
            "scores_sha256": _sha(s), "pool_scores_sha256": _sha(pool_scores),
            "figures": figures}

    comparisons, verdicts = [], {}
    for cid, b, a, decides in COMPARISONS:
        comparisons.append({"id": cid, "b": b, "a": a, "decides_selection": decides,
                            "verdict_auc": _verdict(scores[b], scores[a], lab, pairs, seed),
                            "deltas": _deltas(scores[b], scores[a], lab, pairs, seed)})
    for b, a, _ in BEATS:
        verdicts[(b, a)] = _verdict(scores[b], scores[a], lab, pairs, seed)
    s2 = s2_verdicts(retrieval)
    pool_n = pools.groupby("jd_id").size()
    record = {
        "text_variant": TEXT_VARIANT,
        "text_variant_definition": {"cv_fields": list(corpus.CV_FIELDS_BY_VARIANT[TEXT_VARIANT]),
                                    "jd_text": "Long Description", "decision": "D53"},
        "residual_leak": text_leak.report(pairs, seed),
        "seed": seed,
        "registration": "plan/2026-10-06-extraction-and-rerank/06-registration.md (A1, A2)",
        "reference": {"file": final_set.FINAL_CSV.name, "rule_version": rule_version,
                      "sha256": hashlib.sha256(final_set.FINAL_CSV.read_bytes()).hexdigest(),
                      "labels": "Fit = Good Fit or Potential Fit; No Fit (D47)"},
        "note": ("Per-JD AUC over JDs with both a Fit and a No Fit candidate; CIs resample JDs "
                 f"({resample.N_BOOT:,}). Within-family = candidates sharing the JD's true "
                 "Primary Keyword (a measurement slice; no X rung reads it). `oracle` rows "
                 "read the true fields: an upper bound, never a result. X1's role_match is "
                 "soft, so X2-X4's text term can reorder candidates whose role agreement "
                 "differs by under 0.5 (the registered 1a+2 sum, unchanged)."),
        "x1_profile": profile,
        "systems": systems, "unscoreable": unscoreable,
        "comparisons": comparisons,
        "retrieval": {"pool_size": retrieval_pilot.POOL_SIZE, "n_pools": int(len(pool_n)),
                      "pool_sizes_all_equal": bool(pool_n.eq(retrieval_pilot.POOL_SIZE).all()),
                      "note": ("Pools of retrieval_pilot.build_pools (unchanged members), CV "
                               "text v2. Relevant = final-set Fit among the 10 judged CVs; "
                               "the 490 distractors are unjudged and assumed not fits "
                               "(upper-bound recall). Recall@10 has no bar."),
                      "figures": retrieval},
        "s2": s2,
        "selection": select(s2, verdicts),
    }
    return record, costs


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="rebuild and diff, write nothing")
    args = ap.parse_args(argv)
    record, costs = evaluate(args.seed)
    if args.check:
        committed = RECORD.read_text(encoding="utf-8") if RECORD.exists() else ""
        same = committed == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {RECORD.name}")
        if not same and committed:
            old = json.loads(committed)
            for key in record:
                if old.get(key) != record[key]:
                    print(f"  differs: {key}")
        return 0 if same else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    COSTS.write_text(render(costs), encoding="utf-8")
    for name, block in record["systems"].items():
        f = {(x["metric"], x["scope"]): x for x in block["figures"] if x["subset"] == "all"}
        auc = " / ".join(f"{f[('auc', s)]['value']:.3f}" for s in final_set.SCOPES)
        r = {x["metric"]: x["value"] for x in record["retrieval"]["figures"][name]
             if x["scope"] == "total"}
        print(f"  {name:42s} AUC {auc}  R@10 {r['recall@10']:.3f} R@50 {r['recall@50']:.3f}"
              f"  [{block['status']}]")
    for c in record["comparisons"]:
        print(f"  {c['id']} {c['b']} vs {c['a']}: {c['verdict_auc']}")
    for g, v in record["s2"].items():
        print(f"  S2 {g}: {v['verdict']} (best {v['best']})")
    print(f"  selected: {record['selection']['selected']}")
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
