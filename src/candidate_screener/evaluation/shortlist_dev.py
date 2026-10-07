"""Shortlist recall on the development set (plan 2026-10-07-shortlist-recall, Q2 and Q3).

Everything here runs on `evaluation.dev_pools` (A2 train region; proxy relevance = same
oracle role family). It never reads the final labels (D46). Two parts, one record
(`output/matching/shortlist-dev.json`):

1. **Components (Q2).** C1 (JD title -> family) against the oracle family on the dev JDs,
   and on the 60 final-set JDs (measurement only: the oracle there is the documents'
   metadata, no final label is read). C2 (CV role set) recall and set size on the dev CVs,
   for every option.
2. **Rungs (Q3).** X2 (P7's registered rung), TF-IDF and SBERT weak as references; Y1, Y2,
   Y2t, Y4 for every option (`k` in 0-3, role context off / on). Recall@10 and @50 per JD,
   mean over JDs with a relevant CV, CI resampling JDs; dev-train and dev-select apart.

**Dev rules, fixed before the first run (`RULES`).** For each Y rung, its option is the one
with the highest dev-select proxy Recall@50, ties (6 decimals) to the cheaper option:
smaller `k`, then context off. Y2 vs Y2t: the higher dev-select Recall@50 at their chosen
options, ties to Y2. **Early stop (02 Q3):** the dev-best classical Y rung (Y1, Y2 or Y2t)
must beat X2's dev-select Recall@50 (strictly, point estimate); otherwise stop and record.
A pilot-shaped figure (relevance only among the 10 seeded slots, the final set's shape) is
reported beside and decides nothing.

    uv run python -m candidate_screener.evaluation.shortlist_dev            # write the record
    uv run python -m candidate_screener.evaluation.shortlist_dev --check    # rebuild and diff
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import dev_pools, resample
from candidate_screener.extraction.role_set import KS
from candidate_screener.matching.base import BaseSystem

RECORD = OUTPUT / "matching" / "shortlist-dev.json"
TEXT_VARIANT = "v2"
METRICS = (10, 50)
SELECT_ON = ("dev-select", 50)
DECIMALS = 6
OPTIONS = tuple((k, ctx) for k in KS for ctx in (False, True))      # cheapest first
RULES = {
    "option": "per Y rung, highest dev-select proxy Recall@50; ties (6 dp) to smaller k, then context off",
    "y2_vs_y2t": "higher dev-select Recall@50 at each one's chosen option; ties to Y2",
    "early_stop": "dev-best classical Y (Y1, Y2, Y2t) must beat X2's dev-select Recall@50 "
                  "(strict, point estimate), else stop and go to Q8 with what exists",
    "pilot_shaped": "relevance among the 10 seeded slots only; reported, decides nothing",
    "fixed": "7 Oct 2026, in this module, before its first run",
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Fixed(BaseSystem):
    """A fitted system's scores computed once on the dev frame and replayed by pair_id,
    so the many Y variants share one TF-IDF / SBERT pass."""

    def __init__(self, system: BaseSystem, frame: pd.DataFrame) -> None:
        super().__init__()
        self.inner, self.name, self.rung = system, system.name, system.rung
        self.trained_on = system.trained_on
        self.values = system.score(frame)

    def fit(self, train=None):
        pass

    def _score(self, pairs):
        return self.values.loc[pairs.pair_id.to_numpy()].to_numpy()

    def params(self):
        return self.inner.params()


# --- figures --------------------------------------------------------------------------------

def recall_per_jd(frame: pd.DataFrame, scores: np.ndarray, rel: np.ndarray, k: int
                  ) -> pd.Series:
    """Per-JD recall@k over JDs with a relevant CV; ties broken by pair_id (stable)."""
    df = pd.DataFrame({"jd": frame.jd_id.to_numpy(), "pid": frame.pair_id.to_numpy(),
                       "s": scores, "r": rel})
    df = df.sort_values(["jd", "s", "pid"], ascending=[True, False, True])
    df["rank"] = df.groupby("jd").cumcount()
    n = df.groupby("jd").r.sum()
    hit = df[df["rank"] < k].groupby("jd").r.sum()
    keep = n[n > 0].index
    return (hit.reindex(keep).fillna(0) / n.loc[keep]).sort_index()


def figures(frame: pd.DataFrame, scores: np.ndarray, rel: np.ndarray, seed: int = 0) -> dict:
    out = {}
    for split in dev_pools.SPLITS:
        m = (frame.split == split).to_numpy()
        block = {}
        for k in METRICS:
            v = recall_per_jd(frame[m], scores[m], rel[m], k)
            lo, hi = resample.mean_ci(v, seed=seed)
            block[f"recall@{k}"] = {"value": round(float(v.mean()), DECIMALS),
                                    "ci95": [round(lo, DECIMALS), round(hi, DECIMALS)],
                                    "n_jd": int(len(v))}
        out[split] = block
    return out


def value(fig: dict, split: str = SELECT_ON[0], k: int = SELECT_ON[1]) -> float:
    return fig["proxy"][split][f"recall@{k}"]["value"]


# --- components -----------------------------------------------------------------------------

def _acc_block(right: pd.Series, seed: int = 0) -> dict:
    lo, hi = resample.mean_ci(right.astype(float).reset_index(drop=True), seed=seed)
    return {"value": round(float(right.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n": int(len(right))}


def c1_block(titles: pd.Series, oracle: pd.Series, title_model, jd_texts: pd.Series | None,
             role_model) -> dict:
    """C1 vs the oracle family; P3's text classifier (argmax) on the same JDs beside it."""
    from candidate_screener.extraction import jd_title, role_family as rf
    fam = jd_title.jd_family(titles.tolist(), title_model)
    right = pd.Series(fam.family.to_numpy() == oracle.to_numpy())
    out = {"n": int(len(titles)), "accuracy": _acc_block(right),
           "source_share": {s: round(float((fam.source == s).mean()), 4)
                            for s in ("lookup", "classifier", "none")},
           "accuracy_by_source": {s: round(float(right[(fam.source == s).to_numpy()].mean()), 4)
                                  for s in ("lookup", "classifier")
                                  if (fam.source == s).any()}}
    if jd_texts is not None:
        p = rf.predict_proba(jd_texts.tolist(), "jd", role_model)
        pred = np.array(role_model.labels)[p.argmax(1)]
        out["p3_text_classifier_accuracy"] = _acc_block(pd.Series(pred == oracle.to_numpy()))
    return out


def components(dev, title_model, role_model) -> dict:
    from candidate_screener.extraction import role_family as rf, role_set
    _log("C1 on dev JDs")
    jd = dev["jd"]
    c1_dev = c1_block(jd.title, jd.family, title_model, jd.text, role_model)
    _log("C1 on the 60 final-set JDs (measurement only)")
    pairs, fdocs = rf.load_final_documents()
    fjd = fdocs["jd"]
    oracle = rf.labels_for(fjd)
    strata = pairs.drop_duplicates("jd_id").set_index("jd_id").stratum
    c1_final = {}
    for scope in ("generic", "targeted", "total"):
        ids = fjd.index if scope == "total" else [i for i in fjd.index if strata[i] == scope]
        c1_final[scope] = c1_block(fjd.Position.loc[ids], oracle.loc[ids], title_model,
                                   fjd.text.loc[ids], role_model)
    _log("C2 on dev CVs")
    cv = dev["cv"]
    roles = role_set.cv_roles(cv.text.tolist(), role_model)
    c2 = {}
    for k, ctx in OPTIONS:
        sets = [r.families(k, ctx) for r in roles]
        hit = pd.Series([f in s for f, s in zip(cv.family, sets)])
        c2[f"k={k},context={'on' if ctx else 'off'}"] = {
            "recall": _acc_block(hit), "mean_set_size": round(float(np.mean([len(s) for s in sets])), 4),
            "empty_share": round(float(np.mean([not s for s in sets])), 4)}
    spans_any = pd.Series([bool(r.spans) for r in roles])
    return {"c1_dev_jds": c1_dev,
            "c1_final_set_jds": {**c1_final, "note": "measurement only (D46): oracle = "
                                 "role_family(Primary Keyword, Position) of the final-set JD; "
                                 "no final label is read"},
            "c2_dev_cvs": {"n": int(len(cv)), "any_span": round(float(spans_any.mean()), 4),
                           "options": c2}}


# --- rungs ----------------------------------------------------------------------------------

def evaluate(seed: int = 0) -> dict:
    from candidate_screener.extraction import jd_title, role_family as rf
    from candidate_screener.matching import extracted, lexical, role_filter as yf, sbert
    dev = dev_pools.build()
    frame = dev_pools.pairs(dev)
    rel = dev_pools.labels(dev, frame).to_numpy()
    seeded = dev["pools"].slot.isin(["same_keyword", "other_keyword"]).to_numpy()
    role_model, title_model = rf.load_role_model(), jd_title.load_title_model()

    rungs: dict[str, dict] = {}
    scores: dict[str, np.ndarray] = {}

    def add(name: str, rung: str, status: str, s: np.ndarray, options: dict | None = None):
        scores[name] = s
        rungs[name] = {"rung": rung, "status": status, "options": options,
                       "scores_sha256": hashlib.sha256(np.round(s, 9).tobytes()).hexdigest(),
                       "proxy": figures(frame, s, rel, seed),
                       "pilot_shaped": figures(frame, s, rel * seeded, seed)}
        _log(f"{name}: dev-select R@50 {value(rungs[name]):.4f}")

    _log("TF-IDF")
    tfidf = lexical.Tfidf(TEXT_VARIANT)
    tfidf.fit(None)
    tfidf = Fixed(tfidf, frame)
    add(tfidf.name, "2", "reference", tfidf.values.to_numpy())
    _log("SBERT weak")
    weak = sbert.SbertWeak(TEXT_VARIANT)
    weak.fit(None)
    weak = Fixed(weak, frame)
    add(weak.name, "3b", "reference", weak.values.to_numpy())
    _log("X2")
    x2 = extracted.ExtractedPlus(tfidf, "X2", extracted.ExtractedRule(role_model))
    x2.fit(None)
    add(x2.name, "X2", "reference (P7 registered)", x2.score(frame).to_numpy())

    for k, ctx in OPTIONS:
        opts = {"k": k, "context": ctx}
        rule = yf.RoleFilterRule(k, ctx, role_model, title_model)
        rule.fit(None)
        _log(f"Y rungs, k={k} context={ctx}")
        for rung, system in (("Y1", rule),
                             ("Y2", yf.RoleFilterPlus(tfidf, "Y2", rule)),
                             ("Y2t", yf.RoleFilterTitle(tfidf, "Y2t", rule)),
                             ("Y4", yf.RoleFilterPlus(weak, "Y4", rule))):
            system.fit(None)
            add(f"{rung} {rule.name}", rung, "dev", system.score(frame).to_numpy(), opts)

    choices = choose(rungs)
    return {
        "plan": "plan/2026-10-07-shortlist-recall (Q2, Q3)",
        "dev_set": {"report": dev_pools.REPORT.name,
                    "pools_sha256": hashlib.sha256(
                        dev["pools"].to_csv(index=False).encode()).hexdigest()},
        "models": {"role_family": role_model.fingerprint, "jd_title": title_model.fingerprint},
        "text_variant": TEXT_VARIANT, "jd_title": "A2 JD Position as given (D54, Q-R2)",
        "rules": RULES,
        "note": ("Proxy relevance = same oracle role family (Q-R3); it measures agreement with "
                 "the metadata, not fit. Recall per JD over JDs with a relevant CV; CIs resample "
                 f"JDs ({resample.N_BOOT:,}). TF-IDF's IDF is the shared A2 vectoriser (a seeded "
                 "train-region sample, unsupervised)."),
        "components": components(dev, title_model, role_model),
        "rungs": rungs,
        "choices": choices,
    }


def choose(rungs: dict[str, dict]) -> dict:
    """`RULES`, applied mechanically."""
    out: dict = {}
    for rung in ("Y1", "Y2", "Y2t", "Y4"):
        cands = [(n, r) for n, r in rungs.items() if r["rung"] == rung]
        order = {(o["k"], o["context"]): i for i, o in enumerate(
            {"k": k, "context": c} for k, c in OPTIONS)}
        best = max(cands, key=lambda nr: (round(value(nr[1]), DECIMALS),
                                          -order[(nr[1]["options"]["k"], nr[1]["options"]["context"])]))
        out[rung] = {"name": best[0], "options": best[1]["options"],
                     "dev_select_recall@50": value(best[1]),
                     "dev_train_recall@50": value(best[1], "dev-train")}
    y2, y2t = out["Y2"]["dev_select_recall@50"], out["Y2t"]["dev_select_recall@50"]
    out["Y2_vs_Y2t"] = "Y2t" if round(y2t, DECIMALS) > round(y2, DECIMALS) else "Y2"
    classical = max(("Y1", "Y2", "Y2t"), key=lambda r: (round(out[r]["dev_select_recall@50"],
                                                              DECIMALS), -("Y1", "Y2", "Y2t").index(r)))
    x2 = next(r for r in rungs.values() if r["rung"] == "X2")
    beats = out[classical]["dev_select_recall@50"] > value(x2)
    out["early_stop"] = {"dev_best_classical": classical,
                         "its_dev_select_recall@50": out[classical]["dev_select_recall@50"],
                         "x2_dev_select_recall@50": value(x2), "beats_x2": bool(beats),
                         "decision": "continue" if beats else "stop: record, go to Q8"}
    return out


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
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    for name, r in record["rungs"].items():
        print(f"  {name:75s} sel R@10 {value(r, k=10):.3f} R@50 {value(r):.3f}"
              f"  train R@50 {value(r, 'dev-train'):.3f}")
    print(json.dumps(record["choices"], indent=1))
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
