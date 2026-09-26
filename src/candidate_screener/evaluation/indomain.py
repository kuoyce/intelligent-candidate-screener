"""In-domain evaluation against the final set (D41): Fit / No Fit only (D47).

Rankers get per-JD AUC (primary: is a Fit candidate ranked above a No Fit one for the same
job?) and within-family AUC (only the candidates sharing the JD's `Primary Keyword`, where
metadata cannot help). Classifiers get accuracy, binary macro-F1, F1 of Fit, and kappa.
Every figure names its scope (generic, targeted or total), its n, a CI resampling JDs, and
the final set's rule version. A total is never emitted without both strata beside it.

Usage:
    uv run python -m candidate_screener.evaluation.indomain --seed 0
    uv run python -m candidate_screener.evaluation.indomain --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import final_set, resample

RANK_METRICS = ("auc", "within_family_auc")
CLS_METRICS = ("accuracy", "macro_f1", "f1_fit", "kappa")
SUBSETS = ("all", "llm_free")
RECORD = OUTPUT / "matching" / "indomain-metrics.json"
COSTS = OUTPUT / "matching" / "indomain-costs.json"
DECIMALS = 6


@dataclass(frozen=True)
class IndomainFigure:
    system: str
    metric: str
    scope: str
    subset: str
    value: float
    ci_low: float
    ci_high: float
    n_jd: int
    n_pairs: int
    rule_version: int

    def __post_init__(self) -> None:
        if self.scope not in final_set.SCOPES:
            raise ValueError(f"scope {self.scope!r} is not one of {final_set.SCOPES}")
        if self.subset not in SUBSETS:
            raise ValueError(f"subset {self.subset!r} is not one of {SUBSETS}")
        if not self.n_jd:
            raise ValueError(f"{self.system} {self.metric} ({self.scope}) has no scoreable "
                             "JD — a figure without its n is not reportable")
        if not isinstance(self.rule_version, int):
            raise ValueError("a figure must name the final set's rule_version")

    def as_dict(self) -> dict:
        d = asdict(self)
        for key in ("value", "ci_low", "ci_high"):
            d[key] = round(float(d[key]), DECIMALS)
        return d


# --- labels ------------------------------------------------------------------------

def labels(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The final set as `pair_id -> y` (1 = Fit), with stratum, JD and tier."""
    frame = final_set.load() if frame is None else frame
    out = frame[["pair_id", "stratum", "jd_id", "tier"]].copy()
    out["y"] = (frame.final_label == final_set.FIT).astype(int)
    return out.set_index("pair_id")


def restrict(lab: pd.DataFrame, scope: str, subset: str) -> pd.DataFrame:
    out = lab if scope == "total" else lab[lab.stratum == scope]
    return out[out.tier.isin(final_set.LLM_FREE_TIERS)] if subset == "llm_free" else out


# --- rankers -------------------------------------------------------------------------

def per_jd_auc(y: pd.Series, s: pd.Series, jd: pd.Series) -> pd.Series:
    """AUC within each JD that has both a Fit and a No Fit candidate."""
    frame = pd.DataFrame({"y": y, "s": s, "jd": jd}).dropna()
    return pd.Series({q: roc_auc_score(g.y, g.s) for q, g in frame.groupby("jd", sort=True)
                      if g.y.nunique() == 2}, dtype=float)


def _auc_values(scores: pd.Series, lab: pd.DataFrame, pairs: pd.DataFrame, metric: str
                ) -> tuple[pd.Series, int]:
    sub = lab
    if metric == "within_family_auc":
        same = pairs.set_index("pair_id").keyword_match
        sub = lab[same.reindex(lab.index).fillna(0).astype(bool)]
    values = per_jd_auc(sub.y, scores.reindex(sub.index), sub.jd_id)
    return values, int(sub[sub.jd_id.isin(values.index)].shape[0])


def rank_figures(name: str, scores: pd.Series, lab: pd.DataFrame, pairs: pd.DataFrame,
                 rule_version: int, seed: int = 0,
                 unscoreable: list | None = None) -> list[IndomainFigure]:
    """All scopes of a (metric, subset), or none: a group where some scope has no JD with
    both classes is skipped whole and listed in `unscoreable`, never emitted as a total
    without its strata."""
    out = []
    for subset in SUBSETS:
        for metric in RANK_METRICS:
            group = {scope: _auc_values(scores, restrict(lab, scope, subset), pairs, metric)
                     for scope in final_set.SCOPES}
            empty = [scope for scope, (values, _) in group.items() if not len(values)]
            if empty:
                if unscoreable is not None:
                    unscoreable.append({"system": name, "metric": metric, "subset": subset,
                                        "reason": f"no JD with both classes in {empty}"})
                continue
            for scope, (values, n_pairs) in group.items():
                low, high = resample.mean_ci(values, seed=seed)
                out.append(IndomainFigure(name, metric, scope, subset, float(values.mean()),
                                          low, high, len(values), n_pairs, rule_version))
    return out


def paired_delta(b: pd.Series, a: pd.Series, lab: pd.DataFrame, pairs: pd.DataFrame,
                 metric: str = "auc", scope: str = "total", subset: str = "all",
                 seed: int = 0) -> dict:
    """Mean per-JD difference B − A over the same JDs, with a JD-resampled CI."""
    sub = restrict(lab, scope, subset)
    vb, _ = _auc_values(b, sub, pairs, metric)
    va, _ = _auc_values(a, sub, pairs, metric)
    diff = (vb - va).dropna()
    if diff.empty:
        return {"metric": metric, "scope": scope, "subset": subset, "delta": None,
                "ci": None, "half_width": None, "n_jd": 0}
    low, high = resample.mean_ci(diff, seed=seed)
    return {"metric": metric, "scope": scope, "subset": subset,
            "delta": round(float(diff.mean()), DECIMALS),
            "ci": [round(low, DECIMALS), round(high, DECIMALS)],
            "half_width": round((high - low) / 2, DECIMALS), "n_jd": int(len(diff))}


# --- classifiers ------------------------------------------------------------------------

def _cls_stats(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Binary stats from counts: numpy only, so a 2,000-resample CI is cheap."""
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool)
    tp, fp = np.sum(y & p), np.sum(~y & p)
    fn, tn = np.sum(y & ~p), np.sum(~y & ~p)
    f1_fit = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_no = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return {"accuracy": float((tp + tn) / len(y)), "macro_f1": float((f1_fit + f1_no) / 2),
            "f1_fit": float(f1_fit), "kappa": resample.binary_kappa(y, p)}


def cls_figures(name: str, preds: pd.Series, lab: pd.DataFrame, rule_version: int,
                seed: int = 0, n_boot: int = resample.N_BOOT) -> list[IndomainFigure]:
    """`preds` holds `Fit` / `No Fit` per pair_id."""
    p = (preds == final_set.FIT).astype(int)
    out = []
    for subset in SUBSETS:
        for scope in final_set.SCOPES:
            sub = restrict(lab, scope, subset).assign(p=p).dropna(subset=["p"])
            y, q = sub.y.to_numpy(), sub.p.to_numpy()
            point = _cls_stats(y, q)
            blocks = resample.group_indices(sub.jd_id)
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(n_boot):
                rows = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
                boots.append(_cls_stats(y[rows], q[rows]))
            for metric in CLS_METRICS:
                low, high = np.nanpercentile([bt[metric] for bt in boots], [2.5, 97.5])
                out.append(IndomainFigure(name, metric, scope, subset, point[metric], low, high,
                                          int(sub.jd_id.nunique()), int(len(sub)), rule_version))
    return out


def floor(lab: pd.DataFrame) -> dict:
    """Always-Fit accuracy per scope: the classifier floor every figure sits beside."""
    return {scope: round(float(restrict(lab, scope, "all").y.mean()), DECIMALS)
            for scope in final_set.SCOPES}


# --- the record ----------------------------------------------------------------------------

def assert_total_has_strata(figures: list[dict]) -> None:
    keys = {(f["system"], f["metric"], f["subset"]) for f in figures}
    for key in keys:
        scopes = {f["scope"] for f in figures if (f["system"], f["metric"], f["subset"]) == key}
        if "total" in scopes and not {"generic", "targeted"} <= scopes:
            raise ValueError(f"{key} reports a total without both strata beside it")


def evaluate(rungs: list, classifiers: dict, comparisons: list[tuple[str, str]],
             seed: int = 0) -> tuple[dict, dict]:
    """Score every rung on A2, compute every figure, return (record, costs)."""
    from candidate_screener.matching import corpus
    frame = final_set.load()
    rule_version = int(frame.rule_version.iloc[0])
    lab = labels(frame)
    pairs = corpus.a2_pairs().assign(keyword_match=lambda d: corpus.keyword_match(d))
    pairs = pairs.set_index("pair_id").loc[lab.index].reset_index()

    systems, costs, scores, unscoreable = {}, {}, {}, []
    for spec in rungs:
        system = spec.factory()
        system.fit(None)
        s = system.score(pairs)
        scores[system.name] = s
        figures = [f.as_dict() for f in rank_figures(system.name, s, lab, pairs, rule_version,
                                                     seed, unscoreable)]
        card = system.card()
        costs[system.name] = {k: card[k] for k in ("fit_seconds", "ms_per_pair",
                                                   "ms_per_500_candidate_role")}
        systems[system.name] = {
            "rung": system.rung, "stage": system.stage, "status": spec.status,
            "trained_on": system.trained_on, "params": card["params"],
            "can_explain": card["can_explain"], "deterministic": card["deterministic"],
            "scores_sha256": hashlib.sha256(np.round(s.to_numpy(), 9).tobytes()).hexdigest(),
            "figures": figures,
        }
    cls_block = {}
    for name, (make_preds, status) in classifiers.items():
        preds = make_preds(pairs)
        cls_block[name] = {"status": status,
                           "figures": [f.as_dict() for f in cls_figures(name, preds, lab,
                                                                        rule_version, seed)]}
    deltas = []
    for b, a in comparisons:
        if b in scores and a in scores:
            for subset in SUBSETS:
                for scope in final_set.SCOPES:
                    for metric in RANK_METRICS:
                        deltas.append({"b": b, "a": a, **paired_delta(
                            scores[b], scores[a], lab, pairs, metric, scope, subset, seed)})
    for block in [*systems.values(), *cls_block.values()]:
        assert_total_has_strata(block["figures"])
    record = {
        "seed": seed,
        "reference": {"file": final_set.FINAL_CSV.name, "rule_version": rule_version,
                      "sha256": hashlib.sha256(final_set.FINAL_CSV.read_bytes()).hexdigest(),
                      "labels": "Fit = Good Fit or Potential Fit; No Fit (D47)"},
        "note": ("Agreement with a documented panel rule, not correctness. Per-JD AUC over "
                 "JDs with both a Fit and a No Fit candidate; CIs resample JDs (10,000). "
                 "`llm_free` = pairs whose label needed no LLM tie-break."),
        "classifier_floor_always_fit_accuracy": floor(lab),
        "systems": systems, "classifiers": cls_block, "deltas": deltas,
        "unscoreable": unscoreable,
    }
    return record, costs


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    from candidate_screener.matching import ladder
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="rebuild and diff, write nothing")
    args = ap.parse_args(argv)
    record, costs = evaluate(ladder.rungs(), ladder.classifiers(), ladder.comparisons(),
                             args.seed)
    if args.check:
        committed = RECORD.read_text(encoding="utf-8") if RECORD.exists() else ""
        same = committed == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {RECORD.name}")
        if not same and committed:
            old = json.loads(committed)
            for name, block in record["systems"].items():
                if old.get("systems", {}).get(name) != block:
                    print(f"  differs: system {name}")
        return 0 if same else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    COSTS.write_text(render(costs), encoding="utf-8")
    for name, block in record["systems"].items():
        f = {(x["metric"], x["scope"]): x for x in block["figures"] if x["subset"] == "all"}
        cells = " / ".join(f"{f[('auc', s)]['value']:.3f}" for s in final_set.SCOPES)
        print(f"  {name:28s} AUC {cells}   [{block['status']}]")
    print(f"-> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
