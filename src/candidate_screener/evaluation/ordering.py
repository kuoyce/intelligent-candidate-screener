"""Ordering evaluation on A2 (D52): does a system order each JD's judged CVs the way the
humans grade them, Good Fit (2) > Potential Fit (1) > No Fit (0)?

A secondary figure. Classification stays Fit / No Fit (D47); this module only reads the
raw human labels to grade an order. The references are the humans alone:

    yc         yc's grade, all 600 pairs
    van        van's grade, the 326 pairs van labelled
    mean       the mean of yc and van, only where both exist (NaN elsewhere)
    yc_single  yc's grade on the 274 pairs van did not label (its own, labelled row)

The LLM is never a reference (D43, D51): asking for one raises.

Per JD, over the CVs the reference grades:

    nDCG       linear gain (the grade), log2 discount. Tied scores share their mean gain,
               so a tie is worth exactly the expected value of breaking it at random:
               no tie-break luck, in either direction.
    pairwise   the share of CV pairs with different reference grades that the score puts in
               the same order; a score tie counts 0.5.

A JD with no pair of different grades (so also one with zero ideal DCG) says nothing about
an order: it is excluded from both metrics, and the excluded count is reported. Figures
are means over JDs with a 95% CI that resamples JDs (`resample.mean_ci`), always for
`generic`, `targeted` and `total` together. Comparisons are paired: the CI of the mean
per-JD difference over the JDs both systems are scored on.

The random baseline is analytic, not sampled: expected nDCG under a uniformly random order
is mean(gain) x sum(discounts) / ideal DCG, and expected pairwise agreement is 0.5.

Subsets: `full`; `llm_free` (pairs whose final label needed no LLM, D51); `fit_only` (pairs
the final set calls Fit, the CVs stage 2 orders).

Usage:
    uv run python -m candidate_screener.evaluation.ordering [--text-variant v2] [--seed 0]
    uv run python -m candidate_screener.evaluation.ordering --check
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import final_set, resample

GRADES = {"Good Fit": 2, "Potential Fit": 1, "No Fit": 0}
REFERENCES = ("yc", "van", "mean", "yc_single")
SUBSETS = ("full", "llm_free", "fit_only")
METRICS = ("ndcg", "pairwise")
TEXT_VARIANTS = ("v1", "v2")
RANDOM = "random (expected)"
LLM = "LLM score (0-100)"
LLM_RUNS = (9, 10)
DECIMALS = 6
SCORE_DECIMALS = 9          # scores are rounded before ranking, so float noise is no tie-break
RECORD_DIR = OUTPUT / "matching"


def record_path(text_variant: str):
    """`v2` is the committed record (D53); other variants get a suffixed file."""
    name = "ordering-metrics.json" if text_variant == "v2" else f"ordering-metrics-{text_variant}.json"
    return RECORD_DIR / name


# --- references --------------------------------------------------------------------------

def grade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The final set as `pair_id -> stratum, jd_id, tier, fit, yc, van, mean, yc_single`.

    Grades come from the raw human columns only. `mean` is NaN unless both humans graded
    the pair (skipna=False), and `yc_single` is yc's grade only where van did not grade it.
    """
    for col in ("yc_raw", "van_raw"):
        bad = set(frame[col].dropna()) - set(GRADES)
        if bad:
            raise ValueError(f"{col} has labels outside {tuple(GRADES)}: {sorted(bad)}")
    out = frame[["pair_id", "stratum", "jd_id", "tier"]].copy()
    out["fit"] = frame.final_label == final_set.FIT
    out["yc"] = frame.yc_raw.map(GRADES).astype(float)
    out["van"] = frame.van_raw.map(GRADES).astype(float)
    out["mean"] = out[["yc", "van"]].mean(axis=1, skipna=False)
    out["yc_single"] = out.yc.where(out.van.isna())
    return out.set_index("pair_id")


def reference(grades: pd.DataFrame, name: str) -> pd.Series:
    """One reference's grades; the LLM (or anything not a human reference) is refused."""
    if name not in REFERENCES:
        raise ValueError(f"{name!r} is not a reference: only the human grades {REFERENCES} "
                         "are (the LLM is never the reference, D43/D51)")
    return grades[name]


def restrict(grades: pd.DataFrame, scope: str, subset: str) -> pd.DataFrame:
    if subset not in SUBSETS:
        raise ValueError(f"subset {subset!r} is not one of {SUBSETS}")
    out = final_set.in_scope(grades, scope)
    if subset == "llm_free":
        out = out[out.tier.isin(final_set.LLM_FREE_TIERS)]
    elif subset == "fit_only":
        out = out[out.fit]
    return out


# --- per-JD metrics ------------------------------------------------------------------------

def _discounts(n: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(n) + 2.0)


def ndcg(gains, scores) -> float:
    """nDCG of `scores` against `gains`; tied scores share their block's mean gain.

    NaN when the ideal DCG is zero.
    """
    g = np.asarray(gains, dtype=float)
    s = np.round(np.asarray(scores, dtype=float), SCORE_DECIMALS)
    disc = _discounts(len(g))
    ideal = float((np.sort(g)[::-1] * disc).sum())
    if ideal <= 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    g, s = g[order], s[order]
    dcg, i = 0.0, 0
    while i < len(g):
        j = i
        while j < len(g) and s[j] == s[i]:
            j += 1
        dcg += g[i:j].mean() * disc[i:j].sum()
        i = j
    return dcg / ideal


def pairwise(gains, scores) -> float:
    """Share of differently-graded pairs the score orders correctly (a tie = 0.5).

    NaN when no two grades differ.
    """
    g = np.asarray(gains, dtype=float)
    s = np.round(np.asarray(scores, dtype=float), SCORE_DECIMALS)
    above = g[:, None] > g[None, :]
    n = int(above.sum())
    if n == 0:
        return float("nan")
    ds = s[:, None] - s[None, :]
    return float(((ds > 0) + 0.5 * (ds == 0))[above].sum() / n)


def expected_random(gains, metric: str) -> float:
    """The exact expectation under a uniformly random order (no sampling)."""
    g = np.asarray(gains, dtype=float)
    if metric == "pairwise":
        return 0.5 if (g[:, None] > g[None, :]).any() else float("nan")
    disc = _discounts(len(g))
    ideal = float((np.sort(g)[::-1] * disc).sum())
    return float(g.mean() * disc.sum() / ideal) if ideal > 0 else float("nan")


def _informative(g: np.ndarray) -> bool:
    """A JD says something about an order only if two of its grades differ."""
    return len(g) >= 2 and g.max() > g.min()


def per_jd(grades: pd.DataFrame, ref: str, scores: pd.Series | None, metric: str
           ) -> tuple[pd.Series, int]:
    """(value per informative JD, number of excluded JDs), over the pairs the reference
    grades. `scores=None` is the random baseline's expectation."""
    if metric not in METRICS:
        raise ValueError(f"metric {metric!r} is not one of {METRICS}")
    g_all = reference(grades, ref)
    sub = grades[g_all.notna()]
    if scores is not None:
        missing = sub.index.difference(scores.dropna().index)
        if len(missing):
            raise ValueError(f"{len(missing)} graded pairs have no score")
    values, excluded = {}, 0
    for jd, rows in sub.groupby("jd_id", sort=True):
        g = g_all.loc[rows.index].to_numpy(float)
        if not _informative(g):
            excluded += 1
            continue
        if scores is None:
            values[jd] = expected_random(g, metric)
        else:
            s = scores.loc[rows.index].to_numpy(float)
            values[jd] = ndcg(g, s) if metric == "ndcg" else pairwise(g, s)
    return pd.Series(values, dtype=float), excluded


def _r(x: float) -> float:
    return round(float(x), DECIMALS) + 0.0          # + 0.0: no "-0.0" in the record


def summarise(values: pd.Series, seed: int = 0) -> dict:
    if values.empty:
        return {"value": None, "ci": None, "n_jd": 0}
    low, high = resample.mean_ci(values, seed=seed)
    return {"value": _r(values.mean()), "ci": [_r(low), _r(high)], "n_jd": int(len(values))}


def figures(name: str, scores: pd.Series | None, grades: pd.DataFrame, seed: int = 0
            ) -> list[dict]:
    """Every (reference, subset, metric) with generic, targeted and total side by side."""
    out = []
    for ref in REFERENCES:
        for subset in SUBSETS:
            for metric in METRICS:
                for scope in final_set.SCOPES:
                    sub = restrict(grades, scope, subset)
                    values, excluded = per_jd(sub, ref, scores, metric)
                    n_pairs = int(sub[sub.jd_id.isin(values.index)][ref].notna().sum())
                    out.append({"system": name, "reference": ref, "subset": subset,
                                "metric": metric, "scope": scope, **summarise(values, seed),
                                "n_jd_excluded": excluded, "n_pairs": n_pairs})
    assert_total_has_strata(out)
    return out


def assert_total_has_strata(figs: list[dict]) -> None:
    keys = {(f["system"], f["reference"], f["subset"], f["metric"]) for f in figs}
    for key in keys:
        scopes = {f["scope"] for f in figs
                  if (f["system"], f["reference"], f["subset"], f["metric"]) == key}
        if "total" in scopes and not {"generic", "targeted"} <= scopes:
            raise ValueError(f"{key} reports a total without both strata beside it")


def paired_delta(b: pd.Series | None, a: pd.Series | None, grades: pd.DataFrame, ref: str,
                 metric: str, scope: str = "total", subset: str = "full", seed: int = 0
                 ) -> dict:
    """Mean per-JD difference B - A over the JDs both are scored on, JD-resampled CI."""
    sub = restrict(grades, scope, subset)
    vb, _ = per_jd(sub, ref, b, metric)
    va, _ = per_jd(sub, ref, a, metric)
    diff = (vb - va).dropna()
    out = {"reference": ref, "subset": subset, "metric": metric, "scope": scope}
    if diff.empty:
        return {**out, "delta": None, "ci": None, "n_jd": 0}
    low, high = resample.mean_ci(diff, seed=seed)
    return {**out, "delta": _r(diff.mean()), "ci": [_r(low), _r(high)], "n_jd": int(len(diff))}


# --- the systems ---------------------------------------------------------------------------

def llm_scores(frames: list[pd.DataFrame] | None = None) -> pd.Series:
    """The committed LLM series' 0-100 score per pair: instrument `5d986c82`, haiku, runs
    9 and 10 (D43). Exactly 600 unique pairs, or it raises. A system here, never a reference."""
    frames = [pd.read_csv(p) for p in final_set.LLM_CSVS] if frames is None else frames
    llm = pd.concat(frames, ignore_index=True)
    llm = llm[(llm.agent_sha256 == final_set.LLM_AGENT_SHA256)
              & (llm.model == final_set.LLM_MODEL)
              & llm.run.astype(int).isin(LLM_RUNS)]
    if len(llm) != 600 or not llm.pair_id.is_unique:
        raise ValueError(f"the LLM series must hold exactly 600 unique pairs, "
                         f"got {len(llm)} rows / {llm.pair_id.nunique()} unique")
    return llm.set_index("pair_id").score.astype(float)


def _accepts_variant(factory: Callable) -> bool:
    try:
        return "text_variant" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


def _pairs(text_variant: str) -> pd.DataFrame:
    from candidate_screener.matching import corpus
    if _accepts_variant(corpus.a2_pairs):
        return corpus.a2_pairs(text_variant=text_variant)
    if text_variant != "v1":
        raise SystemExit(f"corpus.a2_pairs has no `text_variant` yet; {text_variant} unavailable")
    return corpus.a2_pairs()


def text_systems() -> list[tuple[str, Callable, bool]]:
    """(name, factory, needs sentence_transformers)."""
    from candidate_screener.matching import lexical
    out: list[tuple[str, Callable, bool]] = [("TF-IDF", lambda: lexical.Tfidf, False)]

    def sb(attr):
        def get():
            from candidate_screener.matching import sbert
            return getattr(sbert, attr)
        return get
    out += [("SBERT (zero-shot)", sb("SbertZeroShot"), True),
            ("SBERT (weak fine-tuned)", sb("SbertWeak"), True)]
    return out


def comparisons() -> list[tuple[str, str]]:
    """(B, A) paired deltas: every system vs the random expectation, and the LLM and SBERT
    rungs vs TF-IDF and SBERT weak (03 P8.4)."""
    systems = ["TF-IDF", "SBERT (zero-shot)", "SBERT (weak fine-tuned)", LLM]
    return [*((s, RANDOM) for s in systems),
            ("SBERT (zero-shot)", "TF-IDF"), ("SBERT (weak fine-tuned)", "TF-IDF"),
            (LLM, "TF-IDF"), (LLM, "SBERT (weak fine-tuned)")]


def score_systems(text_variant: str) -> tuple[dict[str, pd.Series | None], dict[str, str]]:
    """Scores per system on the 600 pairs (`None` = the random expectation), and the reason
    each skipped system was skipped."""
    frame = final_set.load()
    pairs = _pairs(text_variant).set_index("pair_id").loc[frame.pair_id].reset_index()
    scores: dict[str, pd.Series | None] = {RANDOM: None}
    skipped: dict[str, str] = {}
    have_st = importlib.util.find_spec("sentence_transformers") is not None
    for name, get, needs_st in text_systems():
        if needs_st and not have_st:
            skipped[name] = "sentence_transformers is not installed (uv sync --all-groups)"
            continue
        cls = get()
        if _accepts_variant(cls):
            system = cls(text_variant=text_variant)
        elif text_variant == "v1":
            system = cls()
        else:
            skipped[name] = f"{cls.__name__} does not accept text_variant={text_variant!r} yet"
            continue
        system.fit(None)
        scores[name] = pd.Series(system.score(pairs).to_numpy(dtype=float),
                                 index=pairs.pair_id.to_numpy())
    scores[LLM] = llm_scores()
    return scores, skipped


def inter_annotator(grades: pd.DataFrame, seed: int = 0) -> list[dict]:
    """Context, no target (D44): each human's grades scored as an order against the other,
    on the pairs both graded, full set. Grade ties share mean gain like any score tie."""
    both = grades[grades["mean"].notna()]
    out = []
    for scorer, ref in (("van", "yc"), ("yc", "van")):
        for metric in METRICS:
            for scope in final_set.SCOPES:
                sub = final_set.in_scope(both, scope)
                values, excluded = per_jd(sub, ref, sub[scorer], metric)
                out.append({"scorer": scorer, "reference": ref, "metric": metric,
                            "scope": scope, **summarise(values, seed),
                            "n_jd_excluded": excluded})
    return out


def _digest(s: pd.Series | None) -> str | None:
    if s is None:
        return None
    return hashlib.sha256(np.round(s.sort_index().to_numpy(float), SCORE_DECIMALS).tobytes()
                          ).hexdigest()


def evaluate(text_variant: str = "v2", seed: int = 0,
             scored: tuple[dict, dict] | None = None) -> dict:
    if text_variant not in TEXT_VARIANTS:
        raise ValueError(f"text_variant must be one of {TEXT_VARIANTS}")
    frame = final_set.load()
    grades = grade_frame(frame)
    scores, skipped = score_systems(text_variant) if scored is None else scored
    systems = {name: {"scores_sha256": _digest(s), "figures": figures(name, s, grades, seed)}
               for name, s in scores.items()}
    deltas = [{"b": b, "a": a, **paired_delta(scores[b], scores[a], grades, ref, metric,
                                              scope, subset, seed)}
              for b, a in comparisons() if b in scores and a in scores
              for ref in REFERENCES for subset in SUBSETS for metric in METRICS
              for scope in final_set.SCOPES]
    return {
        "seed": seed,
        "text_variant": text_variant,
        "status": "exploratory",
        "reference": {
            "file": final_set.FINAL_CSV.name,
            "rule_version": int(frame.rule_version.iloc[0]),
            "sha256": hashlib.sha256(final_set.FINAL_CSV.read_bytes()).hexdigest(),
            "grades": "human raw labels: Good Fit=2, Potential Fit=1, No Fit=0 (D52); "
                      "the LLM is never the reference",
            "pairs": {ref: int(grades[ref].notna().sum()) for ref in REFERENCES},
        },
        "note": ("Per-JD nDCG (linear gain, tied scores share their mean gain) and pairwise "
                 "agreement over differently-graded pairs (score tie = 0.5); JDs with no two "
                 "different grades are excluded and counted. Mean over JDs, 95% CI resampling "
                 f"JDs ({resample.N_BOOT}). Random is the exact expectation. Subsets: full; "
                 "llm_free = tiers humans_agree, single_human; fit_only = final label Fit."),
        "llm_series": {"agent_sha256": final_set.LLM_AGENT_SHA256, "model": final_set.LLM_MODEL,
                       "runs": list(LLM_RUNS)},
        "skipped": skipped,
        "systems": systems,
        "deltas": deltas,
        "inter_annotator": inter_annotator(grades, seed),
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2) + "\n"


def _headline(record: dict) -> None:
    for name, block in record["systems"].items():
        f = {(x["reference"], x["metric"], x["scope"]): x for x in block["figures"]
             if x["subset"] == "full"}
        cells = []
        for ref, metric in (("yc", "ndcg"), ("yc", "pairwise"), ("mean", "pairwise")):
            x = f[(ref, metric, "total")]
            cells.append(f"{ref}/{metric} {x['value']:.3f} [{x['ci'][0]:.3f},{x['ci'][1]:.3f}]"
                         if x["value"] is not None else f"{ref}/{metric} n/a")
        print(f"  {name:26s} " + "  ".join(cells))
    for name, why in record["skipped"].items():
        print(f"  {name:26s} skipped: {why}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-variant", choices=TEXT_VARIANTS, default="v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="rebuild and diff, write nothing")
    args = ap.parse_args(argv)
    record = evaluate(args.text_variant, args.seed)
    path = record_path(args.text_variant)
    if args.check:
        committed = path.read_text(encoding="utf-8") if path.exists() else ""
        same = committed == render(record)
        print(f"[{'PASS' if same else 'FAIL'}] rebuild reproduces {path.name}")
        if not same and committed:
            old = json.loads(committed)
            for key in ("reference", "skipped", "deltas", "inter_annotator"):
                if old.get(key) != record[key]:
                    print(f"  differs: {key}")
            for name, block in record["systems"].items():
                if old.get("systems", {}).get(name) != block:
                    print(f"  differs: system {name}")
        return 0 if same else 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(record), encoding="utf-8")
    _headline(record)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
