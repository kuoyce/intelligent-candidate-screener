"""P4 measurement of E3 (experience years) and E4 (seniority): the committed record.

Two document sets, measurement only (no labels of the final set are read, nothing is fitted
to the final-set documents):

1. **A2 train** (D19): a seeded sample (`corpus.a2_train_documents`, `text_v2`, 2,000 JDs and
   2,000 CVs). Band accuracy against the metadata bands (`Exp Years` via `JD_EXP_TO_BAND`,
   `Experience Years` via `cv_experience_band`), coverage, confusion; seniority against
   the seniority tokens of `Position`. The two aggregation variants (JD years: `max` /
   `first`; seniority: `first` / `max` span) are compared **here only**, and the one with
   the higher train accuracy is the one applied to the final-set documents (ties keep the
   a-priori default, `max` for JD years and `first` for seniority).
2. **Final-set documents** (the 60 JDs and the CVs of the 600 pairs, `text_v2`): band
   accuracy, seniority accuracy, and pair-level agreement of `experience_compatible` on
   extracted bands with the oracle (true fields), for generic / targeted / total, with
   CIs bootstrapped over JDs.

Document-level shares carry Wilson 95% intervals (one document is one unit).

    uv run python -m candidate_screener.extraction.experience_eval --evaluate
    uv run python -m candidate_screener.extraction.experience_eval --check
"""
from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.extraction import experience as ex
from candidate_screener.extraction.title_profile import wilson

RECORD = OUTPUT / "extraction" / "experience-metrics.json"
TRAIN_N = 2_000
SEED = 0
N_BOOT = 10_000
BANDS = ("0-1", "2-3", "4-6", "7+")
DEFAULTS = {"jd_aggregate": "max", "seniority_pick": "first"}


def _confusion(oracle: Sequence, pred: Sequence, labels: Sequence[str]) -> dict:
    c = collections.Counter(zip(oracle, pred))
    cols = [*labels, "none"]
    return {o: {p: int(c[(o, None if p == "none" else p)]) for p in cols} for o in labels
            if any(c[(o, None if p == "none" else p)] for p in cols)}


def band_block(oracle: Sequence, pred: Sequence, labels: Sequence[str] = BANDS) -> dict:
    """Coverage, accuracy (unknown counted wrong, and among covered), confusion."""
    pairs = [(o, p) for o, p in zip(oracle, pred) if o is not None and not pd.isna(o)]
    n = len(pairs)
    covered = [(o, p) for o, p in pairs if p is not None]
    correct = sum(o == p for o, p in covered)
    order = {b: i for i, b in enumerate(labels)}
    within = sum(abs(order[o] - order[p]) <= 1 for o, p in covered)
    return {"n": n, "coverage": wilson(len(covered), n),
            "accuracy_all": wilson(correct, n),
            "accuracy_covered": wilson(correct, len(covered)),
            "within_one_covered": wilson(within, len(covered)),
            "confusion_oracle_by_extracted": _confusion([o for o, _ in pairs],
                                                        [p for _, p in pairs], labels)}


def jd_bands(texts: Sequence[str], aggregate: str) -> list[str | None]:
    return [ex.jd_band(ex.jd_experience(t, aggregate).min_years) for t in texts]


def cv_extract(texts: Sequence[str]) -> list[ex.CVExperience]:
    return [ex.cv_experience(t) for t in texts]


def seniority_block(positions: Sequence, texts: Sequence[str], years: Sequence,
                    pick: str) -> dict:
    """Accuracy against `Position` seniority tokens, on documents that carry one."""
    gt = [ex.position_seniority(p) for p in positions]
    rows = [(g, *ex.text_seniority(t, y, pick)) for g, t, y in zip(gt, texts, years)
            if g is not None]
    n = len(rows)
    token = [(g, lvl) for g, lvl, src in rows if src == "token"]
    fallback = [(g, lvl) for g, lvl, src in rows if src == "years"]
    return {"documents": len(gt), "n_with_position_token": n,
            "position_token_rate": wilson(n, len(gt)),
            "token_coverage": wilson(len(token), n),
            "token_accuracy": wilson(sum(g == p for g, p in token), len(token)),
            "years_fallback_n": len(fallback),
            "years_fallback_accuracy": wilson(sum(g == p for g, p in fallback), len(fallback)),
            "coverage_with_fallback": wilson(len(token) + len(fallback), n),
            "accuracy_all": wilson(sum(g == p for g, p, _ in rows), n),
            "confusion_position_by_extracted": _confusion(
                [g for g, _, _ in rows], [p for _, p, _ in rows], ex.LEVELS)}


def _source_block(oracle: Sequence, cvx: Sequence[ex.CVExperience]) -> dict:
    out = {}
    for src in ("statement", "ranges"):
        rows = [(o, ex.cv_band(c.years)) for o, c in zip(oracle, cvx) if c.source == src]
        out[src] = {"n": len(rows), "accuracy": wilson(sum(o == p for o, p in rows), len(rows))}
    return out


def _pick(blocks: dict, default: str, key: str) -> str:
    best = max(blocks.values(), key=lambda b: b[key]["share"] or 0.0)[key]["share"]
    return default if blocks[default][key]["share"] == best else next(
        k for k, b in blocks.items() if b[key]["share"] == best)


# --- the two document sets -----------------------------------------------------------

def evaluate_train(jd: pd.DataFrame, cv: pd.DataFrame) -> tuple[dict, dict]:
    jd_texts, cv_texts = list(jd.jd_text), list(cv.cv_text)
    jd_var = {agg: band_block(list(jd.exp_band), jd_bands(jd_texts, agg))
              for agg in ("max", "first")}
    jd_agg = _pick(jd_var, DEFAULTS["jd_aggregate"], "accuracy_all")
    cvx = cv_extract(cv_texts)
    jd_years = [ex.jd_experience(t, jd_agg).min_years for t in jd_texts]
    cv_years = [c.years for c in cvx]
    sen = {side: {pick: seniority_block(list(df.Position), texts, yrs, pick)
                  for pick in ("first", "max")}
           for side, df, texts, yrs in (("jd", jd, jd_texts, jd_years),
                                        ("cv", cv, cv_texts, cv_years))}
    picks = {side: _pick(v, DEFAULTS["seniority_pick"], "accuracy_all") for side, v in sen.items()}
    chosen = {"jd_aggregate": jd_agg, "seniority_pick_jd": picks["jd"],
              "seniority_pick_cv": picks["cv"]}
    block = {
        "n_jd": len(jd), "n_cv": len(cv),
        "jd_years": {"variants": jd_var, "selected": jd_agg},
        "cv_years": {**band_block(list(cv.exp_band), [ex.cv_band(y) for y in cv_years]),
                     "by_source": _source_block(list(cv.exp_band), cvx)},
        "seniority": {side: {"variants": v, "selected": picks[side]} for side, v in sen.items()},
    }
    return block, chosen


def pair_agreement(frame: pd.DataFrame, seed: int = SEED) -> dict:
    """Extracted vs oracle `experience_compatible`, per scope, CI over JDs."""
    from candidate_screener.evaluation.resample import (binary_agreement, binary_kappa,
                                                        pair_stat_ci)
    out = {}
    for scope in ("generic", "targeted", "total"):
        f = frame if scope == "total" else frame[frame.stratum == scope]
        a, b = f.extracted.to_numpy(), f.oracle.to_numpy()
        out[scope] = {
            "n_pairs": int(len(f)), "n_jds": int(f.jd_id.nunique()),
            "agreement": round(binary_agreement(a, b), 4),
            "agreement_ci95": [round(x, 4) for x in pair_stat_ci(
                f, binary_agreement, ["extracted", "oracle"], n_boot=N_BOOT, seed=seed)],
            "kappa": round(binary_kappa(a, b), 4),
            "kappa_ci95": [round(x, 4) for x in pair_stat_ci(
                f, binary_kappa, ["extracted", "oracle"], n_boot=N_BOOT, seed=seed)],
            "compatible_rate_oracle": round(float(b.mean()), 4),
            "compatible_rate_extracted": round(float(a.mean()), 4),
            "extracted_1_oracle_0": int(((a == 1) & (b == 0)).sum()),
            "extracted_0_oracle_1": int(((a == 0) & (b == 1)).sum()),
            "both_bands_extracted": int(f.both_known.sum()),
            "agreement_where_both_extracted": (
                round(binary_agreement(a[f.both_known.to_numpy()], b[f.both_known.to_numpy()]), 4)
                if f.both_known.any() else None),
        }
    return out


def evaluate_final(pairs: pd.DataFrame, chosen: dict, seed: int = SEED) -> dict:
    from candidate_screener.matching.corpus import experience_compatible
    jd = pairs.drop_duplicates("jd_id")
    cv = pairs.drop_duplicates("cv_id")
    jd_years = {i: ex.jd_experience(t, chosen["jd_aggregate"]).min_years
                for i, t in zip(jd.jd_id, jd.jd_text)}
    cv_years = {i: ex.cv_experience(t).years for i, t in zip(cv.cv_id, cv.cv_text)}
    frame = pd.DataFrame({
        "jd_id": pairs.jd_id, "stratum": pairs.stratum,
        "both_known": (pairs.jd_id.map(lambda i: jd_years[i] is not None)
                       & pairs.cv_id.map(lambda i: cv_years[i] is not None)).to_numpy(),
        "oracle": experience_compatible(pairs).to_numpy(),
        "extracted": experience_compatible(pd.DataFrame({
            "jd_exp_band": pairs.jd_id.map(lambda i: ex.jd_band(jd_years[i])),
            "cv_exp_band": pairs.cv_id.map(lambda i: ex.cv_band(cv_years[i]))})).to_numpy(),
    })
    # The public helper must give the same column (it is what a rung will call).
    helper = ex.experience_compatible_extracted(list(pairs.jd_text), list(pairs.cv_text))
    if chosen["jd_aggregate"] == "max" and not np.array_equal(helper.to_numpy(),
                                                              frame.extracted.to_numpy()):
        raise AssertionError("experience_compatible_extracted disagrees with the evaluation")
    return {
        "n_jd": int(len(jd)), "n_cv": int(len(cv)), "n_pairs": int(len(pairs)),
        "jd_years": band_block(list(jd.jd_exp_band), [ex.jd_band(jd_years[i]) for i in jd.jd_id]),
        "cv_years": band_block(list(cv.cv_exp_band), [ex.cv_band(cv_years[i]) for i in cv.cv_id]),
        "experience_compatible_pairs": pair_agreement(frame, seed),
        "seniority": {
            "jd": seniority_block(list(jd.jd_position), list(jd.jd_text),
                                  [jd_years[i] for i in jd.jd_id], chosen["seniority_pick_jd"]),
            "cv": seniority_block(list(cv.cv_position), list(cv.cv_text),
                                  [cv_years[i] for i in cv.cv_id], chosen["seniority_pick_cv"]),
        },
    }


def build_record(seed: int = SEED, n: int = TRAIN_N) -> dict:
    from candidate_screener.matching.corpus import a2_pairs, a2_train_documents
    jd, cv = a2_train_documents(n, n, seed, "v2")
    train, chosen = evaluate_train(jd, cv)
    return {
        "plan": "plan/2026-10-06-extraction-and-rerank (P4, tasks 4.1-4.4)",
        "text_variant": "v2 (CV text without Position; JD text = Long Description)",
        "reference_date": ex.REFERENCE_DATE.isoformat(),
        "rules": "src/candidate_screener/extraction/experience.py (module docstring)",
        "jd_years_to_exp_years": "<1 no_exp, <2 1y, <3 2y, <5 3y, else 5y; then JD_EXP_TO_BAND",
        "cv_years_to_band": "cv_experience_band",
        "unknown_band": "experience_compatible = 0 (the rung-1a rule, unchanged)",
        "seniority_levels": list(ex.LEVELS),
        "seniority_years_fallback": "<2 junior, <5 middle, else senior",
        "selection": {"rule": ("variants compared on A2 train only, by accuracy_all; ties keep "
                               "the default"), "defaults": DEFAULTS, "chosen": chosen},
        "intervals": ("documents: Wilson 95%; pairs: percentile bootstrap over JDs, "
                      f"{N_BOOT} resamples, seed {seed}"),
        "a2_train": {"region": "train (D19)", "seed": seed, **train},
        "final_set_documents": {
            "note": "measurement only: no final-set label is read; oracle = the true fields",
            **evaluate_final(a2_pairs("v2"), chosen, seed)},
    }


def render(record: dict) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evaluate", action="store_true", help="build and write the record")
    mode.add_argument("--check", action="store_true", help="rebuild and diff the record")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=TRAIN_N, help="A2-train documents per side")
    a = ap.parse_args(argv)
    fresh = render(build_record(a.seed, a.n))
    if a.check:
        ok = RECORD.exists() and RECORD.read_text(encoding="utf-8") == fresh
        print(f"experience metrics {'OK' if ok else 'MISMATCH'} ({RECORD.name})")
        return 0 if ok else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(fresh, encoding="utf-8")
    print(f"wrote {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
