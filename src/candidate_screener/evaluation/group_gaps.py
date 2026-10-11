"""Performance gaps across groups for the selected pipeline (plan 04 §2; no target).

Djinni is pre-anonymised: there are no demographic fields, so the groups available are
the JD's experience band and the stratum. Role family (`Primary Keyword`) spreads 60 JDs
over 30+ families, too few per family to estimate anything, so it is reported as counts
only. Metric: per-JD AUC against the final set, CIs resampling JDs.

Usage:
    uv run python -m candidate_screener.evaluation.group_gaps
"""
from __future__ import annotations

import json

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import final_set, indomain, resample

RECORD = OUTPUT / "matching" / "group-gaps.json"


def compute(seed: int = 0) -> dict:
    from candidate_screener import pipeline
    from candidate_screener.matching import corpus
    lab = indomain.labels()
    pairs = corpus.a2_pairs().set_index("pair_id").loc[lab.index].reset_index()
    ranker = pipeline.default_ranker(True)
    ranker.fit(None)
    scores = ranker.score(pairs)
    auc = indomain.per_jd_auc(lab.y, scores.reindex(lab.index), lab.jd_id)
    jd_meta = pairs.drop_duplicates("jd_id").set_index("jd_id")
    out = {"pipeline": pipeline.SELECTED, "reference_rule_version": final_set.RULE_VERSION,
           "note": ("No target (plan 04 §2). No demographic fields exist in Djinni; groups are "
                    "the JD's experience band and the stratum."),
           "by": {}}
    for group, col in (("jd_experience_band", "jd_exp_band"), ("stratum", "stratum")):
        block = {}
        for value, jds in jd_meta.groupby(col).groups.items():
            v = auc.reindex(list(jds)).dropna()
            if len(v) >= 3:
                low, high = resample.mean_ci(v, seed=seed)
                block[str(value)] = {"auc": round(float(v.mean()), 6),
                                     "ci": [round(low, 6), round(high, 6)], "n_jd": int(len(v))}
            else:
                block[str(value)] = {"n_jd": int(len(v)), "note": "fewer than 3 scoreable JDs"}
        out["by"][group] = block
    out["role_family_counts"] = jd_meta.jd_keyword.value_counts().to_dict()
    return out


def main() -> int:
    rec = compute()
    RECORD.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rec["by"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
