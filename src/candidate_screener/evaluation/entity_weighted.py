"""Bounded train-region entity pilot; blocked dependencies never become synthetic scores."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.evaluation import dev_pools, resample
from candidate_screener.evaluation.shortlist_dev import recall_per_jd

RECORD = OUTPUT / "matching" / "entity-weighted-dev.json"
ALPHAS = (0., .25, .5, .75, 1.)
CONFIG = {"version": "entity-weighted-pilot-v1", "alphas": ALPHAS,
          "queries_per_split": 2, "pool_size": 500, "seed": 0,
          "relevance": "same oracle role family, not human fit",
          "selection": "dev-select Recall@50; earlier alpha wins six-decimal tie",
          "adoption": "never from four-query pilot", "idf_cap": 5,
          "idf_population": "first 2000 sorted train CV ids excluding all dev CVs",
          "stage2": "same baseline top 50; denominator full pool"}


def select_queries(frame):
    """Choose query IDs before scoring and retain complete registered pools."""
    chosen = []
    for split in dev_pools.SPLITS:
        ids = sorted(frame.loc[frame.split.eq(split), "jd_id"].unique())[:2]
        if len(ids) != 2:
            raise ValueError("Pilot requires two queries per development split")
        chosen.extend(ids)
    selected = frame[frame.jd_id.isin(chosen)].copy().reset_index(drop=True)
    counts = selected.groupby("jd_id").cv_id.nunique()
    if not counts.eq(500).all() or selected.pair_id.duplicated().any():
        raise ValueError("Pilot requires complete unique 500-CV pools")
    return selected


def choose_alpha(figures):
    """Earlier registered alpha wins a six-decimal primary-metric tie."""
    if tuple(figures) != ALPHAS:
        raise ValueError("Selection requires complete ordered registered alpha grid")
    return max(ALPHAS, key=lambda a: (round(figures[a]["recall@50"], 6), -a))


def fixed_shortlist_recall(frame, baseline, candidate, relevance, shortlist=50, k=10):
    """Rerank baseline shortlist, retaining relevant CVs outside it in denominator."""
    ranking = frame[["jd_id", "pair_id"]].copy()
    ranking["baseline"] = baseline
    ranking["candidate"] = candidate
    ranking["relevant"] = relevance
    ranking = ranking.sort_values(["jd_id", "baseline", "pair_id"], ascending=[True, False, True])
    top = ranking[ranking.groupby("jd_id").cumcount() < shortlist]
    top = top.sort_values(["jd_id", "candidate", "pair_id"], ascending=[True, False, True])
    hit = top[top.groupby("jd_id").cumcount() < k].groupby("jd_id").relevant.sum()
    total = ranking.groupby("jd_id").relevant.sum()
    valid = total[total > 0]
    return (hit.reindex(valid.index).fillna(0) / valid).sort_index()


def summarize(values):
    lo, hi = resample.mean_ci(values, seed=0)
    return {"mean": round(float(values.mean()), 6), "ci95": [round(lo, 6), round(hi, 6)],
            "n_jd": len(values)}


def evaluate_pilot(dev, train_cv, model_factory):
    """Injectable runner; labels remain separate from model inference frames."""
    frame = select_queries(dev_pools.pairs(dev))
    relevance = dev_pools.labels(dev, frame).to_numpy()
    baseline_model = model_factory(0., False, True)
    baseline_model.fit(train_cv)
    baseline = baseline_model.score(frame).to_numpy()
    records, primary = {}, {}
    for weighted in (False, True):
        for alpha in ALPHAS:
            model = model_factory(alpha, True, weighted)
            model.fit(train_cv)
            scores = model.score(frame).to_numpy()
            metrics = {}
            for split in dev_pools.SPLITS:
                keep = frame.split.eq(split).to_numpy()
                x, r = frame[keep], relevance[keep]
                metrics[split] = {}
                for k in (10, 50):
                    v = recall_per_jd(x, scores[keep], r, k)
                    b = recall_per_jd(x, baseline[keep], r, k)
                    metrics[split][f"recall@{k}"] = summarize(v)
                    metrics[split][f"delta_vs_original_x2@{k}"] = summarize(v - b)
                metrics[split]["fixed_top50_recall@10"] = summarize(
                    fixed_shortlist_recall(x, baseline[keep], scores[keep], r))
            key = f"title_{'idf' if weighted else 'unweighted'}_alpha{alpha:g}"
            records[key] = metrics
            if weighted:
                primary[alpha] = {"recall@50": metrics["dev-select"]["recall@50"]["mean"]}
    return {"status": "pilot_complete", "queries": sorted(frame.jd_id.unique()),
            "metrics": records, "selected_alpha_exploratory": choose_alpha(primary),
            "adopted": False, "strata": "development proxy splits; not generic/targeted final strata",
            "gliner_comparison": "not run: complete registered pool extraction cache required"}


def prerequisites():
    from candidate_screener.extraction import role_family
    from candidate_screener.extraction.rules import VOCAB_CSV
    from candidate_screener.data import weak_pairs
    paths = {"role_latest": role_family.CACHE / "LATEST", "vocabulary": VOCAB_CSV,
             "weak_pairs": weak_pairs.MANIFEST}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if not missing:
        fp = paths["role_latest"].read_text().strip()
        paths["role_weights"] = role_family.CACHE / f"role-model-{fp}.pkl"
        key = dev_pools._key(fp)
        for name in ("jd", "cv", "pools"):
            paths[f"dev_{name}"] = dev_pools.CACHE / f"{name}-{key}.parquet"
        missing = [name for name, path in paths.items() if not path.is_file()]
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest()
               for name, path in paths.items() if path.is_file()}
    from pathlib import Path
    from candidate_screener.matching import entity_weighted as matching_module
    for name, path in {"runner_code": Path(__file__), "matching_code": Path(matching_module.__file__)}.items():
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return paths, missing, digests


def report():
    paths, missing, digests = prerequisites()
    result = {"config": json.loads(json.dumps(CONFIG)), "dependency_sha256": digests,
              "adopted": False}
    if missing:
        return {**result, "status": "blocked", "missing_prerequisites": missing,
                "metrics": {}, "reason": "Inherited role model and keyed dev-pool prerequisite restoration pending",
                "resume": "uv run python -m candidate_screener.evaluation.entity_weighted"}
    from candidate_screener.extraction import role_family
    from candidate_screener.matching.entity_weighted import ConceptExtractor, EntityWeightedX2
    from candidate_screener.matching.extracted import ExtractedRule
    dev = {n: pd.read_parquet(paths[f"dev_{n}"]) for n in ("jd", "cv", "pools")}
    # IDF population is independently defined by train-region loader, excluding
    # every development document (both splits), before extracting concepts.
    train = role_family.load_train_documents()["cv"]
    train = (train[~train.id.isin(dev["cv"].id)].sort_values("id").head(2000)
             .loc[:, ["id", "text"]].assign(region="train", doc_type="cv"))
    ex, role = ConceptExtractor(), role_family.load_role_model()
    def factory(alpha, title, weighted):
        return EntityWeightedX2(alpha, title, weighted, extractor=ex,
                                rule=ExtractedRule(role_model=role))
    return {**result, **evaluate_pilot(dev, train, factory)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--warm", action="store_true", help="Accepted registry tier; no training occurs")
    args = parser.parse_args(argv)
    actual = report()
    if args.check:
        if not RECORD.is_file() or json.loads(RECORD.read_text()) != actual:
            raise SystemExit("Entity pilot record differs or is missing")
        print(f"PASS entity pilot integrity ({actual['status']})")
    else:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        print(f"Entity pilot: {actual['status']}; {RECORD}")


if __name__ == "__main__":
    main()
