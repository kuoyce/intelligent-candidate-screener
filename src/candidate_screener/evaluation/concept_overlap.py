"""Stage 1 of plan 2026-10-13-gliner-concepts-in-learned-weights: does a tagger repair overlap?

Label-free: it reads document text only. For each concept source it reports, on a pair frame,
the share of pairs with no JD-CV concept overlap (the gate statistic), the share of JDs with
<= 3 concepts, and the share of CVs with none. The gate (owner-set, 13 Oct 2026): some source
other than the seed rules must cut the no-overlap share by >= 5 percentage points.

    uv run python -m candidate_screener.evaluation.concept_overlap --extract   # fill the caches
    uv run python -m candidate_screener.evaluation.concept_overlap --stage1    # stats + gate
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.extraction import concept_sources as cs

RECORD = OUTPUT / "extraction" / "concept-overlap.json"
GATE_PP = 5.0


def overlap_stats(extractor, pairs: pd.DataFrame) -> dict:
    """Gate statistics for one concept source on a pair frame (`jd_id`, `cv_id`, texts)."""
    jd = {j: extractor.concepts(t, "jd") for j, t in pairs.drop_duplicates("jd_id")[["jd_id", "jd_text"]].itertuples(index=False)}
    cv = {c: extractor.concepts(t, "cv") for c, t in pairs.drop_duplicates("cv_id")[["cv_id", "cv_text"]].itertuples(index=False)}
    shared = np.array([len(jd[j] & cv[c]) for j, c in zip(pairs.jd_id, pairs.cv_id)])
    nj, nc = np.array([len(v) for v in jd.values()]), np.array([len(v) for v in cv.values()])
    return {"pairs": len(pairs), "jds": len(jd), "cvs": len(cv),
            "no_overlap_share": float((shared == 0).mean()),
            "jd_le3_share": float((nj <= 3).mean()), "jd_zero_share": float((nj == 0).mean()),
            "cv_zero_share": float((nc == 0).mean()),
            "median_jd_concepts": float(np.median(nj)), "median_cv_concepts": float(np.median(nc))}


def jaccard(a, b, pairs: pd.DataFrame, side: str = "cv") -> float:
    """Mean per-document Jaccard of the concept sets of two sources."""
    texts = pairs.drop_duplicates(f"{side}_id")[f"{side}_text"]
    vals = []
    for t in texts:
        x, y = a.concepts(t, side), b.concepts(t, side)
        if x | y:
            vals.append(len(x & y) / len(x | y))
    return float(np.mean(vals)) if vals else float("nan")


def gate(stats_by_arm: dict[str, dict]) -> dict:
    """Arms other than R that cut the no-overlap share by >= GATE_PP points versus R."""
    base = stats_by_arm["R"]["no_overlap_share"]
    cut = {a: 100 * (base - s["no_overlap_share"]) for a, s in stats_by_arm.items() if a != "R"}
    return {"base_no_overlap_share": base, "cut_pp": cut,
            "passing": sorted(a for a, v in cut.items() if v >= GATE_PP)}


def pair_sets() -> dict[str, pd.DataFrame]:
    """Pairs whose text the plan extracts: A2 final (text only), A1 LLM-labelled 639, D50."""
    from candidate_screener.evaluation import a1_weights as aw
    from candidate_screener.matching import corpus
    a2 = corpus.a2_pairs(text_variant="v2")
    llm = aw.llm_labels()
    a1 = corpus.a1_pairs("test").drop_duplicates("pair_id").set_index("pair_id", drop=False).loc[llm.index]
    d50, _ = aw.d50_pairs()
    return {"A2 final": a2, "A1 LLM-labelled": a1, "D50": d50}


def idf_documents() -> pd.DataFrame:
    """The CV population `a1_weights.build_scorer` fits IDF on (train sample minus dev-pool CVs)."""
    from candidate_screener.evaluation import dev_pools
    from candidate_screener.matching import corpus
    _, cv = corpus.a2_train_documents(text_variant="v2")
    dev = dev_pools.build()
    cv = cv[~cv.id.isin(dev["cv"].id)]
    return pd.DataFrame({"cv_id": cv.id, "cv_text": cv.cv_text})


def open_sources(which=cs.SOURCES) -> dict[str, cs.CachedEntities]:
    out = {}
    if "gliner" in which:
        from candidate_screener.extraction.model_assets import MODEL_DIR
        from candidate_screener.extraction.transformer import load_model
        out["gliner"] = cs.CachedEntities("gliner", cs.gliner_rows(load_model(MODEL_DIR)))
    if "minilm" in which:
        from candidate_screener.extraction import ner
        out["minilm"] = cs.CachedEntities("minilm", cs.minilm_rows(ner.load_ner()))
    return out


def extract(sets: dict[str, pd.DataFrame], sources: dict[str, cs.CachedEntities],
            sides=("jd", "cv")) -> None:
    """Fill the caches for every distinct document of every set."""
    docs = {}
    for p in sets.values():
        for side in sides:
            for t in p[f"{side}_text"].drop_duplicates():
                docs.setdefault(cs.text_key(t), (t, side))
    print(f"{len(docs)} distinct documents", flush=True)
    for name, src in sources.items():
        for i, (t, side) in enumerate(docs.values(), 1):
            src(t, side)
            if i % 250 == 0:
                src.save(); print(f"{name}: {i}/{len(docs)}", flush=True)
        src.save()


def stage1(sets: dict[str, pd.DataFrame]) -> dict:
    arms = cs.make_extractors({n: cs.CachedEntities(n, None) for n in cs.SOURCES})
    out = {"gate_pp": GATE_PP, "sets": {}}
    for name, pairs in sets.items():
        stats = {a: overlap_stats(e, pairs) for a, e in arms.items()}
        out["sets"][name] = {"stats": stats, "gate": gate(stats),
                             "jaccard_G_N_cv": jaccard(arms["G"], arms["N"], pairs),
                             "jaccard_G_N_jd": jaccard(arms["G"], arms["N"], pairs, "jd")}
    out["gate_passed_on_a2_final"] = out["sets"]["A2 final"]["gate"]["passing"]
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extract", action="store_true")
    p.add_argument("--extract-idf", nargs="+", choices=cs.SOURCES, metavar="SOURCE",
                   help="fill the caches for the IDF population and A1 train (Stage 2 prerequisites)")
    p.add_argument("--stage1", action="store_true")
    a = p.parse_args(argv)
    if a.extract_idf:
        from candidate_screener.matching import corpus
        extract({"idf": idf_documents().assign(jd_text="", jd_id=""),
                 "a1 train": corpus.a1_pairs("train")}, open_sources(a.extract_idf),
                sides=("jd", "cv"))
        return
    sets = pair_sets()
    if a.extract:
        extract(sets, open_sources())
    if a.stage1:
        result = stage1(sets)
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result["gate_passed_on_a2_final"]), RECORD)


if __name__ == "__main__":
    main()
