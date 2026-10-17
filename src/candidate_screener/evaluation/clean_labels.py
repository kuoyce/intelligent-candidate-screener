"""Paired comparisons missing from the stage-1 and stage-2 records, on all pairs and `llm_free`.

Plan 2026-10-17-ladder-on-clean-labels. Reporting only, decides nothing, `exploratory` (none of
these was registered).

1. **X4 vs X2** as AUC (Fit / No Fit), as stage 1 scores them.
2. **The stage-2 rerankers vs X2** as orderings on the human grades (D52), as stage 2 scores
   them: R1, R2 (cross-encoders), R3 (bi-encoder), plus SBERT weak and X4 for context.

Every system's score digest is checked against the committed record it comes from, so this
cannot drift from the frozen rungs. The rerankers load their cached weights; nothing trains.

    uv run python -m candidate_screener.evaluation.clean_labels
"""
from __future__ import annotations

import argparse
import json

from candidate_screener.config import OUTPUT

RECORD = OUTPUT / "matching" / "ladder-clean-labels.json"
X2, X4 = "extracted rule + TF-IDF", "extracted rule + SBERT (weak fine-tuned)"


def run(seed: int = 0) -> dict:
    from candidate_screener.evaluation import final_set, indomain, stage1
    from candidate_screener.matching import corpus
    lab = indomain.labels(final_set.load())
    pairs = corpus.a2_pairs(text_variant=stage1.TEXT_VARIANT).assign(keyword_match=lambda d: corpus.keyword_match(d))
    pairs = pairs.set_index("pair_id").loc[lab.index].reset_index()
    committed = json.loads(stage1.RECORD.read_text())["systems"]
    specs = {s.factory().name: s for s in stage1.specs()}
    scores = {}
    for name in (X2, X4):
        system = specs[name].factory()
        system.fit(None)
        scores[name] = system.score(pairs)
        if stage1._sha(scores[name]) != committed[name]["scores_sha256"]:
            raise ValueError(f"{name}: scores differ from the committed stage-1 record")
    deltas = [indomain.paired_delta(scores[X4], scores[X2], lab, pairs, "auc", scope, subset, seed)
              for subset in indomain.SUBSETS for scope in final_set.SCOPES]
    return {"comparison": f"{X4} vs {X2}", "b": X4, "a": X2, "seed": seed, "decides": "nothing",
            "deltas": deltas, "ordering_vs_x2": ordering_vs_x2(specs, seed)}


SBERT_WEAK = "SBERT (weak fine-tuned)"
ORDER_REFS, ORDER_SUBSETS = ("yc", "mean"), ("full", "llm_free")


def ordering_vs_x2(specs: dict, seed: int = 0) -> dict:
    """R1-R3, SBERT weak and X4 minus X2, per-JD pairwise and nDCG on the human grades."""
    from candidate_screener.evaluation import final_set, ordering, stage1, stage2
    from candidate_screener.matching import corpus, neural
    from candidate_screener.rerank import batch, models
    frame = final_set.load()
    grades = ordering.grade_frame(frame)
    pairs = corpus.a2_pairs(text_variant=stage2.TEXT_VARIANT).set_index("pair_id").loc[frame.pair_id].reset_index()
    s2 = json.loads(stage2.RECORD.read_text())
    s1 = json.loads(stage1.RECORD.read_text())["systems"]
    digest = neural.file_digest(batch.SCORES_CSV)
    scores = {}
    for name in (X2, X4, SBERT_WEAK):
        system = specs[name].factory()
        system.fit(None)
        scores[name] = system.score(pairs)
    if ordering._digest(scores[SBERT_WEAK]) != s2["systems"][SBERT_WEAK]["scores_sha256"]:
        raise ValueError("SBERT weak: scores differ from the committed stage-2 record")
    for r in (models.R1, models.R2, models.R3):
        config = models.grid(r)[s2["batch"]["rungs"][r]["selected"]]
        path = models.weights_dir(config, digest)
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"{r}: no cached weights at {path}; this module never trains")
        system = models.system_for(config, path)
        system.fit(None)
        scores[r] = system.score(pairs)
        del system
        models._release()
        if ordering._digest(scores[r]) != s2["systems"][r]["scores_sha256"]:
            raise ValueError(f"{r}: scores differ from the committed stage-2 record")
    figures = {name: ordering.figures(name, scores[name], grades, seed) for name in (X2, X4)}
    deltas = [{"b": b, "a": X2, **ordering.paired_delta(scores[b], scores[X2], grades, ref, metric, scope, subset, seed)}
              for b in (models.R1, models.R2, models.R3, SBERT_WEAK, X4) for ref in ORDER_REFS
              for subset in ORDER_SUBSETS for metric in ordering.METRICS for scope in final_set.SCOPES]
    return {"names": {r: models.NAMES[r] for r in (models.R1, models.R2, models.R3)},
            "x2_x4_figures": figures, "deltas": deltas}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    result = run(ap.parse_args(argv).seed)
    RECORD.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(RECORD)


if __name__ == "__main__":
    main()
