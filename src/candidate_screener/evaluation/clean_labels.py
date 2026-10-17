"""The one paired comparison missing from the stage-1 record: X4 vs X2, on `all` and `llm_free`.

Plan 2026-10-17-ladder-on-clean-labels. Reporting only, decides nothing. Both systems are scored
as stage 1 scores them; their score digests are checked against `stage1-metrics.json`, so this
cannot drift from the committed rungs.

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
            "deltas": deltas}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    result = run(ap.parse_args(argv).seed)
    RECORD.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(RECORD)


if __name__ == "__main__":
    main()
