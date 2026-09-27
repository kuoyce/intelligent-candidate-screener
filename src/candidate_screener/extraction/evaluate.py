"""The extraction record `output/extraction/extraction-metrics.json` (task 5.5).

Rebuilds every P5 figure from raw data plus the committed vocabulary:

- the vocabulary's digest and the filter tuning (committed constants must equal a re-tune);
- rules vs learned tagger, strict / partial span P/R/F1 with *n* and a bootstrap CI, on
  SkillSpan test (SKILL union headline; skill and knowledge layers separately) and on
  DataTurks `Skills` (the 20 shipped test documents, and all 220: neither system is
  fitted on DataTurks, so every document is out-of-sample);
- normalisation top-1 accuracy, string matching vs SBERT.

Torch-derived figures (paths through `ner` or `sbert`) are compared under
`TOLERANCE`; everything else must reproduce exactly. No timestamps or timings.

    uv run python -m candidate_screener.extraction.evaluate --seed 0
    uv run python -m candidate_screener.extraction.evaluate --seed 0 --check [--reuse-weights]
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from candidate_screener.config import OUTPUT, PROCESSED

RECORD = OUTPUT / "extraction" / "extraction-metrics.json"
CHECK_DIR = PROCESSED / "extraction" / "ner-check"

#: Absolute tolerance on torch-derived figures under `--check`, set from measurement
#: (26 Sep 2026), not preference: two full trainings at 4 threads, under very different
#: CPU contention, gave bit-identical weights (state sha256 `4977c408…`, identical
#: `model.safetensors`), and inference at 1, 4 and 8 threads gave identical tags on all
#: 3,569 SkillSpan test sentences. Every figure is a ratio of integer counts rounded to 4
#: places, so any real change exceeds 1e-4 and a non-zero tolerance would only hide it.
#: Not covered: another CPU or torch build. There, run `--check --reuse-weights` first
#: to separate an inference difference from a training one.
TOLERANCE = 0.0
DETERMINISM = {
    "tolerance_abs": TOLERANCE,
    "applies_to": "figures under a path containing 'ner', 'sbert' or 'ner_minus_rules'",
    "measured": ("two full trainings (seed 0, 4 threads) produced bit-identical weights; "
                 "tagger inference at 1, 4 and 8 threads produced identical tags on all "
                 "SkillSpan test sentences"),
    "training_threads_pinned": 4,
    "not_covered": "a different CPU or torch build; bit-identity is only measured on one machine",
}
TORCH_SEGMENTS = ("ner", "sbert", "ner_minus_rules")
#: The proposal's extraction target, restated as strict span F1 on SkillSpan test (S3,
#: 04-success-measures.md §3). S3 is still open, so the record reports against it
#: without claiming it was adopted.
S3_BAR = 0.85
INFORMATIONAL = ("weights_sha256",)
DATATURKS_NOTE = ("20 test documents decide nothing: the `test` block is the planned figure "
                  "and is reported for completeness; `all` (220 documents) is the one with "
                  "power. Gold `Skills` spans are often whole skills sections, so strict F1 "
                  "is bounded far below 1 for any skill-level extractor, and skills named "
                  "outside an annotated span count as false positives.")
SKILLSPAN_NOTE = ("Token-level: predictions are snapped to SkillSpan tokens. SKILL = skill ∪ "
                  "knowledge; union gold keeps both layers, which overlap in ~2% of spans, so "
                  "a non-overlapping system's strict recall ceiling is slightly below 1. Neither "
                  "system types its spans, so in the per-layer blocks (`skill`, `knowledge`) "
                  "the other layer's correct spans count as false positives: read per-layer "
                  "recall, not per-layer precision. The `validation` block was seen during "
                  "selection (filter tuning, early stopping) and is not a held-out figure.")


def _vocab_block(src) -> dict:
    import hashlib

    from candidate_screener.extraction import vocab
    df = vocab.build(src)
    built = hashlib.sha256(vocab.render(df)).hexdigest()
    on_disk = vocab.digest() if vocab.VOCAB_CSV.is_file() else None
    if built != on_disk:
        raise SystemExit(f"data/vocab/skills.csv ({on_disk}) is not what the build produces "
                         f"({built}); run `extraction.vocab --build` first")
    return {
        "path": "data/vocab/skills.csv", "sha256": built, **vocab.stats(df),
        "esco_scope": ("ESCO v1.2.1 skills in the digital skills collection, or below skill "
                       "group S5 'working with computers', or below knowledge field 06 "
                       "'information and communication technologies'"),
        "esco_scope_concepts": len(vocab.esco_scope(src)),
        "filters": {"min_chars": vocab.FILTERS.min_chars, "max_tokens": vocab.FILTERS.max_tokens,
                    "stopwords": vocab.FILTERS.stopwords,
                    "blacklist_rule": list(vocab.BLACKLIST_RULE) if vocab.BLACKLIST_RULE else None,
                    "blacklist": list(vocab.FILTERS.blacklist)},
        "esco_homonyms_withheld": list(vocab.ESCO_HOMONYMS),
    }


def _tuning_block(src) -> dict:
    from candidate_screener.extraction import vocab
    t = vocab.tune(src, verbose=False)
    ok = vocab.committed_matches_tuning(t)
    if not ok:
        raise SystemExit("vocab.FILTERS / BLACKLIST_RULE differ from what tune() selects; "
                         "re-run `extraction.vocab --tune` and update the constants")
    return {k: v for k, v in t.items() if not k.startswith("_")} | {"committed_matches": ok}


def _skillspan_block(matcher, tagger, seed: int) -> dict:
    from candidate_screener.extraction import ner, skillspan as ss, vocab
    from candidate_screener.extraction.metrics import paired_delta, score
    out: dict = {"note": SKILLSPAN_NOTE}
    for split, scopes in (("test", ss.SCOPES), ("validation", ("union",))):
        rc = {s: vocab.rules_counts(matcher, split, s) for s in scopes}
        tags = ner.skillspan_tags(tagger, split)
        nc = {s: ner.counts_from_tags(tags, split, s) for s in scopes}
        out[split] = {"n_sentences": int(len(ss.load(split))),
                      "rules": {s: score(rc[s], seed=seed) for s in scopes},
                      "ner": {s: score(nc[s], seed=seed) for s in scopes},
                      "ner_minus_rules": {s: paired_delta(rc[s], nc[s], seed=seed)
                                          for s in scopes}}
    out["s3_reference_bar"] = {
        "strict_f1_union": S3_BAR, "status": "S3 open (04-success-measures.md §3)",
        "rules_meets": out["test"]["rules"]["union"]["strict"]["f1"] >= S3_BAR,
        "ner_meets": out["test"]["ner"]["union"]["strict"]["f1"] >= S3_BAR,
        "published_reference": "to be verified against Zhang et al., NAACL 2022 (not quoted "
                               "from memory; no network access in this phase)"}
    return out


def _dataturks_block(matcher, tagger, seed: int) -> dict:
    from candidate_screener.data import dataturks as dt
    from candidate_screener.extraction.metrics import paired_delta, score, unit_counts
    docs, ledger = dt.load()
    ledger = {k: int(ledger.get(k, 0)) for k in (
        "spans_total", "intact", "unescape", "search", "dropped_unrepairable",
        "dropped_pii", "dropped_unlabelled", "malformed_lines")}
    rules_c, ner_c = [], []
    for d in docs:
        gold = d.label_spans(dt.SKILLS)
        rules_c.append(unit_counts([(m.start, m.end) for m in matcher.find(d.text)], gold))
        ner_c.append(unit_counts(tagger.predict_spans(d.text), gold))
    rules_c, ner_c = np.array(rules_c), np.array(ner_c)
    out: dict = {"note": DATATURKS_NOTE, "repair": ledger}
    for name, mask in (("test", [d.split == "test" for d in docs]), ("all", [True] * len(docs))):
        m = np.array(mask)
        out[name] = {"n_docs": int(m.sum()), "n_gold_skills_spans": int(rules_c[m, 1].sum()),
                     "rules": score(rules_c[m], seed=seed), "ner": score(ner_c[m], seed=seed),
                     "ner_minus_rules": paired_delta(rules_c[m], ner_c[m], seed=seed)}
    return out


def _normalisation_block(src, seed: int) -> dict:
    from candidate_screener.extraction import normalise as nz, vocab
    ict = src.skills[src.skills.conceptUri.isin(vocab.esco_scope(src))]
    enc = nz.minilm_encoder()
    return {
        "gold": ("seeded sample of ESCO alternative labels owned by exactly one concept and "
                 "not equal (normalised) to any preferred label; each is mapped back to its "
                 "concept; held-out labels are removed from every index"),
        "headline": "ict_preferred",
        "ict_preferred": nz.evaluate(ict, "preferred", enc, seed),
        "ict_aliases": nz.evaluate(ict, "aliases", enc, seed),
        "all_esco_preferred": nz.evaluate(src.skills, "preferred", enc, seed),
    }


def _tagger(seed: int, *, retrain: bool, out_dir: Path):
    from candidate_screener.extraction import ner
    cfg = ner.NerConfig(seed=seed)
    meta_file = Path(out_dir) / "ner-config.json"
    fresh = retrain or not meta_file.is_file() or \
        json.loads(meta_file.read_text())["config"] != asdict(cfg)
    meta = ner.train(cfg, out_dir) if fresh else json.loads(meta_file.read_text())
    return ner.load_ner(out_dir), meta


def build_record(seed: int = 0, *, retrain: bool = False, ner_dir: Path | None = None,
                 verbose: bool = True) -> dict:
    from candidate_screener.extraction import ner, vocab
    from candidate_screener.extraction.rules import default_matcher
    log = print if verbose else (lambda *a, **k: None)
    log("vocabulary ...")
    src = vocab.load_sources()
    vocab_b = _vocab_block(src)
    log("tuning (re-run) ...")
    tuning_b = _tuning_block(src)
    log("tagger ...")
    tagger, meta = _tagger(seed, retrain=retrain, out_dir=ner_dir or ner.NER_DIR)
    ner.set_determinism(seed, tagger.config.threads)
    matcher = default_matcher()
    log("SkillSpan ...")
    ss_b = _skillspan_block(matcher, tagger, seed)
    log("DataTurks ...")
    dt_b = _dataturks_block(matcher, tagger, seed)
    log("normalisation ...")
    nz_b = _normalisation_block(src, seed)
    return {
        "schema": 1, "seed": seed,
        "plan": "plan/2026-09-26-top-down-reset: tasks 2.1, 2.2, 5.1-5.5; record 08-extraction.md",
        "vocab": vocab_b, "vocab_tuning": tuning_b,
        "skillspan": ss_b, "dataturks": dt_b,
        "ner": {k: meta[k] for k in ("config", "best_epoch", "history", "train_pieces",
                                     "weights_sha256", "base_model_snapshot")},
        "normalisation": nz_b,
        "determinism": DETERMINISM,
    }


def render(record: dict) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def diff(old, new, path: str = "", tol: float = TOLERANCE) -> tuple[list[str], list[str]]:
    """(failures, notes): exact everywhere, `tol` on torch-derived numbers."""
    segs = path.split(".")
    torch_path = any(seg in segs for seg in TORCH_SEGMENTS) and "config" not in segs
    if isinstance(old, dict) and isinstance(new, dict):
        fails, notes = [], []
        for k in sorted(set(old) | set(new)):
            if k not in old or k not in new:
                fails.append(f"{path}.{k}: present in only one record")
                continue
            f, n = diff(old[k], new[k], f"{path}.{k}" if path else k, tol)
            fails += f
            notes += n
        return fails, notes
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        fails, notes = [], []
        for i, (a, b) in enumerate(zip(old, new)):
            f, n = diff(a, b, f"{path}[{i}]", tol)
            fails += f
            notes += n
        return fails, notes
    if old == new:
        return [], []
    num = (int, float)
    if (torch_path and isinstance(old, num) and isinstance(new, num)
            and not isinstance(old, bool) and math.isclose(old, new, rel_tol=0, abs_tol=tol)):
        return [], [f"{path}: {old} -> {new} (within {tol})"]
    if torch_path and path.split(".")[-1] in INFORMATIONAL:
        return [], [f"{path}: {old} -> {new} (informational)"]
    return [f"{path}: {old!r} -> {new!r}"], []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="rebuild (retraining the tagger into a scratch cache) and diff")
    ap.add_argument("--reuse-weights", action="store_true",
                    help="with --check: evaluate the cached tagger instead of retraining")
    ap.add_argument("--weights", type=Path, default=None,
                    help="with --check: evaluate an already-trained tagger directory")
    ap.add_argument("--retrain", action="store_true", help="force retraining on build")
    a = ap.parse_args(argv)
    if a.check:
        from candidate_screener.extraction import ner
        reuse = a.reuse_weights or a.weights is not None
        fresh = build_record(a.seed, retrain=not reuse,
                             ner_dir=a.weights or (ner.NER_DIR if reuse else CHECK_DIR))
        old = json.loads(RECORD.read_text())
        fails, notes = diff(old, fresh)
        for n in notes:
            print("  note:", n)
        for f in fails:
            print("  FAIL:", f)
        print(f"extraction record {'OK' if not fails else 'MISMATCH'} "
              f"({len(fails)} failures, {len(notes)} notes)")
        return 0 if not fails else 1
    record = build_record(a.seed, retrain=a.retrain)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(render(record), encoding="utf-8")
    print(f"wrote {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
