# 01 — Measurements (8 Oct 2026, before any change)

Machine: WSL2, 22 cores, 15 GB RAM. Commit `a05228a`→`b149e35` (no `src/` change between
them). `ml` and `app` groups installed, `data/` fully built. Every run below passed.

## 1. `uv run pytest`

594 passed, 25 warnings, **290 s wall**, 3.5 GB peak RSS. Collection takes 1.8 s.

| File | Time | Share |
|---|---:|---:|
| `test_llm_recheck.py` (74 tests) | 260.3 s | **90 %** |
| `test_baseline_golden.py` (data: refits the baseline) | 11.3 s | 4 % |
| `test_evaluation_ordering.py` (4.3 s is importing sentence-transformers) | 4.9 s | 2 % |
| `test_final_set.py` | 4.3 s | 1 % |
| `test_extraction_vocab.py` (data: rebuilds the vocabulary) | 2.5 s | 1 % |
| the other 41 files | < 7 s | 2 % |

**Cause of the 260 s** (cProfile of the slowest test, 54 s): `llm_recheck.kappa_ci` draws
2,000 bootstrap resamples and calls sklearn's `cohen_kappa_score` on each one. A single
test made 79,790 sklearn calls. About 1.5 ms of each call is sklearn's input validation
(`_check_targets`, `type_of_target`, `check_array`), not arithmetic.

**Prototype** (scratch only, not committed): a numpy confusion-matrix kappa that does the
same operations in the same order. It was compared with the current `kappa_ci` on 80 cases:
random 3-class and binary inputs, identical raters, one-class raters, n = 3 to 120. All
80 matched **bit for bit**, the CI bounds included. n = 200 with 2,000 resamples took
0.079 s against 2.13 s, **27× faster**. The bootstrap indices are unchanged: the same
`rng.integers` call is made in the same order.

## 2. The `--check` commands, run one after another

| Check | Wall (s) | Peak RSS (GB) | Output lines | Dominant cost (profiled where marked †) |
|---|---:|---:|---:|---|
| `extraction.role_family --check` | 780.7 | 4.7 | 12 | refits both candidate models (by design of `--check`) |
| `evaluation.a1_secondary --check` | 773.1 | 2.6 | 9 | cross-encoder inference (not profiled; same rungs as below) |
| `evaluation.shortlist_dev --check` | 671.5 | **6.8** | 51 | † trigram cosine 51 %, TF-IDF pair loop 18 %, role-family featurising 18 % |
| `extraction.evaluate --check` | 640.0 | 2.7 | 29 | **retrains the NER tagger: 412 s** |
| `evaluation.shortlist_dev --variants --check` | 582.7 | 4.2 | 39 | same components as `shortlist_dev` |
| `evaluation.retrieval_pilot --check` | 523.4 | 3.0 | 17 | † cross-encoder predict 63 %, TF-IDF pair loop 12 % |
| `evaluation.indomain --check` (v1) | 452.8 | 3.4 | 25 | † cross-encoder predict 57 %, `per_jd_auc` 35 % |
| `evaluation.stage1 --check` | 180.8 | 1.7 | 4 | (not profiled) |
| `evaluation.shortlist_final --check` | 163.3 | 1.6 | 1 | (not profiled; Z1/Z1L use `role_filter`) |
| `evaluation.dev_pools --check` | 79.2 | 5.1 | 11 | rebuilds the dev-pool cache (`force=True`) |
| `evaluation.retrieval --check` | 67.7 | 1.3 | 2 | |
| `extraction.title_profile --check` | 56.5 | 3.5 | 1 | |
| `evaluation.ordering --check` | 47.2 | 1.2 | 3 | |
| `rerank.scoring_batch --check` | 39.5 | 4.9 | 5 | |
| `evaluation.indomain --check --text-variant v2` | 37.2 | 1.3 | 3 | |
| `data.weak_pairs --check` | 25.2 | 0.9 | 3 | |
| `data.verify --derived` | 15.7 | 3.3 | **120** | |
| `baselines.run --check` | 13.4 | 0.5 | 10 | |
| `extraction.experience_eval --check` | 8.5 | 0.2 | 1 | |
| `evaluation.final_set --check` | 7.6 | 0.1 | 14 | |
| `extraction.vocab --check` | 3.5 | 0.4 | 1 | |
| **Total (serial)** | **5,169 s = 86 min** | max 6.8 | 357 | |

The † profiles ran three at a time under cProfile, so their absolute times are inflated
(indomain 976 s profiled against 453 s plain). Only the **shares** are used below.

Profile detail behind the shares:

- **Cross-encoder** (`matching/cross_encoder.py`). Nothing caches a score. `_fit` predicts the
  validation split once per grid point (2). `classify()` predicts the same pairs a second
  time after `score()` already did. Every check that holds a CE rung (indomain v1,
  a1_secondary, retrieval_pilot) pays the full inference again.
- **`indomain.per_jd_auc`**: 1,432 calls, each a pandas `groupby` with one sklearn
  `roc_auc_score` per JD. `paired_delta` (608 calls) recomputes the AUCs of the **same**
  system and scope for every comparison it appears in.
- **`role_filter.trigram_cosine`**: 37.8 M calls and 75.6 M `_grams` Counter builds. The
  memo in `components()` is per call and keyed by the whole title tuple, so the same
  (JD title, CV title) string pair is recomputed across CVs and across the 32 `components`
  calls.
- **`baselines.tfidf.pair_cosine_scores`**: one sparse `cosine_similarity` call per pair
  (120 k `vector` calls in retrieval_pilot alone).
- **`role_family.predict_proba`**: 8 calls re-featurise the same dev CV texts
  (`title_spans`, `_lowered_with_map`) once per rung.

Output noise: the transformers 5 `Loading weights: 100%|…|` progress bar appears 24 times in
indomain v1's output, and in every check that loads a model.

## 3. How the suite and the checks are actually run (past transcripts)

A scan of the 18 transcript files for this project (both worktrees, subagents included):

| Pattern | Count |
|---|---:|
| full-suite `uv run pytest` invocations | 197 |
| …piped to `tail`/`grep`/`head` | 151 |
| …with no `Edit`/`Write` since the previous full run in the same transcript | 144 |
| targeted `pytest tests/x.py` / `-k` | 88 |
| `--check` invocations | 97 (56 piped) |

The 144 is an **upper bound** on pure re-runs: an edit made by `sed`, or by a subagent in
another transcript, is not seen. Even so, it supports the user's diagnosis. The suite is
re-run, at about 5 min each, to read a different slice of an output that was piped to
`tail -2` the first time. The most common pipes were `tail -2`, `tail -3`,
`grep -E "passed…"` and `tail -15`.

## 4. Other measurements

- `uv run` overhead: 0.05 s, which is negligible. Importing `llm_recheck` plus `indomain`
  takes 1.7 s (sklearn alone is 1.1 s), also negligible against the checks.
- RAM: 15 GB in total and peaks up to 6.8 GB, so **at most two checks can run in
  parallel**, and the heaviest (`shortlist_dev`, `dev_pools`, `scoring_batch`) should run
  alone.
- Running every `--check` left `git status` clean. `dev_pools --check` does rewrite its
  git-ignored cache under `data/`.
