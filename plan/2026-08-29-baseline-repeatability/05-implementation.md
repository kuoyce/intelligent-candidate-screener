# Implementation Record — Phase 4, Baseline Repeatability

**Implemented:** 29 August 2026, on `feat/derived-artefacts`
**Tasks:** 4.1–4.5, all complete
**Result:** the baseline moved out of the notebook into
`src/candidate_screener/baselines/`, was frozen as a committed record, and is now
covered by 58 tests including the three named bug regressions. **No number moved.**

## What was built

| Path | What it is |
|---|---|
| `src/candidate_screener/baselines/skills.py` | `SKILL_PATTERNS`, `extract_skills`, `augment`, `digest` |
| `src/candidate_screener/baselines/lexical.py` | `TOKEN_PATTERN`, `tokenize` — one definition, shared |
| `src/candidate_screener/baselines/tfidf.py` | `fit_shared_vectorizer`, `pair_cosine_scores`, `vocabulary_digest` |
| `src/candidate_screener/baselines/bm25.py` | `fit_bm25`, `score_pair`, `pair_bm25_scores` (D24) |
| `src/candidate_screener/baselines/classifier.py` | `fit_single_feature`, `predict` — solver pinned, raises on non-convergence |
| `src/candidate_screener/baselines/run.py` | `--seed` / `--check` / `--out`; build, cache, record, golden diff |
| `src/candidate_screener/evaluation/classification.py` | `Report`, `Floor`, `report`, `majority_floor` |
| `output/baselines/baseline-metrics.json` | The committed golden record, 4,107 bytes, no timestamp |
| `output/baselines/README.md` | The one command, the incomparability warning, the floor explanation |
| `tests/` (8 files, 59 tests) | Unit, invariant and end-to-end coverage; 55 of them run with no `data/` |
| `notebooks/07-baseline-tfidf-keyword.ipynb` | Rewritten: imports the package, renders from the record |

## Acceptance, task by task

**4.1** — `extract_skills`, `tokenize` and `augment` imported from the package reproduce the
notebook's outputs **element-wise on all 949 unique documents** (4,649 pairs) of the leak-free
split. The comparison was a throwaway script and is not committed *(the Where code goes rule,
and Phase 3's deviation V1)*.

**4.2** — a package-only pipeline reproduces the corrected notebook's figures **exactly**,
confusion matrices included:

| Model | Accuracy | Macro-F1 | Confusion (Good / Potential / No) |
|---|---|---|---|
| TF-IDF | 0.4917 | 0.3818 | `[[92,12,78],[60,8,65],[103,17,224]]` |
| BM25 | 0.4598 | 0.3466 | `[[92,8,82],[46,4,83],[130,7,207]]` |
| Floor | 0.5220 | 0.2286 | — |

**4.3** — `run --seed 0` on a clean tree reproduces the committed JSON **byte-for-byte**;
`git status` shows no change after a re-run.

**4.4** — `uv run pytest`: **59 passed** with data present. Verified on an isolated copy of the
tree with **no `data/` at all**: **55 passed, 4 skipped**, each skip naming the exact build
command to run. Only the four tests that *refit* are skipped — the assertions about the
committed record itself are not, because `baseline-metrics.json` is in git and a fresh clone
can check what it claims about the split and the floor without refitting anything. `run --check`
passes with data present and exits 1 when a metric is perturbed by hand: asserted by two tests,
one moving an accuracy by 1e-6 and one moving a confusion count by 1.

**4.5** — this file, plus `README.md`, `AGENTS.md`, `output/baselines/README.md` and the
predecessor plan's successor pointer, all in the same commit as the code.

## Q23 answered — with measurement, not preference

The work plan required measuring the float question before answering it. Five runs of
`run --seed 0`, diffed against each other:

| Run | Result |
|---|---|
| Repeat 1, 2, 3 (default threading) | byte-identical |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` | byte-identical |
| `PYTHONHASHSEED=12345` | byte-identical |

All five produced sha256 `6caaaf5ec721e4d4…`. **Answer:**

- **Confusion-matrix counts are stable, so exact integer comparison is adopted.** Every integer
  in the record — vocabulary size, confusion counts, *n*, per-class support, `n_iter` — is
  compared exactly. This is the comparison that matters: the risk was never the last bits of a
  float, it was a score sitting on a decision boundary flipping one row's predicted class.
- **No thread pin is necessary**, and none is added to the documented command. Adding one on
  this evidence would be ritual.
- **Floats are compared at `1e-12`** — far tighter than any metric difference that would matter,
  far looser than a BLAS reordering of a sum. This is insurance for assumption A8, which does
  not promise bit-identity across differing BLAS builds and CPU architectures; on this machine
  the record is byte-identical, which is strictly stronger.

**Q23 is closed.** Q24 remains open and deliberately out of scope.

An unplanned second measurement, on the same question: **the `max_features=5000` tie-break
(design spec §3, row 5) does not bite on the real corpus.** Fitting the shared vectoriser from
a shuffled row order gives an identical vocabulary digest and bit-identical scores
(`max |Δ| = 0.0`). The digest stays in the record regardless — it is cheap, and it is what
would make a future numpy bump diagnosable in one diff rather than bisected.

## The three regressions, demonstrated failing

Work plan 4.4 step 2: *a regression test never shown to fail is a test of nothing.* Each named
test was run against a faithful reimplementation of the pre-fix code:

| Test | Pre-fix | Post-fix | What the pre-fix code produced |
|---|---|---|---|
| `test_shared_space_cosine_of_identical_text_is_one` | **FAIL** | PASS | self-cosines of `[0.0] * 6` — two separately-fitted vocabularies share no axes |
| `test_token_pattern_matches_sklearn_default` | **FAIL** | PASS | `['python,', 'java.', 'sql!']` — punctuation glued to every token |
| `test_bm25_scores_document_outside_fit_corpus` | **FAIL** | PASS | `0.0` for a held-out resume against an overlapping JD (post-fix: 3.0797) |

## Found in code review (workflow step 4)

One real defect and three housekeeping items, all fixed in this commit:

| Finding | Fix |
|---|---|
| `evaluation.classification.report` silently accepted a label outside `LABELS`. Accuracy would have counted such a row while the confusion matrix, the per-class table and the floor all excluded it — three of the four reported figures quietly describing a different set of rows than the fourth | Raises now, with the reason in the message. Covered by `test_a_label_outside_the_scheme_raises` |
| `run._differences` treated `True` and `1` as equal, because `True == 1` in Python. A bool where an int was frozen is a schema change, not a matching value | Types must agree for booleans. Covered by `test_booleans_are_not_treated_as_integers` |
| `run.print_environment` would `KeyError` on a record written before `environment` existed — i.e. exactly when the diagnostic is most wanted | Reads through `.get`, printing `unrecorded` for a missing key |
| `NOT_COMPARABLE` was defined below its only use; `hashlib` was imported inside `vocabulary_digest` | Both moved to the top of their modules, matching the house style |

## Deviations from the approved plan

Continuing Phase 3's `W`-numbering.

| # | Deviation | Why |
|---|---|---|
| **W12** | Paths resolve through `config.py` (`OUTPUT`, `BASELINE_METRICS`, `BASELINE_CACHE`) rather than being literals in `run.py` | The design spec §1 named the paths but not where they live. `config.py` already exists so that the committed layout has exactly one definition; a second definition in `run.py` would contradict it |
| **W13** | `baseline-metrics.json`'s `split` block nests `train` and `test` sub-objects instead of the flat `train_pairs` / `test_unique_resumes` keys sketched in design spec §4 | Same content, one shape instead of two parallel key families. The flat form would have needed a new key per split if a val fold is ever reinstated |
| **W14** | `split` gained `note_not_comparable`, and `models.*` gained `classes`, `n_iter` and `n` | `note_not_comparable` puts the Phase 3 incomparability warning *inside* the artefact, so it cannot be separated from the numbers by a copy-paste. `classes` makes the recorded `coef` interpretable (they are per-class rows); `n_iter` makes a near-non-convergence visible before it becomes a raise |
| **W15** | `evaluation.classification.report` takes a leading `model` name argument, not the `(y_true, y_pred, labels)` of design spec §2 | A `Report` that cannot say which model it describes is not printable, and both models are reported side by side everywhere |
| **W16** | `bm25.score_pair` is public alongside `pair_bm25_scores` | It is the single-pair arithmetic, and `test_bm25.py` asserts it agrees with `BM25Okapi.get_scores` on in-corpus documents — the assertion that makes D24 a *removal of an assumption* rather than a change of model. That test needs the function |
| **W17** | Two test files beyond design spec §6's table: `tests/test_run_diff.py` and `tests/conftest.py` | `_differences` is the part of `--check` that decides what counts as a regression, it is testable without `data/`, and Q23's answer lives in it. `conftest.py` holds the shared synthetic corpus |
| **W18** | `test_notebook_is_thin.py` forbids a fourth fragment, `LogisticRegression(`, and adds `test_notebook_reads_the_committed_metrics` | The design spec §5.3 named three fragments; a notebook that imports the scorers but refits its own classifier would still be defining the baseline. The second test enforces the §4 "the notebook cites, it does not restate" rule that was otherwise only prose |
| **W19** | The synthetic test corpus is 6 resumes × 5 JDs with every domain in ≥2 documents, and `test_tfidf.py` gained `test_a_document_with_no_in_vocabulary_term_scores_zero` | Found while writing the tests: under `min_df=2` a document whose vocabulary is unique to it vectorises to all-zeros, so its cosine against *itself* is 0.0 and the shared-space regression would pass for the wrong reason. That property is now asserted explicitly rather than left as a trap for the next person to edit the fixture |
| **W20** | `run.check` reports `environment` as `[info]` and never fails on it | Design spec §4 called `environment` "the one non-content field… it explains [the check]". Failing on it would make every `uv lock` a false alarm, and a check that cries wolf stops being read. The rule is now explicit in the output and asserted by `test_environment_drift_alone_does_not_fail_the_check` |

Nothing in the plan was dropped. `data.build` and `data.verify --derived` gained no
registration, as design spec §5.1 required.

## Open questions after this phase

| # | Question | Status |
|---|---|---|
| Q18 | Pool precision bias | **Open** — inherited from Phase 3, untouched here |
| Q23 | Golden tolerance; pin `OMP_NUM_THREADS=1`? | **Closed 29 Aug** — exact integers, 1e-12 floats, no pin. Measured, above |
| Q24 | Score the baseline on the Phase 3 retrieval pools? | **Open** — deliberately excluded. It is new evaluation, not repeatability, and the random-ranker floor it would be compared against is already measured |
| Q22 | CI | **Closed 29 Aug — none added.** `AGENTS.md` now says so in writing and puts `pytest` / `--check` in the code-review step, so "automatically caught" is not claimed where "caught by a reviewer who runs one command" is what is delivered |

## What is now true that was not this morning

1. **The baseline is reproducible from one documented command** and byte-verifiable against a
   committed record. Before, it was notebook cells writing into a git-ignored directory.
2. **Two of the three bug shapes are unwritable through the new API** — one vectoriser in and
   one out, no corpus lookup to omit — and all three are pinned as named invariant tests rather
   than golden numbers, so they survive a legitimate change to the figures.
3. **The floor travels with every figure.** Accuracy below the majority floor is stated in the
   dataclass, the record, the notebook and `output/baselines/README.md`, so it stops being
   reopened as a bug.
4. **The repository has a test suite**, and it runs on a fresh clone with no data.
