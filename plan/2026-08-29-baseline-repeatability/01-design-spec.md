# Design Specification — Baseline Repeatability

## 1. Target layout

```
src/candidate_screener/
  baselines/                      NEW — the Stage 1 classical baseline
    __init__.py
    skills.py                     SKILL_PATTERNS, extract_skills, augment
    lexical.py                    TOKEN_PATTERN, tokenize — shared by both scorers
    tfidf.py                      fit_shared_vectorizer, pair_cosine_scores
    bm25.py                       fit_bm25, pair_bm25_scores (D24 — no corpus lookup)
    classifier.py                 the single-feature LogisticRegression wrapper
    run.py                        `python -m candidate_screener.baselines.run`
  evaluation/
    classification.py             NEW — accuracy / macro-F1 / confusion, with the floor

tests/                            NEW — pytest, see §6
  test_skills.py  test_lexical.py  test_tfidf.py  test_bm25.py
  test_classification.py  test_baseline_golden.py  test_notebook_is_thin.py

output/baselines/                 NEW — COMMITTED (Q20, 29 Aug: a top-level output/, not docs/)
  baseline-metrics.json           the golden record (§4)
  README.md                       what the numbers are, and the one command

data/processed/baselines/         git-ignored — regenerable cache (§4)
  tfidf_vectorizer.pkl  tfidf_classifier.pkl  bm25_model.pkl  bm25_classifier.pkl
  scores.parquet                  resume_id, jd_id, split, label, tfidf_score, bm25_score
```

The split between the ignored `data/processed/baselines/` and the committed `output/baselines/`
is the same mechanism Phase 3 used for the split and the pools: **the corpus and the fitted
objects are not committed; what is committed is the description of what was built and the
figures it produced**, so a re-run either reproduces them or fails loudly.

`baselines` is a sibling of `data` and `evaluation`, not a subpackage of either. Phase 3's
deviation W11 made the same call for `evaluation`: building an artefact, scoring one, and
*being* a model are three lifetimes, and `data/` stays about data.

## 2. What is pure, and what touches disk

Everything under `baselines/` except `run.py` is a pure function of its arguments. This is what
makes the unit tests in §6 possible without `data/raw/` present — which matters, because
`data/` is git-ignored and a fresh clone has none of it.

| Function | Signature (indicative) | Pure? |
|---|---|---|
| `skills.extract_skills` | `(text: str) -> str` | yes |
| `skills.augment` | `(text: pd.Series) -> pd.Series` — text + " " + extracted | yes |
| `lexical.tokenize` | `(text: str) -> list[str]` | yes |
| `tfidf.fit_shared_vectorizer` | `(resume_texts, jd_texts) -> TfidfVectorizer` | yes |
| `tfidf.pair_cosine_scores` | `(vectorizer, resume_texts, jd_texts) -> np.ndarray` | yes |
| `bm25.fit_bm25(resume_texts)` | `(resume_texts) -> BM25Okapi` — fits `idf`/`avgdl` only | yes |
| `bm25.pair_bm25_scores` | `(model, resume_texts, jd_texts) -> np.ndarray` — scores each pair from the pair's own tokens against `model.idf`/`model.avgdl`; no lookup into the fit corpus (D24) | yes |
| `classifier.fit_single_feature` | `(scores, labels, seed) -> LogisticRegression` | yes |
| `evaluation.classification.report` | `(y_true, y_pred, labels) -> Report` | yes |
| `run.main` | reads a split, writes the cache and the metrics JSON | **no** |

### Two API shapes that make the fixed bugs unrepresentable

This is the point of the refactor, not a side effect of it.

1. **`fit_shared_vectorizer` returns exactly one vectoriser**, and `pair_cosine_scores` takes
   exactly one. There is no signature in which a caller can hold a resume vectoriser and a JD
   vectoriser at once. The bug that shipped until today — two separately-fitted vectorisers, two
   unrelated vocabulary→index maps, and a cosine over unrelated axes — cannot be written through
   this API. §5 tests the invariant anyway, because an API can be worked around.
2. **`lexical.TOKEN_PATTERN` is one module-level constant used by BM25**, and a test asserts it
   equals `TfidfVectorizer().token_pattern`. The bug that shipped until today — BM25 on
   `.lower().split()` while TF-IDF used a word-boundary regex, so `"python,"` and `"python"` were
   different terms — becomes a one-line test failure rather than a quietly weak signal.
3. **`pair_bm25_scores` takes a document's own tokens and scores them against the fitted
   `idf`/`avgdl`; there is no text→row lookup into the fit corpus for it to have.** The bug found
   on 29 Aug while resolving Q19 — `BM25Okapi.get_scores` only scores documents present in the
   corpus it was built from, so a lookup-based design silently returns 0.0 for any document held
   out of that corpus, which is exactly what the leak-free split does for every test resume by
   construction — cannot be written through this API, because there is no lookup step to omit.
   §5 adds an invariant test that scores a document deliberately absent from the fit corpus and
   asserts a non-zero, term-overlap-driven score (D24).

## 3. Determinism inventory

Every place a re-run could legitimately differ, what the verdict is, and what controls it.
Nothing here is asserted from memory — the versions are read from `uv.lock`, the solver
behaviour from scikit-learn's documented contract for `random_state`.

| # | Source | Verdict | Control |
|---|---|---|---|
| 1 | **Library versions.** `uv.lock` pins scikit-learn 1.9.0, numpy 2.5.2, scipy 1.18.1, pandas 3.0.5, rank-bm25 0.2.2, joblib 1.5.3, threadpoolctl 3.6.0 | Controlled, but a `uv lock` bump can move a number | `uv sync --frozen` in the documented command. The five load-bearing versions are **recorded in `baseline-metrics.json`** so a golden-check failure after a bump is diagnosable in one diff rather than bisected |
| 2 | **`LogisticRegression` `random_state`** | **Currently inert, and that is not a bug.** The default solver is `lbfgs`, which is deterministic; `random_state` is consumed only by `sag`, `saga` and `liblinear` | Pin `solver="lbfgs"` **explicitly** rather than relying on the default, and pass `random_state=0` as insurance against a future solver swap making it live. Document in the docstring that it is inert under `lbfgs`, so nobody reads it as a control that exists |
| 3 | **Non-convergence.** `max_iter=5000` today | A silently truncated fit makes the result a function of the iteration cap, and the cap is not a modelling decision anyone made | `fit_single_feature` **raises** if `clf.n_iter_` reaches `max_iter`. A converged fit is reproducible; a truncated one drifts on any perturbation |
| 4 | **Hash-based iteration order.** `set` iteration over strings varies with `PYTHONHASHSEED` between processes | Not currently triggered — the notebook uses `drop_duplicates` (order-preserving) and dicts (insertion-ordered) — but one `set()` in a future edit would introduce cross-run drift that only appears sometimes | **Rule:** no `set` may feed an ordered structure; use `sorted()` where a set is needed. Enforced by a test that fits the vectoriser twice from two different input row orders and asserts an identical vocabulary and identical scores |
| 5 | **`TfidfVectorizer(max_features=5000)` tie-breaking.** sklearn keeps the top 5,000 terms by document frequency using an `argsort` that is not stable, so *which* of several equal-frequency terms survives at the 5,000 boundary is an implementation detail of numpy's sort | Deterministic for a fixed numpy build and input; **not** guaranteed across a numpy bump | The **sha256 of the sorted feature-name list** is recorded in `baseline-metrics.json` alongside the vocabulary size. A boundary shift then fails the check explicitly instead of moving a metric by a hair. Note the index assignment itself *is* canonical — sklearn sorts the vocabulary alphabetically after limiting |
| 6 | **BLAS thread count / float last bits.** Sparse cosine and the lbfgs fit both go through BLAS; thread count can change the last bits of a float | Real but tiny. The risk is not the float — it is a score sitting on a decision boundary flipping one row's predicted class and moving a confusion-matrix count by 1 | Integers (vocabulary size, confusion counts, *n*) compared **exactly**; floats compared with an explicit tolerance. **Task 4.3 measures this** — three runs, and if counts move, `OMP_NUM_THREADS=1` is set in the documented command and the fact recorded. This is Q23, answered with evidence rather than by preference |
| 7 | **Input data.** `data/raw/` is SHA-256 verified by `acquisition-manifest.json` | Controlled | `verify --all` is a stated precondition of the run, exactly as it is for `build` |
| 8 | **Which split.** `load_split("fit", …)` read the **shipped** partition from `data/raw/fit/`, not the leak-free `data/processed/fit/` | **Resolved — Q19, 29 Aug.** Leak-free `data/processed/fit/` (3,990 / 659 pairs, 0% resume/JD overlap) is adopted; the notebook already reads it | `baseline-metrics.json` still records the split source, its row counts and its unique-document counts, so a number can never be read without knowing which split produced it — this stays even though the choice is now fixed, because a future re-split under the same path must still be caught |
| 9 | **Pickle byte-stability** | Not guaranteed across library versions, and not needed | **D23.** Pickles are a cache: never committed, never hashed, never part of an acceptance check |
| 10 | **`_vector_cache` (TF-IDF) and `_query_tokens_cache` / `_doc_tokens_cache` (BM25)**, all keyed by full document text | Deterministic (dicts are insertion-ordered) and a large correctness win — the token caches are what stop BM25 re-tokenising the same document once per row | Kept, moved inside the scorer functions, and covered by a test asserting cached and uncached scoring agree |
| 11 | **BM25 generalisation (D24).** `pair_bm25_scores` computes each score from the corpus-fitted `idf`/`avgdl` plus the pair's own token counts, rather than from `BM25Okapi.get_scores` | Deterministic — same inputs, same formula, no corpus-membership branch to be sensitive to | Covered by §5.2's invariant test; not a source of run-to-run variance, listed here because it replaces a *correctness*-relevant assumption (row 8's split choice) with a *design* guarantee that holds regardless of which split is used |

## 4. Where artefacts and metrics live (D20, D23, D24)

**Q20 resolved 29 Aug: a new top-level `output/` directory, committed to git** — not
`docs/baselines/` as originally proposed below. The reasoning (fixed at sign-off): `run --check`
only has something to diff against if the record is committed, so an ignored `output/` would make
the check meaningless. The path is `output/baselines/baseline-metrics.json`; everything else in
this section (schema, no-timestamp rule, the README beside it) is unchanged, only the parent
directory moved from `docs/` to `output/`.

### Committed: `output/baselines/baseline-metrics.json`

The role `docs/data/profile-metrics.json` already plays for the corpus. Proposed contents —
every field is either a configuration input or a measured output, and **there is no timestamp**
(Phase 3, W6: with a timestamp every re-run diffs and the check tests nothing):

```
{
  "split":   { "source": "processed/fit", "train_pairs": …, "test_pairs": …,
               "train_unique_resumes": …, "train_unique_jds": …,
               "test_unique_resumes": …, "test_unique_jds": …,
               "label_counts": { "Good Fit": …, "Potential Fit": …, "No Fit": … } },
  "config":  { "seed": 0,
               "skill_patterns_sha256": "…",       # catches a silent edit to the pattern list
               "token_pattern": "(?u)\\b\\w\\w+\\b",
               "tfidf": { "max_features": 5000, "min_df": 2, "max_df": 0.8,
                          "ngram_range": [1, 2], "stop_words": "english" },
               "classifier": { "solver": "lbfgs", "max_iter": 5000,
                               "class_weight": "balanced", "random_state": 0 } },
  "vocabulary": { "size": …, "sha256": "…" },      # §3 row 5
  "floor":   { "majority_class": "No Fit", "accuracy": …, "macro_f1": … },
  "models":  { "tfidf": { "accuracy": …, "macro_f1": …,
                          "per_class": { … precision/recall/f1/support … },
                          "confusion": [[…],[…],[…]],
                          "coef": […], "intercept": […] },
               "bm25":  { … same shape … } },
  "environment": { "python": "3.12", "scikit-learn": "1.9.0", "numpy": "2.5.2",
                   "scipy": "1.18.1", "pandas": "3.0.5", "rank-bm25": "0.2.2" }
}
```

`environment` is the one non-content field, and it is deliberate: unlike a timestamp it changes
only when someone deliberately re-locks, so it does not defeat the reproducibility check — it
explains it. Recording the fitted `coef`/`intercept` is what turns "the metrics happen to match"
into "the same model was fitted"; two different models can land on the same accuracy.

### Committed: `output/baselines/README.md`

Short, and it carries the three things a reader needs and cannot infer:

1. The one command that regenerates everything, and `run --check` to verify.
2. **The incomparability warning, Q19-resolved form:** results are on the **leak-free**
   `data/processed/fit/` split, so they *are* comparable to every other Stage 1–4 figure, but are
   **not** comparable to any number published against the shipped `cnamuangtoun` partition
   (including the figures this very plan superseded 29 Aug) — the wording already in
   `docs/data/manifests/README.md`.
3. That both models beat the majority floor on macro-F1 and neither beats it on accuracy, **and
   why that is expected** of one linear threshold over overlapping imbalanced classes. Written
   down once, it stops being reopened as a bug.

### Git-ignored: `data/processed/baselines/`

Unchanged in location from what the notebook writes today, which keeps `.gitignore` untouched
(`data/*` already covers it, and Phase 3's commit policy allows no new exception without a
reason). `scores.parquet` replaces `baseline_predictions.csv` and gains a `split` column so
train scores — needed to refit or to inspect the threshold — are not thrown away.

### The notebook cites, it does not restate

`AGENTS.md` §Notebooks already requires this for `profile-metrics.json`. Notebook 07 loads
`output/baselines/baseline-metrics.json` and renders from it, so the prose cannot drift from the
numbers, and re-running the notebook cannot *become* the definition of the baseline.

## 5. Regression protection

Three layers, cheapest first.

### 5.1 `run --check` — the golden diff

```bash
uv run python -m candidate_screener.baselines.run --seed 0            # rebuild + write
uv run python -m candidate_screener.baselines.run --check            # rebuild + diff, no write
```

`--check` refits in memory, compares field by field against the committed JSON, prints a
`[PASS]`/`[FAIL]` line per group in the same idiom as `verify --derived`, and exits 1 on any
mismatch. It writes nothing, so it is safe to run anywhere. This is the direct analogue of the
`determinism:` check already in `verify_derived.check_fit_split`, which rebuilds the split
manifest in memory and diffs it against the committed one.

**It deliberately does not register in `data.verify --derived`.** That registry is the acceptance
test on *derived data artefacts*, and W11 established that `data/` stays about data. If the team
prefers one umbrella command, that is a one-line change — raised as part of Q20 rather than
decided here.

### 5.2 Named regression tests for the three bugs that shipped

All three are written as invariants, not as golden numbers, so they keep working when the numbers
legitimately change:

- **`test_shared_space_cosine_of_identical_text_is_one`** — score a document against itself
  through `pair_cosine_scores`. With one shared vectoriser this is 1.0 to floating-point; with
  two separately-fitted ones it is not, because the two vocabulary→index maps disagree. This is
  the two-vectoriser bug reduced to a single assertion.
- **`test_token_pattern_matches_sklearn_default`** — assert
  `lexical.TOKEN_PATTERN.pattern == TfidfVectorizer().token_pattern`, and assert
  `tokenize("Python, Java. C++") == ["python", "java"]`. The second half documents an accepted
  property rather than a bug: this pattern drops `c++`, and it drops it for TF-IDF too, so the
  two scorers stay comparable. Writing it down stops it being "fixed" asymmetrically.
- **`test_bm25_scores_document_outside_fit_corpus`** — fit `bm25.fit_bm25` on one set of resumes,
  then score a resume that was **never part of that set** against a query it genuinely overlaps
  with, and assert the score is `> 0`. Under the old lookup-based design this is exactly `0.0` for
  every document outside the fit corpus — which is what the leak-free split triggered for the
  entire test set on 29 Aug. This is the D24 bug reduced to a single assertion, the same way the
  first bullet reduces the two-vectoriser bug.

### 5.3 `test_notebook_is_thin`

Parse `notebooks/07-baseline-tfidf-keyword.ipynb` and assert no code cell contains
`TfidfVectorizer(`, `BM25Okapi(`, or a `SKILL_PATTERNS =` assignment. This is the check that
answers "a future edit to the notebook can't silently change what the baseline means": the
notebook is allowed to plot, tabulate and explain, and is not allowed to build a model. Cheap,
and it fails on exactly the regression it is aimed at — someone pasting model code back inline.

## 6. Test suite

**No test suite and no test dependency exist in this repository today** — verified: no `tests/`
directory, no pytest in `[dependency-groups] dev`, no test file anywhere under `src/`. There is
therefore no convention to follow, which is Q21.

**Recommendation:** pytest, added to the dev group, a `tests/` directory at the repository root,
run as `uv run pytest`.

| File | Covers |
|---|---|
| `test_skills.py` | Doubling (each match appears twice), case-insensitivity, empty and `None` input, a no-match text returning `""`, and a digest test over `SKILL_PATTERNS` that fails on a silent edit |
| `test_lexical.py` | §5.2 tokeniser parity; punctuation stripping; the accepted `c++` loss; lowercase |
| `test_tfidf.py` | §5.2 shared-space invariant; input-row-order invariance (§3 row 4); `min_df`/`max_features` honoured; a hand-built 4-document corpus where the expected ranking is obvious by inspection |
| `test_bm25.py` | §5.2's out-of-corpus invariant (D24) — a resume absent from the fit corpus still scores correctly, not 0.0; cached vs uncached scoring agree (§3 row 10); row-order invariance |
| `test_classification.py` | The majority floor is computed and returned with every report; a report with `n=0` raises, mirroring `metrics.Figure`; macro-F1 against a hand-computed 3-class example |
| `test_baseline_golden.py` | End-to-end: the §5.1 check. **Requires `data/`**, so it is `pytest.mark.skipif`-ed when `data/processed/fit/` is absent, and says so in the skip reason |

Every test except the last runs on tiny inline synthetic corpora and needs no data on disk. That
is a hard requirement, not a nicety: `data/` is git-ignored, and a suite that cannot run on a
fresh clone will not be run.

## 7. Commit policy

Extends Phase 3 §5 without amending it.

**Committed:** `src/candidate_screener/baselines/`, `evaluation/classification.py`, `tests/`,
`output/baselines/baseline-metrics.json`, `output/baselines/README.md`, the rewritten notebook with
its (PII-free) outputs, this plan and its implementation record.

**Never committed:** anything under `data/processed/baselines/` — the pickles and
`scores.parquet`. `.gitignore` needs no change; `data/*` already covers them, and adding an
exception would contradict D23.

**The notebook output cap from D15 applies unchanged:** at most 5 records or 2,000 characters per
cell. Notebook 07 previously printed two full documents (`test_df.loc[1755]` resume and JD) in a
scratch cell, exceeding the cap on both counts — **A1 resumes are real documents**. Removed 29
Aug, ahead of task 4.4, at the user's explicit request alongside the Q19 split switch.
