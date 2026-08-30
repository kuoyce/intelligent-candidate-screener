# Work Plan — Phase 4

Effort is wall-clock for one person. Every task lists an acceptance criterion that a command can
check, so "done" is machine-checkable rather than asserted — the same standard Phase 3 held.

## Sequencing

| Order | Task | Depends on | Effort | Gated on |
|---|---|---|---|---|
| 1 | **4.1** extract the pure functions | — | ½ day | — |
| 2 | **4.2** extract the scorers and the classifier | 4.1 | ½ day | — |
| 3 | **4.3** the entrypoint and the golden freeze | 4.2 | ½ day | — (Q19 resolved 29 Aug) |
| 4 | **4.4** `--check`, the test suite, the notebook rewrite | 4.3 | ½–1 day | — (Q21, Q22 resolved 29 Aug) |
| 5 | **4.5** documentation and the implementation record | 4.4 | ¼ day | — (Q20 resolved 29 Aug) |

4.1 and 4.2 are pure refactors — **their acceptance criterion is that no number moves** relative
to the already-corrected notebook (all three bugs fixed, leak-free split, 29 Aug). Nothing in the
sequence is gated any longer; Q19–Q22 and Q25 are resolved, and only Q23 (measured, not blocking)
and Q24 (deliberately excluded) remain open.

---

## 4.1 — Extract the pure functions

1. Create `src/candidate_screener/baselines/` with `skills.py` and `lexical.py`, moving
   `SKILL_PATTERNS`, `extract_skills` and the BM25 tokeniser out of the notebook **verbatim in
   behaviour**. Add an `augment(text) -> text + " " + extract_skills(text)` helper so the
   `*_with_skills` construction has one definition instead of four inline expressions.
2. Make `TOKEN_PATTERN` a module-level constant in `lexical.py`, and have `tfidf.py` (4.2) and
   `bm25.py` (4.2) both refer to it. Today the notebook restates sklearn's default pattern in a
   comment and hopes it stays true.
3. Write module docstrings in the house style — what the module is *for*, and what failure it
   prevents — following `ids.py` and `metrics.py`, which both lead with the failure mode.
4. Note in `skills.py`'s docstring what the pattern list actually is: **a fixed, hand-written
   keyword list, not a learned or vocabulary-derived one.** `data/vocab/` (task 3.5) will make a
   real skill vocabulary available and is not consumed here — that is a modelling change, out of
   scope under A9, and it should be findable rather than rediscovered.

**Acceptance:** `extract_skills` and `tokenize` imported from the package reproduce the
notebook's current outputs on the full A1 corpus — verified by running both implementations
side by side once, in a throwaway comparison, and asserting element-wise equality. The comparison
is not committed (it is a migration check, not code that outlives the decision — the *Where code
goes* rule, and Phase 3's deviation V1).

---

## 4.2 — Extract the scorers and the classifier

1. `tfidf.py`: `fit_shared_vectorizer(resume_texts, jd_texts)` — the union of the **unique** train
   resume and JD texts, one `TfidfVectorizer`, returned singular; and
   `pair_cosine_scores(vectorizer, resume_texts, jd_texts)` with the per-document vector cache
   moved inside. Write into the docstring *why* the vectoriser is shared, at the length the
   notebook comment currently does — that comment is the only surviving record of a bug that was
   live and invisible, and it must not be lost in the move.
2. `bm25.py`: `fit_bm25(resume_texts)` returning the `BM25Okapi` model (its `idf`/`avgdl` are the
   only state used downstream), and `pair_bm25_scores(model, resume_texts, jd_texts)` scoring each
   pair from **that pair's own tokens** against the fitted `idf`/`avgdl` — never
   `model.get_scores()` plus a text→row lookup into the fit corpus. That lookup-based shape is a
   third bug (D24), found and fixed 29 Aug while resolving Q19: `get_scores` can only score
   documents present in the corpus it was built from, so under the leak-free split — zero resume
   overlap between train and test, by construction — every test lookup missed and BM25 test scores
   were silently `0.0` for the entire test set. The per-JD token cache moved inside
   `pair_bm25_scores` is still a correctness-scale fix, separately: without it, scoring re-tokenises
   the same JD text once per row instead of once per unique JD.
3. `classifier.py`: `fit_single_feature(scores, labels, seed)` with `solver="lbfgs"` pinned
   explicitly, `random_state` passed, and **a raise on non-convergence** (design spec §3, rows 2
   and 3).
4. `evaluation/classification.py`: a `Report` dataclass carrying accuracy, macro-F1, per-class
   figures, the confusion matrix **and the majority-class floor**, refusing construction at
   `n=0` — the same discipline `metrics.Figure` already enforces for retrieval. The floor travels
   *with* the report rather than being recomputed by each caller, because an accuracy below the
   floor is the single most misreadable number this baseline produces.

**Acceptance:** a notebook run using only the package reproduces today's four figures exactly —
TF-IDF 0.4917 / 0.3818, BM25 0.4598 / 0.3466, on the leak-free `data/processed/fit/` split with
all three bugs fixed (this is the corrected notebook as of 29 Aug, not the original). If any
moves, the refactor changed behaviour and is wrong, regardless of which direction it moved.

---

## 4.3 — The entrypoint and the golden freeze

> **Q19 resolved 29 Aug: leak-free.** The 3,990 / 659-pair leak-free `data/processed/fit/` split
> is the baseline's data source, decided and already in effect in the notebook. The 4.2 acceptance
> figures above already reflect it, so this task freezes them as the golden record from the start
> — there is no pre/post-migration reference to reconcile.

1. `run.py` with `--seed`, `--check` and `--out`. `--seed 0` default, matching `build`. No
   `--split-source` flag: Q19 fixed the source to `data/processed/fit/`, and A9 keeps re-splitting
   out of scope, so a flag selecting between splits would be surface area for a choice this plan
   does not offer. It: loads the split (`resume_id`/`jd_id` already present in the processed
   parquet — no `mint_ids` step needed, unlike the shipped-split notebook code this replaces),
   augments, fits the shared vectoriser on train only, scores train and test for both models, fits
   both classifiers, builds both reports, writes `data/processed/baselines/` and
   `output/baselines/baseline-metrics.json`.
2. Register nothing in `data.build` — this is not a data artefact (design spec §5.1).
3. **Measure the float question before answering it (Q23).** Run three times, and once more with
   `OMP_NUM_THREADS=1`, and record whether the confusion-matrix integers and the metric floats
   are stable. Set the tolerance from that measurement and write the measured evidence into the
   implementation record. If counts prove stable, exact integer comparison is adopted and the
   thread pin is unnecessary.
4. Write `baseline-metrics.json` to the schema in the design spec §4 — **no timestamp** (W6), and
   with the vocabulary digest, the skill-pattern digest and the fitted coefficients included.
5. Freeze it: commit the JSON in the same commit as the code that produces it.

**Acceptance:** `run --seed 0` on a clean tree reproduces the committed JSON byte-for-byte, at the
tolerance settled in step 3; `git status` shows no change after a re-run. The tolerance, and the
three-run evidence behind it, are recorded — not chosen by preference.

---

## 4.4 — `--check`, the tests, and the notebook

1. `run --check`: refit in memory, diff against the committed JSON, `[PASS]`/`[FAIL]` per group,
   exit 1 on mismatch, write nothing.
2. Add pytest to `[dependency-groups] dev` (Q21) and create `tests/` per the design spec §6.
   Write the three named bug regressions first — `test_shared_space_cosine_of_identical_text_is_one`,
   `test_token_pattern_matches_sklearn_default`, and `test_bm25_scores_document_outside_fit_corpus`
   (D24) — and **confirm each fails against the old implementation before it passes against the
   new one.** A regression test never shown to fail is a test of nothing.
3. `test_baseline_golden.py` wraps `--check` and skips with an explicit reason when
   `data/processed/fit/` is absent. Every other test runs on inline synthetic corpora and needs no
   data.
4. `test_notebook_is_thin.py` per the design spec §5.3.
5. Rewrite notebook 07: import from `candidate_screener.baselines`, load
   `output/baselines/baseline-metrics.json` and render *from it*, and keep only the analysis the
   package cannot express — the class-wise score distributions, the confusion matrices, the
   interpretation of accuracy-below-floor. **Delete the scratch cell that prints a full resume and
   JD** (`test_df.loc[1755]`): A1 resumes are real documents and that cell breaches D15's 5-record
   / 2,000-character output cap twice over.
6. Execute the notebook with `nbconvert --to notebook --execute --inplace` so outputs are
   committed, per the notebook rule.
7. **Q22.** If CI is wanted, add one workflow running `uv sync --frozen` and `uv run pytest`. If
   not, the definition of done gains a line in the code-review checklist. Either way say which,
   in writing — "automatically caught" and "caught by a human who remembers" are different claims.

**Acceptance:** `uv run pytest` passes on a clone with **no `data/` at all** (the golden test
skips, visibly); `run --check` passes with data present and exits 1 when a metric is perturbed by
hand; all three bug regressions are demonstrated failing against the pre-fix code.

---

## 4.5 — Documentation

Per the documentation rules, and in the same commit as the change:

- `README.md` — a "Run the baseline" block beside the existing fetch / verify / build blocks, and
  a row in *Where to look* pointing at `output/baselines/`.
- `output/baselines/README.md` — new; contents per the design spec §4.
- `AGENTS.md` — the *Where code goes* table gains nothing (the rule already covers this), but the
  Tech Stack command list gains `run`/`run --check`, and the Notebooks section gains
  `baseline-metrics.json` next to `profile-metrics.json` as a thing notebooks cite rather than
  restate. If pytest is adopted, one line on how to run the suite.
- `plan/2026-08-23-derived-artefacts/README.md` — a pointer to this phase as its successor,
  matching how the Phase 3 README points back at the data-strategy plan.
- `docs/data/cards/A1-resume-job-description-fit.md` — **not needed.** Q19 resolved to the
  leak-free split, which the card already documents from Phase 3; this baseline adds no new fact
  about the corpus for the card to carry.
- `05-implementation.md` in this folder — what was built, the Q23 measurement, and every deviation
  in a `W`-numbered table, continuing Phase 3's convention.

**Acceptance:** a reader who has only `README.md` can go from a fresh clone to a reproduced
baseline, and knows before they read a number which split it came from and what it is comparable
to.

---

## Out of scope, stated so it is not drifted into

Under assumption A9 this phase changes **no** number by intent. The following are all reasonable
next steps and none of them belong here:

| Excluded | Why it is tempting, and why not now |
|---|---|
| Using `data/vocab/` (task 3.5) instead of the hand-written `SKILL_PATTERNS` | It is the obvious improvement and the vocabulary is being built for exactly this. It is a **model change**: it would move every metric, and a refactor that moves metrics cannot be verified as a refactor |
| Scoring the baseline on the Phase 3 **retrieval pools** — Recall@10, Precision@5, nDCG@10 | Q24. Those pools were built to score the Stage 1 system and this is that system. But it is new evaluation, and the random-ranker floor it would be compared against is already measured, so it loses nothing by waiting one phase |
| Tuning the classifier, adding features, or replacing the single-feature LR | A9 |
| Anything about the A1 split's design | Phase 3 settled it; Q19 is a question about *which existing split this notebook reads*, not about re-splitting |

---

## Open questions carried into implementation

Restated from the [`README.md`](README.md) so this file stands alone at the bench.

| # | Question | Status |
|---|---|---|
| ~~Q19~~ | Shipped `data/raw/fit/` or leak-free `data/processed/fit/`? | **Closed 29 Aug — leak-free.** Notebook already switched and re-run |
| ~~Q20~~ | Metrics location and naming; one umbrella `verify` or a separate `run --check`? | **Closed 29 Aug — a new top-level `output/baselines/`, committed to git.** The umbrella-vs-separate question defaults to the separate `run --check` proposed in the design spec; not reopened |
| ~~Q21~~ | pytest, `tests/` at root, added to the dev group? | **Closed 29 Aug — yes, per the design spec §6 recommendation** |
| ~~Q22~~ | Does CI exist or should this phase add it? None found in the tree | **Closed 29 Aug — no CI added; deferred** |
| Q23 | Golden tolerance, and whether to pin `OMP_NUM_THREADS=1` | **Open — undecided.** Measured **in** 4.3, answered from that measurement |
| ~~Q25~~ | Keep the four `.pkl` files as a git-ignored cache, or stop writing them? | **Closed 29 Aug — keep, git-ignored** |
