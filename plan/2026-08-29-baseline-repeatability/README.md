# Phase 4 — Baseline Repeatability

**Date:** 29 August 2026
**Status: complete, 29 Aug 2026.** Tasks 4.1–4.5 are implemented — see
[`05-implementation.md`](05-implementation.md). Q19–Q23 and Q25 are closed; **Q24 remains open
and deliberately out of scope**. Q18 stays open in the predecessor plan.
**Predecessor:** [`plan/2026-08-23-derived-artefacts/`](../2026-08-23-derived-artefacts/) — tasks 3.2 and 3.3 done on `feat/derived-artefacts`
**Scope:** Make the classical baseline in `notebooks/07-baseline-tfidf-keyword.ipynb` reproducible from one documented command, and make a silent change to it impossible to merge unnoticed. **No model-quality work.**

## Contents

| File | Purpose |
|---|---|
| [`01-design-spec.md`](01-design-spec.md) | Module layout, the determinism inventory, where artefacts and metrics live, the commit policy |
| [`02-work-plan.md`](02-work-plan.md) | Sequenced tasks with acceptance criteria, and the gate on Q19 |
| [`05-implementation.md`](05-implementation.md) | What was built, the Q23 measurement, the three regressions shown failing, and deviations W12–W20 |

## Why this phase exists

Notebook 07 is the project's first scored system. Three properties it currently does **not** have
are the reason:

- **The baseline exists only as notebook cells.** Skill extraction, the shared vectoriser, the
  cosine scorer, the BM25 tokeniser and the evaluation are all inline. Nothing under
  `src/candidate_screener/` can produce these numbers, so "regenerate the baseline" means
  "re-run a notebook and trust whoever last edited it". That is the exact shape the *Where code
  goes* rule in [`AGENTS.md`](../../AGENTS.md) exists to prevent.
- **Two bugs of this kind were live until today, and both were invisible.** Separately-fitted
  resume and JD vectorisers put the two sides in unrelated vector spaces, so the cosine score was
  noise; and `.lower().split()` left punctuation glued to BM25 tokens, suppressing most of the
  real lexical overlap. Neither raised. Neither changed a row count. Both produced a plausible
  number. There is currently nothing that would catch either one coming back.
- **Nothing is committed that pins what "the baseline" means.** The models and predictions are
  written to `data/processed/baselines/`, which is git-ignored in its entirety, so a future run
  that quietly differs has nothing to differ *from*. Phase 3 solved the same problem for data by
  committing manifests and figures rather than bytes; this phase applies that pattern to a model.

The measured position as of today, on the **leak-free** `data/processed/fit/` test set (Q19,
resolved below) and with the BM25 fix described in the next section:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF cosine + 1-feature LR | 0.4917 | 0.3818 |
| BM25 + 1-feature LR | 0.4598 | 0.3466 |
| Majority class ("always No Fit") | **0.5220** | ~0.229 |

Both beat the majority floor on macro-F1 and neither beats it on accuracy. That is the expected
behaviour of one linear threshold on overlapping, imbalanced classes, not a defect — and it is
precisely the kind of statement that must be **recorded next to the numbers**, or the next reader
reopens it as a bug.

*(Superseded 29 Aug 2026: the figures originally drafted here — TF-IDF 0.4281/0.3521, BM25
0.4304/0.3289, floor 0.4872 — were measured on the shipped, leaking split per the pre-Q19 draft.
They are not wrong as arithmetic, only as a description of the wrong data; kept here per this
repo's superseding convention rather than deleted.)*

### A third bug found while resolving Q19

Switching the notebook to the leak-free split (0% resume/JD overlap between train and test, by
construction) surfaced a bug the shipped split had been hiding: **BM25 test scores came back as
exactly `0.0` for every single row.**

`BM25Okapi.get_scores(query)` can only score documents that were part of the corpus it was fit on
— it indexes into `self.doc_freqs`, built once at fit time. The notebook built that corpus from
train resumes only, then found each row's score by looking its resume up **by exact text match**
in that corpus. Under the shipped split, 99.8% of test resumes also appeared in train, so the
lookup almost always hit and the bug stayed invisible — the same shape as the other two: a
plausible number, no error, no row-count change. Under the leak-free split, the lookup never hits
for a test row (leak-free means zero resume overlap by construction), so BM25 contributed
literally nothing on the corrected data.

**Fixed today, directly in the notebook** (`notebooks/07-baseline-tfidf-keyword.ipynb`, cell
`8ab0a4c3`) by reimplementing BM25 scoring from its own formula: a document's term frequencies and
length, plus the corpus-fitted `idf` and `avgdl`, are sufficient to score *any* document —
whether or not it was part of the fitted corpus. This is the same "fit stats on train, apply to
any document" pattern `TfidfVectorizer.transform()` already uses. Verified against the library's
own `get_scores` source before implementing. Post-fix, BM25 test scores are non-zero, varied, and
separate the three classes (0.4598 accuracy / 0.3466 macro-F1 above).

This is now **D24** below, and needs the same treatment as the other two bugs when 4.1/4.2 extract
`bm25.py`: an invariant test, not a golden number, so it cannot come back silently.

## Inherited decisions

D1–D18 stand. Four patterns from Phase 3 are adopted here without re-litigation, because they
already solved the same problems:

| Inherited | Where it came from | How it applies here |
|---|---|---|
| Bytes are ignored, **figures and manifests are committed** | design spec §5, `docs/data/manifests/` | Pickles stay in git-ignored `data/processed/baselines/`; a `baseline-metrics.json` is committed |
| **No timestamps in a committed artefact** — byte-reproducibility is the acceptance check, and a timestamp makes it test nothing | Phase 3 deviation W6 | `baseline-metrics.json` carries no `generated` field |
| **The reporting discipline is code, not convention** — a figure cannot be constructed without its *n* | `evaluation/metrics.py` | The classification report gets the same treatment, and always carries the majority floor |
| **Scoring lives in `evaluation/`, building lives elsewhere** | Phase 3 deviation W11 | Classification metrics go to `evaluation/classification.py`, not into the baseline package |

## New decisions proposed for sign-off

| # | Decision | Why |
|---|---|---|
| **D19** | The baseline's reusable logic moves to a new **`src/candidate_screener/baselines/`** package with a `python -m candidate_screener.baselines.run` entrypoint. The notebook keeps only loading, reporting and visualisation | The *Where code goes* rule. A notebook that reimplements its own model cannot be regression-tested |
| **D20** | The committed record of the baseline is **`output/baselines/baseline-metrics.json`**, in the role `docs/data/profile-metrics.json` plays for data. Notebook 07 cites it rather than restating numbers. *(Originally proposed as `docs/baselines/`; superseded by the Q20 answer, 29 Aug — a new top-level `output/`, still committed)* | Existing rule in `AGENTS.md` §Notebooks, applied to a model instead of a corpus |
| **D21** | Reproducibility is asserted by **`run --check`**, which rebuilds in memory and diffs against the committed JSON, exiting non-zero on mismatch | Mirrors the `determinism:` check already in `verify_derived.check_fit_split` |
| **D22** | A **`tests/` suite under pytest** is introduced by this phase, with the three fixed bugs (D24 included) written as named regression tests | The repo has no tests at all today. The bugs are the proof that review alone did not catch them |
| **D23** | **Pickles are a cache, never an artefact.** They are not committed, not hashed, and not part of any acceptance check | sklearn pickles are version-fragile and not byte-stable across library versions; treating them as the record would make every `uv lock` a false failure |
| **D24** | **BM25 scoring is reimplemented from `idf`/`avgdl` against a document's own tokens** (`bm25.pair_bm25_scores`), never `BM25Okapi.get_scores` plus a text→row lookup into the fit corpus | `get_scores` can only score documents present in the corpus it was built from; a lookup-based design is silently wrong for any document held out of that corpus — exactly what the leak-free split does by construction. Found and fixed 29 Aug while resolving Q19; see "A third bug found while resolving Q19" above |

## Open questions

Seven of eight are resolved. **Only Q24 remains open**, and it is deliberately excluded from
this phase rather than blocking anything in it.

| # | Question | Status | Blocks |
|---|---|---|---|
| **Q19** | The notebook scored the *shipped* A1 partition, not the leak-free one. Should it be redefined on `data/processed/fit/{train,test}.parquet` (3,990 / 659) instead? | **Closed — leak-free.** Notebook already switched and re-run (29 Aug); see "A third bug found while resolving Q19" above. All figures in this plan now reflect the leak-free split | Unblocks 4.3 |
| **Q20** | Where do the committed metrics live — `docs/baselines/`, `docs/data/`, `docs/models/`? | **Closed — a new top-level `output/` directory**, e.g. `output/baselines/baseline-metrics.json`, **committed to git** (confirmed: it must be committed for `run --check` to have a fixed point to diff against — a git-ignored `output/` would make the check meaningless). This is a naming departure from D20/§4 of the design spec, which proposed `docs/baselines/`; the design spec is updated to match | 4.3, 4.5 |
| **Q21** | pytest, `tests/` at root, added to `[dependency-groups] dev`? | **Closed — recommendation accepted.** A11 is now a decision, not an assumption | 4.4 |
| **Q22** | Should this phase add CI? None exists in the tree | **Closed — no.** Deferred; "automatically caught" stays "caught by `uv run pytest` / `run --check`, run by the reviewer at workflow step 4" (A10) | 4.4 acceptance wording |
| **Q23** | Golden-metric tolerance: exact integer confusion-matrix counts, or allow drift? Pin `OMP_NUM_THREADS=1`? | **Closed 29 Aug — exact integers, 1e-12 floats, no thread pin.** Measured, not chosen: five runs (three repeats, one single-threaded, one with a different `PYTHONHASHSEED`) all produced a byte-identical record. Evidence in [`05-implementation.md`](05-implementation.md) | 4.3 |
| **Q24** | Should the baseline also be scored on the Phase 3 retrieval pools? | **Open — deliberately excluded** from this plan's scope | A follow-on phase |
| **Q25** | Keep the four `.pkl` files as a git-ignored cache, or stop writing them? | **Closed — keep, git-ignored.** Matches D23 and the original default | 4.3 |

Q18 (pool precision bias) remains open in the predecessor plan and is untouched here.

## Progress

| Task | Status |
|---|---|
| Notebook: switch to leak-free split, remove PII cell, fix the BM25 corpus-lookup bug | **Done, 29 Aug** — outside the formal package extraction, at the user's request, because answering Q19 required re-running the notebook to know what "leak-free" actually measured |
| 4.1 extract the pure functions | **Done, 29 Aug** — verified element-wise on all 949 unique documents |
| 4.2 extract the scorers and the classifier | **Done, 29 Aug** — all four figures and both confusion matrices reproduce exactly |
| 4.3 the entrypoint and the golden freeze | **Done, 29 Aug** — byte-reproducible; Q23 measured and closed |
| 4.4 `--check`, the tests, the notebook | **Done, 29 Aug** — 58 tests pass; all three regressions shown failing pre-fix |
| 4.5 documentation | **Done, 29 Aug** — [`05-implementation.md`](05-implementation.md), `README.md`, `AGENTS.md`, `output/baselines/README.md` |

**No number moved.** 4.1 and 4.2 were pure refactors and met their acceptance criterion: the
figures below are the corrected notebook's, reproduced by the package.

## Critical path

```
Q19 resolved (leak-free) ──► 4.3 freeze the golden metrics ──► 4.4 check + tests ──► 4.5 docs
4.1 extract pure functions ──► 4.2 extract scorers ──► 4.3
```

4.1 and 4.2 extract the *already-corrected* notebook logic (all three bugs fixed) into
`src/candidate_screener/baselines/` — their acceptance criterion is still that no number moves
relative to the corrected notebook. 4.3 freezes those numbers as the golden record now that Q19 is
resolved.

## Explicit assumptions

| # | Assumption | Justification | If wrong |
|---|---|---|---|
| A8 | "Repeatable" means: same numbers on the same `data/raw/` (or `data/processed/`) inputs under `uv sync --frozen`, on any machine. **Bit-identical floats across differing BLAS builds and CPU architectures are not promised** | Promising cross-architecture bit-identity would require pinning a BLAS and is not achievable with `uv.lock` alone | The tolerance question (Q23) has to be answered as "tolerant" rather than "exact"; §3 of the design spec already carries the mechanism |
| A9 | "The baseline" is the three now-fixed components as they stand after today — the shared TF-IDF vectoriser, the corpus-generalised BM25 scorer (D24), and a single-feature `LogisticRegression` for each. No hyperparameter, feature or model change is in scope | The task is repeatability; changing the model while making it repeatable would make it impossible to tell a refactor bug from an intended change | Split the change into two commits and re-freeze deliberately |
| A10 | **No CI exists**, so "automatically caught" is delivered as a single command in the definition of done, run by the reviewer at workflow step 4 | Verified by inspection — no `.github/`, no workflow, no pre-commit config in the tree. **Confirmed as the decision, Q22, 29 Aug** | If CI is wanted later, 4.4 gains one workflow file; nothing else in the plan changes |
| A11 | pytest is the runner | Nothing in the repository suggests another, and it is the default for a `uv`-managed `src/` layout. **Confirmed as the decision, Q21, 29 Aug** | — |
| A12 | Notebook 07 stays at its current number and name, and is rewritten in place rather than superseded by an `08-` | It is the same topic; the notebooks are numbered in reading order, not by revision | Renumber at 4.4; no code changes |
