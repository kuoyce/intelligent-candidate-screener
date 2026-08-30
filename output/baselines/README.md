# The Stage 1 classical baseline — committed record

`baseline-metrics.json` is what "the baseline" means. It plays the role
[`docs/data/profile-metrics.json`](../../docs/data/profile-metrics.json) plays for the corpus:
the models themselves are git-ignored regenerable objects, and what is committed is the
description of what was fitted and the figures it produced.

## Regenerate it, and check it

```bash
uv sync --frozen
uv run python -m candidate_screener.data.verify --all                    # inputs
uv run python -m candidate_screener.data.build  --task fit-split --seed 0  # the split
uv run python -m candidate_screener.baselines.run --seed 0               # rebuild + write
uv run python -m candidate_screener.baselines.run --check                # rebuild + diff
uv run pytest                                                            # the regressions
```

`--check` refits everything in memory, diffs it field by field against this file, prints a
`[PASS]`/`[FAIL]` per group and exits 1 on any mismatch. It writes nothing.

The models are `data/processed/baselines/` — four pickles and `scores.parquet` — and are a
**cache, never an artefact** *(D23)*. They are git-ignored, never hashed, and no acceptance
check reads them: sklearn pickles are not byte-stable across library versions, so treating them
as the record would make every `uv lock` a false failure.

## What the numbers are comparable to

Measured on the **leak-free** `data/processed/fit/` split (task 3.2): 3,990 train and 659 test
pairs, with **zero** resume and JD overlap between them by construction.

They are therefore comparable to every other Stage 1–4 figure in this project, and **not**
comparable to any number published against the shipped `cnamuangtoun` train/test partition,
which leaks 99.8% of its test resumes into train. That incomparability is the accepted price of
removing the leakage and must be restated wherever these figures appear.

## The result, and why the accuracy is below the floor

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF cosine + 1-feature LR | 0.4917 | **0.3818** |
| BM25 + 1-feature LR | 0.4598 | **0.3466** |
| Majority class ("always No Fit") | **0.5220** | 0.2286 |

**Both models beat the floor on macro-F1 and neither beats it on accuracy. This is expected,
not a defect.** A1's test split is 52.2% `No Fit`, and `class_weight="balanced"` buys recall on
the two minority classes at the cost of the majority class the floor is made of — so macro-F1
rises well above 0.2286 while accuracy falls below 0.5220. One linear threshold over three
overlapping, imbalanced classes does exactly this.

It is written down here, in `evaluation/classification.py`, and in notebook 07 so that it stops
being reopened as a bug. The floor is a field of every `Report` rather than something a caller
recomputes, precisely because this is the most misreadable number the baseline produces.

The informative failure is `Potential Fit` (F1 0.09 / 0.05): its mean score is *higher* than
`Good Fit`'s with by far the widest spread, so it carries no ordinal information and a monotone
threshold cannot place it. That is the bar the semantic and cross-encoder stages have to clear.

## `retrieval-metrics.json` — the same baseline as a **ranker** *(Q24, decision D27)*

`baseline-metrics.json` measures the baseline as a *classifier*, one pair at a time. That is
not the question the task 3.3 pools were built to ask, and for two phases nobody asked the
other one: **Q24 — score an existing system over the pools — was deferred out of Phase 4 and
deferred again in Phase 5.**

```bash
uv run python -m candidate_screener.evaluation.retrieval --seed 0     # rebuild + write
uv run python -m candidate_screener.evaluation.retrieval --check      # rebuild + diff
```

It re-uses `baselines.run.build` for the fitted vectoriser and BM25 statistics, so the ranker
recorded here is the same object `run --check` guards; there is no second fit path.

### The result

| | Recall@10 strict | Precision@5 strict | % of attainable | nDCG@10 strict |
|---|---|---|---|---|
| TF-IDF | **0.298** [0.216, 0.391] | 0.142 [0.090, 0.194] | 18% | 0.232 |
| BM25 | 0.196 [0.125, 0.279] | 0.103 [0.065, 0.148] | 13% | 0.182 |
| Random floor | 0.128 [0.061, 0.212] | 0.071 [0.039, 0.110] | 9% | 0.084 |

*n* = 31 strict / 64 graded, N100, bootstrap CI resampling queries. **The proposal's Stage 1
target is Recall@10 > 0.80.** TF-IDF reaches 0.298 — above the random floor with a
non-overlapping interval, and nowhere near the target. That is the finding, not a defect to
be tuned away before it is reported.

### What it settles about Q18

Of TF-IDF's 500 top-5 slots, **416 (83.2%) are unjudged distractors** — documents precision
counts as misses that nobody ever looked at. So the bias Q18 raised is *large*, not
negligible, and the raw Precision@5 of 0.142 is a floor rather than an estimate.

The ordering question is the one that was actually in doubt, and it resolves the other way
from the fear. Q18's argument was that the penalty **grows** with system quality, which would
compress or invert the ranking. Measured, the unjudged share falls monotonically as the system
improves — random 94.0%, BM25 86.6%, TF-IDF 83.2% — because a better system puts *more* judged
documents at the top, not fewer. The pools rank these systems in the right order, and the
mechanism that would have made them unfit for that job is not present in the observed
direction.

**This is exposure, not contamination.** How many of the 416 are *genuinely* relevant still
needs a human, and under **D26** that measurement is deliberately not taken: Q18 closes as a
standing limitation, with Recall@10 — which carries a ceiling of 0.983 and is the metric the
success measures name — as the headline. See `plan/2026-08-29-unified-judging-wave/`.

## Reading the file

No timestamp, deliberately *(Phase 3, W6)*: with one, every re-run diffs and the reproducibility
check tests nothing. The git history is the timestamp.

| Field | What it is for |
|---|---|
| `split` | Which data produced the numbers, and its counts — so a figure can never be read without knowing what it was measured on |
| `config` | Every input to the fit, including a **sha256 of the hand-written skill keyword list**, so a silent edit to it becomes a check failure rather than an unexplained metric move |
| `vocabulary.sha256` | Digest of the sorted feature names. `max_features=5000` keeps terms through an unstable `argsort`, so a numpy bump could shift which terms survive at the boundary; this makes that explicit instead of moving a metric by a hair |
| `floor` | The majority-class baseline, carried beside the models rather than left to the reader |
| `models.*.coef` / `intercept` | Turns "the metrics happen to match" into "the same model was fitted" — two different models can land on the same accuracy |
| `environment` | The five load-bearing library versions. The one non-content field, and deliberate: unlike a timestamp it changes only on a re-lock, so it does not defeat the check — it *explains* one. `--check` reports it and never fails on it |

**Comparison policy** *(Q23, measured — see the implementation record)*: integers exactly, floats
at 1e-12. Five runs — three repeats, one with `OMP_NUM_THREADS=1`, one with a different
`PYTHONHASHSEED` — produced a byte-identical file, so no thread pin is needed.
