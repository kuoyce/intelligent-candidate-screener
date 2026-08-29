# Intelligent Candidate Screening System

NUS ISS PLP Practice Module, Group 2. A resume–job-description screening system: ingest
resumes in their original formats, extract structured evidence, and rank candidates against a
job description with the evidence shown back to the recruiter.

**Current phase: baseline repeatability (Phase 4) — in progress.** All eleven adopted sources
are downloaded, verified and documented; the leak-free evaluation split and the retrieval pools
are built; the Stage 1 classical baseline is extracted into the package, frozen as a committed
record and covered by a regression suite.

## Layout

```
src/candidate_screener/     project code
  config.py                 canonical data paths
  data/                     source registry, fetch, verify, profile, splits, pools
  baselines/                the Stage 1 classical baseline (TF-IDF, BM25) and its entrypoint
  evaluation/               retrieval and classification metrics, with their reporting guards
tests/                      pytest — runs without data/ except the golden check
notebooks/                  01-06 the acquisition EDA, 07 the baseline (executed, outputs committed)
output/baselines/           committed model record — figures and config, never bytes
docs/
  data/                     data catalog, per-source dataset cards, manifests
  proposal/                 the module proposal this project implements
plan/                       dated plans and their implementation records
data/                       git-ignored working set — see data/README.md to reconstruct it
```

## Setup

```bash
uv sync                     # Python 3.12, creates .venv and installs the project
```

## Acquire the data (~676 MB)

```bash
uv run python -m candidate_screener.data.fetch   --all   # download every automatable source
uv run python -m candidate_screener.data.verify  --all   # files, row counts, SHA-256 digests
uv run python -m candidate_screener.data.profile --all   # reproduce the catalog's figures
```

Two sources need a human: the Kaggle PDF corpus (account + API token) and ESCO
(free registration). `fetch` prints the exact steps for whichever is missing.
`data/README.md` has the full guide.

```bash
uv run jupyter lab notebooks/       # the EDA
```

## Build the derived artefacts

```bash
uv run python -m candidate_screener.data.build  --all --seed 0   # splits, pools, vocabulary
uv run python -m candidate_screener.data.verify --derived        # acceptance test on them
```

Builders are pure functions of (`data/raw/`, seed). Nothing built is committed; what *is*
committed is the manifest naming each document by content hash, so the artefacts are
reconstructible byte-for-byte — see [`docs/data/manifests/`](docs/data/manifests/).

## Run the baseline

```bash
uv run python -m candidate_screener.baselines.run --seed 0   # fit, score, write the record
uv run python -m candidate_screener.baselines.run --check    # assert it still reproduces
uv run python -m candidate_screener.evaluation.retrieval --seed 0  # the same model, ranked over the pools
uv run pytest                                                # unit + regression suite
```

The Stage 1 classical baseline (TF-IDF cosine and BM25, each over a single-feature logistic
regression) lives in `src/candidate_screener/baselines/`, and its figures are committed to
[`output/baselines/`](output/baselines/README.md). The fitted models are *not* committed — they
are a regenerable cache under git-ignored `data/processed/baselines/`. `--check` refits in
memory and diffs against the committed record, so a run that quietly differs fails loudly.

`pytest` needs no data: every test but the end-to-end golden check runs on inline synthetic
corpora and the golden check skips, visibly, when `data/` is absent.

## Where to look

| Question | Read |
|---|---|
| What data do we have, under what licence, with what defects? | [`docs/data/data-catalog.md`](docs/data/data-catalog.md) and [`docs/data/cards/`](docs/data/cards/) |
| Why these sources, and what did profiling change? | [`plan/2026-08-19-data-strategy/01-requirements-and-findings.md`](plan/2026-08-19-data-strategy/01-requirements-and-findings.md) |
| What happens next with the data? | [`plan/2026-08-23-derived-artefacts/`](plan/2026-08-23-derived-artefacts/README.md) — Phase 3, supersedes the acquisition plan's §3 |
| What does the baseline score, and what is it comparable to? | [`output/baselines/README.md`](output/baselines/README.md) — the committed figures, and why accuracy sits below the majority floor |
| What was actually built, and where did it deviate? | [`plan/2026-08-19-data-strategy/04-acquisition-implementation.md`](plan/2026-08-19-data-strategy/04-acquisition-implementation.md) |
| How should an agent work in this repo? | [`AGENTS.md`](AGENTS.md) |

## Three findings that govern the evaluation design

1. **The core benchmark's effective size is ~640 resumes and ~280 JDs**, not 8,000 examples —
   every confidence interval must be computed against the document counts.
2. **Its shipped split leaks 99.8% of test resumes into train.** The project re-splits so that
   resumes *and* JDs are disjoint, and results are therefore not comparable to published
   numbers on the shipped split. That evaluation is **31 queries**, not 659 pairs — and which
   31 you get moves by a factor of two across random seeds, so the seed is fixed and
   committed.
3. **Every metric ceiling is a property of the split, not of the corpus.** On the shipped split
   the median query has 18 relevant resumes and Recall@10 is unreachable; on the leak-free split
   it has 6, and Recall@10 reaches 0.90 for 93.5% of queries while Recall@50 saturates. The
   adopted metrics are **Recall@10, Precision@5 and nDCG@10**, each reported with its *n* and a
   bootstrap CI — enforced in `candidate_screener.evaluation.metrics`, not by convention.
4. **Precision@k on the pools is capped below 1.0, and Recall@10 is not.** Many queries have
   fewer than *k* judged-relevant resumes, so a perfect ranker cannot fill the top *k*:
   Precision@5 tops out at **0.7806** strict, Recall@10 at 0.983. The cap is a property of the
   split — identical at every pool depth — so every precision figure is reported with its
   ceiling beside it *(D26)*. TF-IDF's Recall@10 is **0.298** [0.216, 0.391] against the
   proposal's 0.80 target; that gap is the finding, not a defect to tune away before reporting.

## The annotation session

One session, ~250 judgements, ~2.5 team-days *(decision D25)* — 200 in-domain Djinni pairs and
50 already-judged A1 pairs mixed in blind to test whether A1's own labels hold up. Read
[`docs/annotation-guide.md`](docs/annotation-guide.md) first.

```bash
uv run python -m candidate_screener.data.build --task indomain judging-queue --seed 0
# dispatch:  data/processed/indomain/judging-queue.csv   (git-ignored, redacted, blind)
# key:       docs/data/manifests/judging-queue.csv       (committed, ids only, no text)
# labels:    docs/data/manifests/judgements.csv          (committed, header until the session runs)
```

The dispatch file carries three columns and nothing else. An annotator who could see why a pair
was selected would know which ones already have a label and would anchor on the expected answer.

## Data handling rules

- Nothing under `data/` is committed. Derived artefacts are reproducible from the scripts.
- **DataTurks (B1) contains direct PII** — `Name` and `Email Address` are annotated entity
  classes. De-identify before use, and never commit un-redacted content, notebook outputs
  included.
- **44 livecareer (C1/C2) resumes overlap the A1 benchmark.** Exclude
  `data/interim/c1_a1_contamination.csv` from any NER training set or distractor pool built
  off that corpus.
