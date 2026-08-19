# Intelligent Candidate Screening System

NUS ISS PLP Practice Module, Group 2. A resume–job-description screening system: ingest
resumes in their original formats, extract structured evidence, and rank candidates against a
job description with the evidence shown back to the recruiter.

**Current phase: data acquisition — complete.** All eleven adopted sources are downloaded,
verified and documented. Modelling has not started.

## Layout

```
src/candidate_screener/     project code
  config.py                 canonical data paths
  data/                     source registry, fetch, verify, profile
notebooks/                  01-06, the data acquisition EDA (executed, outputs committed)
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

## Where to look

| Question | Read |
|---|---|
| What data do we have, under what licence, with what defects? | [`docs/data/data-catalog.md`](docs/data/data-catalog.md) and [`docs/data/cards/`](docs/data/cards/) |
| Why these sources, and what did profiling change? | [`plan/2026-08-19-data-strategy/01-requirements-and-findings.md`](plan/2026-08-19-data-strategy/01-requirements-and-findings.md) |
| What happens next with the data? | [`plan/2026-08-19-data-strategy/03-acquisition-action-plan.md`](plan/2026-08-19-data-strategy/03-acquisition-action-plan.md) §3 |
| What was actually built, and where did it deviate? | [`plan/2026-08-19-data-strategy/04-acquisition-implementation.md`](plan/2026-08-19-data-strategy/04-acquisition-implementation.md) |
| How should an agent work in this repo? | [`AGENTS.md`](AGENTS.md) |

## Three findings that govern the evaluation design

1. **The core benchmark's effective size is ~640 resumes and ~280 JDs**, not 8,000 examples —
   every confidence interval must be computed against the document counts.
2. **Its shipped split leaks 99.8% of test resumes into train.** The project re-splits so that
   resumes *and* JDs are disjoint, and results are therefore not comparable to published
   numbers on the shipped split.
3. **Recall@10 > 0.90 is unreachable for 57% of queries by construction** — the median query
   has 18 relevant resumes. The adopted metrics are Recall@50, Precision@10 and nDCG@10.

## Data handling rules

- Nothing under `data/` is committed. Derived artefacts are reproducible from the scripts.
- **DataTurks (B1) contains direct PII** — `Name` and `Email Address` are annotated entity
  classes. De-identify before use, and never commit un-redacted content, notebook outputs
  included.
- **44 livecareer (C1/C2) resumes overlap the A1 benchmark.** Exclude
  `data/interim/c1_a1_contamination.csv` from any NER training set or distractor pool built
  off that corpus.
