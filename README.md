# Intelligent Candidate Screening System

NUS ISS PLP Practice Module, Group 2. A resume–job-description screening system: ingest
resumes in their original formats, extract structured evidence, and rank candidates against a
job description with the evidence shown back to the recruiter.

**Current phase: the top-down reset** (`plan/2026-09-26-top-down-reset/`). This means:
- one final Fit / No Fit annotation set for the in-domain A2 corpus;
- the matching ladder (rules → TF-IDF/BM25 → SBERT → cross-encoder) scored against it;
- then extraction, the evidence layer and the recruiter UI.

Decisions are logged in [`docs/decisions.md`](docs/decisions.md).

## Results (27 Sep 2026)

Against one final Fit / No Fit panel label per in-domain pair (D41), with per-JD AUC and
95% CIs over 59 JDs. Full detail: [`plan/2026-09-26-top-down-reset/09-findings.md`](plan/2026-09-26-top-down-reset/09-findings.md).

| System | Per-JD AUC | Verdict under the registered rule |
|---|---|---|
| random | 0.556 | — |
| metadata rules (role keyword + experience) | 0.778 | — |
| TF-IDF | 0.797 | — |
| **metadata + TF-IDF** | **0.849** [0.802, 0.891] | **selected (D48)**: nothing richer beats it |
| SBERT, fine-tuned on weak in-domain pairs | 0.854 [0.817, 0.890] | **selected for text-only input**: +0.057 over TF-IDF, justified |
| cross-encoder (A1-trained / weak-trained) | 0.488 / 0.661 | harmful |
| learned combination of all text models | 0.828 | not resolved |

- At 500 candidates per role, metadata + TF-IDF keeps 92.9% of the fits in a
  50-candidate shortlist, in 0.84 s.
- Re-ranking that shortlist with a cross-encoder cuts Recall@10 by a quarter to a third.
- Extraction splits by role (D49): a MiniLM tagger for job postings (SkillSpan strict
  F1 0.523 vs 0.237) and a rules matcher for resumes.

## Layout

```
src/candidate_screener/     project code
  config.py                 canonical data paths
  data/                     source registry, fetch, verify, profile, splits, pools
  baselines/                the Stage 1 classical baseline (TF-IDF, BM25) and its entrypoint
  evaluation/               retrieval and classification metrics, with their reporting guards
tests/                      pytest — runs without data/ except the golden check
notebooks/                  01-06 acquisition EDA, 07 baseline, 08 extraction readiness
output/baselines/           committed model record — figures and config, never bytes
  data/                     source registry, fetch, verify, profile, splits, pools, weak pairs
  baselines/                the frozen Phase 4 classical baseline (TF-IDF, BM25) and its record
  evaluation/               final_set (the reference), indomain (the protocol), retrieval_pilot,
                            a1_secondary, latency, metrics with their reporting guards
  matching/                 the ladder: rules, lexical, SBERT, cross-encoders, hybrids
  extraction/               skill vocabulary, rules matcher, MiniLM NER, normalisation
  evidence/, pipeline.py    requirements, verbatim evidence, end-to-end `screen()`
  app/                      recruiter UI: intents (rules) + Streamlit transport
  annotation/               the frozen annotation instrument (D42)
tests/                      pytest — runs without data/ (data-dependent checks skip visibly)
notebooks/                  01-07 acquisition EDA and baseline; 08-11 the reset's findings
output/                     committed records — figures and config, never model bytes
docs/
  data/                     data catalog, per-source dataset cards, manifests
  decisions.md              the decision log (D1–D49)
  proposal/                 the module proposal this project implements
plan/                       dated plans and their implementation records
data/                       git-ignored working set (except data/vocab/, D15) — see data/README.md
```

## Setup

```bash
uv sync                     # Python 3.12, the classical stack + dev tools
uv sync --all-groups        # + ml (torch CPU, sentence-transformers) + app (streamlit)
```

## Screen candidates (the reset's pipeline)

```bash
uv run python -m candidate_screener.evaluation.final_set --build     # the reference (D41)
uv run python -m candidate_screener.evaluation.indomain --seed 0     # the ladder vs the final set
uv run python -m candidate_screener.evaluation.retrieval_pilot       # 500-candidate pools
uv run python -m candidate_screener.evaluation.a1_secondary          # A1, secondary (A13 caveat)
uv run python -m candidate_screener.extraction.evaluate              # rules vs NER
uv run streamlit run src/candidate_screener/app/streamlit_app.py     # the recruiter UI
```

Each record command takes `--check` to rebuild in memory and diff against the committed file.
Checks come in two tiers (D55):
- **cold** (`--check`) retrains and re-infers, and is the reproducibility proof;
- **warm** (`--check --warm`, where a module supports it) reuses cached model outputs.

One runner drives them all. It prints one line per check and keeps the full output in
`logs/checks/`:

```bash
uv run python -m candidate_screener.checks                 # warm, only checks your uncommitted changes affect
uv run python -m candidate_screener.checks --warm --all    # every live record (~15 min)
uv run python -m candidate_screener.checks --cold --all    # every record from scratch (~50 min)
uv run python -m candidate_screener.checks --list          # the registry, with measured times
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

For preliminary entity-extraction work, start with
[`08-entity-extraction-readiness.ipynb`](notebooks/08-entity-extraction-readiness.ipynb)
and its [execution plan](plan/2026-09-13-entity-extraction/README.md).
The notebook checks what is available in this checkout, reports metadata without displaying
real resumes, and demonstrates the existing keyword helper on synthetic text. Missing NER
sources are reported explicitly; the historical acquisition record does not imply that every
source is installed locally. The plan covers skills, titles, qualifications and experience
evidence, with supervised extraction and theme construction as subsequent work.

Continue with [`09-resume-evidence-and-themes.ipynb`](notebooks/09-resume-evidence-and-themes.ipynb)
and its [execution plan](plan/2026-09-19-resume-themes/README.md) to explore the proposed
evidence records, activity-backed themes and requirement comparisons. Its worked examples
are manually specified synthetic evidence, not extractor predictions or accuracy results.

For executable extraction on local Djinni data, use these notebooks in reading order:

- [`10-entity-extraction-inputs.ipynb`](notebooks/10-entity-extraction-inputs.ipynb): validate inputs and sample only the committed training region.
- [`11-rule-based-entity-extraction.ipynb`](notebooks/11-rule-based-entity-extraction.ipynb): extract skills/tools, titles and qualifications with exact field-relative evidence spans.
- [`12-entity-extraction-evaluation.ipynb`](notebooks/12-entity-extraction-evaluation.ipynb): check offsets, reproducibility and synthetic regressions, and report the status of labelled evaluation.

The [approved implementation plan](plan/2026-10-05-entity-extraction-notebooks/README.md)
uses 1,000 training CVs and 1,000 training JDs by default. Its versioned seed vocabulary
and context rules provide a limited offline baseline; document coverage is descriptive,
not extraction accuracy. DataTurks/SkillSpan gold evaluation remains blocked when those
sources are absent. Real document text and entity surfaces stay in git-ignored local
artifacts; saved notebook outputs contain aggregates and synthetic examples only.

The same extraction can run without Jupyter:

```bash
uv run python -m candidate_screener.extraction.run --sample-size 1000 --seed 0
uv run python -m candidate_screener.extraction.run --sample-size 1000 --seed 0 --check
```

`--check` verifies source/code provenance, local artifact hashes, exact offsets and
reproduced predictions; it does not measure extraction accuracy.

Notebook 11 now includes a dictionary of the 15 stored entity fields and a five-row
redacted preview of actual rule-extracted mentions. Continue with
[`13-transformer-entity-extraction.ipynb`](notebooks/13-transformer-entity-extraction.ipynb)
for locally run GLiNER predictions and a comparison with rules on the same training
documents. GLiNER extracts `KNOWLEDGE`, `SKILL`, `TITLE`, `DEGREE` and
`EXPERIENCE_YEARS`; the rules comparator supports the original four labels only.
Experience spans preserve stated durations, ranges and minimums in CV claims or
JD requirements. They do not infer tenure from dates or determine job fit.
The saved five-label run has its own aggregate record at
`output/extraction/transformer-experience-years-metrics.json`; the original
four-label record and historical notebook 14 results are retained.
After combining the extraction branches, the seed-rule inspection record is
preserved as `output/extraction/rule-inspection-metrics.json`.
`output/extraction/extraction-metrics.json` describes the separately evaluated
skill rules/NER campaign; the two records have distinct schemas and populations.
Neural scores are uncalibrated; context flags still use a separate heuristic, and
unlabelled coverage/disagreement does not establish accuracy or superiority.

The transformer stack uses a separately locked uv environment to support Intel macOS
without downgrading the project's existing dependencies:

```bash
uv sync --project runtime/transformer
uv run python -m candidate_screener.extraction.model_assets
uv run python -m candidate_screener.extraction.transformer_run --sample-size 100
uv run python -m candidate_screener.extraction.transformer_run --sample-size 100 --check
```

Inference is local, with five-per-type smoke testing followed by 100 CVs and 100 JDs
and a 30-minute inference budget. Checkpoint and tokenizer revisions/hashes, runtime
versions, original offsets and field completion are recorded. Real neural evidence
stays in `data/processed/extraction/transformer-runs/<run_id>/entities.parquet`;
notebook 13 prints the actual artifact directory and uses synthetic neural previews.
See the [transformer plan](plan/2026-10-06-transformer-entity-extraction/README.md).

For charts and real examples from the generated outputs, open
[`14-entity-extraction-eda.ipynb`](notebooks/14-entity-extraction-eda.ipynb).
It compares GLiNER and rules on the same 100 CVs and 100 JDs, explores mention
frequencies, per-document counts, scores, field/context patterns and disagreements,
and exports figures to `output/extraction/eda/`. Model scores and threshold-retention
curves describe predictions, not extraction accuracy. This notebook uses existing
artifacts and regenerates only the matching rule predictions; it performs no
additional neural inference or training.

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

   *Superseded 26 Sep 2026 (D40).* The attainable ceilings (0.7806 / 0.7188 / 0.983) are **retired, not restated**. They are computed from A1 labels that do not reproduce (A13), and A1 is now a secondary benchmark. A1 figures are quoted with *n*, CI and the A13 line, with no "% of attainable". The primary benchmark is the A2 final set (D41). See `plan/2026-09-26-top-down-reset/`.

## The annotation session

One session, ~250 judgements, ~2.5 team-days *(decision D25)* — 200 in-domain Djinni pairs and
50 already-judged A1 pairs mixed in blind to test whether A1's own labels hold up. Read
[`docs/annotation-guide.md`](docs/annotation-guide.md) first.

```bash
uv run python -m candidate_screener.data.build --task indomain judging-queue --seed 0
uv run python -m candidate_screener.annotation.ui --annotator <your name>   # label
uv run python -m candidate_screener.annotation.queue --progress             # how far in
# dispatch:  data/processed/indomain/judging-queue.csv   (git-ignored, redacted, blind)
# key:       docs/data/manifests/judging-queue.csv       (committed, ids only, no text)
# labels:    docs/data/manifests/judgements.csv          (committed, header until the session runs)
```

The dispatch file carries three columns and nothing else. An annotator who could see why a pair
was selected would know which ones already have a label and would anchor on the expected answer.

**Pairs are labelled in a local UI** *(decisions D30, D32)*: one screen per query, three
buttons per candidate, and the top-1 shortlist question asked once over a complete field — which
the shuffled flat file could not do, since it scatters a JD's candidates and carries no key to
put them back together. Both corpora are served here and interleave: in-domain groups are 10
candidates, A1 recheck groups 1 to 4. Stop whenever; re-run to resume, since `judgements.csv`
is the only progress there is. Creating a batch picks job titles from the **41** reachable in
the corpus — the generic batch covers 22, which is what an unstratified draw of 40 JDs landed
on rather than a designed scope.

**Each JD gets 10 candidates, 8 of them from its own role family** *(decision D31)*. The first
cut drew 5 from one undifferentiated pool, which put only **5.0%** of pairs in the JD's own
family; a pilot labelled **28 of its first 30 pairs `No Fit`**. A set of obvious negatives costs
the same hours as one that discriminates and tells you nothing about which system is better. The
realised rate is now 80.0%, and every in-domain precision figure has to say that this pool is
deliberately enriched — it is a comparison instrument between systems, not an estimate of
production precision.

### Growing it, pausing it, re-scoping it *(D28)*

The set is a **frozen, append-only batch campaign** — `docs/data/manifests/indomain-batches.json`
is its config. Batch *N* draws only from what batches 1..*N*-1 left behind, so appending can
never change an existing `pair_id`, and labels already collected stay valid.

```bash
uv run python -m candidate_screener.annotation.sample --add-batch --n-jds 20        # wider
uv run python -m candidate_screener.annotation.sample --add-batch --reuse-jds       # deeper
uv run python -m candidate_screener.annotation.sample --add-batch --keywords "Data Science"
uv run python -m candidate_screener.annotation.queue  --build                       # resume
uv run python -m candidate_screener.annotation.queue  --progress                    # where you are
```

`queue --build` is the resume command: it emits only pairs with no row in `judgements.csv`, so
a session that stops at pair 130 of 250 restarts with 120 and no bookkeeping.

**Figures are quoted per stratum, and there are exactly two** *(D29)*. `generic` is the
unstratified batch — *"a typical posting from this board"*. `targeted` pools **every**
keyword-scoped batch — *"the job families we chose to cover"*. Batches within a stratum are
pooled freely; the two strata are never averaged together, because their average describes a
mixture that exists nowhere. `verify --derived` prints the per-stratum counts on every run.

The session is **two annotators, both covering every title**, working from one queue split by
hand, with 30% double-labelled for κ. Assigning titles by expertise was considered and
rejected: it confounds annotator with title, and the confound reaches the pooled figure.

## Data handling rules

- Nothing under `data/` is committed. Derived artefacts are reproducible from the scripts.
- **DataTurks (B1) contains direct PII** — `Name` and `Email Address` are annotated entity
  classes. De-identify before use, and never commit un-redacted content, notebook outputs
  included.
- **44 livecareer (C1/C2) resumes overlap the A1 benchmark.** Exclude
  `data/interim/c1_a1_contamination.csv` from any NER training set or distractor pool built
  off that corpus.
