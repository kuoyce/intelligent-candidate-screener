# Development: setup, records, data and the annotation procedure

This page covers working on the repository natively: installing the environments, the record
commands of the reset's pipeline and the checks runner, acquiring the raw data, building the
derived artefacts, the frozen Phase 4 baseline, the extraction commands, where to look for
what, the findings that govern the evaluation design, and the (closed) annotation session.
The full command list, the evaluation and testing rules, and what each check costs are in
[`AGENTS.md`](../AGENTS.md); read it before changing anything. The notebooks are described in
[`notebooks/README.md`](../notebooks/README.md). Moved from README.md on 1 Nov 2026.

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

Some of these records are **superseded** (AGENTS.md, Testing rule 7): `indomain` without
`--text-variant v2` (indomain v1), `retrieval`, `retrieval_pilot` and `a1_secondary` run only
with `--cold`, or when `--affected` selects them. The live in-domain record is
`evaluation.indomain --check --text-variant v2`.

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

Ask the owner before any `--all`, `--cold` or other long run (AGENTS.md, Testing rule 4).

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

## Extraction commands

The extraction notebooks 10-12 (see [`notebooks/README.md`](../notebooks/README.md)) can run
without Jupyter:

```bash
uv run python -m candidate_screener.extraction.run --sample-size 1000 --seed 0
uv run python -m candidate_screener.extraction.run --sample-size 1000 --seed 0 --check
```

`--check` verifies source/code provenance, local artifact hashes, exact offsets and
reproduced predictions; it does not measure extraction accuracy.

The transformer stack (notebook 13) uses a separately locked uv environment to support Intel
macOS without downgrading the project's existing dependencies:

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
See the [transformer plan](../plan/2026-10-06-transformer-entity-extraction/README.md).

## Build the derived artefacts

```bash
uv run python -m candidate_screener.data.build  --all --seed 0   # splits, pools, vocabulary
uv run python -m candidate_screener.data.verify --derived        # acceptance test on them
```

Builders are pure functions of (`data/raw/`, seed). Nothing built is committed; what *is*
committed is the manifest naming each document by content hash, so the artefacts are
reconstructible byte-for-byte — see [`docs/data/manifests/`](data/manifests/).

## Run the baseline

```bash
uv run python -m candidate_screener.baselines.run --seed 0   # fit, score, write the record
uv run python -m candidate_screener.baselines.run --check    # assert it still reproduces
uv run python -m candidate_screener.evaluation.retrieval --seed 0  # the same model, ranked over the pools (superseded record)
uv run pytest                                                # unit + regression suite
```

The Stage 1 classical baseline (TF-IDF cosine and BM25, each over a single-feature logistic
regression) lives in `src/candidate_screener/baselines/`, and its figures are committed to
[`output/baselines/`](../output/baselines/README.md). The fitted models are *not* committed — they
are a regenerable cache under git-ignored `data/processed/baselines/`. `--check` refits in
memory and diffs against the committed record, so a run that quietly differs fails loudly.

`pytest` needs no data: every test but the end-to-end golden check runs on inline synthetic
corpora and the golden check skips, visibly, when `data/` is absent.

## Where to look

| Question | Read |
|---|---|
| What data do we have, under what licence, with what defects? | [`docs/data/data-catalog.md`](data/data-catalog.md) and [`docs/data/cards/`](data/cards/) |
| Why these sources, and what did profiling change? | [`plan/2026-08-19-data-strategy/01-requirements-and-findings.md`](../plan/2026-08-19-data-strategy/01-requirements-and-findings.md) |
| What happens next with the data? | [`plan/2026-08-23-derived-artefacts/`](../plan/2026-08-23-derived-artefacts/README.md) — Phase 3, supersedes the acquisition plan's §3 |
| What does the baseline score, and what is it comparable to? | [`output/baselines/README.md`](../output/baselines/README.md) — the committed figures, and why accuracy sits below the majority floor |
| What was actually built, and where did it deviate? | [`plan/2026-08-19-data-strategy/04-acquisition-implementation.md`](../plan/2026-08-19-data-strategy/04-acquisition-implementation.md) |
| How should an agent work in this repo? | [`AGENTS.md`](../AGENTS.md) |

## Four findings that govern the evaluation design

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

**Collection is closed (D42).** No new labels, batches or LLM dispatch; the commands below
are kept as the historical procedure that produced `judgements.csv` and the A2 final set, not
as steps to run now (AGENTS.md, Annotation and the LLM).

One session, ~250 judgements, ~2.5 team-days *(decision D25)* — 200 in-domain Djinni pairs and
50 already-judged A1 pairs mixed in blind to test whether A1's own labels hold up. Read
[`docs/annotation-guide.md`](annotation-guide.md) first.

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
