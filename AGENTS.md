# AGENTS.md

Conventions an agent (or a new team member) needs before touching anything in this
repository: what the project is, where code and data live, and how work is planned,
implemented and recorded.

## Project Overview

<!-- TODO: fill in. -->

## Data Structure

<!-- TODO: fill in. -->

## Tech Stack

**Python** is the core language for data processing, analysis and machine learning.

Package management is **`uv` only** — never `pip install` into the venv, never a bare
`python`.

```bash
uv sync                                          # install
uv run python -m candidate_screener.data.verify --all
uv run python -m candidate_screener.baselines.run --seed 0    # fit + freeze the baseline
uv run python -m candidate_screener.baselines.run --check     # assert it still reproduces
uv run python -m candidate_screener.evaluation.retrieval --seed 0   # the same baseline, ranked
uv run python -m candidate_screener.data.build --all --seed 0       # splits, pools, annotation queue
uv run python -m candidate_screener.annotation.ui --annotator <name>   # the labelling UI
uv run python -m candidate_screener.annotation.llm_recheck --agent      # render the LLM judge
uv run python -m candidate_screener.annotation.llm_recheck --dispatch --run 1
uv run python -m candidate_screener.annotation.llm_recheck --judge --run 1   # isolated subprocesses
uv run python -m candidate_screener.annotation.llm_recheck --collect --report
uv run pytest                                                 # unit + regression suite
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0X-*.ipynb
```

The project is an installed package (`src/` layout), so `import candidate_screener` works
everywhere, including in notebook kernels.

It supports:

- Data ingestion and preprocessing of resumes and job descriptions (csv, pdf, html, …)
- …

Where appropriate, commonly used Python libraries (e.g. pandas, NumPy, scikit-learn) are
leveraged to streamline development and ensure maintainability.

## Where code goes

| Kind | Location |
|---|---|
| Reusable project code | `src/candidate_screener/…` — invoked as `python -m candidate_screener.<module>` |
| Tests | `tests/` at the repository root, run as `uv run pytest` |
| Committed model records | `output/…` — figures and config for things we *fit*, never bytes |
| Exploration and visualisation | `notebooks/NN-topic.ipynb`, numbered in reading order |
| Plans and their implementation records | `plan/{date}-{plan_name}/` |
| Data documentation | `docs/data/` — the catalog, and one card per adopted source |
| The annotation instrument | `src/candidate_screener/annotation/` + `docs/annotation-guide.md` |
| The labelling UI | `annotation/session.py` (rules, tested) + `annotation/ui.py` + `ui.html` (transport) |
| The LLM judge | `annotation/llm_recheck.py` + `.claude/agents/a1-judge.md` (generated, committed) |

Do **not** put runnable scripts under `plan/`. A plan is a record of a decision; code that
outlives the decision belongs in the package. (This was deviation V1 of the acquisition
phase.)

## Evaluation and annotation

Two rules that were each bought with a phase of work.

**Every Precision@k figure travels with its ceiling** *(decision D26, closing Q18)*. The pools
carry a judged-supply ceiling — many queries have fewer than *k* judged-relevant resumes, so a
perfect ranker scores below 1.0 by construction. `metrics.attainable_ceiling` computes it; the
ceiling is a property of the **split**, identical at N20/N100/Nfull, so it cannot be mitigated
by choosing another variant. Precision@5 tops out at **0.7806** strict / **0.7188** graded. A
system at 0.45 is at 58% of attainable, not 45%. **Recall@10 is the headline retrieval metric**
— its ceiling is 0.983 and it is what the proposal's success measures name.

Q18 is closed as a **standing limitation**, not by a judging wave. 83.2% of the baseline's
top-5 slots are unjudged distractors, so the bias is large; what makes it tolerable is that the
unjudged share *falls* with system quality (random 94.0%, BM25 86.6%, TF-IDF 83.2%), so the
pools rank systems in the right order.

**Manual annotation is capped at one session** *(decision D25)* — ~250 judgements, ~2.5
team-days, out of a ~30 team-day project. Adding an annotation wave is a budget decision, not a
methodological one, and it needs the same arithmetic doing again. The deferred wave's design is
in `plan/2026-08-29-unified-judging-wave/`.

Nothing reaches an annotator without passing `annotation.redact`, and nothing an annotator sees
carries `selection_reason` — a reader who knows which pairs already have a label anchors on it.

**The in-domain set is a frozen, append-only batch campaign** *(decision D28)*, configured by
`docs/data/manifests/indomain-batches.json`. Grow it by appending a batch spec
(`sample --add-batch`), never by re-running an existing batch with a larger `n_jds` — batch *N*
draws from what 1..*N*-1 left, and that is the only thing keeping already-collected labels
valid. `queue --build` is the resume command: it emits only pairs absent from `judgements.csv`.

**Every figure is quoted per stratum, and there are exactly two** *(decision D29)*: `generic`
(the unstratified draw, D14) and `targeted` (every keyword-scoped batch, pooled). A stratum is
coarser than a batch on purpose — batches within one differ only in which titles they cover, so
pooling them names a real population. Averaging the **two strata** does not, and is forbidden.
`stratum` is stamped on the pair and on the judgement, never looked up from the spec, so editing
a batch's title list cannot re-stratify collected labels.

The session is two annotators, both covering every title, 30% double-labelled. Do not assign
titles by expertise: it confounds annotator with title, and the confound reaches the pooled
figure as well as the per-title ones.

**The A1 recheck gets a third judge, and it is a third opinion** *(decision D33)*.
kappa(A1, yc) = 0.010 with a **bidirectional** disagreement, so it is not a strictness shift
and it does not preserve ranking. Either A1's labels are wrong — **A13 fails** and D26's
ceilings are soft — or `yc` is miscalibrated. One annotator cannot separate the two, and the
two have opposite consequences for every figure quoted against 0.7806. `annotation.llm_recheck`
runs one tool-free `haiku` judge per pair, 10 concurrent (run 1 used `sonnet`).

**It has run, and A13 fails** *(30 Aug 2026, `plan/2026-08-30-llm-recheck/01-findings.md`)*.
kappa(A1, llm) = **0.125** [0.013, 0.241] n=100 and kappa(A1, yc) = **0.034** [-0.162,
0.225] n=50, against kappa(yc, llm) = **0.275** [0.127, 0.457] n=50; under binary collapse
0.113 and 0.031 against **0.425**. The judge's own test-retest reliability is 0.857-1.000
over three runs, so its low agreement with A1 is not a noisy judge — the bound on the kappa
it could have reached was ~0.86, not ~0.3. **All of A1-vs-llm's small positive kappa comes
from one cell**: per class it agrees 0.90 on A1's `No Fit` (n=40), 0.27 on `Potential Fit`
(n=30) and **0.067 on `Good Fit` (n=30)** — 26 of those 30 read as `No Fit`. **D26's
0.7806 / 0.7188 ceilings are therefore soft**, and any figure quoted as a fraction of
attainable inherits the doubt. Restating them is a separate change that supersedes in
place; it has not been made.

**The guide now defines `No Fit` positively, and it moved the judge** *(decision D34, 5 Sep
2026, `plan/2026-09-05-prompt-calibration/02-calibration-run.md`)*. The old instrument's
disagreement with `yc` was not strictness: it matched **57 of `yc`'s 58 `No Fit`s**, and of
its own 82 only **8 cited a different profession** — the rest cited a missing named tool
(20), a missing industry (9) or a years gap (14), all of which the guide's own examples
called `Potential Fit`. The cause was the tie-break: "choose the lower one" reads to a model
as *any doubt → down*, transitively, and nothing required `No Fit` to have a reason. The
guide is now an **ordered decision** — profession, then career stage, then core activity —
with `No Fit` reachable only from the first two steps, the tie-break scoped to the
`Good`/`Potential` boundary and barred from reaching `No Fit`, a missing tool or industry or
a year or two short of a minimum named as a screening question, and **all nine worked
examples removed**. kappa(yc, llm) 0.287 → **0.352** and the binary collapse 0.421 →
**0.614** [0.447, 0.760], n=100. kappa(A1, llm) fell to 0.085: **A13 still fails**. Two
things this does not settle — run 1 was `sonnet` and run 4 `haiku` with no old-guide/haiku
control (**Q32**), and the edit is retroactive against the 500 human labels collected under
guide `842d7b0e`, which `judgements.csv` does not record (**Q33**).

**An instrument is `agent_sha256`, and figures are never pooled across two.** A guide edit
is a new judge. `judge_frames` returns `llm` for the **committed** agent alone, every
instrument on file under `per_instrument`, and `self_consistency_kappa` only for runs that
share one — a per-pair majority across all runs would blend two judges and attribute the
blend to whichever is committed, with no exception and a plausible kappa. `report()` also
quotes binary-collapse kappa beside every three-class figure, because D34 is a run where the
two moved apart.

**A draw is a population, one run holds one of them, and one file holds one of them**
*(decision D35, 6 Sep 2026, `plan/2026-09-06-recheck-file-split/`)*. `agent_sha256` is not
the only axis a figure must not be pooled across. This judge is pointed at two populations —
the 100-pair stratified recheck it is validated on, and option D's sweep of all 659 pairs of
`fit/test.parquet` — and until this change both landed in `llm-recheck.csv`. Option D was
dispatched as **run 2**, which D33 had already used for a 20-pair self-consistency sitting;
dedup keys on `(pair_id, run)`, so the 20 were skipped as "already dispatched" and absorbed.
Nothing raised, no row count looked wrong, and `run1_vs_run2` in the committed report
silently widened from n=20 to **n=100** — a self-consistency figure over 80 pairs that no
two recheck runs had ever both judged. `draw` is now stamped at dispatch and routes the
collected row (recheck → `llm-recheck.csv`, option D → `llm-recheck-full-a1.csv`);
`judge_frames` reads the **recheck file alone**; `assert_run_holds_one_draw` refuses a run
number the other draw already holds, in both directions, and refuses a pre-D35 run outright
rather than guessing (`--backfill-draw` stamps those, deriving run 2's split from
`subsample_ids` over the **50-pair** draw that was live on 30 Aug, checked against run 3's
pair set). Run 2 is the only mixed run there will ever be; `LEGACY_MIXED_RUNS` declares it,
and the guard has closed it to any further dispatch.

**Option D is collected and not analysed** *(Q34)*. `llm-recheck-full-a1.csv` is 639 rows —
the 20 pairs D33 subsampled onto run 2 stay in the recheck file, because that is the
dispatch that sent them, and joining the two files to reach 659 is the pooling D35 forbids.
Every row carries `agent_sha256 = 47fc2e48`, the **pre-D34 instrument**, whose tie-break
D34 showed was driving `No Fit`; kappa(yc, llm) moved 0.287 → 0.352 when the guide was
fixed. So a figure computed over these labels would describe a judge this repository no
longer commits. Q34 stays deferred and unworked unless option D is revisited or a decision
comes to rest on its output — and the first thing it reopens is whether the labels may be
used at all or must be re-judged under the committed instrument (`--dispatch --all-a1
--run 6`, ~659 spawns).

**Both legs are 100 pairs** *(deviation V6, updated 30 Aug 2026)*.
`queue.RECHECK_STRATA` and `LLM_RECHECK_STRATA` are both (30/30/40). The human leg was 50
(15/15/20) while the first sitting was live; raised to 100 after all 50 were labelled, so
the existing labels are a prefix of the new draw and nothing is orphaned. The nesting
assertion (`assert_recheck_nests`, `verify --derived`) is trivially satisfied. The two
consumers must still be handed the *same* draw: units from `load_units(seed, strata)` and
text from `corpus_text(tuple(sorted(strata.items())))`, or half the pairs arrive with no
document.

The raise is also **retroactive on anything seeded off the draw**. D33's `--subsample 20`
was drawn while the machine leg was still 50, so `subsample_ids` over today's 100
reproduces 7 of those 20, not 20 — which is why `backfill_draw` carries
`LEGACY_SUBSAMPLE_STRATA` and checks its derivation against run 3's pair set instead of
assuming the current strata. Measured on 6 Sep 2026, in the course of D35.

**Tool-free is necessary and was not sufficient** *(deviation V4)*. The judge runs as a
`claude -p --agent a1-judge` **subprocess whose cwd is a sandbox holding a copy of the
committed judge and nothing else** — never the `Agent` tool and never inside the repository.
Measured, not assumed: asked whether `AGENTS.md`, `CLAUDE.md` or `a1_label` was in its
context, a judge launched at the repository root answers **YES**. `CLAUDE.md` here is
`@AGENTS.md`, which names D33, quotes the kappa it is chasing and says where `a1_label`
lives. Project memory is a second leak channel and `tools: []` does not close it;
`llm_recheck.assert_isolated` does, by refusing any cwd or ancestor carrying `CLAUDE.md`,
`AGENTS.md`, or a `.claude/` with anything but the judge. **Adding a `CLAUDE.md` above the
sandbox, or judging from the repo, breaks the instrument silently** — which is why the
repository root is a test case.

**Tool-free is the control, not a preference.** The answer key is on disk:
`data/processed/indomain/judging-queue-full.parquet` carries `a1_label` for all 50 recheck
pairs. A judge holding `Read` or `Bash` reaches it in one command, and "we told it not to
look" is not the standard this repo holds human blinding to — `serve_group` redacts its own
output and `assert_clean`s it, with the rules in `session.py` rather than in the caller.
`.claude/agents/a1-judge.md` is **generated** from `docs/annotation-guide.md` (verbatim, minus
the four sections describing the human workflow) and `verify --derived` fails if the committed
judge is not what the guide renders, or if its frontmatter is not `tools: []`.

**Its labels never enter `judgements.csv`.** `annotator` is a free string, so an `llm` row
there would pass every existing check and be silently pooled into the human kappa. They go to
`docs/data/manifests/llm-recheck.csv` (the recheck draw, 240 rows) and
`docs/data/manifests/llm-recheck-full-a1.csv` (option D, 639), quoted as
`llm:{model}:run{N}` (runs 1-3: `claude-sonnet-5`, run 4+: `claude-haiku-4-5`). `collect()`
preserves per-row model provenance from the dispatch record, and routes by `draw` (D35).

**It is collected data, not derived** *(Q31)*. No seed reproduces a subagent run, so
`check_llm_recheck` verifies **provenance** — `agent_sha256`, `prompt_sha256` against the
dispatched prompt, and that every judged pair is one of the 50 — and never determinism, which
would be red on every run.

**Labelling happens in a local UI, both corpora** *(decisions D30, D32)*. `annotation.ui`
serves one query with its candidates and appends straight to `judgements.csv`, which is already
the resume mechanism. In-domain groups are 10 candidates; A1 recheck groups are 1-4, and the
two interleave. D30 first kept the recheck on the flat file so group size could not betray it;
D32 reversed that, because the corpora were **already** distinguishable by length (A1 median
5,134 chars against Djinni's 1,525) and the guide names both. What blinding still protects is
which pairs carry a label and what it says: no `a1_label` and no `selection_reason` reach the
page, and a short group gets no banner. The residual — a reader may infer that short groups are
the rechecks — is a stated limitation on the A13 figure.

Rules about what an annotator may see live in `session.py`, never in the request handler:
`serve_group` redacts its own output and `assert_clean`s it, and a group carries no
`lexical_band`, `selection_reason` or existing label. The shortlist question is gated by
`Group.asks_shortlist` — never asked on a partial re-serve of a double-labelled JD, nor on any
A1 group.

**Every JD gets 10 candidates and 8 of them share its `Primary Keyword`** *(decision D31)*.
The first cut drew 5 per JD from one undifferentiated 200-CV pool, and it measured nothing: a
random pool over 41 role families lands on-category **5.0%** of the time, and the pilot labelled
**28 of its first 30 pairs `No Fit`**. A set of obvious negatives costs the same human hours as
one that discriminates and separates no two systems. The realised rate is now 80.0%. The 2
off-category candidates are taken from the *top* of their pool, not at random — a near miss from
the wrong family is the mistake a real system makes; a random CV is a `No Fit` everything already
ranks last. A thin family (Rust has 40 CVs) cannot fill the quota, so the shortfall is taken
off-category, flagged `keyword_match=False`, and counted as `jds_short_of_quota`.

**A rebuild that would orphan a collected judgement is refused** — `sample.assert_labels_survive`.
Appending a batch cannot orphan anything by construction; *editing* one can, and D31 did exactly
that to batch 1. It was safe only because nothing but the pilot had been labelled, which is the
kind of precondition that stops being true later.

A batch's job titles come from `sample.available_titles()` — **41 reachable, of 45 in the raw
corpus**. The generic batch's 22 titles are what an unstratified draw of 40 JDs happened to
land on (D14), never a designed scope, so `indomain-report.json`'s `titles` array is an
observation. A title a report needs covered arrives as a targeted batch. `validate_keywords`
refuses an unknown one: unmatched keywords scope a batch to zero JDs and append a spec that
produces no pairs at all, with no exception and no warning.

## Data rules

1. **`data/` is git-ignored in its entirety.** Everything in it must be reconstructible from
   `data/README.md` plus the fetch scripts. Traceability comes from
   `docs/data/acquisition-manifest.json` (SHA-256 per file), not from committing bytes.
2. **Add a source by adding a `Source(...)` to `src/candidate_screener/data/sources.py`** —
   it drives `fetch`, `verify` and the card. Record the expected row count so `verify` can
   fail when a publisher re-uploads.
3. **Never trust a dataset card.** Three of them were wrong (see
   `plan/2026-08-19-data-strategy/01-requirements-and-findings.md` §3). Mark figures
   **Verified** only after measuring them yourself with `profile`.
4. **PII.** DataTurks (B1) has `Name` and `Email Address` as annotated classes; livecareer
   (C1/C2) resumes are real documents. Redact before display and before committing any
   notebook output.
5. **Contamination.** 44 livecareer resumes overlap the A1 benchmark — exclude
   `data/interim/c1_a1_contamination.csv` from NER training sets and distractor pools.

## Notebooks

- Numbered, one topic each, executed before committing so the outputs are visible on GitHub.
- Redact PII **inside the notebook** rather than relying on stripping outputs later.
- Cite measured figures from `docs/data/profile-metrics.json` (corpus) and
  `output/baselines/baseline-metrics.json` (models) instead of restating prose — numbers then
  update when the profile or the baseline is re-run, and the notebook cannot drift from them.
- A notebook **reports** a model, it does not define one. `tests/test_notebook_is_thin.py`
  fails if a code cell constructs a vectoriser, a BM25 model or a classifier — that code
  belongs in the package, where it can be regression-tested.
- Cap notebook cell output at 5 records or 2,000 characters *(D15)*. A1 resumes and livecareer
  resumes are real documents; never print one.

## Testing

`uv run pytest`. Two rules hold the suite together:

1. **It must run on a fresh clone with no `data/`.** Every test but the end-to-end golden check
   uses inline synthetic corpora; the golden check is `skipif`-ed with an explicit reason. A
   suite that needs a 676 MB download will not be run.
2. **A regression test must be shown to fail against the code it guards.** Three bugs shipped in
   the classical baseline — two separately-fitted vectorisers, a `.split()` BM25 tokeniser, and a
   BM25 corpus lookup — and each was invisible: no exception, no row-count change, a plausible
   number. Each is now one named invariant test, not a golden number, so it survives a
   legitimate change to the figures.

**There is no CI** *(Q22, deferred deliberately)*. "Automatically caught" therefore means
"caught by `uv run pytest` and `run --check`, run by the reviewer at workflow step 4 below" —
not "caught by a machine". Adding CI is one workflow file if it is ever wanted.

## Agent Workflow Guidelines

### 1. Planning Phase

Always begin by creating a detailed plan and corresponding design specifications.

- Store all plans under the `plan/` directory
- Use the naming convention: `{date}-{plan_name}`

### 2. Approval and Version Control

- Obtain confirmation/approval of the plan before proceeding
- Once approved:
  - Create a new branch
  - Commit the finalized plan to the repository

### 3. Implementation Phase

- Use a sub-agent-driven development approach to implement the solution
- Ensure implementation strictly follows the approved plan and design specifications

### 4. Code Review and Validation

- Conduct a thorough code review after implementation
- Verify that the implementation aligns with the approved plan and design
- Ensure correctness, completeness, and adherence to standards
- **Run `uv run pytest` and, where a committed record exists, its `--check` command**
  (`data.verify --derived`, `baselines.run --check`). With no CI in the repository, this step
  is the only thing that runs them

### 5. Documentation Updates

Update documentation to reflect any changes made during implementation. Ensure any
deviations from the original plan or design are clearly documented.

This includes (but is not limited to):

- `AGENTS.md`
- `README.md`
- Relevant plan files

Any change to what data exists, or to what it is known to contain, updates in the **same
commit**:

- `docs/data/data-catalog.md` — mark a corrected figure *Superseded {date}* in place rather
  than deleting the old one; the audit trail is the point.
- the source's card in `docs/data/cards/`
- the plan's implementation record, if the change is a deviation from the approved plan

## Agent Behaviour

The agent must operate in a structured, transparent, and non-assumptive manner throughout
the project lifecycle.

### 1. No Assumptions

- The agent must **not guess user intent, requirements, or constraints**
- All decisions must be based on explicitly provided information
- If any requirement, context, or ambiguity exists, the agent must:
  - Clearly highlight the uncertainty
  - Avoid proceeding with implicit assumptions

### 2. Clarification-First Approach

- The agent should proactively seek clarification when:
  - Requirements are incomplete or ambiguous
  - Multiple interpretations are possible
  - Key technical or business decisions are unspecified
- Clarifications should be:
  - Concise and specific
  - Organized for easy response

### 3. Open Questions Tracking

- The agent must maintain a **persistent list of open questions** during planning and
  implementation
- This list should:
  - Be explicitly documented (e.g. in plan files or as a dedicated section)
  - Be updated as questions are resolved or new ones arise
- No critical step should proceed without acknowledging unresolved high-impact questions

### 4. Explicit Assumptions (When Unavoidable)

- If progress is necessary and clarification is not immediately available:
  - The agent may proceed with **clearly stated assumptions**
  - Assumptions must be:
    - Explicitly documented
    - Justified
    - Highlighted for later confirmation

### 5. Transparent Reasoning

- All key decisions, trade-offs, and design choices must be:
  - Clearly explained
  - Traceable back to requirements or constraints
- The agent should favour transparency over brevity when reasoning affects outcomes

### 6. Iterative Alignment

- The agent should continuously ensure alignment with the user by:
  - Revisiting assumptions
  - Updating open questions
  - Flagging any deviations from the original plan