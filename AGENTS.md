# AGENTS.md

Conventions an agent (or new team member) needs before touching anything in this
repository. **Rules only.** Why each rule exists, and every superseded one, is in
[`docs/decisions.md`](docs/decisions.md) and the plan records it links to (D45).

## Project Overview

NUS ISS PLP Group 2: an intelligent candidate-screening system. It ranks resumes against a
job description and shows the evidence. The research question (proposal v3,
`docs/proposal/`): **does more sophisticated matching earn its complexity?** The ladder is
rules → TF-IDF/BM25 → Sentence-BERT → cross-encoder. The current plan is
`plan/2026-10-06-extraction-and-rerank/`.

## Data Structure

| Corpus | Role |
|---|---|
| **A2**, Djinni (IT JDs and CVs) | **Primary benchmark.** 600 labelled pairs (60 JDs × 10 CVs) in two strata, `generic` (40 JDs) and `targeted` (20). One final label per pair in `docs/data/manifests/a2-final-labels.csv` (D41). The document-level `train` region may supervise models, the `eval` region never (D19) |
| **A1**, `cnamuangtoun` fit pairs | Training resource and **secondary** benchmark. Its labels do not reproduce (A13), so every A1 figure carries that line (D40) |
| DataTurks, SkillSpan, ESCO | Extraction and skill vocabulary |

## Tech Stack

**Python 3.12**, with **`uv` only** for packages. Never `pip install` into the venv, and
never a bare `python`. The project is an installed package (`src/` layout), so `import
candidate_screener` works everywhere, notebook kernels included.

```bash
uv sync                          # classical stack + dev tools
uv sync --all-groups             # + `ml` (torch CPU, sentence-transformers) + `app` (streamlit)
uv run python -m candidate_screener.data.verify --all          # raw sources
uv run python -m candidate_screener.data.build --all --seed 0  # derived artefacts
uv run python -m candidate_screener.data.verify --derived      # their acceptance checks
uv run python -m candidate_screener.baselines.run --check      # the frozen Phase 4 baseline
uv run python -m candidate_screener.evaluation.final_set --check   # the reference (D41)
uv run python -m candidate_screener.evaluation.indomain --check    # the ladder vs the reference
uv run python -m candidate_screener.evaluation.stage1 --check      # X rungs, K1–K5, S2 (text_v2)
uv run pytest                                                   # the suite (no data needed)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/NN-*.ipynb
```

Torch comes from the PyTorch **CPU** index (`[tool.uv.sources]` in `pyproject.toml`). The
base `uv run pytest` must pass without the `ml` and `app` groups: model-dependent tests
`importorskip`.

## Where code goes

| Kind | Location |
|---|---|
| Reusable project code | `src/candidate_screener/…`, invoked as `python -m candidate_screener.<module>` |
| Data layer (sources, splits, pools, weak pairs) | `data/` |
| The final annotation set and the evaluation protocol | `evaluation/final_set.py`, `evaluation/indomain.py` |
| Matching systems (ladder rungs) | `matching/`: every rung implements `matching.base.System` |
| Extraction (vocabulary, rules, NER) | `extraction/` |
| Evidence layer and end-to-end pipeline | `evidence/`, `pipeline.py` |
| Recruiter UI | `app/`: rules in `intents.py` (tested), Streamlit as thin transport |
| Frozen annotation instrument | `annotation/`: read-only (D42) |
| Tests | `tests/`, run as `uv run pytest` |
| Committed records | `output/…`: config, digests and figures for things we fit, never bytes (D23 baseline) |
| Exploration and reporting | `notebooks/NN-topic.ipynb`, numbered in reading order |
| Plans and their implementation records | `plan/{date}-{plan_name}/` |
| Decision log | `docs/decisions.md` |

No runnable scripts under `plan/`; code that outlives a decision goes in the package.

## Evaluation rules

1. **Systems are scored against one reference: the final set** (D41). It is Fit / No Fit,
   built from the humans with the LLM as a secondary annotator:

   | Case | Final label |
   |---|---|
   | Both humans agree | theirs |
   | Both humans, split | the LLM breaks the tie |
   | One human | that human |
   | No human | the LLM |

   It is frozen. A changed rule is a new `rule_version` and a new file. Do not score systems
   against one annotator.
2. **Fit / No Fit for classification** (D47). `Fit` = `Good Fit` ∪ `Potential Fit`. The
   only graded figure is ordering (nDCG, pairwise) on the human grades (D52).
3. **Report `generic`, `targeted` and `total` together** (D29 amended). Never a total
   without both strata beside it.
4. **Every figure carries its *n* and a CI**, bootstrapped over **queries (JDs)**, never
   over pairs. Rung comparisons use **paired** deltas.
5. **Nothing is fitted or tuned on the final set** (D46): weights, thresholds and
   hyperparameters included. Tune on A1's doubly-disjoint CV folds or on the weak-pair
   validation split. Code that fits a model never reads `a2-final-labels.csv` or
   `judgements.csv`.
6. **No metadata at inference** (D53): systems read `text_v2`; fields are labels or
   oracles. Report within-family AUC beside the headline.
7. **Inter-annotator agreement is reported, with no target** (D44): context, not a goal.
8. **Decide before you look.** A rule or bar that decides between rungs is committed
   in the plan before the held-out rungs are scored. Anything seen
   earlier is labelled `exploratory`.

## Annotation and the LLM

- **Collection is closed** (D42). No new labels, no new batches, and no Claude subprocess
  is ever started from this repository: `llm_recheck --dispatch/--judge` and
  `prompt_calibration` raise `DispatchClosed`. The one exception is the D50 scoring batch,
  scoped in code. Reopening further needs a decision in the log.
- **The LLM is a secondary annotator, not a system** (D43). Exactly one series enters the
  final set: instrument `5d986c82`, `claude-haiku-4-5`, runs 9/10. Only a reranker may
  train on LLM output, and only on the D50 batch (D51). Ordering figures and any LLM
  scored as a system are reported on the `llm_free` subset.
- **LLM labels never enter `judgements.csv`.** They live in `llm-*.csv`, keyed by
  (`agent_sha256`, model, run, draw). Never pool across those axes.
- Every document displayed to a person passes `annotation.redact`.

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
  `output/…` records (models) instead of restating prose, so a notebook cannot drift.
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
2. **A regression test must be shown to fail against the code it guards.** The bugs this
   repository has shipped were silent (no exception, a plausible number), so each is a named
   invariant test, not a golden number.

**There is no CI** *(Q22)*. "Caught" means caught by `uv run pytest` and the `--check`
commands, run by the reviewer at workflow step 4.

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
