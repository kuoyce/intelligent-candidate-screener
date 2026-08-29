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

Do **not** put runnable scripts under `plan/`. A plan is a record of a decision; code that
outlives the decision belongs in the package. (This was deviation V1 of the acquisition
phase.)

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