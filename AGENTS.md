# AGENTS.md

Conventions an agent (or new team member) needs before touching anything in this
repository. **Rules only.** Why each rule exists, and every superseded one, is in
[`docs/decisions.md`](docs/decisions.md) and the plan records it links to (D45). D56–D61 have independent
`(entity/app)` and `(research)` meanings; qualify new citations and resolve
historical citations by their linked plan, as documented in the decision log.

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
uv run python -m candidate_screener.evaluation.dev_pools --check       # shortlist-recall dev set (A2 train)
uv run python -m candidate_screener.evaluation.shortlist_dev --check   # Y rungs on the dev set (~12 min)
uv run python -m candidate_screener.evaluation.shortlist_dev --variants --check  # amendment Z (Z1-Z3) on the dev set
uv run python -m candidate_screener.evaluation.shortlist_final --check   # Q8: frozen Z1/Z1L on the final set
uv run python -m candidate_screener.evaluation.stage1_classifier --check  # S1/S2: learned stage 1 vs X2 (~7 min)
uv run python -m candidate_screener.evaluation.rerank_chat --check --warm  # S3: rerankers on chat-1 (cold ~9 h: retrains L3)
uv run python -m candidate_screener.evaluation.stage1_threshold --check  # step 4a: stage-1 threshold end to end (~4 min)
uv run python -m candidate_screener.evaluation.feature_arms --check  # step 4b: feature arms, both stages (~13 min)
uv run python -m candidate_screener.inference --cache-parts  # saved concept df + TF-IDF space for the API (~40 s)
uv run python -m candidate_screener.inference --freeze       # L1 -> output/rerank/l1-chat3-model.json (done; ~2 min)
uv run python -m candidate_screener.inference --check        # the inference API vs the committed records (~12 min)
uv run python -m candidate_screener.inference --demo JD_ID   # a final-set JD's top 10 through the API (no CV text)
uv run pytest                                                   # the suite (no data needed)
uv run python -m candidate_screener.checks                      # warm checks your uncommitted changes affect
uv run python -m candidate_screener.checks --warm --all         # every live record, warm (D55)
uv run python -m candidate_screener.checks --cold --all         # every record from scratch (D55)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/NN-*.ipynb
```

Every `--check` above is registered in `candidate_screener.checks`, which runs them for you
and prints one line each. `tests/test_checks.py` fails if a new `--check` module is left out
of the registry.

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
| The reported system's inference API for apps (D63/D64, uploads only, loads and never fits) | `inference/`: `Screener` |
| Recruiter UI | `app/`: tested screening/QA services, `intents.py` fallback, Streamlit as thin transport; `portal_inference.py` binds new JDs to `inference.Screener` |
| Conversational inference | `llm/`: frozen local extractive QA/runtime; historical Qwen service; never annotation or training |
| Frozen annotation instrument | `annotation/`: read-only (D42) |
| Tests | `tests/`, run as `uv run pytest` |
| Committed records | `output/…`: config, digests and figures for things we fit, never bytes (D23 baseline) |
| Exploration and reporting | `notebooks/NN-topic.ipynb`, numbered in reading order |
| Plans and their implementation records | `plan/{date}-{plan_name}/` |
| Decision log | `docs/decisions.md` |

No runnable scripts under `plan/`; code that outlives a decision goes in the package.

Recruiter LLM answers use validated references to screening snapshots and redacted
evidence. Conversational answer models cannot author ranks, scores or Fit calls. Application model
selection does not change the research winner; missing weights are an unavailable
state, never a reason to train at startup. See the approved contracts under
`plan/2026-10-12-streamlit-llm-chatbot/`.

Hiring Desk v1 preserves existing CSV JD/CV reviews independently of the live
ranking adapter. Operational JD revisions and live ranking records stay outside
benchmark labels/training. New handoff packages contain ten redacted CVs;
historical eight-CV packages remain readable. Department email captures are local
demonstrations, never reported as mailbox delivery (D60 (entity/app)).

New JDs rank through the inference API (plan `2026-10-24-hiring-desk-inference-api`):
`HR_PORTAL_RANKING_ADAPTER=candidate_screener.app.portal_inference:adapter` with
`HR_PORTAL_CANDIDATE_POOL=a2-final-cvs`, the 600 A2 final-set CVs (the owner's choice), defaulted
by `streamlit_app.py` (exports override; an empty value leaves ranking unconfigured). D63
answers in seconds; D64 is opt-in and runs on the adapter's worker thread (a few minutes, then
**Refresh ranking status**). The adapter passes the model only the JD text and title and
each CV's id and text (D53, D54), reads no labels, and logs failures to
`logs/hiring-desk-adapter.log`. The 60 benchmark JDs stay on the CSV export.

Primary v1 chat uses the existing RoBERTa-SQuAD2 span extractor and private
GLiNER-medium entity caches. Comparisons are assembled from validated source
evidence; Qwen generation and Qwen fine-tuning are outside this flow (D61 (entity/app)).
Preserve qualification context, explicit no-answer/partial states and source
ownership on every cache read. Entities cannot establish skill-specific tenure.

Experimental Hiring Desk v3 (`HR_PORTAL_CHAT_VERSION=v3`, D66) uses frozen Qwen
for plain prose over current source-owned evidence and bounded scoped history.
History is discourse context, never factual evidence. Keep source reauthorization,
qualification context and explicit fallback provenance. Registered paraphrase
checks are conservative grammar checks, not general semantic entailment. Keep v3
opt-in until quality acceptance; no training or model downloads at startup.

The owner-approved Qwen3.5-4B trial uses a pinned local GGUF model and isolated
llama.cpp service (`plan/2026-10-25-hiring-desk-v3/04-qwen35-trial.md`). Explicit
`SCREENER_QWEN_MANIFEST` selects it. Its shared turn budget is at most60seconds;
the extraction reader and existing0.5B runtime retain30-second limits. Preserve
model identity, tokenizer/generation equivalence and source checks on fallback.

Optional Hiring Desk v2 (`HR_PORTAL_CHAT_VERSION=v2`, D65) adds snapshot-scoped
dialogue state and checked criterion comparisons. Qwen can select/order validated
comparison relations; application code writes the factual clauses. Preserve full
qualification context, participant coverage, explicit fallback provenance and
per-turn inference deadlines. V1 remains the default/rollback. V2 authorization
does not authorize model training, downloads or changes to the research ranking.

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
   hyperparameters included. Tune on A1's doubly-disjoint CV folds, on the weak-pair
   validation split, or on a development set drawn from the A2 `train` region (D19). Code that fits a model never reads `a2-final-labels.csv` or
   `judgements.csv`.
6. **No metadata at inference** (D53): systems read `text_v2`; fields are labels or
   oracles. The one exception is the JD title, `jd_title` = the A2 JD `Position` (D54).
   CV `Position`, `Primary Keyword` and experience fields are never inputs. Report
   within-family AUC beside the headline.
7. **Inter-annotator agreement is reported, with no target** (D44): context, not a goal.
8. **Decide before you look.** A rule or bar that decides between rungs is committed
   in the plan before the held-out rungs are scored. Anything seen
   earlier is labelled `exploratory`.

## Annotation and the LLM

- **Collection is closed** (D42). No new labels, no new batches, and no Claude subprocess
  is ever started from this repository: `llm_recheck --dispatch/--judge` and
  `prompt_calibration` raise `DispatchClosed`. The one exception is the D50 scoring batch,
  scoped in code. The recorded research chat collections are separately scoped by
  D60–D61 (research) and D62; they authorize no Claude subprocess. D56 (entity/app)
  separately authorizes human entity-span gold under `entity_annotation/`: confirm
  the source/categories/batch/annotators and freeze its document-disjoint manifest
  before real collection. It does not reopen Fit judgements, the A2 final set,
  frozen instruments or LLM dispatch. Further reopening needs a decision in the log.
- **Entity assistance** (D57 (entity/app)): model suggestions are separate from human
  reviews, explicitly loaded only on untouched primary-annotator entity-train
  documents. Preserve assistance provenance on saves/exports; assisted rows are
  not blind evaluation gold. Entity dev/test and the independent annotator remain
  blind. Reject unrepresentable training spans rather than silently rounding or
  dropping them; never expand correct gold merely to match tokenization.
- **The LLM is a secondary annotator** (D43). Exactly one series enters the final set:
  instrument `5d986c82`, `claude-haiku-4-5`, runs 9/10. Only a reranker or the stage-1
  classifier may train on LLM output: on the D50 batch (D51, D61 (research)), and,
  for the stage-2 reranker, on `chat-3` (48 train-region JDs, D62). For the stage-2
  top-10 task only, `chat-1` is the test reference and is never trained or tuned on
  (D62; steps 3-5d of plan 2026-10-18 trained on it under D61 (research), out of
  fold by JD). Elsewhere the human grades are the reference. Ordering figures and
  any LLM scored as a system are reported on `llm_free` beside `full`. Historical
  exploratory exceptions remain limited to D56 (research)'s linked plans.
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

### Running tests and checks (plan 2026-10-08-test-and-check-runtime)

Know what a command costs before you start it (measured 8 Oct 2026, 22 cores, 15 GB):

| Command | Wall time |
|---|---:|
| `uv run pytest` | ~55 s |
| `uv run pytest tests/test_x.py` | seconds |
| `checks` (warm, affected only) | 0 s to a few min |
| `checks --warm --all` | ~15 min |
| `checks --cold --all` | ~50 min |

1. **Run once, keep the log, read the log.** Redirect the full output to a file, then read
   slices of that file. Never re-run a command to see a different part of its output.

   ```bash
   mkdir -p logs; uv run pytest > logs/pytest.log 2>&1; tail -3 logs/pytest.log
   grep -n "FAILED\|Error" logs/pytest.log
   ```

   `logs/` is git-ignored. The runner already logs each check to
   `logs/checks/<name>.log` and prints one line per check.
2. **Iterate narrowly, verify broadly.** While working, run the test file you are touching,
   or `pytest --lf`. Run the full suite once, before review.
3. **`uv run pytest` alone is enough unless the change touches underlying code** (D58 (research)): code a
   committed record is built from, such as models, extractors, features, scorers or the data
   layer. New standalone tools, tests, docs, plans and notebooks need no checks.
4. **Ask the owner before any long check run.** Before `--all`, `--cold`, or a run expected to
   take more than a few minutes, name the checks, give the estimate from `checks --list`, and
   wait for the owner to confirm there is time. The affected selection can pull in superseded
   records (`retrieval_pilot` alone took 21 min on 17 Oct 2026).
5. **Checks go through the runner.** The default, `checks`, is the warm tier for the checks
   your uncommitted changes affect. Use `--affected REF` to compare against a commit, and
   `--explain` to see why a check was selected.
6. **Cold is mandatory (D55)** before:
   - a commit that touches model or training code;
   - marking a plan complete;
   - any final-set score.

   Warm proves that the code turning cached model outputs into a record is unchanged. It does
   not prove that training reproduces.
7. **Superseded records** (indomain v1, `retrieval`, `retrieval_pilot`, `a1_secondary`) run
   only with `--cold`, or when `--affected` selects them.
8. **RAM.** The runner defaults to `-j 2`. Checks peaking above 4.5 GB (`role_family`,
   `shortlist_dev`, `dev_pools`, `scoring_batch`) always run alone. Do not start other
   heavy jobs alongside it.
9. **Long jobs (training, cold checks): start once in the background, then stop and wait.**
   Anything longer than about 5 minutes runs as one background command, which writes its
   own exit marker and peak RSS:

   ```bash
   /usr/bin/time -v uv run python -m candidate_screener.<module> > logs/<name>.log 2>&1; echo "exit $?" >> logs/<name>.log
   ```

   Then **end the turn**. The harness wakes the agent when the command exits. Never wait in
   the foreground (`until …; do sleep …; done`, `timeout 590 …`, RSS-polling loops). Never
   start a background waiter with a timeout either: one died at 10 minutes and turned into
   foreground polling. Each wake-up re-reads the whole context once the prompt cache has
   expired, so one 3.5-hour training run (P8, 8 Oct 2026) became about 25 full-price
   wake-ups and hit the session limit with nothing new learned. Peak RSS comes from
   `time -v` at the end of the log, not from polling. A sub-agent does not babysit a
   multi-hour job: it starts the job, reports the log path, and returns.
10. **Time one configuration before a grid.** Train the first grid point alone, record its
   wall time, and multiply before launching the rest. If the total is more than about an
   hour, say so before starting it.

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
- **Run `uv run pytest`.** Add the checks for the records the change can affect only when it
  touches underlying code (`python -m candidate_screener.checks`; `--cold` where D55 requires
  it), and ask the owner first if the run is long (Testing rules 3-4). With no CI in the
  repository, this step is the only thing that runs them

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
