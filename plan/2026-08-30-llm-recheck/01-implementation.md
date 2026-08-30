# Implementation record — LLM recheck (D33)

**Built 30 Aug 2026 on `feat/llm-recheck`.** The harness is complete and tested. **The 50
spawns have not run** — see deviation V1.

## What landed

| File | What it is |
|---|---|
| `src/candidate_screener/annotation/llm_recheck.py` | dispatch, collect, agreement, CLI |
| `.claude/agents/a1-judge.md` | the judge — **generated** from the guide, committed, 7,830 chars |
| `tests/test_llm_recheck.py` | 22 tests, 8 of them mutation-demonstrated |
| `data/verify_derived.py` → `check_llm_recheck` | 4-7 provenance checks (Q31) |

`--dispatch --run 1 --seed 0` produced **50 pairs across 32 JDs**, median prompt **8,632
chars**. With the 7,830-char agent definition that is ~4.2k tokens per spawn, ~210k for run 1.

## Deviations from the approved plan

### V1 — the judging step needs a new session

`Agent(subagent_type="a1-judge")` fails in the session that created the definition:

```
Agent type 'a1-judge' not found. Available agents: claude, claude-code-guide, Explore,
general-purpose, Plan, statusline-setup
```

Agent definitions in `.claude/agents/` are read at session start. The judging step therefore
runs in the **next** session, from the dispatched prompts, which is why dispatch and collect
are separate commands and why the prompt file is provenance-stamped rather than rebuilt.

**No fallback was taken, and this is the point.** The available agent types all hold tools —
`claude` and `general-purpose` hold everything, `Explore` holds `Read`. Any of them reaches
`data/processed/indomain/judging-queue-full.parquet`, which carries `a1_label` for all 50
recheck pairs, in one command. Substituting one and instructing it not to look would replace
a structural control with an instructed one, which is the substitution D33 exists to refuse.
A worktree does not help: `data/` is git-ignored, so it is absent from a worktree, but
nothing stops an absolute path back to the main checkout.

### V2 — *One extra question per job description* is also stripped

The plan named three operator sections to drop. The shortlist question is a fourth.
`Group.asks_shortlist` is already `False` for `corpus == "a1"`, so it is a question this
judge is never asked, and leaving it in the system prompt invites a single-pair judge to
compare against a field it cannot see. Six `##` sections remain, all instrument.

### V3 — dispatch is idempotent per (pair, run)

Not in the plan. Dispatch appends, and appending the same (pair, run) twice would let one
spawn's answer be matched against the other's prompt hash — the provenance columns would
then name a prompt that did not produce the label. `--dispatch` is now a no-op on what is
already there, the way `queue --build` is. Covered by
`test_dispatch_is_idempotent_per_pair_and_run`.

## Mutations demonstrated

Every one was applied to the working tree, the guarded test observed to fail, and the tree
restored (AGENTS.md testing rule 2).

| Mutation | Test that caught it |
|---|---|
| `render_prompt` emits `a1_label` | `test_prompt_carries_nothing_that_names_the_answer` |
| judge frontmatter → `tools: Read` | `test_agent_definition_is_tool_free` |
| `a1_groups` reads `load_judgements()` instead of its argument | `test_every_pair_is_served_regardless_of_existing_judgements` |
| `parse_return` stops checking the scheme | `test_bad_returns_are_refused` |
| `collect` also writes `queue.JUDGEMENTS` | `test_collect_never_writes_the_human_judgement_file` |
| `majority` breaks ties by count order | `test_a_tied_majority_is_excluded_rather_than_broken` |
| guide edited without `--agent` | `test_committed_judge_is_what_the_guide_renders` |
| dispatch appends unconditionally | `test_dispatch_is_idempotent_per_pair_and_run` |

## Verification at this commit

- `uv run pytest` — **176 passed**
- `baselines.run --check` — reproduces
- `data.verify --derived` — the new `llm-recheck` section passes 4/4. The suite's one
  `FAIL` is pre-existing and unrelated: `resume: 401 already-judged pairs re-dispatched`
  tests the **committed queue snapshot** rather than a fresh build, so it goes red on the
  first label of any session and stays red until the queue is rebuilt. Rebuilding to clear
  it would shrink the audit trail; re-framing the check to build in a temp directory is the
  fix, and it is not in this plan's scope.

## What the next session does

1. **Probe first — one spawn.** The plan's `tools: []` is asserted by a test and by
   `verify --derived`, but neither can prove Claude Code *honours* it. Data rule 3 applies
   to agent frontmatter as much as to a dataset card: measure it.

   ```
   Agent(subagent_type="a1-judge", prompt=
     'PROBE — do not judge. Reply with one JSON object, no prose:\n'
     '{"label":"No Fit","reason":"<names of every tool you can call, or NONE>"}')
   ```

   If the reason is not `NONE`, **stop**: the blinding control does not hold and the
   experiment cannot be run as designed.

2. **Run 1 — 50 spawns, at most 10 concurrent.** Read
   `data/processed/indomain/llm-recheck-prompts.jsonl`, pass each record's `prompt`
   verbatim, append `{"pair_id", "run", "returned_at", "raw"}` per return to
   `llm-recheck-raw.jsonl`.
3. `--collect`, then **read 10 `reason` fields by hand** before trusting any kappa.
4. `--report`, then the 2x2 reading in `01-findings.md`.
5. Runs 2 and 3 on the seeded 20-pair subsample (`--dispatch --run 2 --subsample 20`) for
   self-consistency, if run 1 lands anywhere the reading depends on precision.

## Still open

- **Q28 — A1's label provenance is not recorded on its card.** If A1's labels are themselves
  machine-generated, a high kappa(A1, llm) is shared-bias artefact rather than independent
  agreement, and the "yc is the outlier" row of the reading table is unreadable. Blocks
  interpretation, not execution.
- **Q29 — truncation.** Answered so far as dispatch can answer it: nothing is truncated on
  the way out (`chars_sent` median 8,632, full documents). Whether anything truncates on the
  way *in* to a subagent is a fact about the harness that only run 1 can establish — a
  `reason` that discusses a resume the prompt did not contain is the signal.
- **9 human A1 pairs outstanding.** Finishing them takes kappa(A1, yc) from n=41 to n=50.
  Recommended before the findings are written, not before the judging runs.
