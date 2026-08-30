# Implementation record — LLM recheck (D33)

**Built 30 Aug 2026 on `feat/llm-recheck`.** The harness is complete and tested, and
**all three runs have executed** — 90 judgements, 90 parsed. The finding is in
`01-findings.md`; it lands on row 1 of the reading table (**A13 fails**). Deviations V1,
V4 and V5 below record how the judging step was actually run: not through the `Agent`
tool.

## What landed

| File | What it is |
|---|---|
| `src/candidate_screener/annotation/llm_recheck.py` | dispatch, **judge**, collect, agreement, CLI |
| `.claude/agents/a1-judge.md` | the judge — **generated** from the guide, committed, 7,830 chars |
| `tests/test_llm_recheck.py` | 35 tests, 16 of them mutation-demonstrated |
| `data/verify_derived.py` → `check_llm_recheck` | 4-7 provenance checks (Q31) |
| `plan/2026-08-30-llm-recheck/01-findings.md` | the result and its reading |

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

**Resolved by V4, not by a fallback.** A fresh `claude` process *does* see the definition
— the registry is read at start-up, and every session that could have judged was started
before the file existed. What the fix could not be is a substitution:

**No tooled fallback was taken, and this is the point.** The available agent types all hold tools —
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

### V4 — the transport is a judge subprocess, not the `Agent` tool

**The approved transport does not survive contact.** Two facts, both measured rather than
assumed:

1. `Agent(subagent_type="a1-judge")` is unavailable in any session started before the
   definition was written (V1), and *this* session is one of them. Waiting for a session
   that postdates the file is not a design.
2. **A judge launched inside the repository reads `AGENTS.md`.** Probed directly — asked
   whether `AGENTS.md`, `CLAUDE.md` or `a1_label` was in its context, a judge running with
   cwd at the repository root answered **YES**. `CLAUDE.md` here is `@AGENTS.md`, which
   names D33, quotes kappa(A1, yc) = 0.010, and says in as many words that
   `judging-queue-full.parquet` carries `a1_label` for all 50 recheck pairs. The plan's
   blinding argument was entirely about *tools*; project memory is a second channel it did
   not consider, and `--exclude-dynamic-system-prompt-sections` does not close it.

The transport is now one `claude -p --agent a1-judge` **subprocess per pair**, with cwd set
to a temporary sandbox holding a byte-identical copy of the committed judge and nothing
else. `assert_isolated` walks the cwd and every ancestor and refuses any `CLAUDE.md`,
`AGENTS.md`, or a `.claude/` carrying anything but the judge. From that sandbox the same
probe answers **NO** for context and **NONE** for tools.

**This is a stricter control than the plan asked for, not a looser one.** Tool-free was the
requirement; the sandbox adds process-level isolation from the repository itself. What is
lost against the approved design is that the judge is a top-level session rather than a
subagent — a different harness system prompt around the same agent definition. Recorded as
an instrument fact, not waved away.

It also closes **Q30** outright: the redacted document text is piped from the prompt file
into a subprocess and never enters the orchestrating session's transcript.

### V5 — `--judge` is a package command

The plan gave the package `--dispatch` and `--collect` and left judging to the
orchestrating session. Under V4 judging is a subprocess loop, which is code, and code that
outlives the decision belongs in the package rather than in an agent's scratch space —
the same rule that keeps scripts out of `plan/`. `llm_recheck --judge --run N` is
idempotent per `(pair_id, run)`, runs `CONCURRENCY = 10`, refuses prompts dispatched
against a different `agent_sha256`, and records a crashed judge rather than dropping it.

The `Verification` block in `README.md` gains a line for it.

### V6 — the machine leg runs to 100 pairs, the human leg stays at 50

Asked for after the first result. The plan sized the recheck at 50 because 50 was what a
human could afford; the machine leg does not draw on that budget, so it is the one worth
widening. It takes kappa(A1, llm) from n=50 to n=100 and — the reason it is worth doing —
**A1's `Good Fit` cell from n=15 to n=30**, which is where the finding actually lives.

Three properties make it safe, and each is asserted rather than assumed:

1. **The draws nest.** `queue.LLM_RECHECK_STRATA` is `{Good 30, Potential 30, No Fit 40}`
   against `RECHECK_STRATA`'s `{15, 15, 20}`. `a1_recheck` takes `order[:n]` from one
   permutation per stratum, so raising `n` appends and never reshuffles — the 100 contain
   the 50 exactly. `assert_recheck_nests` runs in dispatch and `verify --derived` carries
   the same check. Without it, the n=50 figures already in the card, the catalog and
   `AGENTS.md` would stop being a subset of the n=100 ones and the two could not be quoted
   side by side.
2. **The human queue is untouched.** `session.load_units(seed, strata=None)` defaults to
   `RECHECK_STRATA`; only the LLM path passes the wider set. Had the constant simply been
   raised, the outstanding human A1 pairs would have gone from 9 to 59 mid-sitting.
   `test_the_human_queue_is_not_widened_by_the_llm_leg` pins the default.
3. **The report quotes both.** `pairwise_kappa_over_the_human_50` recomputes every leg on
   the original draw, so the published n=50 numbers stay checkable rather than being
   silently replaced by a wider figure computed over pairs no human ever saw.

**One bug this shook out, worth recording because it failed loudly by luck.**
`ui.corpus_text()` is `lru_cache`d and also built its A1 text from `RECHECK_STRATA`. With
units drawn from the wider set and text from the narrower one, half the pairs had no
document — caught by `serve_group`'s "a document in this group has no text" assertion, which
exists for exactly this. `corpus_text` now takes the strata as a hashable key, dispatch
passes one draw to both consumers, and
`test_the_text_lookup_covers_every_pair_the_llm_draw_serves` checks the coverage instead of
relying on the downstream assertion.

**What the widening cannot buy.** kappa(`yc`, llm) and kappa(A1, `yc`) stay capped at n=50:
the additional 50 were never in the human queue. Widening the human leg is a D25 budget
decision and is not made here.

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
| `assert_isolated` is a no-op | `test_the_judge_refuses_a_working_directory_that_carries_project_context` |
| judging drops its `done` filter | `test_judging_is_idempotent_per_pair_and_run` |
| the `agent_sha256` comparison is dropped | `test_judging_refuses_prompts_dispatched_against_a_different_judge` |
| a crashed judge is skipped | `test_a_judge_that_crashes_is_recorded_rather_than_dropped` |
| the wide draw taken at another seed | `test_a_diverged_draw_is_refused_rather_than_quoted` |
| `load_units` defaults to `LLM_RECHECK_STRATA` | `test_the_human_queue_is_not_widened_by_the_llm_leg` |
| `corpus_text` ignores its `strata_key` | `test_the_text_lookup_covers_every_pair_the_llm_draw_serves` |
| `pairwise_kappa_over_the_human_50` dropped | `test_the_report_still_quotes_the_human_50_alongside_the_wider_draw` |

## Verification at this commit

- `uv run pytest` — **187 passed** (`test_llm_recheck.py`: 22 at the harness commit, **35**
  now; **16** mutation-demonstrated)
- `baselines.run --check` — reproduces
- `data.verify --derived` — the `llm-recheck` section passes **7/7**, including the two
  provenance checks that only have data to check now that the runs have happened. The suite's one
  `FAIL` is pre-existing and unrelated: `resume: 401 already-judged pairs re-dispatched`
  tests the **committed queue snapshot** rather than a fresh build, so it goes red on the
  first label of any session and stays red until the queue is rebuilt. Rebuilding to clear
  it would shrink the audit trail; re-framing the check to build in a temp directory is the
  fix, and it is not in this plan's scope.

## What was run

1. **Probe first — measured, not assumed.** `tools: []` is asserted by a test and by
   `verify --derived`, but neither proves the harness *honours* it (data rule 3 applies to
   agent frontmatter as much as to a dataset card). Asked to name every tool it can call,
   the judge returned `{"label":"No Fit","reason":"NONE"}`. A second probe asked whether
   `AGENTS.md`, `CLAUDE.md` or `a1_label` was in its context: **YES** from the repository
   root, **NO** from the V4 sandbox. The second probe was not in the plan and is the reason
   V4 exists.
2. **Run 1 — 50 pairs, concurrency 10.** 50/50 returned, 50/50 parsed. Later extended to
   **100** under deviation V6; 100/100 parsed.
3. **`--collect`, then 10 `reason` fields read by hand.** All ten cite specific content
   from both documents — job-specific stacks against CV-specific employers and skills. The
   judge answered the question it was asked; it did not pattern-match length.
4. **Runs 2 and 3 — the seeded 20-pair subsample, 40 spawns.** Taken because run 1 gave
   kappa(`yc`, llm) = 0.259 with a CI overlapping kappa(A1, llm)'s, which is exactly the
   condition §5 named for going to three runs. Self-consistency came back 0.857 / 0.857 /
   1.000.
5. **`--report`, then the reading in `01-findings.md`.** 140 rows, 3 runs, 0 unparsed.
6. **V6 — the machine leg widened to 100**, and the 50 additional `reason` fields
   spot-checked the same way. The 9 human A1 pairs outstanding at the previous commit were
   labelled in parallel, so the human leg is complete at 50 and no figure here is
   provisional.

**Row 1 of the reading table. A13 fails.** kappa(A1, llm) = 0.125 [0.013, 0.241] n=100;
kappa(A1, `yc`) = 0.034 [-0.162, 0.225] n=50; kappa(`yc`, llm) = 0.275 [0.127, 0.457] n=50.
Under binary collapse the A1 legs are 0.113 and 0.031 against 0.425 between the blind
judges. The judge agrees 0.90 with A1's `No Fit` (n=40) and **0.067 with its `Good Fit`**
(n=30), reading 26 of those 30 as `No Fit` — the whole of the small positive kappa comes
from the majority-negative class.

## Still open

- **Q28 — A1's label provenance is not recorded on its card.** No longer blocks
  interpretation: shared bias would *inflate* kappa(A1, llm), and the observed value is
  0.029 (0.000 collapsed). Still worth recording on the card.
- **Q29 — truncation — closed.** Nothing truncates on the way out (`chars_sent` median
  8,632, full redacted documents) and nothing on the way in: the `reason` fields cite
  content from both documents, and 0.95-1.00 run-to-run agreement is not what a judge
  reading a variably-clipped prompt produces.
- **Q30 — closed by V4**, and more strongly than the plan closed it: the document text
  never enters the orchestrating session's transcript at all.
- **Q31 — closed.** `check_llm_recheck` checks provenance, never determinism.
- **The human leg is closed at 50.** kappa(A1, `yc`) moved 0.010 (n=41) to 0.034 (n=50)
  when the last 9 landed — still at chance, and the reading is unchanged.
- **D26's ceilings are now in doubt and have not been changed.** Restating 0.7806 / 0.7188
  is a separate change that supersedes in place, and it is out of this plan's scope.
- **New — the judge's own bias is unmeasured.** One model, one prompt, no temperature
  control, and markedly severe (40/50 `No Fit`). The finding rests on the *ordering* of
  three kappas, not on any single label of its being right.
