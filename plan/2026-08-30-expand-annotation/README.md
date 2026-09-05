# Expand annotation: A1 recheck to 100, targeted A2 batches, option D

> **Status: draft, 30 Aug 2026.**

## Motivation

Three objectives, the first two arising from the D33 finding (A13 fails), the third
extending it:

1. **Double the human A1 recheck from 50 to 100.** The machine leg already covers 100 pairs
   (LLM_RECHECK_STRATA = 30/30/40). The human leg is 50 (RECHECK_STRATA = 15/15/20). If the
   pattern holds at n=100 — yc-vs-llm agreement still dominates A1-vs-either — it confirms
   A13 has failed. If it reverses, the n=50 finding was a sampling artefact and the ceilings
   stand.

2. **Add targeted A2 batches.** Batch 1 is generic (unstratified D14 draw, 22 of 41 reachable
   titles). 19 titles have no coverage. Targeted batches fill the gap, reported as the
   `targeted` stratum (D29), never pooled with `generic`.

3. **Option D — LLM-judge all 659 A1 pairs.** Running concurrently with objectives 1-2, not
   gated on them. The n=100 human leg *confirms or denies* the LLM's calibration after the
   fact, but does not need to complete first — the labels are collected data, and their value
   is decided when the agreement numbers land, not when the judge runs.

## Objective 1 — A1 recheck human leg to 100

### What changes

Set `RECHECK_STRATA` to `{"Good Fit": 30, "Potential Fit": 30, "No Fit": 40}` — matching
`LLM_RECHECK_STRATA` exactly. This adds 50 new A1 pairs to the annotation queue.

### Why this is safe

- **The draws nest by construction.** `a1_recheck` permutes each stratum with
  `SeedSequence(seed)`, and taking a prefix of a permutation is stable under extension.
  The existing 50 labels are on pairs that remain in the expanded 100. Verified:
  `pair_ids(human_50) ⊂ pair_ids(machine_100)` holds.
- **The human sitting is complete.** All 50 original pairs are labelled. The warning in
  `AGENTS.md` about `RECHECK_STRATA` ("silently adds pairs to a live human sitting") is
  about a *mid-sitting* change; the sitting is finished.
- **Nesting assertions continue to hold.** human_100 == machine_100, so
  `assert_recheck_nests` is trivially satisfied.

### What it costs

50 additional human judgements on A1 pairs, appearing in the UI interleaved with in-domain
groups. At the current pace (~40 pairs/session), roughly 1-2 sessions.

### Confirmation gate (no longer a decision gate)

Option D runs concurrently (objective 3), so this gate *confirms or invalidates* labels
already collected, rather than deciding whether to collect them.

After the 50 new human labels are collected, recompute the three-way comparison on n=100:

| If | Then |
|---|---|
| yc-vs-llm agreement still dominates (within-1-step > 80%) and A1-vs-either stays < 75% | Option D labels confirmed usable; proceed to restate D26 ceilings |
| A1-vs-yc agreement rises substantially on the new 50 | The n=50 finding was sampling-dependent; option D labels are collected but not acted on |

## Objective 2 — targeted A2 batches

### Titles not covered by batch 1

19 of 41 reachable titles are absent from the generic stratum:

| Title | JDs available |
|---|---|
| Android | 675 |
| Artist | 123 |
| Business Analyst | 910 |
| C++ | 968 |
| Data Analyst | 174 |
| Data Science | 543 |
| Flutter | 91 |
| iOS | 624 |
| Lead | 475 |
| Product Manager | 477 |
| Recruiter | 138 |
| Rust | 42 |
| Salesforce | 38 |
| Scala | 124 |
| Scrum Master | 64 |
| SEO | 119 |
| Security | 205 |
| Technical Writing | 95 |
| Unity | 281 |

### Batch design

A targeted batch draws JDs scoped to specific keywords. Rules from D28/D29/D31:
- Append-only — never edit an existing batch.
- Each targeted batch goes to the `targeted` stratum.
- 10 candidates per JD, 8 on-category (D31).
- JDs are drawn from those not already used by earlier batches.

### Open question

**Q32 — which titles to cover and how many JDs per batch.** The 19 uncovered titles range
from 968 JDs (C++) to 38 (Salesforce). Thin families (< 100 JDs) may not fill 8 on-category
candidates per JD. Options:

- **One big batch, all 19 titles.** Maximum coverage, but some titles may be too thin and the
  annotation cost is high (~190 JDs × 10 = 1,900 pairs).
- **Prioritise by relevance to the proposal.** Cover only titles the proposal's success
  measures need, in 1-2 batches of 10-20 JDs each.
- **Split into two batches.** Batch 2: the larger families (> 200 JDs). Batch 3: the smaller
  ones, accepting some titles may fall short of the on-category quota.

The choice depends on what the proposal needs and how much annotation budget remains under
D25. **This is the user's decision.**

## Objective 3 — Option D: LLM-judge all 659 A1 pairs

### What this is

The A1 benchmark (`data/processed/fit/test.parquet`) contains 659 JD–CV pairs with labels
from the original A1 dataset (182 Good Fit, 133 Potential Fit, 344 No Fit, across 100 JDs
and 158 CVs). The LLM recheck (D33) showed kappa(A1, llm) = 0.125 on 100 stratified
pairs. Option D extends the same judge — same agent definition, same sandbox, same
provenance — to every A1 pair.

The purpose is to produce a **complete LLM opinion** on A1, so that:

1. Every D26 ceiling figure can be restated against the LLM's labels (if confirmed).
2. Per-class agreement with A1 is measured at full population, not a 100-pair sample.
3. The comparison with the human leg (objectives 1) is available at whatever n it reaches.

### Why concurrent with objective 1

The LLM labels are **collected data** — append-only, provenance-stamped, stored in
`llm-recheck.csv`. Their value depends on the confirmation gate (objective 1), but their
*collection* does not. If the gate confirms, the labels are ready to use immediately. If it
denies, they are recorded but not acted on. Running the judge costs API time, not human
time, and costs nothing to discard.

### Instrument

The same `a1-judge` agent definition, the same isolation controls:

- **Agent:** `.claude/agents/a1-judge.md`, generated from `docs/annotation-guide.md`
  (verbatim minus operator sections). `tools: []`. `model: sonnet`.
- **Sandbox:** `judge_sandbox` — a temporary directory holding only the agent file, with
  `assert_isolated` checking every ancestor for `CLAUDE.md`, `AGENTS.md`, or `.claude/`
  context leaks.
- **Provenance:** every row carries `agent_sha256`, `prompt_sha256`, `run`, `model`,
  `judged_at`. The agent definition must not change between dispatch and judging.
- **Blinding:** `ANCHORING_FIELDS` are asserted absent from every rendered prompt.
  `redact.assert_clean` runs on the full prompt batch before dispatch.

### Scope: 659 pairs, 559 new

100 pairs already have LLM labels (the recheck, run 1). Option D dispatches the remaining
559 as new prompts. They share the same `run` number (a new run, not run 1) so the
provenance is distinct.

### Batching: ~100 pairs per `--dispatch` / `--judge` invocation

659 total pairs. 100 already judged. 559 new pairs, dispatched as **one run** and judged in
**batches of ~100 concurrent subagent calls**:

| Batch | Pairs | Notes |
|---|---|---|
| 1 | 100 | Already done (recheck run 1, all 3 classes stratified) |
| 2 | 100 | run 2, first dispatch |
| 3 | 100 | run 2, resume (judge picks up where it left off) |
| 4 | 100 | run 2, resume |
| 5 | 100 | run 2, resume |
| 6 | 100 | run 2, resume |
| 7 | 59 | run 2, final |

The batching is operator-side pacing, not a code change. `--judge --run 2 --concurrency 10`
is idempotent per `(pair_id, run)` — calling it repeatedly processes the next unfinished
batch of `CONCURRENCY` pairs. Each invocation launches up to 10 concurrent subagent
processes (the existing `CONCURRENCY = 10`), so **~100 pairs ≈ 10 sequential waves of 10
concurrent judges** per invocation, taking roughly 15-25 minutes at current rates.

The operator runs `--judge` 6 times, each processing ~100 pairs. No code change is needed
for this — the existing idempotency in `judge()` (skip `(pair_id, run)` already in
`llm-recheck-raw.jsonl`) makes repeated invocation the resume mechanism.

**Alternatively**, raise `CONCURRENCY` to 100 and run `--judge` once. This would launch
100 concurrent `claude -p` subprocesses. Whether the machine and rate limits tolerate this
is an operational question. The code supports it; the default is conservative.

### What changes in `llm_recheck.py`

The current dispatch path (`dispatch()`) serves only the recheck draw (100 pairs from
`a1_recheck`). Option D needs **all 659 pairs** from `test.parquet`. Changes:

1. **New function `dispatch_all_a1(run, seed)`** — loads all 659 pairs from
   `test.parquet`, renders prompts through the same `render_prompt` + `assert_not_anchoring`
   + `redact.assert_clean` pipeline, and writes to the same `PROMPTS` file. Idempotent:
   skips `(pair_id, run)` already dispatched.

2. **New CLI flag `--all-a1`** — selects `dispatch_all_a1` instead of the recheck-only
   `dispatch`. Example: `--dispatch --all-a1 --run 2`.

3. **`collect()` and `report()`** — already work over the full `PROMPTS`/`RAW` files with
   no run filter, so they pick up option D rows automatically. The `stratum` column stays
   `a1_recheck` (it describes the corpus, not the draw). The `annotator` string becomes
   `llm:claude-sonnet-5:run2`.

4. **`report()`** — add a section for the full-A1 comparison: kappa(A1, llm) over all 659
   pairs, broken down per class. The existing recheck-vs-human comparison stays separate.

5. **`assert_recheck_nests`** — unchanged; it governs the recheck draw (objectives 1),
   not the full-A1 dispatch.

### What does NOT change

- The agent definition (`.claude/agents/a1-judge.md`).
- The sandbox and isolation controls.
- The judge subprocess machinery (`judge_one`, `judge`, `judge_sandbox`).
- The `llm-recheck.csv` schema (`LLM_COLUMNS`).
- The recheck-specific dispatch (`dispatch()`) — it stays for reruns of the recheck.

### Output

All labels go to `docs/data/manifests/llm-recheck.csv`, the same file as the recheck.
The `run` column distinguishes option D (run 2) from the recheck (run 1). `collect()`
already handles multiple runs.

After collection, `--report` produces `output/annotation/llm-recheck-report.json` with:
- `pairwise_kappa` section: `a1_vs_llm` over all 659 pairs (run 1 + run 2 majority vote).
- `per_class_llm_vs_a1` section: agreement by A1 class over all 659.
- The existing recheck-specific sections are unchanged.

### Cost estimate

559 new pairs × median 7,801 chars per prompt ≈ 4.4M input chars. At sonnet rates, this
is modest. Wall clock depends on concurrency:
- At `CONCURRENCY=10`: ~6 batches × ~20 min = ~2 hours total, run in 6 sessions.
- At `CONCURRENCY=100`: ~1 hour, run in 1 session (if rate limits permit).

### Risk

The judge has already run on 100 stratified pairs with 0.857–1.000 test-retest reliability.
The only new risk is **rate limiting** at higher concurrency. The existing `JUDGE_TIMEOUT_S
= 600` (10 min) per pair and error capture in `judge()` handle transient failures — a
failed pair is recorded as unparsed and the batch continues.

## Implementation steps

### Step 1 — expand A1 recheck (code change)

1. In `src/candidate_screener/annotation/queue.py`, change `RECHECK_STRATA` from
   `{"Good Fit": 15, "Potential Fit": 15, "No Fit": 20}` to
   `{"Good Fit": 30, "Potential Fit": 30, "No Fit": 40}`.
2. Update the docstring/comment explaining why RECHECK_STRATA was left at 50 — it is now
   intentionally raised because the human sitting is complete.
3. Rebuild the queue: `uv run python -m candidate_screener.annotation.queue --build`.
4. Update `AGENTS.md` — the warning about never raising `RECHECK_STRATA` needs a dated note
   that it was raised after the first sitting completed.

### Step 2 — add targeted A2 batch(es)

Once Q32 is answered:

```bash
uv run python -m candidate_screener.annotation.sample --add-batch \
    --n-jds <N> --keywords "<Title1>" "<Title2>" ...
```

This appends the batch to `indomain-batches.json`, draws the pairs, and rebuilds the queue.
The queue rebuild emits only pairs absent from `judgements.csv`, so existing labels are
unaffected.

### Step 3 — annotate

Run the UI: `uv run python -m candidate_screener.annotation.ui --annotator yc`

Both the new A1 recheck pairs and the targeted A2 pairs will appear interleaved in the queue.

### Step 4 — option D: dispatch all 659 A1 pairs (code change + run)

Runs concurrently with steps 1-3.

1. Add `dispatch_all_a1(run, seed)` to `llm_recheck.py`:
   - Loads all 659 pairs from `test.parquet` directly (no stratified draw).
   - Renders each through `render_prompt` → `assert_not_anchoring` → `redact.assert_clean`.
   - Writes to `PROMPTS` (`llm-recheck-prompts.jsonl`), skipping any `(pair_id, run)`
     already there. The 100 recheck pairs are already dispatched under run 1 — they are
     **not re-dispatched** under run 2; the two runs are separate instruments on the same
     pairs, and run 1's majority-vote label is the recheck, not option D.

2. Add `--all-a1` CLI flag for dispatch mode.

3. Dispatch: `uv run python -m candidate_screener.annotation.llm_recheck --dispatch --all-a1 --run 2`

4. Judge in batches (~100 pairs per invocation, 10 concurrent subprocesses each):
   ```bash
   # Repeat until all 659 are judged (idempotent — each picks up where the last stopped)
   uv run python -m candidate_screener.annotation.llm_recheck --judge --run 2
   uv run python -m candidate_screener.annotation.llm_recheck --judge --run 2
   uv run python -m candidate_screener.annotation.llm_recheck --judge --run 2
   uv run python -m candidate_screener.annotation.llm_recheck --judge --run 2
   uv run python -m candidate_screener.annotation.llm_recheck --judge --run 2
   uv run python -m candidate_screener.annotation.llm_recheck --judge --run 2
   ```
   Or raise concurrency: `--concurrency 100` and run once.

5. Collect + report:
   ```bash
   uv run python -m candidate_screener.annotation.llm_recheck --collect
   uv run python -m candidate_screener.annotation.llm_recheck --report
   ```

### Step 5 — confirmation gate

After *both* objective 1 (n=100 human) and objective 3 (659 LLM) are complete:

| If | Then |
|---|---|
| yc-vs-llm agreement still dominates at n=100 | Option D labels confirmed; restate D26 ceilings |
| A1-vs-yc agreement rises substantially | Option D labels collected but not acted on; reassess |

### Step 6 — update `report()` for full-A1 comparison

Extend `report()` to add a `full_a1` section:
- kappa(A1, llm) over all 659 pairs (run 2 labels only — not majority-voted with run 1).
- Per-class agreement over all 659.
- The existing recheck sections (pairwise over the 100, over the human 50) stay unchanged.

## Verification

```bash
uv run pytest
uv run python -m candidate_screener.data.verify --derived
```

## Open questions

**Q33 — run 2 vs. majority vote for the full-A1 figure.** The 100 recheck pairs have
labels from both run 1 and run 2. For the full-A1 comparison (659 pairs), should we:

- Use **run 2 only** for all 659, so every pair is judged by the same instrument invocation?
- Use **majority vote** across runs where available (better on the 100, but mixes instruments)?

Recommendation: run 2 only for the headline figure, majority vote as a robustness check.
**This is the user's decision.**

## Out of scope

- Changing the LLM recheck strata (machine leg stays at 100).
- Restating D26 ceilings (that awaits the confirmation gate, step 5).
- Recruiting a second annotator (D25 budget decision, separate).
