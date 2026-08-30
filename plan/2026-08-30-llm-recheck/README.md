# LLM recheck — a third judge on the 50 A1 pairs

> **Status: implemented and run, 30 Aug 2026.** The result is in `01-findings.md`
> (row 1 of the reading table — **A13 fails**); how it was built and where it deviates is
> in `01-implementation.md`. The transport below is superseded by deviation **V4**: the
> judge runs as an isolated subprocess, not through the `Agent` tool, because a judge
> launched inside the repository loads `AGENTS.md`.
>
> **Revised 30 Aug 2026, before approval.** The first draft called the Messages API from
> `annotation/llm_recheck.py` via `requests`. **There is no API key available**, so the
> transport becomes the `Agent` tool: one tool-free `sonnet` subagent per pair. The draft's
> reasoning about *why* to do this at all is unchanged and kept below; §§1-7 are rewritten.
> The superseded transport is recorded in *What changed*, not deleted — the repo convention
> is to keep the audit trail.

## Why this exists

κ(A1, yc) = **0.010**, 95% CI [-0.198, 0.223], agreement 16/41. Disagreement is
**bidirectional** — 7 of A1's `Good Fit` became our `No Fit` and 7 of A1's `No Fit` became
our `Good Fit` — so it is not a strictness shift, which would have preserved ranking. Binary
collapse does not rescue it (agreement 0.537, κ 0.076).

Plumbing is ruled out: text mis-mapping (hash-verified through `ui.corpus_text()`),
ambiguous source labels (0 of 659 A1 test pairs carry two labels), and within-group index
scrambling (size-1 groups agree at 0.47, n=17 — indistinguishable from the rest).

Two explanations survive, and they have opposite consequences:

- **A1's labels are wrong** → **A13 fails**, and D26's 0.7806 / 0.7188 ceilings are soft.
- **`yc` is miscalibrated** → A13 survives, and the in-domain set needs recalibrating before
  anything is quoted off it.

One human annotator cannot separate them. A third judge can.

## The reading table

| κ(A1, llm) | κ(yc, llm) | Reading |
|---|---|---|
| low | **high** | The LLM and `yc` agree, A1 does not → **A13 fails**; the ceiling is soft |
| **high** | low | `yc` is the outlier → A13 survives; recalibrate before Stage 2 |
| high | high | Contradicts κ(A1,yc)=0.010 → a bug we have not found. **Stop and find it** |
| low | low | The 3-class scheme is not reproducible on A1 by any judge — itself the finding |

## What this is not

- **Not ground truth.** An LLM is a third judge with its own bias, known to exist and not
  measured here. It breaks a tie between two accounts; it does not adjudicate one correct.
- **Not a screening-system evaluation.** That is Stage 4, on held-out data, against the
  frozen pools. This judges the *answer key*, not a system.
- **Not part of the human budget.** D25's 588 judgements are unaffected. This costs no
  annotator time.

## What changed from the API draft

| | API draft (superseded 30 Aug) | This plan |
|---|---|---|
| Transport | one `POST` to the Messages API via `requests` | the `Agent` tool |
| Judge | `claude-opus-5`, temperature 0 | `claude-sonnet-5` subagent, no temperature control |
| Unit | one group (a JD and all its candidates) | **one pair, one subagent** |
| Reproducible from the package | yes, from a seed | **no** — see below |
| Who executes | `annotation.llm_recheck`, end to end | the package does dispatch + collect; the **orchestrating session** does the judging |

### The consequence that matters: this is collected data, not derived

No seed reproduces a subagent run. `llm-recheck.csv` is therefore of the same *kind* as
`judgements.csv` — append-only, provenance-stamped, never re-derived — and `verify --derived`
must not attempt a determinism check on it. What it checks instead is **provenance**: every
row's `prompt_sha256` matches the committed prompt template, every row's `agent_sha256`
matches the committed agent definition, and every `pair_id` is one of the 50.

This is a real loss against the API design and it is worth stating plainly: a reader cannot
re-run this and get the same numbers. What they can do is re-run it and get numbers from a
recorded instrument, which is the same standard the human session meets.

## Decision to record

**D33 — the A1 recheck gets a third judge, and it is a third opinion, not an answer key.**
Its labels live in their own file, are never pooled into human κ, and are quoted as
`llm:claude-sonnet-5` wherever they appear. The judge runs tool-free so that blinding is
structural rather than instructed.

---

## 1. The judge: `.claude/agents/a1-judge.md`

A committed agent definition. `model: sonnet`. **`tools:` empty.**

The tool-free requirement is the load-bearing part of this plan, and it is not caution.
The answer key is on disk in plaintext-adjacent form:

- `data/processed/indomain/judging-queue-full.parquet` — carries `a1_label` for all 50
  recheck pairs (verified: 450 rows, 50 non-null `a1_label`).
- `data/processed/fit/test.parquet` — the source of those labels.

A subagent with `Read` or `Bash` reaches either in one command. The repo's existing standard
for human blinding is structural, not instructed: `serve_group` redacts its own output and
`assert_clean`s it, and the rules live in `session.py` rather than in the request handler,
precisely so that a caller cannot forget. **"We told the judge not to look it up" does not
meet that standard.** A tool-free agent cannot look anything up.

The agent's system prompt is `docs/annotation-guide.md` **verbatim**, minus three operator
sections that describe the human workflow and would only confuse a single-pair judge:

- *How the pairs reach you* (describes the UI, `--annotator`, saving and resuming)
- *How the work is split* (two annotators, title coverage)
- *Double labelling and adjudication*

Everything that defines the instrument is kept, unedited: the three class definitions and
their tests, all nine worked examples, the tie-breaking rule, the what-not-to-consider list,
*Two corpora, one scheme* (A16), and the "domain-literate, not professional recruiters"
statement. Paraphrasing would measure the paraphrase. The comparison only means something if
both judges were handed the same instrument.

**Pre-flight, in the dispatch step:** assert that none of the guide's nine worked examples is
one of the 50 recheck pairs. A judge shown the answer is not a judge.

## 2. Dispatch — `annotation.llm_recheck --dispatch --seed 0`

Writes `data/processed/indomain/llm-recheck-prompts.jsonl` (git-ignored — it carries document
text). One record per pair per run: `pair_id`, `run`, `prompt`, `prompt_sha256`, `chars_sent`.

**Built by calling `session.serve_group` with an empty judgements frame.** This needs **no
change to `session.py`**: `serve_group` takes `pairs` and `judgements` as arguments, so an
empty frame returns all 50 A1 units instead of the 20 `outstanding()` would allow, and the
text goes through the identical `redact.redact` + `redact.assert_clean` path a human screen
goes through. `Group.asks_shortlist` is already `False` for `corpus == "a1"`, so no
group-level question is lost by dissolving the groups.

**Blinding assertions on the rendered prompt**, not on the frame it came from: no `a1_label`,
no `selection_reason`, no `lexical_band`, no `batch`, no `stratum`.

### Why one pair per subagent is lossless here — and better

A1 recheck groups are 1-4 candidates, and `asks_shortlist` is `False` for A1, so no question
in the instrument spans candidates. Nothing is lost by dissolving the groups.

Something is gained. The guide's single most emphasised habit is:

> **Label each candidate against the standard in this guide, not against the other four.**

A fresh context per pair enforces that by construction. It also removes order effects and
within-group anchoring entirely.

**State it as an instrument difference, because it is one.** `yc` saw A1 candidates 1-4 at a
time; the LLM sees one. On this axis the two judges are not identical, and the difference
runs in the direction of the LLM following the guide more strictly than a human can.

## 3. The judging loop — the orchestrating session

Read the jsonl; spawn one `a1-judge` subagent per record. **At most 10 concurrent.**

### Context economy

Every spawn pays for its own context, so the per-pair prompt carries nothing but the
instrument and the two documents:

- **The guide lives in the agent definition**, not in the per-pair prompt. It is authored
  once and committed; the dispatch step renders only the pair.
- **`tools:` empty removes the tool schemas** — the single largest fixed cost in a subagent's
  context — as a side effect of the blinding requirement in §1.
- **No preamble, no restatement, no role-play.** The user message is exactly:

  ```
  JOB DESCRIPTION
  <jd_text>

  CANDIDATE
  <cv_text>

  Reply with one JSON object, no prose, no markdown fence:
  {"label":"Good Fit"|"Potential Fit"|"No Fit","reason":"<=120 chars"}
  ```

- The `reason` cap is enforced on the way in (the instruction) and on the way out (collect
  truncates and flags anything longer). It exists so the judge can be spot-checked, not so it
  can explain itself.

Raw returns are appended to `data/processed/indomain/llm-recheck-raw.jsonl`, one line per
spawn, with `pair_id`, `run`, `returned_at`, `raw`.

**Read 10 `reason` fields by hand** before trusting any kappa computed from the set — it is
the only cheap check that the judge answered the question it was asked rather than
pattern-matching document length.

## 4. Collect — `annotation.llm_recheck --collect`

Validates each raw return against `session.LABELS` and against the exact dispatched
`pair_id` set. One retry per pair; a second failure records `parsed_ok=False`, excludes the
row, and the count is reported rather than silently dropped.

Writes `docs/data/manifests/llm-recheck.csv` — `queue.JUDGEMENT_COLUMNS` plus `model`,
`agent_sha256`, `prompt_sha256`, `run`, `chars_sent`, `parsed_ok`, `judged_at`. `annotator`
is `llm:claude-sonnet-5:run{N}`.

**It never writes `judgements.csv`.** `annotator` is a free string, so an `llm` row appended
there would pass every existing check and then be silently pooled into the human κ that
`verify --derived` computes and the report quotes.

## 5. Runs, and the gate

Self-consistency is not optional garnish: the judge's test-retest reliability **bounds the κ
it can reach with anyone**. κ(A1, llm) = 0.3 means one thing if κ(llm, llm) = 0.9 and nothing
at all if κ(llm, llm) = 0.35.

Staged, because the second and third runs cost 100 spawns to sharpen a number:

1. **Run 1 — all 50 pairs (50 spawns).** Gives κ(A1, llm) and κ(yc, llm), and locates the
   result in the 2x2.
2. **Runs 2 and 3 — a fixed 20-pair subsample (40 spawns).** Gives κ(llm, llm) with a wide
   but usable CI, reported explicitly as *n=20*.

**Total ~90 spawns.** Three full runs (150 spawns) tightens the self-consistency CI and is
the option if the run-1 result is close to a cell boundary. Record which was chosen and why.

Where all three runs exist, the per-pair label is the majority; ties are excluded and *n* is
stated.

## 6. Analysis and output

- Pairwise κ over {A1, yc, llm}, each with a bootstrap 95% CI, on the pairs all three cover.
- κ(llm, llm) self-consistency, with its *n*.
- The 2x2 reading, named explicitly, with the cell the result lands in.
- **Every disagreement by `pair_id` with all three labels** and the LLM's `reason`.
- Per-class breakdown — A1's `No Fit` is the majority class, and a judge that agrees only
  there is a different finding from one that agrees everywhere.

Written to `output/annotation/llm-recheck-report.json` and `plan/2026-08-30-llm-recheck/01-findings.md`.

## 7. Tests — each shown to fail against the code it guards

Inline synthetic corpora, fresh-clone safe, no network, no `data/`, no subagent spawned.

| # | Invariant | Mutation that must break it |
|---|---|---|
| 1 | The rendered prompt carries no `a1_label`, `selection_reason` or `lexical_band` | add `a1_label` to the template |
| 2 | All 50 pairs dispatch regardless of `judgements.csv` | pass the real judgements frame — count drops to 20 |
| 3 | **The agent definition is tool-free** | add `Read` to its frontmatter `tools:` |
| 4 | `--collect` rejects an out-of-scheme label, an unknown `pair_id`, and a missing one | widen the validator to accept any string |
| 5 | `--collect` writes only to `llm-recheck.csv` | point the writer at `queue.JUDGEMENTS` |
| 6 | Every row's `prompt_sha256` matches the committed template | edit the template without re-dispatching |
| 7 | κ and its bootstrap reproduce on a synthetic frame of known κ | off-by-one in the agreement matrix |

Test 3 is the blinding control. It deserves a test for the same reason `assert_clean` has
one: it is the assertion that stops being true quietly.

## 8. Open questions

- **Q28 — how were A1's labels produced?** The A1 card records the label mix (50.4 / 24.9 /
  24.7%), the 6 conflicting-label pairs and the 7 train duplicates, but **not the
  provenance**. If A1's labels are themselves machine-generated, a high κ(A1, llm) is
  shared-bias artefact rather than independent agreement, and row 2 of the reading table is
  unreadable. **Blocks interpretation, not building.**
- **Q29 — truncation.** A1 resumes run to a 5,134-char median. Dispatch records `chars_sent`;
  collect asserts it equals the redacted source length. If anything truncates, the LLM is a
  different instrument from the one `yc` used and the comparison is void.
- **Q30 — redacted A1 resume text in the orchestrating session's transcript.
  **Closed 30 Aug 2026: accepted by the operator.** A tool-free judge must be handed its
  documents inline. The alternative — prompt files on disk, read by the subagent — keeps the
  text out of the transcript but hands the judge `Read`, which reopens the
  `judging-queue-full.parquet` leak. The text is already redacted, stays on this machine, and
  is never committed; the leak is the one thing the experiment cannot survive.
- **Q31 — registration.** Confirm `llm-recheck.csv` is registered in `verify --derived` as
  **collected data with a provenance check**, not as a derived artefact with a determinism
  check. A determinism check on it would be red on every run.

## Effort and sequencing

| | |
|---|---|
| Harness (dispatch, collect, agent definition, tests) | ~½ day, no annotator time |
| Judging | ~90 subagent spawns, 10 concurrent |
| Analysis and findings write-up | ~¼ day |

**Recommended first, not required:** finish the 9 outstanding human A1 pairs. One short
sitting takes κ(A1, yc) from n=41 to n=50 and strengthens the human leg of every comparison
in the table above.

## Verification

```bash
uv run pytest
uv run python -m candidate_screener.data.verify --derived      # must be unchanged by this work
uv run python -m candidate_screener.annotation.llm_recheck --dispatch --run 1 --seed 0
uv run python -m candidate_screener.annotation.llm_recheck --judge --run 1   # deviation V4
uv run python -m candidate_screener.annotation.llm_recheck --collect
uv run python -m candidate_screener.annotation.llm_recheck --report
```

## Out of scope

- **Any change to `judgements.csv`, the pools, or D26's ceiling figures.** Those change only
  if the finding says so, in a separate commit that supersedes in place.
- **Using the LLM as an annotator for the in-domain set.** This plan judges the answer key on
  50 A1 pairs. Extending it to the 450-pair in-domain campaign is a different decision with a
  different risk profile, and it is not made here.
- **Any modelling.** Stage 4 remains frozen and untouched.
