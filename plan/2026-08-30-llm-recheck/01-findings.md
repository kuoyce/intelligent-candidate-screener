# Findings — a third judge on the 50 A1 recheck pairs (D33)

**Run 30 Aug 2026 on `feat/llm-recheck`.** 90 judgements: run 1 over all 50 pairs, runs 2
and 3 over a seeded 20-pair subsample. 90/90 parsed, 0 refused.
`docs/data/manifests/llm-recheck.csv`, `output/annotation/llm-recheck-report.json`.

## The result, in one line

**Row 1 of the reading table. A13 fails.** The two judges who never saw A1's labels agree
with each other; neither agrees with A1 above chance; and the LLM's own test-retest
reliability is high enough that it *could* have agreed with A1 if A1 were reproducible.

## The numbers

| | kappa | 95% CI | n | agreement |
|---|---|---|---|---|
| A1 vs `yc` | 0.010 | [-0.197, 0.223] | 41 | 0.390 |
| **A1 vs llm** | **0.029** | [-0.134, 0.190] | 50 | 0.400 |
| **`yc` vs llm** | **0.291** | [0.115, 0.492] | 41 | 0.634 |

Self-consistency, on the 20-pair subsample:

| | kappa | 95% CI | agreement |
|---|---|---|---|
| run 1 vs run 2 | 0.857 | [0.417, 1.000] | 0.95 |
| run 1 vs run 3 | 0.857 | [0.417, 1.000] | 0.95 |
| run 2 vs run 3 | 1.000 | [1.000, 1.000] | 1.00 |

Three runs were taken rather than the plan's two-then-decide, because run 1 landed with
kappa(`yc`, llm) = 0.259 whose CI overlapped kappa(A1, llm)'s — the reading depended on
precision, which is the condition §5 named for going to three.

### Why self-consistency is the load-bearing row

kappa(llm, llm) **bounds** the kappa the judge can reach with anyone. At 0.86-1.00 the
bound is not binding: a judge this reproducible reaching 0.029 with A1 is not a noisy
judge, it is a judge measuring something A1 does not contain. Had self-consistency come
back near 0.3, every other number here would have been unreadable.

## Binary collapse removes the remaining excuses

The three-class boundary is the usual explanation for a low kappa, and a strictness shift
is the second. Collapsing to *fit at all* (`Good`∪`Potential`) vs `No Fit` removes both:

| | binary kappa | agreement |
|---|---|---|
| A1 vs `yc` | 0.076 | 0.537 |
| **A1 vs llm** | **0.000** | 0.440 |
| **`yc` vs llm** | **0.424** | 0.732 |

kappa(A1, llm) collapses to **exactly chance**. On the coarsest question the instrument
asks, two independent judges agree at 0.42 and neither recovers A1's answer.

## Where it breaks: A1's positive classes

Marginals on the 50 recheck pairs — the judge is markedly stricter than either human:

| | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| A1 (n=50) | 15 | 15 | 20 |
| `yc` (n=41) | 12 | 6 | 23 |
| llm (n=50) | 1 | 9 | 40 |

A1 (rows) against the judge (columns), n=50:

| | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| A1 `Good Fit` | 0 | 1 | **14** |
| A1 `Potential Fit` | 0 | 5 | 10 |
| A1 `No Fit` | 1 | 3 | **16** |

`yc` (rows) against the judge, n=41:

| | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| `yc` `Good Fit` | 1 | 5 | 6 |
| `yc` `Potential Fit` | 0 | 2 | 4 |
| `yc` `No Fit` | 0 | 1 | **22** |

Per-class agreement with A1's label: `No Fit` 0.60 (`yc`) / 0.80 (llm), `Potential Fit`
0.27 / 0.36, **`Good Fit` 0.10 / 0.00**.

Everyone agrees on A1's negatives. **Neither independent judge reproduces a single one of
A1's 15 `Good Fit` labels as `Good Fit`**, and 14 of the 15 are read as `No Fit` outright.
The judge's `reason` fields on those pairs are not hedged — *"Administrative/clerical
background, no data engineering, SQL, cloud, or pipeline experience"* against a data
engineering JD; *"Graphics/compiler engineer with no Salesforce, Vlocity, or Telecom CPQ
experience"*. These are not boundary calls.

**The strictness objection does not survive this table.** A stricter judge would push
*everyone's* positives down uniformly. It pushes 14/15 of A1's `Good Fit` to `No Fit` but
only 6/12 of `yc`'s, and it agrees with 22 of `yc`'s 23 `No Fit` labels. The disagreement
with A1 is about *which* pairs are positive, not about where the bar sits.

## Consequences

1. **A13 fails.** A1's labels are not reproducible from A1's own documents by two
   independent judges working from the annotation guide. They cannot be treated as an
   answer key.
2. **D26's ceilings are soft.** Precision@5 = **0.7806** strict / **0.7188** graded and
   Recall@10 = 0.983 are computed against pools whose A1 leg carries these labels. Every
   figure quoted as a fraction of attainable inherits the doubt. *This finding does not
   itself restate the ceilings* — that is a separate change, superseding in place, and it
   is out of this plan's scope.
3. **`yc` is not exonerated, only un-indicted.** kappa(`yc`, llm) = 0.291 is *fair*, not
   substantial. The finding is an ordering — the two blind judges agree with each other
   more than either does with A1 — not a certificate on `yc`.
4. **Part of row 4 is also true.** No pair of judges reaches even moderate agreement on
   this corpus. The three-class scheme is hard on A1 for everyone; what separates row 1
   from row 4 is that one leg (`yc` vs llm) does clear zero with its CI and the two A1
   legs do not.

## Open questions, resolved and standing

- **Q28 — A1's label provenance — no longer blocks interpretation.** The worry was that
  machine-generated A1 labels would make a *high* kappa(A1, llm) a shared-bias artefact,
  rendering the "yc is the outlier" row unreadable. Shared bias inflates that kappa; the
  observed value is 0.029, and 0.000 collapsed. The reading does not depend on Q28 either
  way. It remains unrecorded on the A1 card and is still worth recording.
- **Q29 — truncation — closed.** Nothing truncates on the way out (`chars_sent` median
  8,632 for run 1, full redacted documents). Nothing truncates on the way in either: the
  `reason` fields cite specific content from both documents (job-specific stacks, CV-specific
  employers and skills), and run-to-run agreement of 0.95-1.00 is not what a judge reading
  a variably-clipped prompt produces.
- **Q30 — redacted A1 text in the orchestrating session's transcript.** Closed differently
  from the plan, and better: under deviation V4 the documents never enter this session's
  transcript at all. They are piped from the prompt file into a judge subprocess.
- **Q31 — registration — closed.** `check_llm_recheck` verifies provenance
  (`agent_sha256`, `prompt_sha256`, the pair set), never determinism.
- **Standing: 9 human A1 pairs remain unlabelled.** kappa(A1, `yc`) and kappa(`yc`, llm) are
  n=41; kappa(A1, llm) is n=50. Finishing them costs one short sitting and would put all
  three legs on the same 50.
- **New: the judge's own bias is unmeasured.** It is one model, one prompt, one temperature
  we do not control. Its severity (40/50 `No Fit`) is a property of this judge, and D33's
  "third opinion, not an answer key" is doing real work here — the finding rests on the
  *ordering* of three kappas, not on any one of the judge's labels being right.

## Blinding, as executed

- **Tool-free, measured not assumed.** Probed before the run: `{"label":"No
  Fit","reason":"NONE"}` when asked to name every tool it can call.
- **No repository context.** The same probe asked whether `AGENTS.md`, `CLAUDE.md` or
  `a1_label` appeared in its context. From the repository root the judge answered **YES** —
  `CLAUDE.md` is `AGENTS.md`, which names the recheck, the kappa it is chasing and the
  parquet that holds `a1_label`. From the isolated sandbox of deviation V4 it answered
  **NO**. This leak was not anticipated by the plan and would have contaminated the
  instrument; `assert_isolated` now refuses any working directory that carries it, and the
  repository root is a test case.
- **Nothing that names the answer reaches the prompt**: no `a1_label`, `selection_reason`,
  `lexical_band`, `batch` or `stratum`, asserted on the rendered prompt.
- **Instrument difference, restated.** `yc` saw A1 candidates 1-4 at a time; the judge sees
  one pair in a fresh context. Nothing in the A1 instrument spans candidates
  (`asks_shortlist` is `False` for `corpus == "a1"`), so no question is lost — but on the
  co-presence axis the two judges are not identical.
