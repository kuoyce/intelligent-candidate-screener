# Findings — a third judge on the A1 recheck pairs (D33)

**Run 30 Aug 2026 on `feat/llm-recheck`.** 140 machine judgements: run 1 over **100** pairs,
runs 2 and 3 over a seeded 20-pair subsample of the original 50. 140/140 parsed, 0 refused.
`docs/data/manifests/llm-recheck.csv`, `output/annotation/llm-recheck-report.json`.

The machine leg was widened from 50 to 100 pairs after the first result (deviation **V6**).
It costs no annotator time, it nests the original 50 exactly, and it doubles the cell the
finding actually lives in — A1's `Good Fit`, from n=15 to n=30.

The human leg is **complete at 50 of 50** — the 9 pairs outstanding at the previous
commit were labelled while this ran, which is why kappa(A1, `yc`) is quoted here at
0.034 (n=50) rather than the 0.010 (n=41) the earlier record carries.

## The result, in one line

**Row 1 of the reading table. A13 fails.** The two judges who never saw A1's labels agree
with each other roughly four times as strongly as either agrees with A1, and the LLM's own
test-retest reliability is high enough that it *could* have agreed with A1 if A1 were
reproducible.

## The numbers

| | kappa | 95% CI | n | agreement | binary-collapse kappa |
|---|---|---|---|---|---|
| A1 vs `yc` | 0.034 | [-0.162, 0.225] | 50 | 0.380 | 0.031 |
| **A1 vs llm** | **0.125** | **[0.013, 0.241]** | **100** | 0.460 | **0.113** |
| **`yc` vs llm** | **0.275** | [0.127, 0.457] | 50 | 0.640 | **0.425** |

Over the original 50, so that the figures already published stay quotable
(`pairwise_kappa_over_the_human_50` in the report):

| | kappa | 95% CI | n | binary |
|---|---|---|---|---|
| A1 vs llm | 0.029 | [-0.134, 0.190] | 50 | **0.000** |
| A1 vs `yc` | 0.034 | [-0.162, 0.225] | 50 | 0.031 |
| `yc` vs llm | 0.275 | [0.127, 0.457] | 50 | 0.425 |

Self-consistency, on the seeded 20-pair subsample:

| | kappa | 95% CI | agreement |
|---|---|---|---|
| run 1 vs run 2 | 0.857 | [0.417, 1.000] | 0.95 |
| run 1 vs run 3 | 0.857 | [0.417, 1.000] | 0.95 |
| run 2 vs run 3 | 1.000 | [1.000, 1.000] | 1.00 |

### What doubling the draw changed, stated plainly

**kappa(A1, llm) rose from 0.029 to 0.125 and its CI now just clears zero** (lower bound
0.013). At n=50 the judge was indistinguishable from chance against A1; at n=100 it is
*slightly* better than chance. That is a real change and it is not hidden here.

It does not move the reading, for two reasons:

1. **0.125 is still negligible agreement**, and it is less than half kappa(`yc`, llm) — on
   a leg with twice the *n* and therefore a *tighter* interval. The ordering is firmer than
   it was at n=50, not weaker.
2. **All of it comes from one cell.** Per class, against A1's own label over the 100:
   `No Fit` n=40, judge agrees **0.900**; `Potential Fit` n=30, **0.267**; `Good Fit`
   n=30, **0.067**. A judge that agrees on the majority-negative class and nowhere else
   scores a small positive kappa without reproducing anything the label scheme is for.

### Why self-consistency is the load-bearing row

kappa(llm, llm) **bounds** the kappa the judge can reach with anyone. At 0.86-1.00 the
bound is not binding: a judge this reproducible reaching 0.125 with A1 is not a noisy
judge, it is a judge measuring something A1 does not contain. Had self-consistency come
back near 0.3, every other number here would have been unreadable.

## Binary collapse removes the remaining excuses

The three-class boundary is the usual explanation for a low kappa, and a strictness shift
is the second. Collapsing to *fit at all* (`Good`∪`Potential`) vs `No Fit` removes both:
A1 vs llm **0.113** over the 100 (exactly **0.000** over the original 50) and A1 vs `yc`
**0.031**, against **0.425** between the two blind judges. On the coarsest question the
instrument asks, neither blind judge recovers A1's answer above the noise, and they recover
each other's an order of magnitude better.

## Where it breaks: A1's positive classes

Marginals over the 100 — the judge is markedly stricter than either human:

| | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| A1 (n=100) | 30 | 30 | 40 |
| llm (n=100) | 4 | 14 | 82 |
| A1 (n=50) | 15 | 15 | 20 |
| `yc` (n=50) | 14 | 7 | 29 |

A1 (rows) against the judge (columns), n=100:

| | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| A1 `Good Fit` | 2 | 2 | **26** |
| A1 `Potential Fit` | 1 | 9 | 20 |
| A1 `No Fit` | 1 | 3 | **36** |

`yc` (rows) against the judge, n=50:

| | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| `yc` `Good Fit` | 1 | 6 | 7 |
| `yc` `Potential Fit` | 0 | 2 | 5 |
| `yc` `No Fit` | 0 | 1 | **28** |

**26 of A1's 30 `Good Fit` labels are read as `No Fit` by a blind judge**, and only 2 are
confirmed. Over the original 50 the human confirms **3 of A1's 15** `Good Fit` labels and
the judge **none of them**. Both agree with A1's negatives (0.60 human / 0.80 judge) and
with each other on theirs — 28 of `yc`'s 29 `No Fit`.

The judge's `reason` fields on the rejected positives are not hedged — *"Medical
coding/billing background, not senior GL accounting; no SAP, no 10+ yrs, no IFRS/GAAP
evidence"*; *"No Salesforce/Vlocity/Communications cloud experience; candidate is
Python/Java QA, not SFDC BA"*. These are not boundary calls.

**The strictness objection does not survive this table.** A stricter judge would push
*everyone's* positives down uniformly. It pushes 26/30 of A1's `Good Fit` to `No Fit` but
only 7/14 of `yc`'s, and it agrees with 28 of `yc`'s 29 `No Fit` labels. The disagreement
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
3. **`yc` is not exonerated, only un-indicted.** kappa(`yc`, llm) = 0.275 is *fair*, not
   substantial. The finding is an ordering — the two blind judges agree with each other
   more than either does with A1 — not a certificate on `yc`.
4. **Part of row 4 is also true.** No pair of judges reaches even moderate agreement on
   this corpus. The three-class scheme is hard on A1 for everyone; what separates row 1
   from row 4 is that the `yc`-llm leg clears zero comfortably in both the 3-class and the
   binary collapse, and the A1 legs do not.

## Open questions, resolved and standing

- **Q28 — A1's label provenance — no longer blocks interpretation.** The worry was that
  machine-generated A1 labels would make a *high* kappa(A1, llm) a shared-bias artefact,
  rendering the "yc is the outlier" row unreadable. Shared bias inflates that kappa; the
  observed value is 0.125 at n=100, driven entirely by the negative class. The reading
  does not depend on Q28 either way. It remains unrecorded on the A1 card and is still
  worth recording.
- **Q29 — truncation — closed.** Nothing truncates on the way out (`chars_sent` median
  8,632 run 1 over the original 50, 7,265 over the additional 50 — full redacted
  documents). Nothing truncates on the way in either: the `reason` fields cite specific
  content from both documents, and run-to-run agreement of 0.95-1.00 is not what a judge
  reading a variably-clipped prompt produces.
- **Q30 — redacted A1 text in the orchestrating session's transcript.** Closed differently
  from the plan, and better: under deviation V4 the documents never enter this session's
  transcript at all. They are piped from the prompt file into a judge subprocess.
- **Q31 — registration — closed.** `check_llm_recheck` verifies provenance
  (`agent_sha256`, `prompt_sha256`, the pair set), never determinism.
- **Standing: the human leg cannot be widened for free.** The additional 50 are machine-only
  by construction — they were never in the human queue and `RECHECK_STRATA` stays at 50, so
  kappa(`yc`, llm) is capped at n=50 whatever the machine leg does — and the human 50 are
  now all labelled, so that cap is reached. Widening the human leg
  is a D25 budget decision, not a methodological one.
- **New: the judge's own bias is unmeasured.** It is one model, one prompt, one temperature
  we do not control. Its severity (82/100 `No Fit`) is a property of this judge, and D33's
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
- **The two draws nest, and it is asserted.** `assert_recheck_nests` and a `verify
  --derived` check both hold that the human 50 are a subset of the machine 100 — the only
  thing that makes an n=50 figure quotable beside an n=100 one.
- **Instrument difference, restated.** `yc` saw A1 candidates 1-4 at a time; the judge sees
  one pair in a fresh context. Nothing in the A1 instrument spans candidates
  (`asks_shortlist` is `False` for `corpus == "a1"`), so no question is lost — but on the
  co-presence axis the two judges are not identical.
