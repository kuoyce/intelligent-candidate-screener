# A2 LLM pre-labelling — can the judge stand in for the annotator?

**Status:** draft, awaiting approval
**Date:** 6 Sep 2026
**Supersedes nothing. Depends on:** D33 (the judge exists), D34 (the calibrated guide),
D35 (one draw per file), D29 (two strata, never pooled).

## The question

The manual annotation of A2 batch 2 is pending. Can the LLM judge label A2 now — and if
its agreement with the human clears some threshold *X*, can its labels replace A1 as the
evaluation ground truth?

## The answer this plan proposes

**Yes to pre-labelling, immediately, and it costs nothing to find out.** But the plan
starts somewhere better than the pending batch: **400 A2 pairs are already labelled by
`yc` and the judge has never seen them.** kappa(yc, llm) on the A2 corpus is computable
today at n=400 — four times the n of the A1 recheck figure, on the corpus the project
actually targets.

**Not yet to replacement**, and the blocker is not the judge. It is that *X* cannot be set
defensibly right now, because this repository has **no human–human kappa**. Every kappa on
file is `yc` against something else. Comparing kappa(yc, llm) to a threshold taken from a
1977 adjective table is exactly the kind of unearned figure the rest of this repo refuses.
The plan therefore makes establishing the human–human ceiling a gate, not an afterthought.

## What is actually pending

Measured 6 Sep 2026, `queue.indomain_rows()` against `judgements.csv`:

| Stratum | Batch | Pairs | Human-labelled |
|---|---|---|---|
| `generic` | 1 | 400 | **400** (`yc`) |
| `targeted` | 2 | 200 | **0** ← what is pending |
| | | **600** | 400 |

The judge's committed instrument is `75e00ea3` on `claude-haiku-4-5` (post-D34). Runs 1–4
are spent. Runs 5+ are free.

## Documents

- [`01-design-spec.md`](01-design-spec.md) — draws, files, thresholds, what must not be pooled
- [`02-work-plan.md`](02-work-plan.md) — tasks, order, acceptance checks

## A latent bug this plan found before dispatching anything

Tracing what happens if credits run out mid-run: **a failed spawn is currently recorded as
a completed pair.** `judge()` resumes from `llm-recheck-raw.jsonl`, and `judge_one` writes
infrastructure failures there as data, so a re-run judges nothing and reports
`remaining: 0`. The report then quotes a kappa over the survivors — which are in
completion-latency order, not random — with no signal that anything was lost.

No run has hit it (all 879 raw returns on file are clean), so option D got lucky at 639
spawns. Design spec §6 has the trace; fixes **F1–F4** are prerequisites to Phase 0 and
apply to the A1 path too, since the bug is not A2-specific.

## Open questions

- **Q35** — kappa(yc, second annotator) on A2. **Answered 6 Sep 2026: a second annotator is
  confirmed and arrives after this plan is approved.** Phases 0–1 run without waiting; the
  relative gate in design spec §4 is the expected path, and the absolute fallback survives
  only as contingency.
- **Q36** — Does the judge's agreement with `yc` differ between the A2 corpus and the A1
  recheck it was calibrated on? Documents differ by 3x in length (A1 median 5,134 chars,
  A2 1,549 JD / 1,742 CV) and D34's guide was tuned against A1 pairs. **Confirmed in scope
  — Phase 0 answers it.**
- **Q37** — If replacement proceeds, may the 400 `yc` labels and the LLM's labels coexist
  in one evaluation layer, or does mixing provenances re-create the problem the
  `label_source` column was invented for? **Deferred by decision, 6 Sep 2026** — revisit at
  Phase 3, not worked before then.
