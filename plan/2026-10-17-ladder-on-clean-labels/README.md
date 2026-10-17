# The ladder on clean labels

Status: **approved by the owner 17 Oct 2026 (idea 1); done the same day.** Reporting only:
nothing is fitted, scored anew or selected.

## Question

Error analysis 2 found that the 107 A2 pairs where the two humans split (the LLM broke the tie,
D41) cap every score: dropping them lifts X2 from 0.850 to 0.900. **Does the order of the ladder
rungs change on the clean pairs?** If the sophisticated rungs gain more there, label noise was
hiding their value; if not, "more sophisticated matching does not earn its complexity" gets
stronger.

## Method

- "Split pairs dropped" is exactly the existing `llm_free` subset (tiers `humans_agree` and
  `single_human`, 460 of 600 pairs counted); A2 has no `llm_only` pairs.
- Every rung's AUC and the registered paired comparisons (K1-K5) already carry `all` and
  `llm_free` figures in `output/matching/stage1-metrics.json`; the cross-encoder rerankers (R1-R3)
  carry ordering figures on `full` and `llm_free` in `stage2-metrics.json`. No new run is needed.
- [Notebook 23](../../notebooks/23-ladder-on-clean-labels.ipynb) reads those records and puts
  `all` beside `llm_free` for generic, targeted and total (rule 3).
- **Not in the records:** a paired X4-vs-X2 comparison on `llm_free`. Unpaired figures are shown;
  a paired delta needs the cached scores re-run (stage1 warm, a few minutes) and is left out
  unless the owner asks.

## Result

See notebook 23. In short: on clean pairs the rungs keep their order, and the gaps **widen in
favour of the simple hybrid**. X2 (extracted rule + TF-IDF) vs the fine-tuned SBERT goes from
+0.016 [−0.025, +0.058] on all pairs to +0.048 [+0.001, +0.097] on `llm_free` (targeted +0.128).
SBERT (weak fine-tuned) gains least from clean labels (0.834 → 0.852); the top rung swaps from X4
(rule + SBERT) to X2. The one sophisticated rung that gains is the cross-encoder R3 (pairwise
agreement 0.730 → 0.808, just above TF-IDF's 0.794 on clean pairs; unpaired). Label noise was not
hiding the value of the sophisticated rungs.
