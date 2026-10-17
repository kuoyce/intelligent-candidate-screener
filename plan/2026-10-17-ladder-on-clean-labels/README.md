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
- **Not in the records:** a paired X4-vs-X2 comparison. Added at the owner's request by
  `evaluation/clean_labels.py` (re-scores both rungs, checks their digests against the stage-1
  record) into `output/matching/ladder-clean-labels.json`.

## Result

See notebook 23. In short: on clean pairs the rungs keep their order, and the gaps **widen in
favour of the simple hybrid**. X2 (extracted rule + TF-IDF) vs the fine-tuned SBERT goes from
+0.016 [−0.025, +0.058] on all pairs to +0.048 [+0.001, +0.097] on `llm_free` (targeted +0.128).
SBERT (weak fine-tuned) gains least from clean labels (0.834 → 0.852); the top rung swaps from X4
(rule + SBERT) to X2, but paired they are level: X4 − X2 = +0.004 [−0.026, +0.033] on all
pairs, −0.009 [−0.054, +0.032] on `llm_free`. The cross-encoders (R1, R2) stay below TF-IDF; the
bi-encoder R3 gains a little (pairwise 0.730 → 0.808, just above TF-IDF's 0.794 on clean pairs;
unpaired). Label noise was not hiding the value of the sophisticated rungs.

## Amendment (owner, 17 Oct 2026): every reranker against X2

Stage 2 compared R1-R3 with SBERT weak and TF-IDF only (registration O1-O5). The owner asked for
X2 as an independent yardstick: it never saw an LLM score. `clean_labels.py` adds paired
pairwise and nDCG deltas against X2 on the human grades (`yc`, `mean`; `full`, `llm_free`), with
every score digest checked against the committed records. Not registered, so `exploratory`.

Result (pairwise vs `yc`, total; all / clean): R1 −0.197 / −0.178, R2 −0.201 / −0.222 (both
resolved, worse); R3 −0.007 / −0.001, SBERT weak −0.009 / −0.015, X4 +0.008 / +0.006 (level).
R3's one resolved gain is generic nDCG (+0.043 [+0.01, +0.08]). No reranker trained on the LLM
batch orders candidates better than X2.
