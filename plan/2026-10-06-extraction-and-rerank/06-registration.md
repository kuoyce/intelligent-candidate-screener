# 06 — Registration: decided before the held-out rungs are scored

**Committed 7 Oct 2026, before any X rung (P7) or trained reranker (P8) is scored.** A bar,
comparison or reference changed after a held-out result exists is recorded here as
**post-hoc**, with its date. `AGENTS.md` evaluation rule 8.

**References:** `docs/data/manifests/a2-final-labels.csv` (rule v1) for AUC and recall;
the human grades `yc_raw` / `van_raw` for ordering (D52). **Text:** `text_v2` (D53).

## 1. Already seen (exploratory: none of these tests a rule)

- The ordering check of 6 Oct (01 §4): random, keyword match, metadata rules, TF-IDF,
  SBERT zero-shot and the LLM 0–100 score, against `yc`, `van` and their mean, on `text_v1`.
- Every rung re-scored on `text_v2` in P1 (random, TF-IDF, BM25, SBERT zero-shot, SBERT
  weak, and the metadata rungs).
- P2's title profile, and P3/P4's extractor measurements on the final-set documents
  (role-family and experience-band accuracy vs the oracle). These measure extractors; they do
  not score a matching rung.
- 15 batch returns (run 11) were parsed for a smoke test; only their label counts were read.

## 2. Held-out rungs (built under this plan)

| Rung | System | Stage |
|---|---|---|
| X1 | extracted metadata rule: `2 × role_match + experience_compatible`, both extracted (02 §3.4). **An unknown experience band on either side counts as compatible (1)** (amendment A1) | 1 |
| X2 | X1 + TF-IDF (fixed rank-normalised sum, as 1a+2) | 1 |
| X3 | X1 + skill overlap | 1 |
| X4 | X1 + SBERT weak | 1 |
| R1 | cross-encoder trained on the D50 batch, pointwise (regression on the score) | 2 |
| R2 | cross-encoder trained on the D50 batch, pairwise (within-JD score gaps ≥ δ) | 2 |
| R3 | bi-encoder trained on the D50 batch (the better of the two forms on validation) | 2 |

R1 and R2 are both scored; the form used downstream is the one with the better
**validation** pairwise agreement on the batch's validation split, chosen before the final
set is read. δ and every hyperparameter are chosen there too (D46).

## 3. Decision rule (unchanged from the 26 Sep registration)

For B against the cheaper A named in §4, on the stated metric:

| Verdict | Condition |
|---|---|
| **Justified** | paired Δ(B − A) 95% CI on the **total** above 0, **and** the Δ point estimate > 0 in both `generic` and `targeted` |
| **Justified, one stratum** | total CI above 0, one stratum's point estimate ≤ 0. Reported; does not count for selection |
| **Justified, LLM-dependent** | justified, but the total Δ on the `llm_free` subset ≤ 0. Reported; does not count for selection |
| **Harmful** | total CI entirely below 0 |
| **Not resolved** | anything else; the total CI half-width is quoted beside it |

Within-family AUC is reported beside every AUC verdict and does not decide it.

## 4. Registered comparisons

**Stage 1 (metric: per-JD AUC against the final set, `text_v2`).**

| # | B | A | Question |
|---|---|---|---|
| K1 | X2 | TF-IDF | Do extracted fields add to text, as the real fields did? |
| K2 | X2 | oracle 1a+2 (true fields) | What does dropping the metadata cost? (expected ≤ 0; reported, decides nothing) |
| K3 | X4 | SBERT weak | Do extracted fields add to the best text-only rung? |
| K4 | X2 | SBERT weak | Cheap extracted rule + lexical vs the fine-tuned bi-encoder |
| K5 | X3 | X1 | Does skill overlap add to the extracted rule? |

**Stage 1 retrieval (S2, unchanged bars, text-only):** Recall@50 of final-set fits in the
500-candidate pools of `evaluation/retrieval_pilot.py`: **≥ 0.80** for the classical
retriever (best of TF-IDF, X2), **≥ 0.90** for the advanced retriever (best of SBERT weak,
X4). Pass if the point estimate meets the bar; "CI above the bar" is stated separately.
**Recall@10 is reported with no bar.**

**Stage 2 (metric: per-JD pairwise agreement against the mean human grade where two humans
labelled, and against `yc`; nDCG reported beside it).** Primary reference: `yc` (600 pairs,
the only reference on all JDs). The mean-human figure is reported beside it and, where the
two disagree in verdict, both are stated.

| # | B | A | Question |
|---|---|---|---|
| O1 | selected cross-encoder (R1/R2) | SBERT weak | Does a cross-encoder trained on LLM scores order candidates better than the selected text rung? |
| O2 | selected cross-encoder | TF-IDF | Against the cheap lexical ranker |
| O3 | R3 bi-encoder | SBERT weak | Is it the supervision or the architecture? |
| O4 | selected cross-encoder | R3 | The architecture, at equal supervision |
| O5 | each reranker | random | **The floor** (user, 6 Oct): it must beat guessing: the **paired** per-JD Δ(system − random) 95% CI lower bound > 0, total scope (amendment A2) |

Each ordering figure is reported on `full`, `llm_free` and `fit_only` (02 §5). The
**fit_only** figure is the one that matches the stage-2 role (ordering the Fit CVs).

Also reported, deciding nothing: the LLM score as a ranker (on `llm_free`, D43/D51);
R1 vs R2; the end-to-end stage 1 + stage 2 Recall of fits in the top 10.

## 5. Selection

Cost order, cheapest first: random → X1 → X3 → TF-IDF / X2 → SBERT zero-shot → SBERT weak /
X4 → bi-encoder (R3) → cross-encoder (R1/R2).

- **Stage 1** adopts the cheapest rung no justified, more expensive rung beats on K1–K4,
  among those that pass S2.
- **Stage 2** adopts the cheapest reranker that no justified, more expensive reranker beats
  on O1–O4, among those that pass O5. If none beats SBERT weak, stage 2 is SBERT weak
  ordering (no new model), and that is the finding.

## 6. Bars on the training data itself (decide nothing about rungs)

- The batch is used even if its score is noisy; the replicate (if run, Q-A1b) reports the
  run-to-run noise.
- **Stop rule (04 §3):** if every trained reranker fails O5 on the batch's own validation
  split, the batch is reported as insufficient; it is **not** extended to chase a result.

## 7. Amendments (all before any held-out rung was scored)

| # | Date | Change | Why | Post-hoc? |
|---|---|---|---|---|
| A1 | 7 Oct 2026 | X1's experience term: an unknown band (no experience stated) on either side counts as compatible, not 0 | P4 measured that 40–55% of documents state no experience; under unknown → 0 the extracted check agrees with the true one on only 0.435 of pairs (0.863 where both sides state it). User decision | No: X1 not yet scored. Based on extractor measurements, not on a matching result |
| A2 | 7 Oct 2026 | O5 "beats random" = paired Δ vs random, CI lower bound > 0 | The P5 record left the test ambiguous. User decision | No: no reranker scored |
