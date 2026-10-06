# 01 — Context: what was measured and learned before this plan

Everything here was read from the repository or measured on 6 Oct 2026. Items marked
**exploratory** were seen before any bar was registered, so they do not test a rule.

## 1. Where the ladder ended (27 Sep 2026, D48)

Per-JD AUC against the final Fit / No Fit set, 59 JDs (one of the 60 has no AUC; the reason was not checked), 95% CIs bootstrapped over JDs.

| System | AUC (total) | Note |
|---|---|---|
| random | 0.556 | |
| metadata rules (1a) | 0.778 | `2 × keyword_match + experience_compatible`, fixed |
| TF-IDF | 0.797 | |
| metadata + skill overlap | 0.822 | the only rung that uses skills (1b) |
| **metadata + TF-IDF** | **0.849** | selected with metadata (D48) |
| SBERT zero-shot | 0.774 | |
| **SBERT weak fine-tuned (3b)** | **0.854** [0.817, 0.890] | selected for text only (D48) |
| cross-encoder A1 / weak | 0.488 / 0.661 | harmful as trained |
| learned hybrid | 0.828 | not resolved |

Retrieval at 500 candidates per role: metadata + TF-IDF keeps 92.9% of fits in a
50-candidate shortlist (S2 bar 0.80, pass); metadata + SBERT weak 0.936 (bar 0.90, pass).
Reranking that shortlist with a cross-encoder **cut** Recall@10 by 0.254 (weak) and 0.328
(A1): C12, harmful.

**Why the metadata rung is so strong, and why it cannot be assumed elsewhere.** Of the 10
candidates per JD, 8 share the JD's `Primary Keyword` (D31). The selection splits by input:
metadata + TF-IDF where the field exists, SBERT where it does not. Both depend on how the
pools were built.

## 2. What the cross-encoder was and why it lost

Rung 4: MiniLM cross-encoder reading JD and CV together, one logit. 4a trained on A1 fit
pairs collapsed to Fit / No Fit (its first run learned nothing, loss stuck at ln 2, until a
larger learning rate and more epochs were chosen on its own validation split). 4b trained on
the weak pairs (8,388 rows over 3,000 JDs). It was tested standalone and as a reranker of a
shortlist (4c). **The reranker test used 500-candidate pools**, not a top-10 shortlist.

Two causes are visible, neither verified as *the* cause:
- The weak label is "same `Primary Keyword` and compatible experience". It teaches family
  matching, not ordering inside a family.
- A1's labels do not reproduce (A13).

## 3. A2 facts used in this plan

| Fact | Value |
|---|---|
| Final set | 60 JDs × 10 CVs = 600 pairs; 40 `generic`, 20 `targeted`; 8 of 10 CVs share the JD's keyword (D31) |
| `yc` | labelled **all 600** (134 Good, 189 Potential, 277 No Fit) |
| `van` | labelled **326** (125 / 105 / 96): 126 generic and all 200 targeted |
| LLM (`5d986c82`, haiku, runs 9/10) | all 600: 63 Good, 306 Potential, 231 No Fit, plus a **0–100 score** per pair |
| Two humans | 326 pairs; the LLM breaks the tie on 107 of them (D41) |
| `shortlist_pick` | one top-1 pick per JD in `judgements.csv`: 59 picks over 58 JDs (`yc` 39, `van` 20) |
| Raw A2 titles | `Position` on JDs: 141,897 rows, 100% filled, 69,671 distinct. On CVs: 210,250 rows, 100%, 52,913 distinct. `Primary Keyword`: 45 JD classes, 41 CV classes |
| Train/eval partition (D19, document level) | JDs 106,423 train / 35,474 eval; CVs 157,688 train / 52,562 eval |
| **Leak to remove** | `cv_text` is built from `Position`, `CV`, `Highlights`, `Moreinfo`, `Looking For` (`annotation/sample.py:70`), so the CV's title is **in the text a system reads**. The JD text is `Long Description` only |

Other title sources (from the data catalog; **not yet checked**): ESCO occupations
(3,043 occupations, 30,417 alternative labels), `data_jobs` `job_title_short`, livecareer
(24 categories) and resume-atlas (43 categories) as CV role categories.

## 4. The ordering check (exploratory, 6 Oct 2026)

Question: does an ordering of the 10 CVs per JD agree with the human order
Good (2) > Potential (1) > No Fit (0)? Systems: the existing LLM 0–100 score (no new LLM
calls), random, keyword match, metadata rules, TF-IDF, SBERT zero-shot. nDCG with tied
scores sharing their mean gain; pairwise agreement counted over pairs whose human grades
differ (a score tie counts 0.5). Total, JDs with at least two different grades (nDCG: n = 56 for
all three references; pairwise: 56 for `yc`, 51 for `van`, 53 for the mean), 95% CI over JDs.

| System | nDCG vs `yc` | pairwise vs `yc` | nDCG vs mean(yc, van) | pairwise vs mean(yc, van) |
|---|---|---|---|---|
| random | 0.737 | 0.5 (by construction) | 0.839 | 0.5 |
| keyword match | 0.791 | 0.685 | 0.880 | 0.628 |
| metadata rules | 0.795 | 0.697 | 0.892 | 0.653 |
| TF-IDF | 0.873 | 0.785 | 0.921 | 0.726 |
| SBERT zero-shot | 0.871 | 0.761 | 0.921 | 0.706 |
| **LLM score** | **0.896** [0.853, 0.932] | **0.843** [0.805, 0.880] | **0.928** [0.900, 0.953] | **0.770** [0.692, 0.838] |

Readings:
- The LLM's lower CI bound is above random on every cell. That meets the user's bar
  ("better than guessing").
- On pairwise against `yc` the LLM leads TF-IDF (0.843 vs 0.785). On nDCG the intervals
  overlap, so it cannot be separated from TF-IDF or SBERT.
- `targeted` is harder (pairwise vs `yc`: 0.778 targeted, 0.874 generic).
- nDCG is inflated by many Good and Potential CVs per JD (random scores 0.74–0.84). Read
  pairwise first.
- The script lived in the scratchpad. It is promoted to the package in P5.

## 5. Why D43 exists (audit §5)

The LLM leg was meant to give a **secondary annotator** merged with the humans into one set.
It never became a replacement (D36 never made). Human–human three-class agreement is low
(kappa 0.24). Adding the LLM label to the other human moves the combination closer to the
hidden human in all four cases (Spearman 0.44 → 0.53 / 0.57 generic, 0.35 → 0.43 / 0.48
targeted). Its score alone does slightly better than its label.

The design spec does not state the reason for "no system trained on LLM labels". The reading
used here is **circularity**: the LLM decides 107 of 326 double-labelled pairs, so a system
trained on LLM output and scored on the final set is partly scored against its teacher
(02 spec §3.5). This plan therefore reports every ordering figure on the `llm_free` subset
as well, and trains only on a batch **outside** the final set.

## 6. The existing LLM scores cannot be the training data

They cover only the 600 final-set pairs, so training on them breaks D46 (nothing fitted on
the final set) regardless of D43. Training at useful scale needs new LLM scores on pairs
from the A2 `train` region. That is the batch in [04](04-llm-scoring-batch.md).
