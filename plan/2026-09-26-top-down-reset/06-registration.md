# 06 — Registration (P1.5): the rule, the comparisons and the bars, fixed before the held-out rungs

**Committed before any held-out rung is scored.** Git history is the proof: this file
precedes the first figure for skill overlap (1b), fine-tuned SBERT (3b), the
cross-encoders (4a, 4b), retrieve-then-rank (4c) and the learned hybrid (5).

**Reference:** `docs/data/manifests/a2-final-labels.csv`, rule v1. The sha256 is recorded
in `output/matching/indomain-metrics.json`. Fit / No Fit only (D47). The metric is per-JD
AUC unless stated.

## 1. What has already been seen (exploratory)

These were measured before this file existed, in the audit probe or in the first P1.4
record:

- random;
- keyword match and metadata rules (1a);
- TF-IDF and BM25 (2);
- metadata + TF-IDF and metadata + BM25;
- SBERT zero-shot with chunked mean pooling (3a), and metadata + SBERT, seen in the probe.

Their figures are reported, **but none of them tests the rule.**

Current reading (total, 59 JDs):
- metadata + TF-IDF − metadata rules = **+0.070** [0.043, 0.099];
- TF-IDF − metadata rules = +0.019 [−0.042, 0.084], not resolved;
- BM25 − TF-IDF = **−0.161** [−0.215, −0.107], harmful.

## 2. The decision rule (02 §5.4, restated)

For rung B against the cheaper rung A named in §3, on per-JD AUC:

| Verdict | Condition |
|---|---|
| **Justified** | the paired Δ(B − A) 95% CI on the **total** lies above 0, **and** the Δ point estimate is > 0 in both `generic` and `targeted` |
| **Justified, one stratum** | the total CI lies above 0, but one stratum's point estimate is ≤ 0. Reported; does not count for selection |
| **Justified, LLM-dependent** | justified, but the total Δ on the `llm_free` subset is ≤ 0. Reported; does not count for selection |
| **Harmful** | the total CI lies entirely below 0 |
| **Not resolved** | anything else. The minimal detectable effect (the total CI half-width, currently about 0.03–0.06) is quoted beside it |

**Within-family AUC** is reported beside every verdict but does not decide it.

## 3. The registered comparisons (B vs A)

**A is fixed now, not chosen after results.** Text rungs are compared with the current
cheap text rung (TF-IDF). Their metadata hybrids are compared with the current cheap
champion (metadata + TF-IDF).

| # | B | A | Held-out? | Question |
|---|---|---|---|---|
| C1 | skill overlap (1b) | random | yes | Does the proposal's Stage-1 rule work at all? |
| C2 | metadata + skill overlap | metadata rules | yes | Does skill overlap add to metadata? |
| C3 | SBERT zero-shot (3a) | TF-IDF | no (seen) | Semantic vs lexical, standalone |
| C4 | metadata + SBERT zero-shot | metadata + TF-IDF | no (seen) | The same, under metadata |
| C5 | SBERT weak-fine-tuned (3b) | TF-IDF | yes | Does in-domain weak supervision make semantic retrieval pay? |
| C6 | metadata + SBERT weak-fine-tuned | metadata + TF-IDF | yes | The same, under metadata |
| C7 | cross-encoder, A1-trained (4a) | TF-IDF | yes | Does the proposal's Stage 4 pay, trained as proposed? |
| C8 | metadata + cross-encoder (A1) | metadata + TF-IDF | yes | The same, under metadata |
| C9 | cross-encoder, weak-trained (4b) | TF-IDF | yes | Stage 4 with in-domain weak supervision |
| C10 | metadata + cross-encoder (weak) | metadata + TF-IDF | yes | The same, under metadata |
| C11 | learned hybrid (5) | metadata + TF-IDF | yes | Does learning the combination beat the fixed one? |
| C12 | retrieve-then-rank (4c) | its retriever alone | yes | On the 500-candidate pools: does re-ranking the shortlist pay? |

These are also reported, but decide nothing:
- chain comparisons: 3b vs 3a, 4b vs 4a;
- metadata + BM25 vs metadata + TF-IDF.

## 4. Selection (D48, applied in P4.6)

**Cost order**, cheapest first, fixed by what each rung needs to run:

1. random
2. keyword match / metadata rules
3. skill overlap
4. TF-IDF
5. BM25
6. SBERT zero-shot
7. SBERT fine-tuned
8. cross-encoder
9. learned hybrid

A metadata hybrid costs what its text rung costs.

**The pipeline adopts the cheapest rung that no justified, more expensive rung beats**
(§2), among the rungs that meet the quality floor in §5, if one is set.

## 5. Success bars: provisional defaults (Q53 S0–S4)

The user left S0–S4 open. These are the recommendations in
[04](04-success-measures.md) §3, registered **provisionally** so that no bar is chosen
after the results are visible. They set the **pass/miss column** of the findings. They do
not change what is built or which rung the rule selects.

If the user changes a bar after held-out results exist, the change is recorded here as
post-hoc, with its date.

| Item | Bar | Applies to |
|---|---|---|
| S0 | The proposal's system-vs-human agreement target stays, measured against the final set | Interpretation, pending confirmation |
| S1 | **Accuracy ≥ 0.85** and **binary macro-F1 ≥ 0.85** against the final set, total scope | Rungs that emit Fit / No Fit: the metadata rule, the cross-encoders' heads, the learned hybrid. Unsupervised rankers are not classifiers and carry no S1 figure |
| S2 | **Recall@50 ≥ 0.80** (classical retriever) and **≥ 0.90** (advanced retriever) of final-set fits, over the 500-candidate pools | P3.6 / P4.3 |
| S3 | **Strict span F1 ≥ 0.85** on SkillSpan test | P5, reported as pass or miss beside rules vs learned |
| S4 | Report the measured end-to-end latency per candidate and per 500-candidate role against the proposal's 2-minute-per-candidate budget. No claim about recruiter time | P7.3 |

**Class calls:** a rung that emits Fit / No Fit does so with a threshold fitted on its own
training data's validation split (A1 CV folds, or the weak validation split), never on the
final set (D46).
