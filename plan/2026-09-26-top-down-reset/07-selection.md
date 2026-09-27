# 07 — Selection (P4.6, D48): the registered rule applied

Source: `output/matching/indomain-metrics.json`. The reference is the final set, rule v1
(D41), Fit / No Fit (D47). The metric is per-JD AUC. The rule and comparisons were
registered in [06](06-registration.md) (commit `c29445a`) before any held-out rung was
scored.

## 1. The ladder, against the final set

Total over 59 JDs, with the 95% CI resampling JDs. Within-family = only the candidates
sharing the JD's `Primary Keyword`.

| Rung | System | Status | generic | targeted | **total** [CI] | within-family |
|---|---|---|---|---|---|---|
| 0 | random | exploratory | 0.558 | 0.552 | 0.556 [0.505, 0.606] | 0.580 |
| 1a | metadata rules | exploratory | 0.776 | 0.784 | 0.778 [0.732, 0.819] | 0.540 |
| 1b | skill overlap | held-out | 0.719 | 0.676 | 0.705 [0.652, 0.757] | 0.648 |
| 1a+1b | metadata + skill overlap | held-out | 0.818 | 0.830 | 0.822 [0.776, 0.864] | 0.644 |
| 2 | TF-IDF | exploratory | 0.792 | 0.808 | 0.797 [0.750, 0.843] | 0.724 |
| 2 | BM25 | exploratory | 0.613 | 0.681 | 0.636 [0.584, 0.687] | 0.625 |
| **1a+2** | **metadata + TF-IDF** | exploratory | 0.848 | 0.849 | **0.849** [0.802, 0.891] | 0.708 |
| 3a | SBERT zero-shot | exploratory | 0.789 | 0.746 | 0.774 [0.725, 0.822] | 0.755 |
| 1a+3a | metadata + SBERT zero-shot | exploratory | 0.849 | 0.846 | 0.848 [0.800, 0.891] | 0.721 |
| **3b** | **SBERT, weak fine-tuned** | held-out | 0.872 | 0.820 | **0.854** [0.817, 0.890] | 0.755 |
| 1a+3b | metadata + SBERT, weak fine-tuned | held-out | 0.845 | 0.839 | 0.843 [0.796, 0.886] | 0.705 |
| 4a | cross-encoder, A1-trained | held-out | 0.532 | 0.403 | 0.488 [0.437, 0.539] | 0.532 |
| 1a+4a | metadata + cross-encoder (A1) | held-out | 0.800 | 0.789 | 0.796 [0.745, 0.842] | 0.589 |
| 4b | cross-encoder, weak-trained | held-out | 0.667 | 0.649 | 0.661 [0.616, 0.705] | 0.643 |
| 1a+4b | metadata + cross-encoder (weak) | held-out | 0.818 | 0.829 | 0.821 [0.773, 0.865] | 0.641 |
| 5 | metadata + learned text combination | held-out | 0.829 | 0.825 | 0.828 [0.775, 0.875] | 0.673 |

## 2. The registered comparisons (06 §3)

| # | B vs A | Δ total [CI] | Verdict |
|---|---|---|---|
| C1 | skill overlap vs random | +0.149 [0.079, 0.218] | **justified** |
| C2 | metadata + skill overlap vs metadata rules | +0.044 [0.024, 0.065] | **justified** |
| C3 | SBERT zero-shot vs TF-IDF | −0.023 [−0.081, 0.037] | not resolved |
| C4 | metadata + SBERT zero-shot vs metadata + TF-IDF | −0.001 [−0.024, 0.022] | not resolved |
| C5 | SBERT weak fine-tuned vs TF-IDF | **+0.057** [0.009, 0.106] | **justified** |
| C6 | metadata + SBERT weak fine-tuned vs metadata + TF-IDF | −0.006 [−0.030, 0.017] | not resolved |
| C7 | cross-encoder (A1) vs TF-IDF | −0.309 [−0.370, −0.245] | **harmful** |
| C8 | metadata + cross-encoder (A1) vs metadata + TF-IDF | −0.052 [−0.083, −0.022] | **harmful** |
| C9 | cross-encoder (weak) vs TF-IDF | −0.136 [−0.201, −0.071] | **harmful** |
| C10 | metadata + cross-encoder (weak) vs metadata + TF-IDF | −0.027 [−0.060, 0.007] | not resolved |
| C11 | learned hybrid vs metadata + TF-IDF | −0.021 [−0.045, 0.002] | not resolved |
| C12 | retrieve-then-rank vs retriever alone (Recall@10, 500-candidate pools) | metadata + TF-IDF → CE (weak): −0.254 [−0.333, −0.168]; → CE (A1): −0.328 [−0.406, −0.251] | **harmful** |

Every justified verdict is positive in both strata and on the `llm_free` subset, so none
is LLM-dependent.

Reported only, deciding nothing:
- E4: metadata + TF-IDF vs metadata rules +0.070 [0.043, 0.099];
- E6: SBERT weak vs zero-shot +0.080 [0.048, 0.116];
- E3: BM25 vs TF-IDF −0.161, harmful.

## 3. Selection (06 §4), applied exhaustively

**The rule:** the cheapest rung that no justified, more expensive rung beats. Every more
expensive rung in the context is checked, not only the listed comparisons, because a
missing comparison is not an acquittal. The full trail is in the record under
`selection`.

**With metadata** (both sides carry a `Primary Keyword`):

| Cost | Rung | Beaten by (justified) |
|---|---|---|
| 1 | random | every other rung |
| 2 | metadata rules / keyword match | metadata + skill overlap, metadata + TF-IDF, and 6 more |
| 3 | metadata + skill overlap (0.822) | metadata + SBERT zero-shot, +0.026 [0.002, 0.050] |
| 3 | skill overlap | TF-IDF and 9 more |
| **4** | **metadata + TF-IDF (0.849)** | **none.** Every more expensive rung is harmful or not resolved: BM25, SBERT zero-shot and every cross-encoder are harmful; metadata + SBERT (zero-shot or weak), metadata + CE (weak), the learned hybrid and standalone SBERT weak are not resolved, within ±0.02–0.05 |

**Text only** (no category field, e.g. uploaded files):

| Cost | Rung | Beaten by |
|---|---|---|
| 3 | skill overlap | TF-IDF, SBERT zero-shot, SBERT weak |
| 4 | TF-IDF | SBERT weak (+0.057 [0.009, 0.106]) |
| 5–6 | BM25, SBERT zero-shot | SBERT weak |
| **7** | **SBERT, weak fine-tuned (0.854)** | **none**: both cross-encoders are harmful against it |

## 4. D48

**The selected pipeline:**
- **With metadata: metadata + TF-IDF.** The keyword rule decides the level, and TF-IDF
  (baseline config, IDF from the A2 train region) orders candidates within it.
- **Text only: SBERT, fine-tuned on weak in-domain pairs.**

No cross-encoder and no learned combination earned its place. The fit call is the
metadata rule's (Fit iff the keyword matches). No rung with a threshold beat it on the
class metrics:

| Classifier (total) | accuracy | macro-F1 | kappa |
|---|---|---|---|
| metadata rule | **0.768** | **0.721** | 0.465 |
| CE (weak), threshold 0.53 | 0.575 | 0.575 | 0.191 |
| CE (A1), threshold 0.636 | 0.537 | 0.533 | 0.081 |
| always-Fit floor | 0.612 | — | — |

**Cost, for the record** (`indomain-costs.json`; CPU, this machine):
- metadata + TF-IDF: ~0.6 s per 500-candidate role.
- cross-encoder: ~35 s per 500-candidate role.
- learned hybrid: ~72 s per 500-candidate role, since it runs every component.
- SBERT's logged time uses a warm embedding cache. Cold, encoding costs about 90 ms per
  document on CPU (953 documents in 85 s, 01 §6).

## 5. What this answers

The research question was whether more sophisticated matching earns its complexity. On
this in-domain set, against this panel, the answer is **only where metadata is absent**:
- **Where a category field exists,** a keyword rule plus TF-IDF is as good as anything
  richer at this resolution. The minimal detectable effect is about ±0.02–0.03 AUC, and
  no richer rung comes within it on the positive side.
- **Where it does not,** fine-tuning the bi-encoder on free weak in-domain pairs is the
  one sophistication that paid: +0.057 over TF-IDF, resolved in total and positive in both
  strata.
- **The cross-encoder is harmful as trained here:**
  - on A1 labels that do not reproduce (A13), its standalone AUC is below chance;
  - on weak pairs, it is below TF-IDF;
  - re-ranking a good shortlist with either cuts Recall@10 by a quarter to a third.

  That is a statement about these training signals on CPU budgets, not about
  cross-encoders in general.

Every figure is agreement with the documented panel rule of notebook 08, not correctness.
