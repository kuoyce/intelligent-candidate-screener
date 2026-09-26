# What is a true label? — the reference design for A2

**Status:** **proposal, awaiting approval.** Nothing here is implemented and no figure in
the repository moves until it is.
**Date:** 10 Sep 2026
**Depends on:** D29 (two strata, never pooled), D37 (consensus is a reference and licenses
measurement, not substitution), D39 (Q45 closed, `5d986c82` committed), and Phase 2 of
`plan/2026-09-06-a2-llm-prelabel/` (an annotator is an axis).
**Supersedes nothing.** **D36 stays unmade.**

---

## Why this plan exists

The judge-calibration plan is complete and it ended in a negative result: the `Good Fit`
boundary is not reachable by editing the judge (move 3, outcome B) or by changing its model
(move 2, outcome D). The user has decided there will be **no third annotator**.

So the question stops being *"how do we make the judge agree with the humans?"* and becomes
**"what do we score anything against?"** — on **A2**, because A1's labels fail their own
reproducibility check and cannot be repaired by the judge (§2).

## 1. The decision, stated plainly

**There is no ground truth on this corpus, and the design has to stop implying there is
one.** Every candidate reference below is a *choice of whose judgement counts*, with a
measurable coverage cost and a measurable bias. The deliverable of this plan is to pick
one, name what it excludes, and make the code refuse to quote a figure without it — the
way `assert_consensus_carries_coverage` already does for consensus.

## 2. Why the LLM cannot be the reference — settled, recorded here so it is not re-asked

Two independent arguments, either sufficient.

**(a) On A1 it does not repair the labels, it replaces them.** Against A1's own labels at
n=100, the committed-at-the-time instrument agreed on **0 of 30** `Good Fit`s (17 → `No
Fit`, 13 → `Potential Fit`), 0.500 on `Potential Fit` and 0.650 on `No Fit`. Adopting its
labels for A1 is not a corrected dataset; it is a different one, of unknown quality.

**(b) The disagreement is predicted by the judge's own measured bias, so it is not evidence
about A1.** On A2 — where human labels exist to check against — the committed judge issues
`Good Fit` at **11.0%** / **9.5%** where `yc` issues it at 25.2% / 16.5% and `van` at
51.6% / 30.0%. It is stingy with that class against *every* reference ever measured. A
rater whose threshold on one class is measured as mis-set cannot audit another rater's use
of exactly that class.

And the seductive variant — **use the humans where they agree, the LLM where they
disagree** — is already forbidden by **D37**, in terms:

> the excluded 52% is exactly the population a substituting judge would label alone, with
> no reference and no appeal, and is the harder half by construction.

The contested pairs are where the judge is *least* reliable, not most. That option is
closed.

**What the LLM may still do**, unchanged: act as a **second annotator** in a future
double-labelling wave (licensed by the interchangeability gate), and act as a **system
component** — a reranker — where it has to work but does not have to be correct by
definition. Its score is good at that: AUC **0.832** / **0.756** separating consensus
`Good` from `Potential`, **0.982** / **0.967** separating `Good` from `No Fit`.

## 3. The root finding this design has to accommodate

**Two trained annotators following the same guide reach kappa 0.214-0.250 three-class.**
That is poor agreement, and it is not noise — it is a systematic threshold difference:

| stratum | annotator | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|---|
| `generic` (n=126 shared) | `van` | **51.6%** | 29.4% | 19.0% |
| | `yc` | **25.2%** | 34.5% | 40.2% |
| `targeted` (n=200 shared) | `van` | 30.0% | 34.0% | **36.0%** |
| | `yc` | 16.5% | 25.5% | **58.0%** |

`van` is more generous than `yc` on both strata and at both boundaries. **Picking a single
annotator as the reference is picking a threshold**, and it is the threshold the whole
`Good Fit` argument has been about since D33.

The honest reading is that **the three-class scheme is not reliably applied by humans under
this guide.** That is a finding about the instrument, not about the annotators, and it
bounds what any evaluation on this corpus can claim.

## 4. The candidates, with their real coverage

Measured, not estimated:

| reference | `generic` coverage | `targeted` coverage | bias |
|---|---|---|---|
| **A. Three-class consensus** | 60 / 400 = **15.0%** | 97 / 200 = **48.5%** | easy half by construction; class distribution is neither annotator's |
| **B. Binary consensus** (fit / no fit) | 93 / 400 = **23.2%** | 126 / 200 = **63.0%** | same selection effect, weaker |
| **C. Single annotator** | 100% (`yc`) | 100% | picks a threshold; `van` and `yc` differ by 26 points on `Good Fit` |
| **D. Per-annotator, never pooled** *(status quo)* | 100% | 100% | none — but yields two of every number and no single label |
| **E. Graded from endorsement count** | 100% | 100% | none introduced; resolution varies by pair |
| **F. LLM labels** | — | — | **closed, §2** |
| **G. LLM tie-break on contested** | — | — | **closed, §2 / D37** |

The number that reorders this table is **A on `generic`: 15.0%**. Only 126 of 400 `generic`
pairs were ever double-labelled, so three-class consensus is not a corpus-level reference
there — it is a fifteen-percent sample of the easy pairs. It is fine for *measurement*, as
D37 says, and unusable as *the* evaluation reference.

**Binary nearly doubles it** (15.0% → 23.2%, 48.5% → 63.0%) because `Good ↔ Potential`
disagreement — the boundary both the humans and the judge fail on — does not cross the
fit/no-fit line.

## 5. The recommendation

**A two-tier reference, plus a graded relevance signal for ranking metrics.**

**Tier 1 — the headline evaluation reference: binary consensus (option B).** Fit vs no fit,
both annotators agreeing. It is the operationally relevant question for a *screener*
(shortlist or don't), it has the best coverage of any unbiased-by-construction option, and
it collapses exactly the boundary that D33, D34, move 2 and move 3 all failed to pin down.
Quoted always with its coverage and its selection note, like every consensus figure.

**Tier 2 — three-class consensus (option A), diagnostic only.** Retained because it is what
the guide actually asks for and because the `Good Fit` question does not disappear by being
collapsed. **Never quoted as a headline on `generic`** at 15.0% coverage.

**Alongside — graded relevance from endorsement count (option E), for retrieval metrics.**
Recall@10 and Precision@k need a relevance *ordering*, not a three-class label. Endorsement
count gives one at 100% coverage without inventing agreement: 2 of 2 annotators = strong,
1 of 2 = weak, 0 = negative, and a single-labelled pair carries its own resolution honestly.
This fits the strict/graded machinery D26 already has.

**What stays refused.** Options F and G. And a headline figure on `generic` three-class.

## 6. Open questions

- **Q46 — is binary the scheme this project should have used?** If two trained humans
  cannot apply three classes reliably (kappa 0.21-0.25) but can apply two better (0.29-0.42,
  still not good), then the three-class guide is measuring something humans do not agree
  exists. Adopting Tier 1 answers this operationally without settling it.
- **Q47 — does `generic`'s 31.5% double-labelling rate make it a second-class stratum?**
  D29 forbids pooling the strata and forbids picking the one that suits. But `targeted` is
  100% double-labelled and `generic` is 31.5%, so they do not support the same references.
  A design that quotes Tier 1 on both must say that `generic`'s rests on 23.2% and
  `targeted`'s on 63.0%.
- **Q48 — what is the A1 corpus for now?** The proposal is: **weak supervision for training
  only, never an evaluation reference again**, and D26's ceilings retired rather than
  restated. Not yet decided.
- **Q49 — does the judge's score belong in Tier 1's construction?** No, under §2. But it
  may belong in the *system*, and the boundary between "component" and "reference" needs
  stating once, in code, so it cannot erode.

## 7. What approval would authorise

1. A `reference` module making Tier 1 / Tier 2 / graded explicit, with an assertion that
   refuses any figure quoted without its coverage and tier.
2. Re-quoting the existing A2 figures under it — **no new spawns, no new human hours.**
3. A decision record for Q48 (A1's demotion), which is the change that touches
   `data-catalog.md`, the A1 card and D26.

**It authorises no annotation, no dispatch and no re-judging.** Every number this plan
needs is already on file.
