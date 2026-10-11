# 04 — Success measures (Q53)

**Status: partly decided, 26 Sep 2026.** The remaining items (§3) must be fixed **before
P1.5** (registration). Targets set after the results are visible would be goalposts moved
to fit, and the report could not defend them.

## 1. Decided (user, 26 Sep 2026)

1. **Three classes are dropped** (D47). Every measure is **Fit / No Fit**, where `Fit` =
   `Good Fit` ∪ `Potential Fit`. This includes the success measures.
2. **Inter-annotator agreement is reported, with no target.**
   - Agreement and kappa for `yc`–`van`, `yc`–LLM and `van`–LLM, for generic, targeted and
     total (02 §3.6).
   - They appear beside every system figure as context. Nothing is expected to hit them,
     and no system target is derived from them.
3. **Reporting** covers generic, targeted and total (Q56).

*Interpretation, to confirm (S0 below).* "Inter-annotator agreement" is taken to mean the
agreement **between annotators**. The proposal's separate target, **system** agreement
with human judgement, is treated as still open (S1). It is now measured against the final
Fit / No Fit set.

## 2. The proposal's targets, as they now stand

| Proposal target (§3.4, §7) | Restated Fit / No Fit | Status | Evidence so far (scratch probe) |
|---|---|---|---|
| ≥ 85% candidate–role fit agreement with human judgement | Accuracy of a system's Fit / No Fit call against the final set | **Open: S1** | Keyword match as a classifier 0.77; A1 baseline 0.72; always-`Fit` floor 0.61 |
| Fit-classification macro-F1 > 85% | Binary macro-F1 against the final set | **Open: S1** | Keyword match 0.72; A1 baseline 0.71; always-`Fit` floor ≈ 0.38 |
| *(new)* Ranking quality | Per-JD AUC, `Fit` vs `No Fit`, against the final set | Decided by the relative rule (02 §5.4). Absolute target open: S1 | Keyword + TF-IDF 0.85 [0.81, 0.90]; keyword + SBERT 0.86 [0.81, 0.90]; random 0.54. **Already seen** |
| Retrieval Recall@10 > 80% (classical) / > 90% (advanced) | Recall@50 of final-set fits in 500-candidate pools (02 §5.2) | **Open: S2** | **Not yet measured**, so a target fixed now is not post-hoc |
| Extraction F1 > 85% | Span F1 on SkillSpan test (and DataTurks, with its n=20 caveat) | **Open: S3** | Not yet measured |
| Screening time > 80% lower (25 → 2 min) | System latency per candidate and per 500-candidate role | **Open: S4** | Not yet measured. Human time needs a user study, which is excluded |
| 500+ applications per role | Throughput and latency on the 500-candidate pools | Proposed: keep as is | Not yet measured |
| Performance gaps across groups | Per-JD AUC gaps across experience band, English level, role family and stratum (no demographics in Djinni) | Proposed: report, no target | — |
| Stable core-intent responses (UI) | One scenario test per intent passes. Evidence-faithfulness invariant holds | Proposed: keep as a pass/fail gate | — |
| Model-selection conclusion | The registered decision rule (02 §5.4) | Decided | — |
| *Inter-annotator agreement* | Reported (02 §3.6) | **Decided: no target** | `yc`–`van` 0.672 / kappa 0.353 (n=326) |

## 3. Still open, with a recommendation each

**S0. Confirm the interpretation in §1.** Is the system-vs-human agreement target (the
proposal's "≥ 85% agreement") still a candidate target, measured against the final set?
Or was it meant to be dropped along with inter-annotator agreement?

**S1. An absolute target for Fit / No Fit quality against the final set?**

| Option | Meaning | Consequence on today's evidence |
|---|---|---|
| (a) Keep the proposal's 85% | Accuracy ≥ 0.85 and binary macro-F1 ≥ 0.85 against the final set, reported as pass or miss | The number was fixed before any result existed, so it is not post-hoc. Today's cheapest systems are at 0.72–0.77 and 0.71–0.72, so it is a real stretch target and may be missed |
| (b) No absolute target | Success is the relative rule only (02 §5.4). Absolute figures are reported without a bar | No goalpost risk. The report cannot say "good enough" in absolute terms |
| (c) An AUC target | For example, per-JD AUC ≥ 0.85 | **Not recommended.** Cheap systems were already seen at 0.85–0.86, so any value picked now is chosen with the answer in view |

*Recommendation:* **(a) together with the relative rule.** Keep the proposal's own 85% for
binary accuracy and macro-F1, reported honestly as pass or miss. Let the §5.4 rule decide
which rung is selected. The absolute bar then says whether anything is "good enough", and
the relative rule says whether sophistication paid.

**S2. Retrieval target at pilot scale.** Keep the proposal's thresholds, applied to the
measurable version: Recall@50 ≥ 0.80 for the classical retriever and ≥ 0.90 for the
advanced one, over the 500-candidate pools.
- K = 50 is a 10% shortlist.
- *Recommendation:* adopt. Nothing has been measured, so fixing the numbers now is
  exactly what pre-registration is for.

**S3. Extraction F1 target.** Keep 85% strict span F1 on SkillSpan test as the
proposal's figure, reported as pass or miss, beside the relative comparison (learned
vs rules).
- The dataset's own paper reports its best systems well below 0.85. The exact figure is
  to be checked and cited in P5, not quoted here from memory. So expect a miss that
  reflects task difficulty.
- *Recommendation:* same pattern as S1: keep the number, and decide by the relative
  comparison.

**S4. Screening time.** Report measured system latency (per candidate, per
500-candidate role) against the proposal's 2-minute budget. Make no claim about recruiter
time saved, since that needs a user study.
- *Recommendation:* adopt.

## 4. What each answer changes in the plan

- **S1 (a):** P1.5 registers the 0.85 bars. P8's findings table gains a pass/miss column.
  No extra compute.
- **S1 (b):** P1.5 registers the relative rule only.
- **S2 / S4:** P3.6 and P7.3 already measure these. Registration adds the thresholds.
- **S3:** P5.5 reports against the bar.

## 5. Why the dropped items were dropped (for the report)

- **Three classes.**
  - Two trained annotators agree on 48% of three-class labels (kappa 0.21–0.25).
  - The `Good`/`Potential` boundary could not be pinned down by guide edits, a model
    change or a scoring instrument (Q38).
  - Fit / No Fit is the operational question a screener answers, and the one the
    annotators agree on most (67% agreement).
- **An inter-annotator target.** Agreement between annotators describes the difficulty
  of the task and the instrument. It is not something the system can move. Reporting it
  gives every system figure its context without turning it into a goal.
- The evidence for both is in [05-annotator-vs-system.md](05-annotator-vs-system.md).
