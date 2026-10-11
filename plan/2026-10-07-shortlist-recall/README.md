# Shortlist recall: role as a filter, the JD title, main-tech matching

**Status: approved (7 Oct 2026)**, with the answers to Q-R1–Q-R5 in [04](04-open-questions.md).
Branch `feat/shortlist-recall`. **Outcome (8 Oct):** hard role filters lost to X2 on dev (Q3 early stop); a larger soft role weight beat X2 on dev but tied it on the final set (Recall@50 0.781 vs 0.779), so S2 is still missed and nothing is adopted.
Progress and findings: [05](05-findings.md).

Follows [P7](../2026-10-06-extraction-and-rerank/16-p7-stage1.md): text-only stage 1 missed
S2 (X2 Recall@50 0.779 < 0.80; X4 0.761 < 0.90). The exploratory diagnosis (16 §6a) traced
the loss to the role signal: the JD's family is guessed wrong for 18 of 59 JDs (Recall@50
0.64 there vs 0.84 where right), same-role fits get a low soft agreement (median 0.05 vs
0.64), and the soft sum lets other-role CVs with similar wording fill the shortlist.

## Objective

Raise text-only Recall@50 on the 500-candidate pools to the registered S2 bars, without
tuning on the final set, by fixing **how the role is obtained and used**, then report
whether it worked. The existing registered P7 result (the miss) stands and is reported
beside whatever this plan produces.

## Decisions taken (user, 7 Oct 2026)

| # | Item | Decision |
|---|---|---|
| 1 | JD title | **Allowed at inference.** A recruiter always types the job title. The A2 JD `Position` becomes an input, `jd_title`. Amends D53 (new **D54**). CV `Position`, `Primary Keyword` and the experience fields stay excluded |
| 1b | CV roles | A person may have held several roles. The CV side is a **set** of families; a pair matches if **any** of them matches the JD's |
| 2 | Filter vs fuzzy | **Categorical families on both sides → hard match is fine. Matching extracted text (title strings) → fuzzy is fine** |
| 3 | Main-tech / weighted skills | Explore |
| 4 | Related roles (partial credit) | One simple test on the development set; continue only if it shows a clear early gain |
| 5 | Shortlist wider, then rerank | **Deferred**: only if the options here are exhausted, or when the shortlist step is finalised |
| 6 | Learned combination | Try it; the concern (too little data) is addressed by the development set (02 §1) |
| 7 | Better role classifier | Do it |

## Files

| File | What |
|---|---|
| [01-design.md](01-design.md) | Development set, components, rungs |
| [02-work-plan.md](02-work-plan.md) | Phases, acceptance, early stops, effort |
| [03-registration.md](03-registration.md) | What is fixed before any final-set scoring |
| [04-open-questions.md](04-open-questions.md) | Open questions and assumptions |
| [05-findings.md](05-findings.md) | Implementation record: what was built, dev figures, deviations |

## Relation to the running plan

Stage 1 only. P8 (stage-2 rerankers) and the LLM batch in
`2026-10-06-extraction-and-rerank` are unaffected and continue; the batch runs between
phases here, never beside a heavy job.
