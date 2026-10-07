# 04 — Open questions and assumptions

Updated 7 Oct 2026.

## Answered (user, 7 Oct 2026)

| ID | Question | Answer |
|---|---|---|
| Q-R1 | May a post-hoc Y rung be **adopted** as the stage-1 retriever if it passes S2? | **Yes**, for building the pipeline, always labelled post-hoc beside the registered P7 miss |
| Q-R2 | JD title scope: the A2 JD `Position` as given? | **Use as given** |
| Q-R3 | Dev relevance proxy = same oracle role family; add "and experience compatible"? | **As recommended: role family only** |
| Q-R4 | Time: core only, or core + Q4 + Q5, before returning to P8? | **Review again after this phase.** Read as: build Q0–Q3 (the core up to the dev results), then stop for a review that decides Q4/Q5 before Q8 freezes anything. Q8 is not run in this pass, since freezing 03 §2 first would shut Y2m and Y5 out |
| Q-R5 | Fuzzy title similarity: character 3-gram cosine, or SBERT on titles? | **3-gram cosine** |

## Open

None from this plan's original list. New ones are added to [05](05-findings.md) §Open as they arise.

Carried from the running plan: Q-P7-1, Q-P7-2 (P8.5's retriever: this plan may supply it),
Q-A1b (replicate), Q-B2 (title hand-check: more relevant now, since titles carry the match).

## Assumptions (to confirm)

1. The A2 JD `Position` is a fair stand-in for a recruiter-entered title (no Djinni-only
   artefacts). Checked in Q2 by reading a sample of titles; findings recorded.
2. The P3 validation part (≈ 10.6k JDs, 15.8k CVs) has enough JDs with a non-empty title for
   400 dev pools after the exclusions. Measured in Q1.
3. "Any of the CV's roles matches" will over-match on CVs that mention other roles
   (colleagues, teams). C2's role-context option and the set-size measurement are there to
   show how much.
