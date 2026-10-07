# 02 — Work plan

Order is by value: the core (Q0–Q3) answers "does the JD title + a hard role match fix the
shortlist?"; the rest is ordered so it can be cut if time runs short. About 19 days remain
(Q-D1, as of 7 Oct), and P8 (3 d) and P9 (1.5 d) of the running plan still need room.

| Phase | Work | Acceptance | Effort | Tier |
|---|---|---|---|---|
| **Q0** | D54 in `docs/decisions.md` (JD title at inference; amends D53). `AGENTS.md` evaluation rule 6 updated. Commit this plan | Decision entry + plan link; `test_decisions` passes | 0.25 d | core |
| **Q1** | Development set (01 §1): `evaluation/dev_pools.py`, cached, with a `--check` report (sizes, relevant per pool, disjointness). X2 and TF-IDF scored on it as the reference | Named tests: dev documents are disjoint from the role classifier's fit part, the calibration draws, the weak pairs and the eval region; the module never reads `a2-final-labels.csv` / `judgements.csv` | 1 d | core |
| **Q2** | C1 JD title → family; C2 CV role set; C3 `family_match`, `title_similarity`. Accuracy of C1 and C2 on dev (and C1 on the 60 final-set JDs, measurement only) | Named tests: CV `Position`, `Primary Keyword` and experience fields are never read (scrambled columns → identical scores); a CV with two role titles matches a JD of either | 1.5 d | core |
| **Q3** | Y1, Y2, Y2t, Y4 on dev; choose `k`, role-context, Y2 vs Y2t on dev-select. **Early stop:** if the dev-best classical rung's proxy Recall@50 does not beat X2's on dev, stop, record, and go to Q8 with what exists | Dev record with every option's figure | 1 d | core |
| **Q4** | C4 main tech: the check (JD main tech vs `Primary Keyword`, dev), then Y2m on dev if the check passes (agreement ≥ 0.6 on non-generic keywords, fixed here) | Check figure; Y2m dev figure or the stop recorded | 1.5 d | next |
| **Q5** | C7 learned combination on dev-train, C on dev-select | Coefficients and dev figures recorded | 1 d | next |
| **Q6** | C6 classifier candidates (b), (c) on P3's split and select draw | Selection table; swap or keep | 1.5 d | later |
| **Q7** | C5 related roles: one dev run | Gain ≥ 0.02 → Y-rel joins; else recorded and closed | 0.5 d | later |
| **Q8** | Freeze 03 §2 (one configuration per rung, committed). Then score once on the final set: S2 (Recall@50 vs bar), Recall@10, per-JD AUC, comparisons L1–L5. Record `output/matching/shortlist-recall.json` with `--check`; findings record | Committed freeze precedes the scoring commit; the record reproduces | 1 d | core |
| **Q9** | Docs: README, the running plan's progress record, `AGENTS.md` command list | Every figure read from the record | 0.25 d | core |

Core = Q0–Q3, Q8, Q9: about 5 days. With Q4 and Q5: about 7.5 days. With everything: 9.5.

**The LLM batch** continues between phases (sequential batches, a check between each,
never beside an SBERT job or a running sub-agent).

**Implementation** follows the repository workflow (sub-agents per phase), except where a
phase is short enough that a sub-agent would cost more session limit than it saves; any
such phase is named in the findings record.
