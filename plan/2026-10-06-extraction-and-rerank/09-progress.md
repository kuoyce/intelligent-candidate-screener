# 09 — Progress and next steps (handoff, 7 Oct 2026)

Branch `feat/extraction-and-rerank`. Everything below is committed; the working tree is clean
except the uncommitted LLM bank under `data/processed/rerank/` (git-ignored by design).

## Done

| Phase | Commit | Record |
|---|---|---|
| Plan | `1f4bbc9` | README, 01–05 |
| P0 decisions D50–D53, `AGENTS.md` | `8984c12` | `docs/decisions.md` |
| P1 `text_v2`, leak report, v2 re-score (exploratory) | `0c10c95` | [10](10-p1-text-v2.md) |
| P2 title data checks | `0d0758c` | [11](11-p2-data-checks.md) |
| P3 role family (E2), Q-B3 accepted | `4baf7f0` | [12](12-p3-role-family.md) |
| P4 experience and seniority (E3, E4) | `3b35468` | [13](13-p4-experience.md) |
| P5 ordering module + records | `52a115e` | [14](14-p5-ordering.md) |
| P5.3 registration + amendments A1, A2 | `911a4b3`, `4ddbece` | [06](06-registration.md) |
| P6 batch code + manifest | `204185a` | [15](15-p6-batch.md) |
| P7 stage 1: X1–X4, K1–K5, S2 (text-only) | `799268f` | [16](16-p7-stage1.md) |
| Shortlist recall (post-hoc, branch `feat/shortlist-recall`): Q0–Q3, early stop on dev | see its record | [shortlist-recall 05](../2026-10-07-shortlist-recall/05-findings.md) |

## The LLM batch (run 11)

**436 / 1,000 banked** (7 Oct, 16:20 SGT). 564 owed, 5 of them quarantined by a transient DNS
error (`EAI_AGAIN`, exit 1); the 24 lost to the 11:44 session limit were retried and banked. Rate measured:
50 pairs per ~6.5 min at concurrency 5. The user wants batches run **sequentially**, with a
check between batches, and only when no sub-agents are running.

One batch, then check:

```bash
uv run python -m candidate_screener.rerank.scoring_batch --judge --limit 50 --concurrency 5
uv run python -m candidate_screener.rerank.scoring_batch --coverage
```

Stop at any non-zero exit (a quota exit lists `kinds`; wait for the reset, then re-run the
same command — it is idempotent). When coverage is full: `--collect`. The replicate (run 12,
`--replicate`) waits on Q-A1b.

## Next steps, in order

1. ~~**P7 (stage 1).**~~ Done ([16](16-p7-stage1.md)): K1 justified (X2 > TF-IDF, +0.061);
   K2–K5 not resolved; **S2 missed** for both groups (X2 R@50 0.779, X4 0.761); no stage-1
   rung adopted. Q-P7-1, Q-P7-2 open.
2. **Finish the batch**, `--collect` → `llm-train-scores.csv`.
3. **P8 (stage 2).** R1/R2/R3 trained on the batch (train split), selected on its validation
   split; scored with `evaluation/ordering.py` against O1–O5 (full, `llm_free`, `fit_only`).
   Watch the P5 finding: every system orders Fit CVs weakly (pairwise 0.67–0.72), and the LLM
   on targeted Fit CVs is not separable from random. Stop rule in 06 §6.
4. **P9.** Pipeline and UI read extracted fields only; unify the two title normalisers
   (`title_profile` vs `text_leak`) and the two v2 CV-text builders (`corpus.build_cv_text`
   vs `role_family`); findings record; README; full `uv run pytest` and every `--check`.

## Open

Q-A1b (replicate), Q-B2 (span hand-check), Q-D2, Q-D3, Q-P7-1, Q-P7-2 (P8.5 retriever) — see [05](05-open-questions.md).
