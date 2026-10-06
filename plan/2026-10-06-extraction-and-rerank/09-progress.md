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

## The LLM batch (run 11)

**365 / 1,000 banked, 0 failures, 0 unparsed** (7 Oct, 07:41). 635 owed. Rate measured:
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

1. **P7 (stage 1).** `Retriever` interface; X1–X4 per 02 §3.4 and registration §2 (X1 uses
   amendment A1: unknown experience = compatible; `role_match` = P3's soft score). Score on
   `text_v2` against K1–K5; retrieval pilot text-only for S2 and Recall@10. Record
   `16-p7-stage1.md`. Needs no LLM data. Memory: SBERT needs ~6.5 GB; don't run it beside
   another heavy job.
2. **Finish the batch**, `--collect` → `llm-train-scores.csv`.
3. **P8 (stage 2).** R1/R2/R3 trained on the batch (train split), selected on its validation
   split; scored with `evaluation/ordering.py` against O1–O5 (full, `llm_free`, `fit_only`).
   Watch the P5 finding: every system orders Fit CVs weakly (pairwise 0.67–0.72), and the LLM
   on targeted Fit CVs is not separable from random. Stop rule in 06 §6.
4. **P9.** Pipeline and UI read extracted fields only; unify the two title normalisers
   (`title_profile` vs `text_leak`) and the two v2 CV-text builders (`corpus.build_cv_text`
   vs `role_family`); findings record; README; full `uv run pytest` and every `--check`.

## Open

Q-A1b (replicate), Q-B2 (span hand-check), Q-D2, Q-D3 — see [05](05-open-questions.md).
