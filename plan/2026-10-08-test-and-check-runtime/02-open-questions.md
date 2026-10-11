# 02 — Open questions

All six resolved by the user on 8 Oct 2026.

| # | Question | Answer (8 Oct) |
|---|---|---|
| Q1 | When is the **cold** tier mandatory? | **(b)**: before a commit that touches model or training code, before a plan is marked complete, and before a final-set score. Warm for the everyday loop |
| Q2 | Add a read-only "check from cached model" mode to `role_family`? | **Yes** |
| Q3 | Run the superseded records (indomain v1, `retrieval`, `retrieval_pilot`, `a1_secondary`) only in the cold tier, or when `--affected` hits them? | **Yes** |
| Q4 | Where do the check logs and the cross-encoder score cache live? | **Logs: `logs/`** at the repository root, added to `.gitignore`. Score cache: `data/processed/matching/ce-scores/` (already git-ignored; this part was not contested) |
| Q5 | Is `-j 2` acceptable as the runner's default? | **"Sure"**, read as: `-j 2` is the default. Checks marked run-alone (peak RSS ≥ 4.5 GB) never share the machine. *Explicit assumption: `-j 1` stays available as a flag* |
| Q6 | Attempt W6 (vectorised TF-IDF)? | **Skip.** W6 is dropped |

Branching (user, 8 Oct): `chore/test-check-runtime` is cut from `feat/shortlist-recall`, whose
feature is complete. Once implemented and reviewed, it is merged back into
`feat/shortlist-recall`.

## Open during implementation

(none yet)
