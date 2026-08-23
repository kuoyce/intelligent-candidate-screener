# Derived-artefact manifests

`data/` is git-ignored, so these manifests are how a derived artefact stays reproducible
without committing a corpus. A manifest names documents by **content-addressed ID** and
records the seed; running the builder against the same `data/raw/` reproduces the artefact.

```bash
uv run python -m candidate_screener.data.build  --all --seed 0   # rebuild
uv run python -m candidate_screener.data.verify --derived        # acceptance test
```

Document IDs are `sha256(normalise(text))[:12]` prefixed `r_` / `j_` — A1 ships no publisher
ID, so identity is minted from the text (design spec §2). **No manifest carries a timestamp:**
the git history is the timestamp, so a diff is a real change in the data, never a re-run.

## Task 3.2 — the leak-free A1 split

| File | What it is |
|---|---|
| `fit-split.csv` | `doc_id, doc_type, split` — the authoritative assignment. Every stage and every team member must score the same documents |
| `fit-split-yield.json` | Seed, fractions, per-split counts, and the pooling record (what was dropped and why) |
| `fit-split-survey.json` | What each hold-out fraction buys, across 20 seeds — the evidence for choosing 30% |
| `fit-pair-conflicts.csv` | The 7 duplicate `(resume, jd)` pairs found by pooling, 6 of them labelled two ways, and how each was resolved |
| `fit-c4-overlap.csv` | The 7 A1 resumes whose text also appears in the **rejected** `vm-structured-onet` set, with the split each landed in |

> **Results on this split are NOT comparable to any published number on the shipped
> `cnamuangtoun` train/test partition.** That partition leaks 99.8% of its test resumes into
> train; this one is doubly disjoint — nothing in test shares a resume *or* a JD with train.
> The incomparability is the accepted price of removing the leakage (D8) and must be stated
> wherever Stage 1–4 results appear.

**`doc_type` takes only `train`, `val` and `test`.** `discarded` is a *pair*-level outcome —
a pair whose two documents were assigned to different splits — and is counted in
`fit-split-yield.json`, not in this file. A document is never discarded; it is held out.

**`fit-c4-overlap.csv` is a guard, not a defect.** Those 7 resumes belong in A1. The file
exists so that if the rejected C4 derivatives are ever re-imported as extra training data,
the exclusion can be *applied* rather than remembered — the same people would otherwise be
trained on and scored in test at once.
