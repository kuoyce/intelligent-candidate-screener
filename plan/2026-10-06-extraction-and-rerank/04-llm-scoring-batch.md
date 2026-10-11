# 04 — The LLM scoring batch (reopening D42, once)

## 1. What it is

New 0–100 scores from the existing instrument, on pairs from the A2 **train** region, to
train the stage-2 reranker. It is the only LLM dispatch this plan makes.

- **Instrument:** `5d986c82` under `claude-haiku-4-5`, the same series the final set uses.
  The prompt is not changed.
- **Unit:** **one call per pair** (a JD and one CV), pointwise, returning a label and a
  0–100 score. This is the form the order check validated (01 §4); a listwise
  "rank these ten" prompt would be a different, unvalidated instrument and is not used.
- **Corpus:** A2 `train` region only (D19). **No** final-set JD or CV, **no** eval-region
  document. A test enforces both.
- **Output:** `docs/data/manifests/llm-train-scores.csv`, ids and scores only (no document
  text), keyed by (`agent_sha256`, model, run, draw). Never written to `judgements.csv`.

## 2. How many calls

Measured on 8–10 Sep 2026: 600 pair-calls (runs 9 and 10) took **nine sittings**, each
sitting answering roughly **40–130 pairs** before the account's session limit (about 67 on
average). The calls draw on the **account's session limit**, `/low-priority` does not extend
to them, and a quota failure must be waited out (the repo's own note). A call sent a median
of 3,792 characters (mean 4,157, max 11,300).

| Option | JDs × CVs | Calls | Sittings at ~67 / sitting | Training signal |
|---|---|---|---|---|
| A | 60 × 10 | **600** | ~9 | 48 train JDs: 2,160 within-JD pairs before the δ filter |
| **B (recommended)** | 100 × 10 | **1,000** | **~15** | 80 train JDs: **3,600** within-JD pairs; 20 validation JDs |
| C | 200 × 10 | **2,000** | ~30 | 160 train JDs: 7,200 pairs; 40 validation JDs |

**Plus a replicate:** re-score a random 10% (100 pairs for B) once more, to measure the
run-to-run noise of the score. That is an extra **~100 calls**, and the figure goes in the
record. It is optional; without it the score's noise is unknown.

The recommendation is B. A is the size of the final set and may be too small to fine-tune a
cross-encoder on (the A1-trained one needed a larger learning rate just to leave chance).
C is the safer size for a cross-encoder but costs about twice B in sittings. **This number
is an estimate from one prior run, not a promise; the account's limit may differ today.**

## 3. Are there more scoring batches?

**No second batch is planned.** This is the one batch the user approved. Inside it:
- the **validation split** (20 JDs of 100, disjoint by JD and CV) is carved from the same
  calls, so no separate validation batch is needed;
- the optional **replicate** is part of the same scoring run.

A second batch would be needed only if validation shows the first is too small. There is no
automatic extension: **any further batch needs its own decision in the log.** The stop rule,
fixed now: if the batch's validation figures are below the registered bar, the plan reports
that and does not scale the batch to chase a result.

## 4. Calendar

The batch is the critical path (about 15 sittings for B). P6 therefore starts right after
P0, while P1–P5 proceed. Each sitting banks answers as they arrive, so a lost tool result
loses nothing. Coverage is checked before collecting; a short run is never collected.

## 5. Safeguards (D42 stays closed except for this)

- The reopen is **scoped in code** by corpus (A2 train), instrument (`5d986c82`), model
  (haiku) and a hard pair cap (the chosen N plus the replicate). Every other entry point
  (`llm_recheck --dispatch` and `--judge` outside this batch, `prompt_calibration`) still
  raises `DispatchClosed`, with a test.
- Where the code lives is settled in P6.1 after reading `llm_recheck` (the annotation
  package is frozen and read-only, D42). The existing append-only banking and idempotent
  retry are reused; how is not assumed here.
- The batch manifest has a digest; a `--check` fails if a document id is in the final set
  or the eval region.

## 6. How the scores are used and kept honest

- Trained on: the batch's train split only. Tuned on: its validation split only (D46).
- Scored on: the final set, with the `llm_free` subset beside the total (D43), because the
  final label uses the LLM on 107 tie-breaks.
- The reference ordering for scoring is **always the human grades**, never an LLM score.
- The training data's quality is itself reported: the score's replicate noise, and (on the
  final-set pairs only) its agreement with the humans (01 §4).
