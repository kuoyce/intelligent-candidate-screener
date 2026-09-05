# The guide was rewritten for the judge, and it moved (D34)

**Run 5 Sep 2026 on `feat/judge-calibration`.** Run 4: 100 pairs, `claude-haiku-4-5`,
100/100 parsed, 0 refused. `docs/data/manifests/llm-recheck.csv`,
`output/annotation/llm-recheck-report.json`.

The target changed with this run. D33 asked whether A1's labels are reproducible, and the
answer was no. This run asks a different question — **can the judge be made to apply the
three classes the way `yc` does** — and is calibrated against `yc` over the 100 recheck
pairs, not against A1.

## Why the old instrument disagreed

Diagnosed from the run-1 returns before anything was changed.

`yc` and the judge already agreed on rejection: of `yc`'s 58 `No Fit`s the judge matched
**57**. The whole disagreement sat *inside* a matched profession. Of the judge's 82
`No Fit`s only **8 cited a different profession** — the rest cited a missing named tool
(20), a missing industry (9) or a years gap (14). Under the guide's own worked examples all
three of those shapes were `Potential Fit`.

The cause was the tie-breaking rule. "When you cannot decide between two labels, choose the
lower one" was written for humans anchoring high; a model reads it as *any unresolved doubt
→ down* and applies it transitively, so a missing sixth bullet demoted a well-evidenced
accountant past `Potential` to `No Fit`. Nothing in the guide required `No Fit` to have a
reason.

`yc` did not follow the guide either, in the opposite direction. All 12 pairs where `yc`
said `Good Fit` and the judge said `No Fit` were the same broad occupation differing in
specialisation or seniority — the guide's own `Potential Fit` shape. `yc` uses `Potential`
on 34.5% of Djinni pairs and 17% of A1 pairs, with `Good` outnumbering it on A1 only.

Independent of either judge: `yc`'s labels track JD-CV TF-IDF cosine at rho = 0.494 and the
judge's at 0.365, while **A1's are non-monotone** — mean cosine 0.060 / 0.072 / 0.068 for
`No` / `Potential` / `Good`. A1's `Good Fit` scores below its `Potential Fit`. That is not
an ordinal scale, and it does not depend on trusting either blind judge.

## What changed in the guide

`docs/annotation-guide.md` sha256 `842d7b0e` -> `3a51b7a8`;
`.claude/agents/a1-judge.md` `47fc2e48` -> `75e00ea3`.

- **All nine worked examples removed.** They were clean, and none of them was the modal hard
  pair in this corpus — right profession, wrong specialisation.
- **New `## Decide in this order`**: profession, then career stage, then core activity.
  `No Fit` is reachable only from the first two steps.
- **`No Fit` is a claim, not a residue.** Assign it only when step 1 or step 2 can be named.
- **A missing requirement is a screening question, not a rejection** — a missing tool,
  industry, or a year or two short of a minimum is `Potential Fit` when the profession and
  the core activity match.
- **The tie-break is scoped** to the `Good`/`Potential` boundary and explicitly barred from
  reaching `No Fit`.
- **Boilerplate clause.** 13 of the 60 CVs in the recheck draw carry a header that
  contradicts the document — six open "Highly motivated Sales Associate with extensive
  customer service and sales experience" above an accountant's or a .NET developer's actual
  history, and seven carry a "Guest services / Inventory control / Merchandising / Loss
  prevention" skills list. The guide now says to judge the Experience section.
- **Not a domain expert.** "You cannot check whether a technical claim on a CV is true, only
  whether it is relevant" — the useful half of the HR-specialist candidate, folded in rather
  than kept as a separate instrument.

## The numbers

Against `yc`, n=100, over the same draw throughout:

| instrument | runs | model | kappa | 95% CI | binary kappa | binary CI | agreement |
|---|---|---|---|---|---|---|---|
| `47fc2e48` | 1, 2, 3 | sonnet | 0.287 | [0.155, 0.426] | 0.421 | [0.266, 0.587] | 0.65 |
| **`75e00ea3`** | **4** | **haiku** | **0.352** | [0.229, 0.474] | **0.614** | [0.447, 0.760] | 0.61 |

Marginals over the 100 — `yc` is 25 `Good` / 17 `Potential` / 58 `No`:

| | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| old judge | 4 | 14 | 82 |
| new judge | 3 | 42 | 55 |

`yc` (rows) against the new judge:

| | `Good Fit` | `Potential Fit` | `No Fit` |
|---|---|---|---|
| `yc` `Good Fit` (25) | 3 | **20** | 2 |
| `yc` `Potential Fit` (17) | 0 | 11 | 6 |
| `yc` `No Fit` (58) | 0 | 11 | **47** |

**The binary collapse is where it moved: 0.421 -> 0.614**, and the two intervals barely
overlap. On the coarsest question the instrument asks — reject or not — the judge and `yc`
now agree substantially rather than fairly. Three-class kappa rose less and raw agreement
fell slightly, both for the same reason: the judge stopped defaulting to `No Fit` and
started using the middle class, which is where `yc` mostly is not.

The 26 `No Fit`s that were the whole disagreement are gone. Twenty of `yc`'s 25 `Good Fit`s
now return `Potential Fit` with reasons of the intended shape — *"General accounting
experience present, but missing cost accounting methodology and manufacturing environment"*;
*"Strong data analysis and reporting skills match core role activities, but lacks telecom
industry experience"*. The two still rejected are both IC engineer against engineering
manager, where absent management experience is a core-activity absence rather than a missing
tool. The ladder working as written.

The cost is 11 of `yc`'s 58 `No Fit`s promoted to `Potential`, against 1 of 58 before.

**kappa(A1, llm) fell to 0.085** [-0.038, 0.206]. Nothing here rescues A1 and **A13 still
fails**; D26's ceilings stay soft.

## Open questions

- **Q32 — the model is confounded with the guide.** Run 1 was `sonnet`, run 4 is `haiku`,
  and no old-guide/`haiku` run exists anywhere on file. Runs 1 and 2 (both `sonnet`, old
  guide, different dispatch paths) agree at 0.287 / 0.276, so the old instrument is stable
  — but that does not isolate the model. Closing it is one 100-pair `haiku` run against the
  old agent definition, and it has not been done.
- **Q33 — the guide edit is retroactive.** All 500 human labels in `judgements.csv` were
  collected under guide `842d7b0e`, and nothing in `judgements.csv` records which guide an
  annotator worked from. The old text is recoverable from git and its sha is above; a
  `guide_sha256` column on the judgement would make that structural rather than
  archaeological. Not added.
- **Standing — closing the last gap would be `yc`-fitting.** The judge is now systematically
  one step below `yc` at the `Good`/`Potential` boundary. Changes 1-4 above are not
  `yc`-fitting: they make the judge apply the guide as it was already written, and they move
  `yc` down as much as the judge up. Encoding `yc`'s actual `Good` bar — profession plus
  level, core-activity evidence not required — would be, and would forfeit D33's third
  opinion. It belongs in a separate prompt, not in this guide.

## Reporting, and what it now refuses to do

`judge_frames` took a per-pair majority across **every run on file**. With two instruments
that blends the old judge's labels into the `llm` series and attributes the blend to
whichever agent is committed — no exception, no row-count change, a plausible kappa. It now
keys on `agent_sha256`: `llm` is the current instrument alone, every instrument is reported
under `per_instrument`, and `self_consistency_kappa` compares only runs that share one.
Guarded by `test_judge_frames_never_pools_two_instruments`, shown to fail against the
pooling it replaces.

Binary-collapse kappa is now computed by `report()` beside every three-class figure rather
than by hand. This run is why: the two moved apart.

## Not done

`uv run pytest` is green (188). `verify --derived` has two failures, both pre-existing in
the working tree this branch started from and neither from this change:

- `resume: 50 already-judged pairs re-dispatched` — the judging-queue rebuild.
- `scope: 559 judged pair(s) are not among the 100 A1 recheck pairs` — the option-D
  `dispatch_all_a1` run, which judges all 659 A1 pairs by design. `check_llm_recheck`'s
  scope check predates it and needs a decision from that workstream: either the check learns
  about option D, or option D's rows belong in a file of their own.
