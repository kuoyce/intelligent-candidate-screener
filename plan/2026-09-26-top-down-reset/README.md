# Top-down reset: one final Fit / No Fit set, then the screening system

**Status:** **approved 26 Sep 2026 (v4); executed 27 Sep 2026.** Findings: [09-findings.md](09-findings.md). Selection (D48): [07-selection.md](07-selection.md). Revised 26 Sep 2026 after three rounds of
review. The final-set rule and the binary scheme are approved. A few success-measure items
remain open (04 §3).

Nothing is implemented. This directory is uncommitted on `feat/top-down-reset` (worktree
`../intelligent-candidate-screener-reset`, branched from `b39fce0`).
**Date:** 26 Sep 2026 (proposal week 7 of 12; about 30 days remain)
**Supersedes:** `plan/2026-09-10-a2-reference-design/` (never approved), and D26's headline
use.
**Constraints (user, 26 Sep 2026):**
- Breaking changes and a change of direction are allowed.
- No new data: no external sources and no new LLM output.
- Claude subagents must not be re-run programmatically (`llm_recheck`,
  `prompt_calibration`).
- Running code to reproduce results is allowed.
- Packages are installed through uv.

## The diagnosis

The proposal asks whether more sophisticated matching earns its complexity. Seven weeks
in, the repository has the data layer and one classical rung. It has **no semantic model,
no cross-encoder, no extraction component, no evidence layer and no UI**.

The annotation campaign collected 1,034 human labels and 1,800 LLM labels on the in-domain
set, but never took the step it existed for: **reconciling them into one label per pair**.
53% of the source code, 73% of the tests and 84% of `AGENTS.md` went to calibrating the
judge instead. See [01-audit.md](01-audit.md).

## The direction

1. **P1: build the final annotation set** (D41, D47). Fit / No Fit, by the user-approved
   rule:

   | Case | Label | Pairs today |
   |---|---|---|
   | Both humans agree | theirs | 219 |
   | Both humans, split | the LLM breaks the tie | 107 |
   | One human | that human | 274 |
   | No human | the LLM | 0 |

   The LLM is the committed instrument `5d986c82` (haiku). The set is frozen as
   `a2-final-labels.csv` and guarded by invariants.
2. **P1–P4: build the ladder and score every rung against that one set**, for generic,
   targeted and total:
   - rules → TF-IDF/BM25 → SBERT (zero-shot and weakly fine-tuned) → cross-encoder (A1-
     vs weak-trained, Fit / No Fit head) → hybrid;
   - the primary metric is per-JD AUC, `Fit` vs `No Fit`;
   - a decision rule is registered before the held-out rungs run;
   - the `llm_free` subset flags any verdict that depends on the LLM's tie-breaks.
3. **P5–P7:** extraction (rules vs MiniLM NER), the evidence layer, and a Streamlit
   conversational UI, on top of the selected rung.
4. A1 becomes training data plus a secondary benchmark. Annotation collection and judge
   calibration close.

See [02-design-spec.md](02-design-spec.md) and [03-work-plan.md](03-work-plan.md)
(21.5 team-days).

**Where the systems stand today** (scratch probe, against the final set, total over 59
JDs):
- keyword + TF-IDF: per-JD AUC 0.85 [0.81, 0.90];
- keyword + SBERT: 0.86 [0.81, 0.90];
- TF-IDF alone or SBERT alone: 0.77;
- random: 0.54.
- Inter-annotator agreement, reported with no target: `yc`–`van` 0.67 (kappa 0.35),
  `yc`–LLM 0.76, `van`–LLM 0.71.

## Decisions requested

| # | In one line |
|---|---|
| D40 | A2 is the primary benchmark. A1 is demoted, and D26's ceilings are retired (superseded in place) |
| D41 | One final set by the approved rule is the only thing systems are scored against. Reported for generic, targeted and total. Frozen, versioned by `rule_version` |
| D42 | Annotation collection and judge calibration are closed. `annotation/` is frozen, and dispatch refuses in code |
| D43 | The LLM is a secondary annotator (one series, `5d986c82`/`haiku`), not a system. No system is trained on LLM labels |
| D44 | Success measures partly decided: three-class dropped; inter-annotator agreement reported with no target; S0–S4 open ([04](04-success-measures.md) §3) |
| D45 | `AGENTS.md` holds live rules only (≤ 1,800 words). History moves to `docs/decisions.md` |
| D46 | Nothing is fitted or tuned on the final set, weights and thresholds included. Its construction reads annotator labels only |
| D47 | Fit / No Fit is the only label scheme, everywhere. Raw three-class labels are kept for audit only |

## Questions

| # | Question | Status |
|---|---|---|
| Q50 | Time remaining | **Answered:** about 30 days. The plan fits |
| Q51 | Package installs | **Answered:** yes, through uv. PyTorch CPU index, Python 3.12, cached MiniLM weights only |
| Q52 | Merge `feat/a2-judge-calibration` first | **Answered:** yes. P0.1, local, nothing pushed unless asked |
| Q53 | Success measures | **Partly answered** (D44, D47). Open: **S0** (does the system-vs-human 85% target survive?), **S1** (absolute bar for Fit / No Fit accuracy and macro-F1), **S2** (Recall@50 ≥ 0.80 / 0.90 at pilot scale), **S3** (extraction F1 bar), **S4** (latency instead of recruiter time). Details and recommendations in [04](04-success-measures.md) §3. Needed before P1.5 |
| **Q54** | Allow one explicitly non-held-out upper-bound experiment: a cross-encoder cross-fitted by JD on the final set? | Open. Default: not run |
| **Q55** | Who owns track A (matching) and track B (extraction, evidence, UI)? | Open |
| Q56 | Strata or pooled | **Answered:** generic, targeted **and** total. The total carries the decision |
| Q57 | Single-human pairs | **Answered:** the human's label stands (rule 3) |

**Assumptions** (stated so they can be overturned):
- **A1:** the scratch probe numbers are indicative only. P1 recomputes everything in the
  package.
- **A2:** "no new LLM output" also excludes an LLM at UI run time. The UI is
  template-based, which proposal §4.6 allows.
- **A3:** the LLM's existing labels may be used as a secondary annotator because no new
  output is produced.

## Contents

| File | What |
|---|---|
| [01-audit.md](01-audit.md) | Promised vs built, where the effort went, the evidence, keep/freeze/retire |
| [02-design-spec.md](02-design-spec.md) | Decisions, the final annotation set, the ladder, the evaluation protocol and decision rule, extraction, evidence, UI, code layout |
| [03-work-plan.md](03-work-plan.md) | Phases P0–P8, acceptance criteria, effort, risks |
| [04-success-measures.md](04-success-measures.md) | Q53: what is decided, the proposal's targets restated as Fit / No Fit, what is still open |
| [06-registration.md](06-registration.md) | The rule, comparisons and provisional bars, committed before any held-out rung |
| [07-selection.md](07-selection.md) | The ladder, the verdicts, and D48 |
| [08-extraction.md](08-extraction.md) | Extraction component and D49 |
| [09-findings.md](09-findings.md) | The answer to the research question, success measures, limitations |
| [10-implementation.md](10-implementation.md) | What was built and every deviation |
| [05-annotator-vs-system.md](05-annotator-vs-system.md) | Evidence shown during review: every system and annotator against each annotator and the final set. Three-class tables kept as the record of what was shown |
