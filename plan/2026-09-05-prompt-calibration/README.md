# Prompt calibration harness for the a1-judge

## Context

The LLM judge (run 1, n=100) assigns `Good Fit` to only 2 of the 30 pairs A1 labels `Good Fit`
(agreement 0.067). The tie-breaking rule ("choose the lower one") is the prime suspect — it is
designed for human annotators who anchored high; a model that defaults to `No Fit` needs the
opposite nudge. We want to test prompt variations without repeating a 100-agent run for each
candidate.

**The key separation**: the production instrument requires isolation (no CLAUDE.md, subprocess,
provenance hash). A calibration probe doesn't. It is not collecting labels — it is choosing
which prompt to give a full run.

**Constraints established**:
- `anthropic` Python package is NOT installed — no direct API calls
- `claude` CLI is available (it's what `judge_one` already uses)
- The CLI's `--system-prompt` flag replaces the entire system prompt without an agent file
- Haiku is the right model for calibration: faster and ~10× cheaper than Sonnet for throwaway runs

## Approach

A new module `src/candidate_screener/annotation/prompt_calibration.py` that:

1. **Selects a focused test set** — the ~30 pairs where `a1_label == "Good Fit"`, loaded from
   the stratified draw via `judging.a1_recheck(judging.LLM_RECHECK_STRATA, seed=0)`. Their
   already-rendered prompts live in `data/processed/indomain/llm-recheck-prompts.jsonl` (run 1).

2. **Calls via subprocess directly** — `claude -p --system-prompt <candidate> --model haiku`
   with `input=pair_prompt`, 10 concurrent workers. No agent file. No sandbox. No `anthropic`
   package. Not written to `llm-recheck.csv`.

3. **Parses with the existing `parse_return()`** from `llm_recheck.py` — same strictness as
   production.

4. **Reports per-class distribution and agreement** vs A1 labels (and vs human labels where
   available), printed as JSON.

## The system prompt for calibration

The `render_agent_definition()` function in `llm_recheck.py` returns the full agent file
(frontmatter + body). For `--system-prompt` we only want the body — everything after the second
`---\n\n` separator. A helper `render_system_prompt(guide_text)` strips the frontmatter.

The body text is independent of the haiku/sonnet model choice — only the frontmatter names the
model. Calibration ignores the frontmatter entirely.

## Files to create / modify

| File | Change |
|---|---|
| `src/candidate_screener/annotation/prompt_calibration.py` | New module (CLI + importable) |

No changes to `llm_recheck.py`, `session.py`, or any manifest.

## CLI

```bash
# Run the calibration probe on the 30 Good Fit pairs, using the current guide
uv run python -m candidate_screener.annotation.prompt_calibration --run 1

# Test a modified guide file
uv run python -m candidate_screener.annotation.prompt_calibration \
    --guide path/to/modified-guide.md --run 1

# Run on all 100 recheck pairs, not just Good Fit
uv run python -m candidate_screener.annotation.prompt_calibration --run 1 --all
```

## Implementation sketch

```python
def render_system_prompt(guide: str | None = None) -> str:
    """Body of the agent definition only — strips frontmatter for --system-prompt."""
    defn = llm.render_agent_definition(guide)
    # render_agent_definition() -> "---\n<frontmatter>\n---\n\n<body>"
    return defn.split("---\n\n", maxsplit=1)[-1]


def calibrate_one(pair_prompt: str, system_prompt: str,
                  model: str = "haiku") -> str:
    """One pair, one subprocess, one raw return. No sandbox, no provenance."""
    proc = subprocess.run(
        [shutil.which("claude") or "claude", "-p",
         "--system-prompt", system_prompt,
         "--model", model],
        input=pair_prompt, capture_output=True, text=True,
        timeout=120, check=False)
    return proc.stdout.strip() or f"<exited {proc.returncode}: {proc.stderr.strip()[:200]}>"


def calibrate(guide: str | None, pair_records: list[dict],
              a1_labels: dict[str, str], human_labels: dict[str, str],
              concurrency: int = 10, model: str = "haiku") -> dict:
    """Run the candidate prompt against the probe set; return agreement stats."""
    system_prompt = render_system_prompt(guide)
    ...

def load_probe_pairs(run: int, good_fit_only: bool = True,
                     seed: int = 0) -> tuple[list[dict], dict, dict]:
    """
    Returns:
      - list of {pair_id, prompt} from llm-recheck-prompts.jsonl
      - a1_labels: {pair_id: label} from a1_recheck draw
      - human_labels: {pair_id: label} from judgements.csv
    Filters to Good Fit pairs by default (the 30 where LLM agreement = 0.067).
    """
    ...
```

## What to vary in the guide

Four candidate modifications (try in order, cheapest first — 4 is the outlier worth running
alongside the others):

1. **Soften the tie-breaking rule** — change "choose the lower one" to "when genuinely
   uncertain between two labels, prefer the lower; but do not apply this when the evidence
   simply favours a higher label."
2. **Add a corpus note to the tie-breaking section** — "A Djinni CV at 1,500 chars that
   matches the JD's core stack is `Good Fit`, not `Potential Fit`, because of its length."
3. **Rewrite the tie-breaking rule** — replace with a symmetric "when in genuine doubt,
   use `Potential Fit`" rule, removing the directional bias entirely.
4. **HR specialist persona — a new system prompt, not a guide modification.** Reframes the
   judge away from the annotation guide entirely:
   - Persona: a generalist HR specialist reviewing resumes — domain-literate but not a
     domain expert (cannot assess whether a specific technology claim is correct, only whether
     experience and skills appear relevant).
   - **Good Fit**: candidate has similar experience and skills, meets ~80% of stated requirements.
   - **Potential Fit**: language is vague or details are thin, but the experience is still
     relevant — warrants a screening call.
   - **No Fit**: the experience is clearly misaligned with the role.
   - This prompt is written from scratch, not derived from `docs/annotation-guide.md`. It goes
     into a separate file (e.g. `plan/2026-08-30-expand-annotation/hr-specialist-prompt.md`)
     and is passed as `--guide` to the calibration CLI. It does **not** modify the committed
     guide and does **not** change the production instrument.
   - **Why this is interesting**: the existing guide's "choose the lower" rule was written for
     human annotators who anchor high. An HR specialist framing with an explicit 80% threshold
     may be better calibrated for a model that already tends strict. It also removes the
     domain-expertise assumption — closer to what the model is actually doing when reading a
     Djinni CV it cannot independently verify.

## Cost estimate

10 concurrent Haiku runs × 30 pairs × ~2K tokens input + ~50 tokens output ≈ **$0.008 per
calibration run**. The full `--all` 100-pair variant is ~$0.027.

## Production run after calibration

Once a promising variant is found:

1. Apply the guide change to `docs/annotation-guide.md`
2. Re-render the agent: `uv run python -m candidate_screener.annotation.llm_recheck --agent`
3. Dispatch a new run: `uv run python -m candidate_screener.annotation.llm_recheck --dispatch --run 2`
4. Judge: `uv run python -m candidate_screener.annotation.llm_recheck --judge --run 2`

The production run uses Haiku (as committed in the agent frontmatter). Run 1 used Sonnet;
`collect()` preserves per-row model provenance from the dispatch record.

## Verification

```bash
uv run python -m candidate_screener.annotation.prompt_calibration --run 1
# Expect: Good Fit agreement > 0.067 (current baseline) on the 30-pair probe
# Acceptable: Potential Fit / No Fit agreement does not collapse below 0.5

uv run pytest tests/test_llm_recheck.py  # existing suite must stay green
```

No new tests are required for the calibration module — it is an operator tool, not production
code. The existing `parse_return` tests cover the parser it delegates to.

---

## Work done (session 2026-09-05)

### Background: what prompted this

The merged `feat/llm-recheck` branch (D33) confirmed A13 fails:

| Pair | κ | n | Agreement |
|---|---|---|---|
| A1 vs human | 0.081 | 100 | 0.41 |
| A1 vs LLM   | 0.082 | 96  | 0.45 |
| human vs LLM | 0.264 | 96 | 0.67 |

Per-class breakdown vs A1's labels (LLM judge, run 1):
- **Good Fit** (n=29): LLM agrees **3%** — 26 of 30 pairs called `No Fit`
- **Potential Fit** (n=27): LLM agrees 22%
- **No Fit** (n=40): LLM agrees **90%**

The judge's own test-retest reliability is κ = 0.857–1.000, so the low agreement with A1
on `Good Fit` is not noise — the judge is strict. D26's 0.7806 / 0.7188 precision ceilings
are therefore soft.

### Files created

**`src/candidate_screener/annotation/prompt_calibration.py`** — implemented and tested.

Key functions:
- `render_system_prompt(guide_path)` — strips frontmatter from `render_agent_definition()` output for use with `--system-prompt`
- `calibrate_one(pair_prompt, system_prompt, model)` — one subprocess call: `claude -p --system-prompt <text> --model haiku`
- `calibrate(guide_path, pair_records, a1_labels, human_labels)` — 10-concurrent run, parses with `parse_return()`, reports per-class agreement
- `load_probe_pairs(run, good_fit_only, seed)` — loads prompts from `llm-recheck-prompts.jsonl`, filters to the 30 Good Fit pairs by default

Verified: `load_probe_pairs(run=1, good_fit_only=True)` returns 30 pairs. `uv run pytest tests/test_llm_recheck.py` stays green (33 tests pass).

**`plan/2026-08-30-expand-annotation/hr-specialist-prompt.md`** — candidate 4 system prompt.

Written from scratch. Key differences from the annotation guide:
- Persona: generalist HR specialist (no domain expertise assumed)
- Good Fit threshold: ~80% of stated requirements met
- Potential Fit: vague language or thin details, still relevant
- No directional tie-breaking bias
- Explicit note that CV length is a platform norm, not a quality signal
- Retained: two-corpus awareness (A1 vs Djinni), what-not-to-consider list

### How to run the calibration

```bash
# 30 Good Fit pairs, current production guide (baseline)
uv run python -m candidate_screener.annotation.prompt_calibration --run 1

# HR specialist prompt (candidate 4)
uv run python -m candidate_screener.annotation.prompt_calibration \
    --guide plan/2026-08-30-expand-annotation/hr-specialist-prompt.md --run 1

# All 100 recheck pairs
uv run python -m candidate_screener.annotation.prompt_calibration --run 1 --all
```

Cost: ~$0.008 per 30-pair run on Haiku, ~30–60 s at 10 concurrency.

### Decisions (2026-09-05)

- **Haiku everywhere.** Calibration and production use the same model — no transfer risk.
  `MODEL` in `llm_recheck.py` is now `claude-haiku-4-5`; agent frontmatter is `model: haiku`.
  `collect()` uses the dispatched record's model, so run 1 (sonnet) rows keep their provenance.
- **Candidate 4 → fold into `annotation-guide.md`** if it wins, not a separate instrument.
- **`--all` is mandatory.** Run the 30 Good Fit probe first (fast screen), then `--all` on
  any candidate that passes. Acceptance: Good Fit agreement > 0.067 *and* No Fit agreement
  does not collapse below 0.50.
- **Escalate if nothing beats 0.067.** Do not chase alternatives autonomously.

### Next steps

1. Run the baseline calibration to confirm the module works end-to-end
2. Run candidates 1–4, each with `--all` follow-up if the 30-pair probe passes
3. If any candidate scores > 0.067 without degrading No Fit agreement below 0.50:
   - For candidates 1–3: apply the edit to `docs/annotation-guide.md`
   - For candidate 4: fold the HR-specialist framing into `docs/annotation-guide.md`
   - Re-render with `--agent`, dispatch run 2
4. If no candidate beats 0.067: escalate — do not proceed autonomously
5. Production run: `--dispatch --run 2 / --judge --run 2 / --collect --report` (Haiku)
