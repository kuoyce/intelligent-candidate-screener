"""A third judge on the 50 A1 recheck pairs *(decision D33)*.

kappa(A1, yc) = 0.010, CI [-0.198, 0.223], and the disagreement is **bidirectional** —
7 of A1's `Good Fit` became our `No Fit` and 7 of its `No Fit` became our `Good Fit`. That
is not a strictness shift, so it does not preserve ranking. Two explanations survive and
they have opposite consequences: either A1's labels are wrong, and **A13 fails** so D26's
0.7806 / 0.7188 ceilings are soft, or `yc` is miscalibrated and the in-domain set needs
recalibrating before anything is quoted off it. One human annotator cannot separate them.

**This is a third opinion, not an answer key.** An LLM is a judge with its own bias, known
to exist and not measured here. Its labels never enter `judgements.csv`: `annotator` is a
free string, so an `llm` row there would pass every existing check and then be silently
pooled into the human kappa that `verify --derived` computes and the report quotes.

**Why the judge runs tool-free.** The answer key is on disk:
`data/processed/indomain/judging-queue-full.parquet` carries `a1_label` for all 50 recheck
pairs, and `data/processed/fit/test.parquet` is its source. A subagent holding `Read` or
`Bash` reaches either in one command. The repo's standard for human blinding is structural
rather than instructed — `serve_group` redacts its own output and `assert_clean`s it, and
the rules live in `session.py` rather than in the request handler precisely so a caller
cannot forget — and "we told the judge not to look it up" does not meet that standard.

**This file is collected data, not derived.** No seed reproduces a subagent run, so
`llm-recheck.csv` is of the same kind as `judgements.csv`: append-only, provenance-stamped,
never re-derived. `verify --derived` must check *provenance* (`prompt_sha256`,
`agent_sha256`, the pair set) and never determinism, which would be red on every run.

**One draw per file, one draw per run** *(decision D35)*. This judge is pointed at two
populations: the 100-pair stratified recheck it is validated on, and option D's every-pair
sweep of `fit/test.parquet`. `draw` is stamped at dispatch and routes the collected row —
recheck to `llm-recheck.csv`, option D to `llm-recheck-full-a1.csv` — and `judge_frames`
reads the first alone. `dispatch` refuses a run number the other draw already holds, which
is the check that was missing when option D was sent as run 2 and silently absorbed the 20
pairs D33 had dispatched under that number.

Usage:
    uv run python -m candidate_screener.annotation.llm_recheck --agent      # (re)write the judge
    uv run python -m candidate_screener.annotation.llm_recheck --dispatch --run 1
    uv run python -m candidate_screener.annotation.llm_recheck --dispatch --run 5 --subsample 20
    uv run python -m candidate_screener.annotation.llm_recheck --dispatch --all-a1 --run n
    uv run python -m candidate_screener.annotation.llm_recheck --backfill-draw   # pre-D35 prompts
    uv run python -m candidate_screener.annotation.llm_recheck --judge --run 1 --limit 100
    uv run python -m candidate_screener.annotation.llm_recheck --coverage --run 1
    uv run python -m candidate_screener.annotation.llm_recheck --collect
    uv run python -m candidate_screener.annotation.llm_recheck --report

**A run that lost spawns is not a smaller sample.** `--judge` quarantines any pair whose
process never answered and exits non-zero rather than reporting itself complete *(fixes
F1/F2, `plan/2026-09-06-a2-llm-prelabel`)*. Recovery is to fix the cause — usually an
empty balance — and re-run the same `--judge` command, which retries exactly those pairs.
`--coverage` shows dispatched against answered; do not `--collect` a run that is short.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.annotation import queue as judging
from candidate_screener.annotation import redact, session
from candidate_screener.config import DOCS, OUTPUT, PROCESSED, PROJECT_ROOT
from candidate_screener.data.fit_split import MANIFESTS

GUIDE = DOCS / "annotation-guide.md"
AGENT_DEF = PROJECT_ROOT / ".claude" / "agents" / "a1-judge.md"
PROMPTS = PROCESSED / "indomain" / "llm-recheck-prompts.jsonl"
RAW = PROCESSED / "indomain" / "llm-recheck-raw.jsonl"

#: Spawns that never produced an answer — credit exhaustion, a rate limit, a timeout, a
#: missing CLI *(fix F1, plan/2026-09-06-a2-llm-prelabel)*. They are **not** returns, and
#: the distinction is the whole point of the file: `judge()` resumes from `RAW`, so a
#: failure written there is indistinguishable from a finished pair and is never retried.
#: A run that lost 150 pairs to an empty balance would report `remaining: 0`, and the
#: kappa would be computed over whatever `as_completed` returned before the wall — which
#: is completion-latency order, not a random sample. Quarantining here keeps the pair in
#: `todo`, so recovery is: top up, re-run `--judge`, done.
FAILED = PROCESSED / "indomain" / "llm-judge-failed.jsonl"
RECHECK_CSV = MANIFESTS / "llm-recheck.csv"
FULL_A1_CSV = MANIFESTS / "llm-recheck-full-a1.csv"
REPORT = OUTPUT / "annotation" / "llm-recheck-report.json"

#: The A2 figures get their own file for the reason they get their own frame builder: the
#: recheck report backs published numbers, and appending a section to it would let this
#: plan move them.
A2_REPORT = OUTPUT / "annotation" / "llm-a2-report.json"

#: The two populations this judge is ever pointed at *(decision D35)*. `a1_recheck` is the
#: 100-pair stratified draw the instrument is validated on; `a1_full` is option D, every
#: pair in `fit/test.parquet`. They are different measurements over different populations
#: and they live in different files.
RECHECK_DRAW = "a1_recheck"
FULL_A1_DRAW = "a1_full"

#: The in-domain draws *(plan/2026-09-06-a2-llm-prelabel)*. Two, not one, because D29's
#: strata are two populations whose average names nothing: `generic` is the unstratified
#: D14 sample, `targeted` pools the keyword-scoped batches. Composing D29 with D35 —
#: one file, one draw — makes the pooled A2 figure impossible rather than merely
#: forbidden. `a2_generic` is retrospective: those 400 pairs already carry `yc` labels the
#: judge has never seen, so kappa(yc, llm) on the in-domain corpus is computable without
#: any pending annotation. `a2_targeted` is dispatched *before* its human session, which
#: is what makes it a prospective, held-out measurement.
A2_GENERIC_DRAW = "a2_generic"
A2_TARGETED_DRAW = "a2_targeted"
A2_GENERIC_CSV = MANIFESTS / "llm-a2-generic.csv"
A2_TARGETED_CSV = MANIFESTS / "llm-a2-targeted.csv"

#: The stratum each A2 draw samples. `indomain_rows()` stamps `stratum` on the pair, so
#: this maps a draw to the rows it may contain and nothing else selects them.
A2_DRAW_STRATUM = {A2_GENERIC_DRAW: "generic", A2_TARGETED_DRAW: "targeted"}

DRAW_FILES = {RECHECK_DRAW: RECHECK_CSV, FULL_A1_DRAW: FULL_A1_CSV,
              A2_GENERIC_DRAW: A2_GENERIC_CSV, A2_TARGETED_DRAW: A2_TARGETED_CSV}

#: The judge's default model. Named in every output row, because a different model is a
#: different judge — and, since M2 (8 Sep 2026), **passed to the subprocess** rather than
#: merely recorded beside it. Before that the argv carried no `--model` at all, so the
#: column asserted what the constant said rather than what answered; runs 1-6 are not
#: re-stamped and their model provenance rests on the constant having been right at the
#: time (**Q43**). `--model` overrides it per dispatch, which is what makes the model an
#: axis a figure may be quoted along (D38) instead of a property of the whole file.
MODEL = "claude-haiku-4-5"

#: Sections of the guide that describe the *human workflow* and are dropped from the
#: judge's system prompt. The shortlist question goes with them: `asks_shortlist` is
#: already `False` for `corpus == "a1"`, so it is a question this judge is never asked,
#: and leaving it in would invite a single-pair judge to compare against a field it
#: cannot see. Everything that defines the instrument — the three class
#: definitions, the ordered decision, the tie-breaking rule, the what-not-to-consider
#: list, `Two corpora, one scheme` (A16) — is kept **verbatim**. Paraphrasing would
#: measure the paraphrase; the comparison only means something if both judges were handed
#: the same instrument.
OPERATOR_SECTIONS = ("How the pairs reach you",
                     "One extra question per job description",
                     "How the work is split",
                     "Double labelling and adjudication")

#: How many judges run at once. The plan's number; raise it and the only thing that
#: changes is wall-clock and the chance of a rate limit.
CONCURRENCY = 10

#: One judge, one pair, one process. Generous — an 8.7k-char pair is a single turn, but a
#: cold start plus a retry inside the CLI is not instant.
JUDGE_TIMEOUT_S = 600

#: The judge is spot-checked, not asked to explain itself. Enforced on the way in (the
#: instruction) and on the way out (`parse_return` flags anything longer).
REASON_CHARS = 120

#: `queue.JUDGEMENT_COLUMNS` plus provenance. `run` and the two hashes are what make a row
#: attributable to an instrument, since no seed will reproduce it. `draw` names the
#: population that was sampled *(D35)* — run 2 held two of them under one run number and one
#: `agent_sha256`, so it is not derivable from the columns that preceded it. `score` is the
#: graded instrument's second answer (move 3); it is empty for every run dispatched under
#: `SCHEME_LABEL`, which is runs 1-8, and a run dispatched under `SCHEME_SCORED` has no
#: parsed row without one.
LLM_COLUMNS = judging.JUDGEMENT_COLUMNS + (
    "model", "agent_sha256", "prompt_sha256", "run", "chars_sent", "parsed_ok",
    "judged_at", "draw", "score")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the judge -------------------------------------------------------------

def strip_operator_sections(guide: str,
                            sections: tuple[str, ...] = OPERATOR_SECTIONS) -> str:
    """Drop each named `## ` section and every `### ` beneath it, keeping the rest as-is."""
    out, skipping = [], False
    for line in guide.splitlines():
        if line.startswith("## "):
            skipping = any(line[3:].strip().startswith(s) for s in sections)
        elif line.startswith("# ") and not line.startswith("## "):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


def render_agent_definition(guide: str | None = None) -> str:
    """The committed `.claude/agents/a1-judge.md`, rendered from the guide.

    `tools: []` is the blinding control, not a tidiness preference — see the module
    docstring. It also happens to be the largest single saving in a subagent's context,
    since the tool schemas go with it.
    """
    body = strip_operator_sections(guide if guide is not None
                                   else GUIDE.read_text(encoding="utf-8"))
    return f"""---
name: a1-judge
description: Judges one job-description/CV pair against the annotation guide's three classes and a 0-100 score. Returns one JSON object and nothing else. Tool-free by design.
tools: []
---

You are one annotator in a labelling session. You will be shown **one** job description
and **one** candidate CV, and you return **one** label and **one** score.

Two rules override anything else:

1. **Reply with exactly one JSON object and no other text.** No prose before it, no
   markdown fence around it, no explanation after it.
   `{{"label":"Good Fit"|"Potential Fit"|"No Fit","score":<0-100 integer>,"reason":"<={REASON_CHARS} chars"}}`
2. **Judge only what is in front of you.** You have no tools and nothing to look up. If
   the CV is thin, that is evidence about the CV, not a reason to seek more.

Everything below is the annotation guide, unedited. It is the instrument; apply it as
written, including the tie-breaking rule and the what-not-to-consider list.

---

{body}"""


def agent_definition_is_current() -> bool:
    return (AGENT_DEF.exists()
            and AGENT_DEF.read_text(encoding="utf-8") == render_agent_definition())


def assert_agent_definition_current() -> str:
    """The committed judge must be the one this guide renders.

    A guide edited after dispatch would leave rows on disk attributed to an instrument
    that no longer exists, and nothing else would notice.
    """
    if not agent_definition_is_current():
        raise AssertionError(
            f"{AGENT_DEF} is missing or stale against {GUIDE}. Re-render it with "
            "`--agent` and re-dispatch: a row's agent_sha256 has to name the instrument "
            "that actually judged it.")
    return sha(AGENT_DEF.read_text(encoding="utf-8"))


# --- the per-pair prompt ---------------------------------------------------

#: What a dispatch asks the judge to return, stamped on every prompt record.
#:
#: Move 3 of `plan/2026-09-08-judge-calibration` added the score, and `collect` requires
#: one of a row whose dispatch asked for one and of no other row. Without the stamp,
#: re-collecting runs 1-8 would mark every one of their rows `parsed_ok=False` for
#: withholding a field they were never asked for — and the frame builders drop unparsed
#: rows, so the committed A1 and A2 figures would silently recompute over nothing. Same
#: shape as `draw` *(D35)*: a property of the dispatch that is not derivable from the
#: columns that preceded it, so it is written at dispatch and read back, never guessed.
#: A record without one is pre-move-3 and asked for a label alone.
SCHEME_LABEL = "label"
SCHEME_SCORED = "label+score"
RESPONSE_SCHEME = SCHEME_SCORED

#: Inclusive, and whole numbers only — the guide says so and `parse_score` enforces it.
SCORE_MIN, SCORE_MAX = 0, 100


def render_prompt(jd_text: str, cv_text: str) -> str:
    """The whole user message. No preamble — the guide is in the agent definition.

    The scheme it names is the scheme the agent definition names, and the two move in one
    edit. A prompt asking for `label` alone while the guide asked for a score would leave
    the judge to choose which instruction to follow, and `parse_return` would refuse
    whichever it did not pick — 600 spawns spent on a contradiction this file wrote.
    """
    return (f"JOB DESCRIPTION\n{jd_text}\n\nCANDIDATE\n{cv_text}\n\n"
            "Reply with one JSON object, no prose, no markdown fence:\n"
            f'{{"label":"Good Fit"|"Potential Fit"|"No Fit",'
            f'"score":<{SCORE_MIN}-{SCORE_MAX} integer>,'
            f'"reason":"<={REASON_CHARS} chars"}}')


#: Fields that would tell the judge what the answer is, or that a label already exists.
#: Asserted against the **rendered prompt**, not against the frame it came from: the
#: frame is not what gets sent.
ANCHORING_FIELDS = ("a1_label", "selection_reason", "lexical_band", "a1_recheck",
                    "second_opinion")


def assert_not_anchoring(prompt: str, what: str) -> None:
    found = [f for f in ANCHORING_FIELDS if f in prompt]
    if found:
        raise AssertionError(
            f"{what}: {found} reached the judge's prompt. A judge shown which pairs "
            "already carry a label anchors on the expected answer.")


def guide_example_pairs(guide: str | None = None) -> set[str]:
    """Ids named in the guide's worked examples, if any ever are.

    The examples are sketches today, not real pairs. This is the check that notices the
    day someone replaces one with a real document — a judge shown the answer is not a
    judge.
    """
    text = guide if guide is not None else GUIDE.read_text(encoding="utf-8")
    return set(re.findall(r"\b[jr]_[0-9a-f]{4,}\b", text))


# --- dispatch --------------------------------------------------------------

EMPTY_JUDGEMENTS = pd.DataFrame(columns=list(judging.JUDGEMENT_COLUMNS))


def a1_groups(units: pd.DataFrame, jd_text: pd.Series, cv_text: pd.Series,
              judgements: pd.DataFrame = EMPTY_JUDGEMENTS,
              seed: int = 0) -> list[session.Group]:
    """Every A1 unit, served through the path a human screen goes through.

    `serve_group` takes `judgements` as an argument, so passing an **empty** frame returns
    every one rather than the 20 `outstanding()` would allow a fresh annotator — and the text
    still goes through the same `redact.redact` + `assert_clean`. No change to
    `session.py` is needed for any of this. `Group.asks_shortlist` is already `False` for
    `corpus == "a1"`, so dissolving the groups into single pairs loses no question.
    """
    units = units[units.corpus == "a1"]
    served: set[str] = set()
    groups: list[session.Group] = []
    while True:
        remaining = units[~units.pair_id.isin(served)]
        if remaining.empty:
            break
        group = session.serve_group(remaining, judgements, "llm", jd_text, cv_text,
                                    seed=seed)
        if group is None:
            break
        groups.append(group)
        served |= {f"{group.jd_id}__{c.cv_id}" for c in group.candidates}
    return groups


def subsample_ids(pair_ids: list[str], k: int, seed: int) -> list[str]:
    """A fixed subsample for runs 2 and 3, drawn from the seed rather than from order."""
    ordered = sorted(pair_ids)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    return sorted(np.array(ordered)[rng.permutation(len(ordered))[:k]].tolist())


class MixedDraw(ValueError):
    """One run number, two populations *(D35)*."""


class MixedModel(ValueError):
    """One run, two models — or a judge asked to run a model the dispatch did not name.

    Model is an axis a figure may not be pooled across (**D38**), and the run is where
    that starts: a run whose prompts name `haiku` and whose spawns answered as `sonnet`
    produces rows no later check can separate, because `collect()` copies the model out
    of the dispatch record. The refusal is here, at the only two points where the two
    could diverge.
    """


def assert_run_holds_one_draw(run: int, draw: str) -> None:
    """A run number is a sitting of **one** draw, and dispatch refuses to widen it.

    This is the check that was missing when option D was dispatched onto run 2. Dedup keys
    on `(pair_id, run)`, so the 20 pairs D33 had already sent under that number were skipped
    as "already dispatched" and their labels became part of what reads as one 659-pair run.
    Nothing raised, no row count looked wrong, and `run1_vs_run2` in the committed report
    silently widened from n=20 to n=100 — a self-consistency figure over 80 pairs that were
    never judged twice by the same draw.

    A record with no `draw` is a pre-D35 dispatch and is refused rather than assumed: a
    default would refile option D into the recheck file on the first `--collect` run on a
    machine that had not been backfilled.
    """
    on_file = {r.get("draw") for r in read_jsonl(PROMPTS) if r["run"] == run}
    if None in on_file:
        raise MixedDraw(
            f"run {run} carries prompts dispatched before `draw` existed. Run "
            "`--backfill-draw` first — which draw they belong to is not guessable from "
            "the run number (run 2 held both).")
    other = on_file - {draw}
    if other:
        raise MixedDraw(
            f"run {run} already carries {sorted(other)} prompts and this dispatch is "
            f"{draw!r}. A run number holds one draw (D35) — choose an unused run number.")


def dispatch(run: int, seed: int = 0, subsample: int | None = None,
             model: str = MODEL) -> dict:
    from candidate_screener.annotation import ui

    agent_hash = assert_agent_definition_current()
    assert_run_holds_one_draw(run, RECHECK_DRAW)
    assert_recheck_nests(seed)
    # One draw, two consumers: the text lookup and the unit frame must come from the same
    # strata or `serve_group` meets a pair whose documents it cannot find.
    strata = judging.LLM_RECHECK_STRATA
    jd_text, cv_text = ui.corpus_text(tuple(sorted(strata.items())))
    units = session.load_units(seed, strata)
    groups = a1_groups(units, jd_text, cv_text, seed=seed)

    records = []
    for group in groups:
        for candidate in group.candidates:
            pair_id = f"{group.jd_id}__{candidate.cv_id}"
            prompt = render_prompt(group.jd_text, candidate.text)
            assert_not_anchoring(prompt, f"pair {pair_id}")
            records.append({"pair_id": pair_id, "query_id": group.jd_id,
                            "doc_id": candidate.cv_id, "run": run,
                            "draw": RECHECK_DRAW,
                            "prompt": prompt, "prompt_sha256": sha(prompt),
                            "chars_sent": len(prompt),
                            "agent_sha256": agent_hash, "model": model,
                            "response_scheme": RESPONSE_SCHEME})

    if subsample:
        keep = set(subsample_ids([r["pair_id"] for r in records], subsample, seed))
        records = [r for r in records if r["pair_id"] in keep]

    # The examples are the judge's calibration; a pair that is also an example has been
    # shown its own answer.
    overlap = guide_example_pairs() & {r["pair_id"] for r in records}
    if overlap:
        raise AssertionError(f"{len(overlap)} recheck pairs appear as worked examples in "
                             f"{GUIDE}: {sorted(overlap)[:3]}")

    redact.assert_clean(pd.Series([r["prompt"] for r in records]), "llm dispatch")

    # Dispatch appends, and appending the same (pair, run) twice would let one spawn's
    # answer be matched against the other's prompt hash. Re-running `--dispatch` is
    # therefore a no-op on what is already there, the way `queue --build` is.
    already = {(r["pair_id"], r["run"]) for r in read_jsonl(PROMPTS)}  # noqa: F821
    skipped = [r for r in records if (r["pair_id"], r["run"]) in already]
    records = [r for r in records if (r["pair_id"], r["run"]) not in already]

    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with PROMPTS.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"run": run, "seed": seed, "draw": RECHECK_DRAW, "pairs": len(records),
            "already_dispatched": len(skipped),
            "jds": len({r["query_id"] for r in records}),
            "subsample": subsample, "agent_sha256": agent_hash, "model": model,
            "response_scheme": RESPONSE_SCHEME,
            "chars_sent_median": int(np.median([r["chars_sent"] for r in records]))
            if records else 0,
            "prompts": str(PROMPTS)}


def dispatch_all_a1(run: int, seed: int = 0, model: str = MODEL) -> dict:
    """Option D: dispatch every A1 pair (all 659) through the same judge.

    Uses the same prompt rendering, anchoring check, and redaction as the recheck path
    but loads directly from test.parquet rather than the stratified draw. Idempotent per
    (pair_id, run).

    Its output is a **different population** from the recheck's and goes to a different
    file *(D35)*. `assert_run_holds_one_draw` is what stops this dispatch from landing on a
    run number the recheck is already using, which is how the two came to be mixed.
    """
    agent_hash = assert_agent_definition_current()
    assert_run_holds_one_draw(run, FULL_A1_DRAW)
    test = pd.read_parquet(PROCESSED / "fit" / "test.parquet")

    records = []
    for _, row in test.iterrows():
        jd_id, cv_id = str(row.jd_id), str(row.resume_id)
        pair_id = f"{jd_id}__{cv_id}"
        jd_text = redact.redact_text(row.job_description_text)
        cv_text = redact.redact_text(row.resume_text)
        prompt = render_prompt(jd_text, cv_text)
        assert_not_anchoring(prompt, f"pair {pair_id}")
        records.append({"pair_id": pair_id, "query_id": jd_id, "doc_id": cv_id,
                        "run": run, "draw": FULL_A1_DRAW,
                        "prompt": prompt, "prompt_sha256": sha(prompt),
                        "chars_sent": len(prompt), "agent_sha256": agent_hash,
                        "model": model, "response_scheme": RESPONSE_SCHEME})

    overlap = guide_example_pairs() & {r["pair_id"] for r in records}
    if overlap:
        raise AssertionError(f"{len(overlap)} A1 pairs appear as worked examples in "
                             f"{GUIDE}: {sorted(overlap)[:3]}")

    redact.assert_clean(pd.Series([r["prompt"] for r in records]), "option D dispatch")

    already = {(r["pair_id"], r["run"]) for r in read_jsonl(PROMPTS)}
    skipped = [r for r in records if (r["pair_id"], r["run"]) in already]
    records = [r for r in records if (r["pair_id"], r["run"]) not in already]

    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with PROMPTS.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"run": run, "seed": seed, "draw": FULL_A1_DRAW, "pairs": len(records),
            "already_dispatched": len(skipped),
            "total_a1": len(test),
            "jds": len({r["query_id"] for r in records}),
            "agent_sha256": agent_hash, "model": model,
            "response_scheme": RESPONSE_SCHEME,
            "chars_sent_median": int(np.median([r["chars_sent"] for r in records]))
            if records else 0,
            "prompts": str(PROMPTS)}


def dispatch_a2(stratum: str, run: int, seed: int = 0, model: str = MODEL) -> dict:
    """Dispatch one in-domain stratum through the committed judge.

    The judge's agreement with `yc` has only ever been measured on A1 pairs, which are 3x
    longer than these (median 5,134 chars against 1,549 JD / 1,742 CV) and are the corpus
    D34's guide was tuned against. Whether it carries to A2 is **Q36**, and this is the
    dispatch that answers it.

    `stratum` selects the draw, and the draw selects the file: no figure over `generic`
    may be pooled with one over `targeted` *(D29)*, and after this call nothing has to
    remember that — the rows are in different files.

    **The prompt cannot carry a label.** `indomain_rows()` returns `a1_label = None`
    throughout for A2 and joins no human judgement, so `yc`'s existing labels on the
    `generic` pairs are unreachable from here by construction; `assert_not_anchoring` is
    the second line, not the first.
    """
    draw = next((d for d, s in A2_DRAW_STRATUM.items() if s == stratum), None)
    if draw is None:
        raise ValueError(
            f"stratum {stratum!r} is not one of {sorted(A2_DRAW_STRATUM.values())}. "
            "D29 defines exactly two, and a third would be a new population with no "
            "file to go to.")

    agent_hash = assert_agent_definition_current()
    assert_run_holds_one_draw(run, draw)

    rows = judging.indomain_rows()
    rows = rows[rows.stratum.astype(str) == stratum]
    if rows.empty:
        raise AssertionError(
            f"no in-domain pairs in stratum {stratum!r} — `sample --build` first")

    records = []
    for row in rows.itertuples():
        jd_id, cv_id = str(row.query_id), str(row.doc_id)
        pair_id = f"{jd_id}__{cv_id}"
        prompt = render_prompt(redact.redact_text(row.query_text),
                               redact.redact_text(row.candidate_text))
        assert_not_anchoring(prompt, f"pair {pair_id}")
        records.append({"pair_id": pair_id, "query_id": jd_id, "doc_id": cv_id,
                        "run": run, "draw": draw,
                        # Carried so `collect` files the row as what it is rather than
                        # defaulting it to the recheck's A1 identity.
                        "corpus": "a2", "batch": int(row.batch), "stratum": stratum,
                        "selection_reason": str(row.selection_reason),
                        "prompt": prompt, "prompt_sha256": sha(prompt),
                        "chars_sent": len(prompt), "agent_sha256": agent_hash,
                        "model": model, "response_scheme": RESPONSE_SCHEME})

    overlap = guide_example_pairs() & {r["pair_id"] for r in records}
    if overlap:
        raise AssertionError(f"{len(overlap)} A2 pairs appear as worked examples in "
                             f"{GUIDE}: {sorted(overlap)[:3]}")

    redact.assert_clean(pd.Series([r["prompt"] for r in records]), f"a2 {stratum} dispatch")

    already = {(r["pair_id"], r["run"]) for r in read_jsonl(PROMPTS)}
    skipped = [r for r in records if (r["pair_id"], r["run"]) in already]
    records = [r for r in records if (r["pair_id"], r["run"]) not in already]

    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with PROMPTS.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"run": run, "seed": seed, "draw": draw, "stratum": stratum,
            "pairs": len(records), "already_dispatched": len(skipped),
            "total_in_stratum": len(rows),
            "jds": len({r["query_id"] for r in records}),
            "agent_sha256": agent_hash, "model": model,
            "response_scheme": RESPONSE_SCHEME,
            "chars_sent_median": int(np.median([r["chars_sent"] for r in records]))
            if records else 0,
            "prompts": str(PROMPTS)}


#: D33 dispatched its second and third self-consistency sittings as `--subsample 20`, and
#: option D later landed on run 2's number. `subsample_ids` is seeded, so which 20 they
#: were is still computable — which is what makes the backfill below a derivation rather
#: than a guess about file order.
#:
#: It must be computed over the draw that was **live on 30 Aug 2026**, which was 50 pairs,
#: not the 100 of today: V6 raised the machine leg later the same day, and `subsample_ids`
#: permutes the list it is handed. Over the 100 the same call reproduces 7 of the 20. The
#: 50 nest inside the 100 (`assert_recheck_nests`), so this names a subset of the current
#: draw, not a draw that has gone away.
LEGACY_SUBSAMPLE = 20
LEGACY_SUBSAMPLE_STRATA = {"Good Fit": 15, "Potential Fit": 15, "No Fit": 20}

#: The run numbers that hold two draws, and the only ones that ever will. Run 2 was D33's
#: 20-pair subsample until option D was dispatched onto it; `assert_run_holds_one_draw`
#: now refuses that, in both directions, so this set is closed. It is declared rather than
#: inferred because "a run appearing in both files" is otherwise indistinguishable from the
#: mistake the guard exists to prevent — and run 2 itself is permanently un-dispatchable:
#: the guard sees both draws on it and refuses whichever one asks.
LEGACY_MIXED_RUNS = frozenset({2})


def backfill_draw(seed: int = 0) -> dict:
    """Stamp `draw` on prompt records dispatched before D35. One time, in place.

    `llm-recheck-prompts.jsonl` is collected data in a git-ignored directory: no seed
    reproduces it, so it is stamped rather than rebuilt. The rewrite adds a field and
    touches nothing a hash is taken over — `prompt_sha256` covers the prompt text alone —
    so every provenance check still holds against the same bytes it held against before.

    The split is **derived and then asserted**, never taken from file order:

    * A run whose pairs are all inside the recheck draw is that draw, whole.
    * A run carrying pairs outside it is a mixed legacy run — only run 2 ever was. Its
      recheck part is exactly `subsample_ids` over `LEGACY_SUBSAMPLE_STRATA`, the draw D33
      sent; everything else is option D. Both halves are checked: the subsample must
      be present in the run, and every pair outside the recheck draw must be a pair of
      `fit/test.parquet`, or the backfill refuses rather than filing a row by default.
    * Where a pure run has exactly `LEGACY_SUBSAMPLE` pairs it must *be* that subsample —
      an independent read on whether the seed still reproduces D33's draw. Run 3 is that
      run, and it is the only reason the mixed split can be trusted: the same derivation
      that names run 2's recheck half reproduces run 3's pair set exactly.
    """
    records = read_jsonl(PROMPTS)
    if not records:
        raise FileNotFoundError(f"{PROMPTS} is empty — nothing to backfill")
    if all("draw" in r for r in records):
        counts = collections.Counter((r["run"], r["draw"]) for r in records)
        return {"stamped": 0, "already_stamped": len(records),
                "by_run": {f"run{run}:{draw}": n for (run, draw), n in sorted(counts.items())},
                "prompts": str(PROMPTS)}

    ids = lambda f: set(f.query_id.astype(str) + "__" + f.doc_id.astype(str))  # noqa: E731
    recheck_ids = ids(judging.a1_recheck(judging.LLM_RECHECK_STRATA, seed))
    legacy_ids = ids(judging.a1_recheck(LEGACY_SUBSAMPLE_STRATA, seed))
    subsample = set(subsample_ids(sorted(legacy_ids), LEGACY_SUBSAMPLE, seed))
    test = pd.read_parquet(PROCESSED / "fit" / "test.parquet")
    test_ids = set(test.jd_id.astype(str) + "__" + test.resume_id.astype(str))

    by_run: dict[int, list[dict]] = {}
    for record in records:
        by_run.setdefault(record["run"], []).append(record)

    stamped = 0
    for run, group in sorted(by_run.items()):
        pairs = {r["pair_id"] for r in group}
        outside = pairs - recheck_ids
        if not outside:
            if len(pairs) == LEGACY_SUBSAMPLE and pairs != subsample:
                raise AssertionError(
                    f"run {run} has {LEGACY_SUBSAMPLE} recheck pairs but they are not "
                    f"subsample_ids(a1_recheck({LEGACY_SUBSAMPLE_STRATA}), "
                    f"{LEGACY_SUBSAMPLE}, seed={seed}) — the seed no longer reproduces "
                    "D33's subsample, so the mixed run cannot be split by it either")
            draw_of = dict.fromkeys(pairs, RECHECK_DRAW)
        else:
            unknown = outside - test_ids
            if unknown:
                raise AssertionError(
                    f"run {run} carries {len(unknown)} pair(s) in neither the recheck draw "
                    f"nor fit/test.parquet, e.g. {sorted(unknown)[:3]} — this backfill only "
                    "knows the two draws D35 names")
            if not subsample <= pairs:
                raise AssertionError(
                    f"run {run} is mixed but does not contain D33's {LEGACY_SUBSAMPLE}-pair "
                    "subsample, so its recheck half cannot be identified. Refusing rather "
                    "than filing 100 option-D rows as a recheck")
            draw_of = {p: (RECHECK_DRAW if p in subsample else FULL_A1_DRAW) for p in pairs}
        for record in group:
            if "draw" not in record:
                record["draw"] = draw_of[record["pair_id"]]
                stamped += 1

    tmp = PROMPTS.with_suffix(PROMPTS.suffix + ".backfill")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(PROMPTS)

    counts = collections.Counter((r["run"], r["draw"]) for r in records)
    return {"stamped": stamped, "already_stamped": len(records) - stamped,
            "by_run": {f"run{run}:{draw}": n for (run, draw), n in sorted(counts.items())},
            "prompts": str(PROMPTS)}


def assert_recheck_nests(seed: int = 0) -> None:
    """The LLM's 100 contain the human's 50, exactly.

    Every figure this repo has already published — kappa(A1, llm) = 0.029 at n=50, the
    `Good Fit` cell at n=15 — is quoted beside a figure over the wider draw. That is only
    honest if the narrow draw is a subset of the wide one. `a1_recheck` makes it so by
    taking `order[:n]` from one permutation per stratum, which is exactly the kind of
    property that stops being true quietly when someone changes a seed or a sort.
    """
    narrow = judging.a1_recheck(judging.RECHECK_STRATA, seed)
    wide = judging.a1_recheck(judging.LLM_RECHECK_STRATA, seed)
    ids = lambda f: set(f.query_id.astype(str) + "__" + f.doc_id.astype(str))  # noqa: E731
    missing = ids(narrow) - ids(wide)
    if missing:
        raise AssertionError(
            f"{len(missing)} of the human 50 are absent from the LLM {len(ids(wide))} — "
            "the two draws have diverged, and no figure over one may be quoted beside a "
            "figure over the other")


# --- judge -----------------------------------------------------------------

def read_jsonl(path) -> list[dict]:
    """JSONL is delimited by `\\n`, and by nothing else.

    This used `str.splitlines()`, which also splits on U+2028, U+2029, `\\x0b`, `\\x0c`,
    `\\x1c`, `\\x1d`, `\\x1e` and `\\x85`. Prompts are written with `ensure_ascii=False`, so
    a document containing any of those puts one **inside** a JSON string and the record is
    torn in half on the way back in. Djinni CV text carries 16 U+2028s; A1 text carried
    none, which is the only reason runs 1-4 worked.

    It surfaced as a `JSONDecodeError` on the first A2 judging pass, which is the good
    outcome. Had the loader skipped unparseable fragments instead, both callers that
    matter would have gone quietly wrong: dispatch dedups on `(pair_id, run)` read from
    here, and `judge`'s resume set is built from here — so torn records would have meant
    pairs re-dispatched and pairs silently never judged.
    """
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


#: Files whose presence in an ancestor of the judge's working directory would put this
#: repository's own prose into its context. `AGENTS.md` names the recheck, the kappa it is
#: chasing and the parquet that holds `a1_label`; a judge that has read it is not the
#: instrument `yc` was.
CONTEXT_FILES = ("CLAUDE.md", "AGENTS.md", ".claude")


class JudgeNotIsolated(RuntimeError):
    pass


class JudgeUnavailable(RuntimeError):
    """The spawn never produced an answer. Infrastructure, not data *(F1)*.

    Raised by `judge_one` and caught by `judge`, which quarantines the pair instead of
    filing it as a return. `kind` is a coarse cause used only for the operator summary —
    every kind is handled identically, because the pair is retried either way.
    """

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail = kind, detail


class JudgeFailures(RuntimeError):
    """`judge()` lost pairs to infrastructure *(fix F2)*.

    It raises rather than returning, because every downstream step reads as success: the
    successful returns are already in `RAW`, `collect()` files them without complaint, and
    a report over them carries an `n` that looks like a smaller sample rather than a
    truncated one. The operator has to learn about this here or not at all.
    """

    def __init__(self, summary: dict):
        self.summary = summary
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(summary["kinds"].items()))
        super().__init__(
            f"run {summary['run']}: {summary['failed']} of "
            f"{summary['attempted']} spawns produced no answer ({kinds}). "
            f"{summary['judged_now']} returns were kept and "
            f"{summary['remaining']} pairs are still unjudged. Nothing was lost — the "
            f"failed pairs stayed out of {summary['raw']} and are quarantined in "
            f"{summary['failed_log']}, so re-running `--judge --run {summary['run']}` "
            "retries exactly them. Do not `--collect` a run that is short: a kappa over "
            "the survivors is a kappa over whichever pairs answered fastest.")


#: Substrings that name an exhausted balance or a throttle rather than a broken install.
#: Used only to label the operator summary — the handling is identical either way.
#:
#: **Do not rely on these firing.** Measured 6 Sep 2026, when run 5 hit a usage limit
#: 193 pairs in: all 202 failed spawns exited 1 with **both streams empty**, so there was
#: nothing to match and they were labelled `exit1`. An unexplained `exit1` in bulk is what
#: a quota looks like from here; the label is a convenience, not a diagnosis.
QUOTA_MARKERS = ("credit balance", "usage limit", "rate limit", "quota", "429",
                 "insufficient", "billing", "too many requests")


def classify_failure(returncode: int, stderr: str) -> str:
    lowered = (stderr or "").lower()
    if any(marker in lowered for marker in QUOTA_MARKERS):
        return "quota"
    return f"exit{returncode}"


def assert_isolated(cwd: Path) -> None:
    """The judge's cwd, and every ancestor, carries no project context but the agent.

    Structural, not instructed — the same standard `serve_group.assert_clean` holds. A
    judge launched inside the repository loads `CLAUDE.md`, and `CLAUDE.md` here is
    `AGENTS.md`, which describes this very experiment. Measured, not assumed: running the
    probe from the repository root answers *yes* to "is AGENTS.md in your context", and
    from an isolated directory it answers *no*.
    """
    cwd = Path(cwd).resolve()
    allowed = (cwd / ".claude" / "agents" / AGENT_DEF.name).resolve()
    for d in (cwd, *cwd.parents):
        for name in CONTEXT_FILES:
            found = d / name
            if not found.exists():
                continue
            if found.is_dir():
                strays = [f for f in found.rglob("*")
                          if f.is_file() and f.resolve() != allowed]
                if strays:
                    raise JudgeNotIsolated(f"{found} carries {strays[0]}")
                continue
            raise JudgeNotIsolated(f"{found} would reach the judge's context")


def judge_sandbox(stack) -> Path:
    """A working directory holding the committed judge and nothing else."""
    cwd = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="a1-judge-")))
    agents = cwd / ".claude" / "agents"
    agents.mkdir(parents=True)
    shutil.copy2(AGENT_DEF, agents / AGENT_DEF.name)
    assert_isolated(cwd)
    return cwd


def judge_argv(record: dict) -> list[str]:
    """The subprocess command for one dispatched pair, **naming its model** *(M2)*.

    Until 8 Sep 2026 this was `claude -p --agent a1-judge` with no `--model`, so the
    model column recorded what the `MODEL` constant said at dispatch time and nothing
    made the spawn honour it. That is fine while there is one model and fatal the moment
    there are two: move 2 of `plan/2026-09-08-judge-calibration` dispatches the same
    instrument under `sonnet`, and a run whose prompts say `sonnet` while the CLI answers
    on its own default is a model comparison with one arm missing — no exception, no
    wrong row count, a plausible kappa. The record is the authority; a record with no
    model is a pre-M2 dispatch and falls back to the constant it was written under.
    """
    model = str(record.get("model") or MODEL).strip()
    return [shutil.which("claude") or "claude", "-p", "--agent", "a1-judge",
            "--model", model]


def judge_one(record: dict, cwd: Path) -> dict:
    """One pair, one process, one label. The process holds no tools and no repository.

    **A non-zero exit is never a return** *(F1)*. A healthy judge exits 0; anything else is
    the CLI reporting that it could not do the task, and filing that as data would let an
    exhausted balance enter the corpus as a judgement the parser then rejects — at which
    point it is indistinguishable from a judge that answered badly. Only what a
    successfully-exited process wrote is data, and an empty stdout from a clean exit is
    still data: the judge ran and said nothing, which `parse_return` should see.
    """
    try:
        proc = subprocess.run(
            judge_argv(record),
            input=record["prompt"], cwd=str(cwd), capture_output=True, text=True,
            timeout=JUDGE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise JudgeUnavailable("timeout", f"no answer in {JUDGE_TIMEOUT_S}s") from exc
    except OSError as exc:                                       # `claude` not on PATH
        raise JudgeUnavailable("spawn", f"{type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        said = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        raise JudgeUnavailable(
            classify_failure(proc.returncode, proc.stderr),
            f"exit {proc.returncode}: {said[:300]}" if said else
            f"exit {proc.returncode}, both streams empty (a usage limit looks like this)")
    return {"pair_id": record["pair_id"], "run": record["run"],
            "returned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw": proc.stdout.strip()}


def assert_run_holds_one_model(run: int, model: str | None = None) -> str:
    """The model a run is judged under is the one its dispatch named *(M2, D38)*.

    Two refusals, both about the same axis. A run whose prompts name two models could
    not be quoted as either, because `collect()` takes the model from the dispatch
    record per row and no later check compares it with what answered. And a `--judge`
    told to run a model the dispatch did not name would write rows labelled with the
    dispatch's model and produced by a different judge — the D35 shape, one axis along.
    """
    named = {str(r.get("model") or MODEL)
             for r in read_jsonl(PROMPTS) if r["run"] == run}
    if not named:
        raise FileNotFoundError(
            f"no run {run} in {PROMPTS} — `--dispatch --run {run}` first")
    if len(named) > 1:
        raise MixedModel(
            f"run {run} was dispatched under {sorted(named)}. A run is one sitting of one "
            "judge, and a model is part of which judge it is (D38): the rows would carry "
            "two models under one run number and no figure could name a population.")
    dispatched_model = named.pop()
    if model is not None and model != dispatched_model:
        raise MixedModel(
            f"run {run} was dispatched against {dispatched_model!r} but --judge was asked "
            f"for {model!r}. `collect()` copies the model out of the dispatch record, so "
            "the rows would name the model that was dispatched and hold the answers of "
            "the one that ran. Re-dispatch under a free run number instead.")
    return dispatched_model


def judge(run: int, concurrency: int = CONCURRENCY, limit: int | None = None,
          model: str | None = None) -> dict:
    """Run every dispatched pair of `run` that has no raw return yet.

    Idempotent per `(pair_id, run)` for the reason dispatch is: a second return matched
    against the first return's prompt hash would let the provenance columns name a prompt
    that did not produce the label.

    `limit` caps how many pairs this invocation judges, for operator-side pacing.
    """
    import contextlib

    dispatched = [r for r in read_jsonl(PROMPTS) if r["run"] == run]
    if not dispatched:
        raise FileNotFoundError(f"no run {run} in {PROMPTS} — `--dispatch --run {run}` first")
    run_model = assert_run_holds_one_model(run, model)

    on_disk = {r["agent_sha256"] for r in dispatched}
    if on_disk != {sha(AGENT_DEF.read_text(encoding="utf-8"))}:
        raise JudgeNotIsolated(
            "the committed judge is not the one these prompts were dispatched against; "
            "re-dispatch rather than judging with a different instrument")

    done = {(r["pair_id"], r["run"]) for r in read_jsonl(RAW)}
    todo = [r for r in dispatched if (r["pair_id"], r["run"]) not in done]
    remaining = len(todo)
    if limit is not None:
        todo = todo[:limit]

    returned = 0
    kinds: collections.Counter[str] = collections.Counter()
    with contextlib.ExitStack() as stack:
        cwd = judge_sandbox(stack)
        RAW.parent.mkdir(parents=True, exist_ok=True)
        fh = stack.enter_context(RAW.open("a", encoding="utf-8"))
        failed_fh = stack.enter_context(FAILED.open("a", encoding="utf-8"))
        with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
            futures = {pool.submit(judge_one, rec, cwd): rec for rec in todo}
            for future in concurrent.futures.as_completed(futures):
                rec = futures[future]
                try:
                    got = future.result()
                except JudgeUnavailable as exc:
                    kinds[exc.kind] += 1
                    sink, row = failed_fh, {
                        "pair_id": rec["pair_id"], "run": rec["run"],
                        "failed_at": datetime.now(timezone.utc)
                                     .isoformat(timespec="seconds"),
                        "kind": exc.kind, "detail": exc.detail}
                except Exception as exc:                        # noqa: BLE001
                    # Anything unforeseen is infrastructure too. The old branch wrote it
                    # to RAW as a `<judge failed: ...>` string, which is the bug F1 fixes.
                    kinds["error"] += 1
                    sink, row = failed_fh, {
                        "pair_id": rec["pair_id"], "run": rec["run"],
                        "failed_at": datetime.now(timezone.utc)
                                     .isoformat(timespec="seconds"),
                        "kind": "error", "detail": f"{type(exc).__name__}: {exc}"}
                else:
                    sink, row = fh, got
                    returned += 1
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()

    summary = {"run": run, "model": run_model, "dispatched": len(dispatched),
               # Scoped to this run. `done` spans every run in RAW, so the unscoped count
               # read 884 on a 400-pair run — a number that invites the wrong conclusion
               # about what just happened.
               "already_judged": len({p for p, r in done if r == run}),
               "attempted": len(todo), "judged_now": returned,
               "failed": sum(kinds.values()), "kinds": dict(kinds),
               "remaining": remaining - returned,
               "concurrency": concurrency, "limit": limit, "raw": str(RAW),
               "failed_log": str(FAILED)}
    if summary["failed"]:
        raise JudgeFailures(summary)
    return summary


def run_coverage(run: int) -> dict:
    """Dispatched, answered, quarantined — for one run *(F3's primitive)*.

    A kappa is a figure over a population, and a run that lost pairs to infrastructure
    sampled a different one. Any report quoting this judge must carry `n_dispatched`
    beside `n_scored` and refuse to print a kappa while they differ; this is the function
    that lets it, and `assert_run_complete` is the refusal.
    """
    dispatched = {r["pair_id"] for r in read_jsonl(PROMPTS) if r["run"] == run}
    answered = {r["pair_id"] for r in read_jsonl(RAW) if r["run"] == run}
    quarantined = {r["pair_id"] for r in read_jsonl(FAILED) if r["run"] == run}
    return {"run": run, "n_dispatched": len(dispatched), "n_answered": len(answered),
            "n_quarantined": len(quarantined - answered),
            "n_missing": len(dispatched - answered),
            "complete": dispatched == answered and bool(dispatched)}


def assert_run_complete(run: int) -> dict:
    """Refuse to quote a figure over a run that is short *(F2/F3)*."""
    cov = run_coverage(run)
    if not cov["complete"]:
        raise JudgeFailures({
            "run": run, "attempted": cov["n_dispatched"],
            "judged_now": cov["n_answered"], "failed": cov["n_missing"],
            "kinds": {"unanswered": cov["n_missing"]},
            "remaining": cov["n_missing"], "raw": str(RAW),
            "failed_log": str(FAILED)})
    return cov


# --- collect ---------------------------------------------------------------

class BadReturn(ValueError):
    pass


def parse_score(value) -> int:
    """A whole number in `SCORE_MIN..SCORE_MAX`, or a `BadReturn`.

    As strict as the label check and for the same reason. A judge that answered `"high"`,
    `85.5` or `120` did not answer the question the guide asks, and a cut point fitted
    over a column holding those describes no instrument. `85.0` is accepted as 85 — JSON
    has one number type and a model writing a trailing `.0` has still given a whole
    number — but nothing else that merely looks numeric is.
    """
    if isinstance(value, bool):                    # bool is an int in Python; not a score
        raise BadReturn(f"score {value!r} is not a number")
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise BadReturn(f"score {value!r} is not a number") from exc
    if number != int(number):
        raise BadReturn(f"score {value!r} is not a whole number")
    number = int(number)
    if not SCORE_MIN <= number <= SCORE_MAX:
        raise BadReturn(f"score {number} is outside {SCORE_MIN}-{SCORE_MAX}")
    return number


def parse_return(raw: str, require_score: bool = False) -> dict:
    """One JSON object, one in-scheme label. Anything else is a `BadReturn`.

    Deliberately strict. A judge that returned prose, or a fourth class, did not do the
    task the guide describes, and averaging its output into a kappa would hide that.

    `require_score` is the dispatch's `response_scheme`, never a default read off this
    module's constant. A row is parsed under the scheme it was *asked* for: runs 1-8 were
    asked for a label alone and are still complete returns, and a run 9 row that answered
    without a score is **`parsed_ok=False`, not a row with a null score** — a null would
    survive into the frame and be skipped by whatever later took a mean, which is the
    silent-partial-sample shape F1 closed one axis along.
    """
    text = str(raw).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BadReturn(f"not JSON: {exc}") from exc
    if not isinstance(obj, dict) or "label" not in obj:
        raise BadReturn("no `label` key")
    label = str(obj["label"]).strip()
    if label not in session.LABELS:
        raise BadReturn(f"label {label!r} is not one of {session.LABELS}")
    if require_score and obj.get("score") is None:
        raise BadReturn("no `score` key, and this run was dispatched under "
                        f"{SCHEME_SCORED!r}")
    score = None if obj.get("score") is None else parse_score(obj["score"])
    reason = str(obj.get("reason", "")).strip()
    return {"label": label, "score": score, "reason": reason[:REASON_CHARS],
            "reason_overlong": len(reason) > REASON_CHARS}


def collect() -> dict:
    """Validate the raw returns and write one committed CSV **per draw**.

    Never `judgements.csv`: `annotator` there is a free string, so an `llm` row would pass
    every existing check and be pooled into the human kappa.

    And never one file for two draws *(D35)*. `llm-recheck.csv` is the file the instrument
    is validated from — `judge_frames` reads it and nothing else — so a row over a
    population the recheck never sampled does not belong in it, whatever run number it
    carries. Routing is by the `draw` stamped at dispatch, and a record without one is
    refused: defaulting would refile option D as a recheck on the first `--collect` after a
    fresh clone.
    """
    dispatched = {(r["pair_id"], r["run"]): r for r in read_jsonl(PROMPTS)}
    if not dispatched:
        raise FileNotFoundError(f"{PROMPTS} is empty — run `--dispatch` first")

    rows, bad = [], []
    for got in read_jsonl(RAW):
        key = (got["pair_id"], got["run"])
        if key not in dispatched:
            raise BadReturn(f"{key} was never dispatched — a judged pair that was not "
                            "served is a bookkeeping error, not a label")
        sent = dispatched[key]
        # The scheme the dispatch asked for, not the one this module asks for today. A
        # pre-move-3 record has none and asked for a label alone; requiring a score of it
        # would fail 1,679 complete returns on a field they were never asked for.
        scheme = str(sent.get("response_scheme", SCHEME_LABEL))
        try:
            parsed = parse_return(got["raw"], require_score=(scheme == SCHEME_SCORED))
            ok, label, reason = True, parsed["label"], parsed["reason"]
            score = parsed["score"]
        except BadReturn as exc:
            ok, label, reason, score = False, "", f"unparsed: {exc}", None
            bad.append(key)
        row_model = sent.get("model", MODEL)
        draw = sent.get("draw")
        if draw not in DRAW_FILES:
            raise BadReturn(
                f"{key} was dispatched with draw={draw!r}. A row is filed by the draw it "
                "was sent under, and pre-D35 prompts carry none — run `--backfill-draw` "
                "before collecting.")
        # Identity comes from the dispatch record, defaulting to the recheck's A1 shape.
        # The defaults are what runs 1-4 were filed under and are kept so re-collecting
        # reproduces the two committed files byte-for-byte; the A2 dispatch stamps its own.
        rows.append({
            "pair_id": sent["pair_id"],
            "batch": sent.get("batch", 0),
            "stratum": sent.get("stratum", "a1_recheck"),
            "corpus": sent.get("corpus", "a1"),
            "query_id": sent["query_id"], "doc_id": sent["doc_id"],
            "selection_reason": sent.get("selection_reason", "a1_recheck"),
            "annotator": f"llm:{row_model}:run{got['run']}",
            "label": label, "shortlist_pick": "", "notes": reason,
            "model": row_model, "agent_sha256": sent["agent_sha256"],
            "prompt_sha256": sent["prompt_sha256"], "run": got["run"],
            "chars_sent": sent["chars_sent"], "parsed_ok": ok,
            "judged_at": got.get("returned_at", ""), "draw": draw,
            "score": score})

    frame = pd.DataFrame(rows, columns=list(LLM_COLUMNS))
    written = {}
    for draw, path in DRAW_FILES.items():
        part = frame[frame.draw == draw]
        # A draw with no rows here was not dispatched on this machine — `RAW` is git-ignored
        # and a clone may hold only part of it. Skipping leaves that draw's committed file
        # alone; truncating it would delete collected labels nothing can reproduce.
        if part.empty:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(path, index=False)
        written[draw] = {"path": str(path), "rows": len(part),
                         "runs": sorted(part.run.unique().tolist())}
    return {"rows": len(frame), "parsed_ok": int(frame.parsed_ok.sum()),
            "scored": int(frame.score.notna().sum()),
            "unparsed": len(bad), "unparsed_pairs": bad[:10],
            "runs": sorted(frame.run.unique().tolist()) if len(frame) else [],
            "written": written}


# --- agreement -------------------------------------------------------------

def cohen_kappa(a: list[str], b: list[str],
                labels: list[str] | None = None) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b,
                                   labels=labels or list(session.LABELS)))


#: `Good` u `Potential` against `No Fit` — the coarsest question the instrument asks, and the
#: one a three-class kappa cannot separate from a boundary disagreement. Reported beside
#: every three-class figure since the run-4 calibration, where the two moved apart: kappa
#: rose 0.287 -> 0.352 while the binary collapse rose 0.421 -> 0.614.
BINARY = ["fit", "no fit"]


def collapse(labels) -> list[str]:
    return ["no fit" if x == "No Fit" else "fit" for x in labels]


def kappa_ci(a: list[str], b: list[str], n_boot: int = 2000,
             seed: int = 0, labels: list[str] | None = None) -> dict:
    """Kappa with a bootstrap 95% CI over pairs. n is always reported alongside."""
    a, b = list(a), list(b)
    if len(a) < 2:
        return {"kappa": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": len(a), "agreement": float("nan")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        sa, sb = [a[i] for i in idx], [b[i] for i in idx]
        if len(set(sa)) > 1 or len(set(sb)) > 1:
            draws.append(cohen_kappa(sa, sb, labels))
    lo, hi = (np.nanpercentile(draws, [2.5, 97.5]) if draws
              else (float("nan"), float("nan")))
    return {"kappa": cohen_kappa(a, b, labels), "lo": float(lo), "hi": float(hi),
            "n": len(a),
            "agreement": float(np.mean([x == y for x, y in zip(a, b)]))}


def majority(labels: list[str]) -> str | None:
    """The per-pair label across runs. A tie is excluded, and the count is reported."""
    counts = pd.Series(labels).value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return None
    return str(counts.index[0])


def judge_frames(seed: int = 0) -> dict[str, pd.Series]:
    """A1's own label, the human's, and the LLM's — one series each, indexed by pair_id.

    **`llm` is the *current* instrument only.** A guide edit is a new judge: it gets a new
    `agent_sha256`, and its labels are a different measurement from the previous judge's.
    Taking a per-pair majority across every run on file would average two instruments into
    one series and quietly attribute the blend to whichever agent happens to be committed —
    the same failure mode as an `llm` row in `judgements.csv`, one level up. Runs of other
    instruments are still returned, under `_instruments`, and reported separately.

    **It reads `RECHECK_CSV` and nothing else** *(D35)*. `llm-recheck-full-a1.csv` holds
    option D — the same judge over a different population — and pooling the two would
    repeat the instrument mistake one axis across: a self-consistency figure computed over
    pairs judged once by the recheck and once by a dispatch that was not the recheck. That
    is not a hypothetical. It is what `run1_vs_run2` reported at n=100 for six days.
    """
    recheck = judging.a1_recheck(judging.LLM_RECHECK_STRATA, seed)
    recheck["pair_id"] = (recheck.query_id.astype(str) + "__"
                          + recheck.doc_id.astype(str))
    a1 = recheck.set_index("pair_id").a1_label

    judged = session.load_judgements()
    human = judged[(judged.corpus == "a1") & judged.label.notna()]
    human = human.drop_duplicates("pair_id").set_index("pair_id").label

    llm = pd.Series(dtype=str)
    runs: dict[int, pd.Series] = {}
    instruments: dict[str, dict] = {}
    current = sha(AGENT_DEF.read_text(encoding="utf-8")) if AGENT_DEF.exists() else ""
    if RECHECK_CSV.exists():
        frame = pd.read_csv(RECHECK_CSV)
        frame = frame[frame.parsed_ok.astype(str).str.lower().isin(("true", "1"))]
        for run, chunk in frame.groupby("run"):
            runs[int(run)] = chunk.drop_duplicates("pair_id").set_index("pair_id").label
        for agent, chunk in frame.groupby("agent_sha256"):
            instruments[str(agent)] = {
                "runs": sorted(int(r) for r in chunk.run.unique()),
                "models": sorted(chunk.model.dropna().astype(str).unique().tolist()),
                "rows": int(len(chunk)),
                "is_current": str(agent) == current,
                "labels": chunk.drop_duplicates("pair_id").set_index("pair_id").label}
        live = frame[frame.agent_sha256.astype(str) == current]
        if len(live):
            grouped = live.groupby("pair_id").label.apply(lambda s: majority(list(s)))
            llm = grouped.dropna()
    return {"a1": a1, "human": human, "llm": llm, "_runs": runs,
            "_instruments": instruments, "_current": current}


def per_class_agreement(reference: list[str], other: list[str]) -> dict:
    """Per class of `reference`, how often `other` said the same thing.

    A pooled kappa can be carried entirely by the easy class: A1-vs-llm scored 0.90 on
    `No Fit`, 0.27 on `Potential Fit` and **0.067 on `Good Fit`**, and still came out
    positive overall. The substitution gate therefore has a per-class clause, and this is
    what evaluates it. Asymmetric on purpose — it is conditioned on the human's label,
    because the human is the reference the judge would be standing in for.
    """
    out = {}
    for label in session.LABELS:
        idx = [i for i, x in enumerate(reference) if x == label]
        out[label] = {
            "n": len(idx),
            "agreement": (float(np.mean([other[i] == label for i in idx]))
                          if idx else float("nan"))}
    return out


def confusion(reference: list[str], other: list[str]) -> dict:
    counts = {a: {b: 0 for b in session.LABELS} for a in session.LABELS}
    for a, b in zip(reference, other):
        if a in counts and b in counts[a]:
            counts[a][b] += 1
    return counts


#: The per-class bar the substitution gate has always applied, unchanged *(D33)*: a
#: pooled kappa carried entirely by the easy class is not agreement about the task. What
#: changed on 8 Sep 2026 is who it is applied to — the **human pair** as well as the
#: judge. A class the two annotators cannot hold between themselves is not a class the
#: judge is failing; it is a class the instrument does not identify.
PER_CLASS_BAR = 0.40

#: The declared indifference band on the interchangeability gate. Not a significance
#: test: at n=126-200 every interval in that table overlaps every other, so nothing here
#: is *separated*, and a verdict of `pass` means "at least as good a partner as the human
#: it replaces, by more than the band", never "better". Stated as a number so the reading
#: is fixed before the run rather than chosen after it.
INTERCHANGEABILITY_BAND = 0.05


def consensus_of(humans: dict[str, pd.Series]) -> tuple[pd.Series, dict]:
    """The label **every** annotator present gave, and the coverage that qualifies it.

    Written for *n* annotators, not two, so a third does not change the API. A pair with
    one label is not consensus and is not in the denominator either: the question the
    coverage answers is "of the pairs where two or more people looked, on how many did
    they agree", and a single-labelled pair had no opportunity to disagree.

    **The two travel together and this function is why** *(design spec §1)*. A consensus
    figure without its coverage tells the reader something false by omission: kappa
    0.401 is over the 48.5% of `targeted` pairs two annotators agreed on, and the other
    51.5% is not missing at random. It is the harder half, selected by exactly the
    property being measured, so a consensus figure is an **upper** estimate of the
    judge's performance on the stratum and never an unbiased one.
    """
    if not humans:
        return pd.Series(dtype=str), {
            "annotators": [], "n_multi_labelled": 0, "n_agreed": 0,
            "coverage": float("nan")}
    wide = pd.DataFrame(humans)
    multi = wide[wide.notna().sum(axis=1) >= 2]
    if multi.empty:
        agreed = multi
        series = pd.Series(dtype=str)
    else:
        agreed = multi[multi.apply(lambda row: row.dropna().nunique() == 1, axis=1)]
        series = (agreed.apply(lambda row: row.dropna().iloc[0], axis=1)
                  if not agreed.empty else pd.Series(dtype=str))
    series.name = "label"
    return series, {
        "annotators": sorted(humans),
        "n_multi_labelled": int(len(multi)),
        "n_agreed": int(len(agreed)),
        "coverage": (float(len(agreed) / len(multi)) if len(multi)
                     else float("nan"))}


#: Attached to every consensus block *(design spec §1)*.
CONSENSUS_SELECTION_NOTE = (
    "The pairs two annotators agreed on are not a random subsample: they are the pairs "
    "both found unambiguous. A judge scores higher there for the reason a student scores "
    "higher on the easy half of an exam, so this is an UPPER estimate of its performance "
    "on the stratum, never an unbiased one. It is also why consensus cannot license "
    "substitution — the pairs it excludes are exactly the population a substituting judge "
    "would label alone, with no reference and no appeal (design spec §3).")


def assert_consensus_carries_coverage(block: dict) -> None:
    """A consensus figure is refused unless its coverage is beside it *(design spec §1)*.

    The same shape as `run_coverage` guarding a kappa over a short run: the figure is
    computable either way and is a different statement depending on a number that is not
    in it. Enforced here rather than in prose so a report cannot be emitted without it.
    """
    missing = [k for k in ("n_multi_labelled", "n_agreed", "coverage", "note_selection")
               if k not in block]
    if missing:
        raise ValueError(
            f"a consensus block is missing {missing}. A consensus kappa is over the "
            "pairs two annotators agreed on — a non-random, easier subset of the "
            "stratum — and quoting it without saying what fraction that was, and that "
            "it is an upper estimate, is false by omission (design spec §1).")
    if block["n_agreed"] > block["n_multi_labelled"]:
        raise ValueError("more agreed pairs than multi-labelled ones — not a consensus")


def interchangeability_gate(humans: dict[str, pd.Series], llm: pd.Series,
                            seed: int = 0) -> dict:
    """Would the judge, in one annotator's seat, partner the other as well as they did?

    This replaces the old gate's relative clause *(Q39)*. That clause divided by the
    human ceiling, and on `targeted` the ceiling is 0.214 with the judge at 0.204-0.249:
    one overlapping cluster, so 0.75 x ceiling separated nothing and "the gate passed"
    and "the gate is not measuring" produced the same output. The deeper problem is that
    both old clauses assume a ground truth this corpus does not have — a question of the
    form "does the judge match the true label" is unanswerable where two people who read
    the same guide match each other at 0.214.

    So the question is changed rather than the threshold. Each row **retains** one
    annotator and asks whether the judge, standing in for the other, is as good a partner
    to the retained one as the person it replaced. Every figure is computed on the pairs
    **all three series share**, which is what makes the two comparable: kappa(yc, llm) is
    0.380 at n=400 and 0.455 on the 126 pairs `van` also labelled, and reading the first
    against a ceiling measured on the second is comparing two populations.

    **What passing licenses**: the judge may serve as a *second* annotator in a future
    double-labelling wave, a role whose specification is being about as unreliable as a
    human. It does **not** license the judge as an answer key. No agreement figure
    between two fallible raters can support a claim about correctness, and D36 stays
    unmade *(design spec §3-4)*.
    """
    names = sorted(humans)
    if len(names) < 2 or llm.empty:
        return {}
    shared = llm.index
    for who in names:
        shared = shared.intersection(humans[who].index)
    if len(shared) < 2:
        return {}

    band = INTERCHANGEABILITY_BAND

    def verdict(judge_k: float, human_k: float) -> str:
        if not np.isfinite(judge_k) or not np.isfinite(human_k):
            return "undefined"
        if judge_k > human_k + band:
            return "pass"
        if judge_k < human_k - band:
            return "fail"
        return "tie"

    rows = {}
    for retained in names:
        for replaced in names:
            if replaced == retained:
                continue
            keep = humans[retained].loc[shared]
            human_fig = _pairwise(keep, humans[replaced].loc[shared], seed)
            judge_fig = _pairwise(keep, llm.loc[shared], seed)
            if human_fig is None or judge_fig is None:
                continue
            rows[retained] = {
                "retained": retained, "replaced": replaced, "n": len(shared),
                "human_pair": {"three_class": human_fig["three_class"],
                               "binary": human_fig["binary"],
                               "per_class": human_fig["per_class"]},
                "judge": {"three_class": judge_fig["three_class"],
                          "binary": judge_fig["binary"],
                          "per_class": judge_fig["per_class"]},
                "verdict": {
                    "three_class": verdict(judge_fig["three_class"]["kappa"],
                                           human_fig["three_class"]["kappa"]),
                    "binary": verdict(judge_fig["binary"]["kappa"],
                                      human_fig["binary"]["kappa"])}}

    # The absolute per-class clause, applied to the human pair as well *(design spec §4)*.
    # A class both humans miss against each other is not a class the judge failed; it is
    # one the instrument does not identify, and that is a finding about the guide.
    per_class = {}
    for label in session.LABELS:
        human_scores = {r: rows[r]["human_pair"]["per_class"][label]["agreement"]
                        for r in rows}
        judge_scores = {r: rows[r]["judge"]["per_class"][label]["agreement"]
                        for r in rows}
        # Per-class agreement is asymmetric — it is conditioned on the reference — so
        # the human pair produces two numbers per class, and the class is unidentified
        # if **either** falls below the bar. `van` recovers 33.8% of `yc`'s `Good Fit`s
        # while `yc` recovers 75.9% of `van`'s; that is not one of them being right, it
        # is the class being drawn in two different places, and a judge cannot be scored
        # against a boundary the instrument does not fix. Requiring both directions to
        # fail would call `Good Fit` identified on the strength of the direction that
        # happens to point from the rarer user of the class to the more frequent one.
        finite = [v for v in human_scores.values() if np.isfinite(v)]
        unidentified = bool(finite) and any(v < PER_CLASS_BAR for v in finite)
        per_class[label] = {
            "bar": PER_CLASS_BAR,
            "human_pair": human_scores,
            "judge": judge_scores,
            "unidentified_by_instrument": unidentified,
            "judge_clears_bar": {r: bool(np.isfinite(v) and v >= PER_CLASS_BAR)
                                 for r, v in judge_scores.items()}}
    return {"n": len(shared), "band": band, "rows": rows, "per_class": per_class,
            "note_not_separated": (
                "Every interval in this block overlaps every other at this n. `pass` "
                "means the judge partners the retained annotator better than the "
                "replaced one did by more than the declared band, not that the two are "
                "distinguishable."),
            "note_licences": (
                "Passing licenses the judge as a *second* annotator in a double-"
                "labelling wave, where being as unreliable as a human is the "
                "specification. It does not license the judge as an answer key: no "
                "agreement figure between two fallible raters is a claim about "
                "correctness. D36 stays unmade.")}


#: The score is defined to increase with fit, so a cut point is a pair of thresholds and
#: the classes are contiguous. That is an assumption the guide makes explicit — the bands
#: describe one scale — and it is what makes "fit two numbers" the whole search.
CUT_POINT_SEED = 0

FITTED_NOTE = (
    "These cut points were fitted against the consensus reference on the pairs named in "
    "`fitted_on`. A kappa on the pairs a parameter was fitted on is not an out-of-sample "
    "result and may not be compared with the emitted label's kappa as though the two "
    "were measured the same way. `held_out` is the only figure here that may.")


def assert_score_figure_names_its_fit(block: dict) -> None:
    """A fitted figure carries the pairs it was fitted on, or it is not emitted.

    Same rule as `assert_consensus_carries_coverage`, one parameter along. A cut point is
    fitted, and a fitted parameter quoted without saying what it was fitted on reads as a
    measurement of the judge — which is the one thing it is not. Enforced here rather
    than in prose so a later edit to `a2_report` cannot drop it and still build.
    """
    for key in ("cut_points", "kappa", "n", "fitted_on"):
        if key not in block:
            raise AssertionError(
                f"a fitted cut-point block is missing {key!r}. A fitted parameter travels "
                "with the pairs it was fitted on (move 3, work-plan task 3.4), the way a "
                "consensus figure travels with its coverage (D37).")
    if not block["fitted_on"]:
        raise AssertionError("`fitted_on` is empty — name the pairs or do not emit the "
                             "figure")


def labels_from_score(scores: pd.Series, lo: float, hi: float) -> list[str]:
    """`No Fit` below `lo`, `Good Fit` at or above `hi`, `Potential Fit` between."""
    return ["No Fit" if v < lo else ("Good Fit" if v >= hi else "Potential Fit")
            for v in scores]


def fit_cut_points(scores: pd.Series, reference: pd.Series,
                   seed: int = CUT_POINT_SEED) -> dict | None:
    """The two thresholds that best reproduce `reference` from the score.

    Exhaustive over the midpoints between observed scores: the search space is at most
    101 values and the criterion is three-class kappa, so there is nothing to be gained
    by a smarter optimiser and something to be lost — a local optimum here would be
    indistinguishable from a score that does not separate the classes.

    Ties are broken by the wider `Potential Fit` band, which is the conservative choice:
    it is the class the judge already over-issues, so a tie resolved that way cannot
    manufacture the `Good Fit` recall this move is measuring.
    """
    m = pd.concat([pd.to_numeric(scores, errors="coerce").rename("s"),
                   reference.rename("r")], axis=1).dropna()
    if m.empty or m.r.nunique() < 2:
        return None
    values = sorted(set(m.s.tolist()))
    edges = ([values[0] - 0.5]
             + [(a + b) / 2 for a, b in zip(values, values[1:])]
             + [values[-1] + 0.5])
    best = None
    for i, lo in enumerate(edges):
        for hi in edges[i:]:
            fitted = labels_from_score(m.s, lo, hi)
            k = cohen_kappa(m.r.tolist(), fitted)
            k = -1.0 if pd.isna(k) else float(k)
            width = hi - lo
            if best is None or (k, width) > (best["kappa"], best["_width"]):
                best = {"kappa": k, "_width": width, "lo": float(lo), "hi": float(hi)}
    fitted = labels_from_score(m.s, best["lo"], best["hi"])
    out = {"cut_points": {"no_fit_below": best["lo"], "good_fit_at_or_above": best["hi"]},
           "n": int(len(m)),
           "fitted_on": sorted(m.index.astype(str).tolist()),
           "kappa": kappa_ci(m.r.tolist(), fitted, seed=seed),
           "binary": kappa_ci(collapse(m.r.tolist()), collapse(fitted), seed=seed,
                              labels=BINARY),
           "per_class": per_class_agreement(m.r.tolist(), fitted),
           "label_distribution": {label: fitted.count(label) for label in session.LABELS},
           "note_fitted": FITTED_NOTE}
    assert_score_figure_names_its_fit(out)
    return out


def rank_separation(scores: pd.Series, reference: pd.Series,
                    higher: str, lower: str) -> dict | None:
    """P(score of a `higher` pair > score of a `lower` pair), ties at 0.5.

    The registered S1 statistic. It asks whether the score separates a boundary the
    emitted label collapses, and it needs no threshold to do it — which is the point,
    since a threshold is the thing move 3 wants to stop having to re-dispatch for.
    """
    m = pd.concat([pd.to_numeric(scores, errors="coerce").rename("s"),
                   reference.rename("r")], axis=1).dropna()
    a = m[m.r == higher].s.tolist()
    b = m[m.r == lower].s.tolist()
    if not a or not b:
        return None
    wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return {"auc": wins / (len(a) * len(b)), "n_higher": len(a), "n_lower": len(b),
            "higher": higher, "lower": lower}


def score_analysis(scores: pd.Series, reference: pd.Series,
                   seed: int = CUT_POINT_SEED) -> dict | None:
    """S1, S2 and S3 of `plan/2026-09-08-judge-calibration/06-graded-registration.md`.

    The held-out half is the only figure in here that is an out-of-sample estimate, and
    it is computed by a seeded split fixed before the scores existed. Reporting the
    full-data fit alone would let a wide held-out interval be quietly replaced by the
    number that cannot miss.
    """
    m = pd.concat([pd.to_numeric(scores, errors="coerce").rename("s"),
                   reference.rename("r")], axis=1).dropna()
    if m.empty:
        return None
    out: dict = {"n": int(len(m)), "seed": seed}
    out["distribution_by_reference_class"] = {
        label: {"n": int(len(chunk)), "mean": float(chunk.s.mean()),
                "median": float(chunk.s.median()),
                "q1": float(chunk.s.quantile(0.25)), "q3": float(chunk.s.quantile(0.75))}
        for label, chunk in m.groupby("r") if label in session.LABELS}
    out["separation"] = {
        name: fig for name, (hi, lo) in {
            "good_vs_potential": ("Good Fit", "Potential Fit"),
            "potential_vs_no": ("Potential Fit", "No Fit"),
            "good_vs_no": ("Good Fit", "No Fit")}.items()
        if (fig := rank_separation(m.s, m.r, hi, lo)) is not None}
    fitted = fit_cut_points(m.s, m.r, seed=seed)
    if fitted is not None:
        out["fitted_full"] = fitted
    # S3. The split is over the shared pairs in sorted order under a fixed seed, so it is
    # reproducible from the report alone and was fixed before any score existed.
    ids = sorted(m.index.astype(str).tolist())
    if len(ids) >= 8:
        order = np.random.default_rng(seed).permutation(len(ids))
        half = len(ids) // 2
        fit_ids = [ids[i] for i in sorted(order[:half])]
        test_ids = [ids[i] for i in sorted(order[half:])]
        fit_part = fit_cut_points(m.s.loc[fit_ids], m.r.loc[fit_ids], seed=seed)
        if fit_part is not None:
            cuts = fit_part["cut_points"]
            held = labels_from_score(m.s.loc[test_ids],
                                     cuts["no_fit_below"], cuts["good_fit_at_or_above"])
            ref = m.r.loc[test_ids].tolist()
            out["held_out"] = {
                "cut_points": cuts,
                "fitted_on": fit_ids, "quoted_on": test_ids,
                "n_fit": len(fit_ids), "n": len(test_ids),
                "kappa": kappa_ci(ref, held, seed=seed),
                "binary": kappa_ci(collapse(ref), collapse(held), seed=seed,
                                   labels=BINARY),
                "per_class": per_class_agreement(ref, held),
                "note": ("fitted on one seeded half of the consensus pairs and quoted on "
                         "the other. This is the only out-of-sample figure in this "
                         "block, and its n is half the consensus coverage, so its "
                         "interval is wide by construction and is reported as it fell.")}
            if set(out["held_out"]["fitted_on"]) & set(out["held_out"]["quoted_on"]):
                raise AssertionError(
                    "the held-out half shares pairs with the fitted half — that is a "
                    "fitted figure wearing an out-of-sample name")
    out["note_fitted"] = FITTED_NOTE
    return out


def a2_frames(stratum: str, model: str | None = None) -> dict:
    """Each annotator's labels, separately, and the judge's, for **one** in-domain stratum.

    Deliberately not `judge_frames`. That function reads `RECHECK_CSV` and nothing else
    and its output backs the committed D33/D34 figures; an A2 row reaching it would move
    published numbers. Two functions, two reports, two files — the same separation D35
    drew between the recheck and option D, one axis across.

    `stratum` is a single stratum and there is no argument that pools two *(D29)*. The
    average of `generic` and `targeted` describes no population, so the API cannot express
    it.

    **`humans` is a dict and there is no `human` key.** Until 7 Sep 2026 there was one:
    every A2 label was `yc`'s, so `drop_duplicates("pair_id")` was a no-op and the series
    was named `human` and read everywhere as `yc`. A second annotator arrived and that
    stopped being true silently — `drop_duplicates` keeps whichever row is first in
    `judgements.csv`, which is file order, not a rule. It resolved to all-`yc` on
    `generic` and **all-`van` on `targeted`**, so `a2_report` would have gone on emitting
    a field the plan and AGENTS.md both read as kappa(yc, llm) while computing
    kappa(van, llm) — a different number (0.204 against 0.377) under the same name. Same
    shape as D35's `run1_vs_run2`: no exception, no row count that looks wrong, a
    plausible figure. An annotator is now an axis a figure may not be pooled across, like
    `agent_sha256` and `draw`, and the API cannot express the pooled version.

    **`model` is an axis too** *(fix M1, proposed D38)*. `llm` used to be a per-pair
    majority across every run sharing the committed `agent_sha256` — and a guide is not
    the whole judge. The same guide under a second model has the *same* `agent_sha256`,
    so two runs of it are a two-element majority: `majority` returns `None` on a tie and
    `.dropna()` deletes it, which means **every pair the two models disagreed on would be
    silently dropped** and the survivors would be exactly the pairs two models agreed on.
    A non-random, easier sample, a smaller n nobody ordered, and a kappa that rises for
    the wrong reason — the D35 shape one axis along. It has never fired: runs 5 and 6 are
    the only A2 runs and both are `haiku`. Move 2 of
    `plan/2026-09-08-judge-calibration` fires it on its first dispatch, which is why this
    is a prerequisite to that dispatch rather than a cleanup after it.

    So `llm` is the committed instrument under **one** model — `model` if given, else the
    model of the highest-numbered run of that instrument — and every other model on file
    is returned under `per_model`, the way `judge_frames` returns `_instruments`.

    **`per_instrument` is the axis above it.** Keyed by `agent_sha256`, each value the
    same model-keyed dict, it is what keeps a superseded instrument quotable after the
    guide moves on: move 3's edit gives the judge a new hash, and a builder that returned
    only the committed one would have dropped runs 5-8 out of the report on the commit
    that changed the guide — a figure disappearing is as misleading as a figure blending,
    and a reader cannot tell "withdrawn" from "superseded" by absence. `scores` is the
    same shape once more, holding the graded instrument's 0-100 column for the runs that
    have one; it is empty for every pre-move-3 run and carries no pair the parser refused.
    """
    if stratum not in A2_DRAW_STRATUM.values():
        raise ValueError(
            f"{stratum!r} is not an in-domain stratum. D29 defines exactly two — "
            f"{sorted(A2_DRAW_STRATUM.values())} — and they are never pooled: a figure "
            "over both describes no population. Ask for one, then the other.")
    draw = next(d for d, s in A2_DRAW_STRATUM.items() if s == stratum)
    path = DRAW_FILES[draw]

    judged = session.load_judgements()
    rows = judged[(judged.corpus == "a2") & (judged.stratum.astype(str) == stratum)
                  & judged.label.notna()]
    humans = {
        str(who): chunk.drop_duplicates("pair_id").set_index("pair_id").label
        for who, chunk in rows.groupby(rows.annotator.astype(str))}
    consensus, consensus_coverage = consensus_of(humans)

    llm = pd.Series(dtype=str)
    per_model: dict[str, pd.Series] = {}
    per_instrument: dict[str, dict[str, pd.Series]] = {}
    scores: dict[str, dict[str, pd.Series]] = {}
    runs: dict[int, pd.Series] = {}
    run_meta: dict[int, dict] = {}
    live_model = ""
    current = sha(AGENT_DEF.read_text(encoding="utf-8")) if AGENT_DEF.exists() else ""
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame[frame.parsed_ok.astype(str).str.lower().isin(("true", "1"))]
        for run, chunk in frame.groupby("run"):
            runs[int(run)] = chunk.drop_duplicates("pair_id").set_index("pair_id").label
            # Both axes, per run. `self_consistency` is a property of one judge repeating
            # itself, and two runs that differ on either are a comparison, not a
            # reliability check.
            run_meta[int(run)] = {
                "models": sorted(chunk.model.dropna().astype(str).unique().tolist()),
                "agents": sorted(chunk.agent_sha256.dropna().astype(str)
                                 .unique().tolist()),
                "rows": int(len(chunk)),
                "scored": int(chunk.score.notna().sum()) if "score" in chunk else 0}
        # Every instrument on file, each split by model — the two axes that make a judge,
        # nested in the order a reader asks them. A guide edit supersedes an instrument
        # (move 3 is one); it does not delete the runs made under the old one, and a
        # report that showed only the committed hash would drop runs 5-8 from the file
        # entirely the moment the guide changed, leaving the reader to infer that the
        # figures had been withdrawn rather than superseded. No figure pools two of
        # these, which is the whole reason they are separate keys *(D38, one axis up)*.
        for agent, agent_chunk in frame.groupby(frame.agent_sha256.astype(str)):
            by_model = {}
            for name, chunk in agent_chunk.groupby(agent_chunk.model.astype(str)):
                by_model[str(name)] = chunk.groupby("pair_id").label.apply(
                    lambda s: majority(list(s))).dropna()
                if "score" in chunk:
                    numeric = pd.to_numeric(chunk.score, errors="coerce").dropna()
                    if len(numeric):
                        scores.setdefault(str(agent), {})[str(name)] = (
                            chunk.loc[numeric.index]
                            .drop_duplicates("pair_id").set_index("pair_id")
                            .score.astype(float))
            per_instrument[str(agent)] = by_model
        per_model = dict(per_instrument.get(current, {}))
        live = frame[frame.agent_sha256.astype(str) == current]
        if per_model:
            # The default is the **committed** model, not the most recent run *(D38)*.
            # Defaulting to run order would mean that dispatching a control run silently
            # changed what `vs_llm` refers to — the field would go on being read as the
            # committed judge's figure while holding the challenger's, which is the
            # rename-under-one-field this module refuses one axis at a time. A file with
            # no rows under the committed model falls back to the newest run, and the
            # report names whichever it got.
            newest = live.loc[live.run.astype(int).idxmax()]
            default = MODEL if MODEL in per_model else str(newest.model)
            live_model = str(model) if model is not None else default
            if live_model not in per_model:
                raise ValueError(
                    f"the committed instrument has no rows under model {live_model!r} in "
                    f"{path.name}. On file: {sorted(per_model)}. A model is part of which "
                    "judge this is (D38) and there is no series that blends two.")
            llm = per_model[live_model]
    return {"humans": humans, "consensus": consensus,
            "consensus_coverage": consensus_coverage,
            "llm": llm, "per_model": per_model, "per_instrument": per_instrument,
            "scores": scores,
            "_runs": runs, "_run_meta": run_meta, "_model": live_model,
            "_draw": draw, "_current": current, "_path": path}


def _pairwise(a: pd.Series, b: pd.Series, seed: int) -> dict | None:
    """Three-class, binary, per-class and both confusions over the pairs `a` and `b` share.

    Binary sits beside three-class on every figure without exception — D34 is the run
    where the two moved apart, and A2 is a corpus where they say different things about
    the same judge (0.380 against 0.637 on `generic`). Per-class sits there because the
    A1 recheck's whole positive kappa came out of one cell and the headline hid it.
    """
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if m.empty:
        return None
    x, y = m.a.tolist(), m.b.tolist()
    return {"n": len(m),
            "three_class": kappa_ci(x, y, seed=seed),
            "binary": kappa_ci(collapse(x), collapse(y), seed=seed, labels=BINARY),
            "per_class": per_class_agreement(x, y),
            "confusion_rows_first": confusion(x, y),
            "confusion_rows_second": confusion(y, x)}


def a2_report(seed: int = 0, strict: bool = True) -> dict:
    """Per in-domain stratum: each annotator against the judge, and against each other.

    Never an average of the two strata *(D29)*, and **never an average of two annotators**.
    Every judge figure is keyed by who it is against, because on this corpus the answer
    depends on that: on `generic`, the same judge scores 0.380 against `yc` and 0.142
    against `van` over the same pairs. A field called `human_vs_llm` would be one of those
    two numbers wearing a name that claims to be both, which is what this report emitted
    until 7 Sep 2026 — see `a2_frames`.

    `human_ceiling` is kappa(annotator, annotator) on the pairs the two share. It is the
    denominator of the substitution gate in `plan/2026-09-06-a2-llm-prelabel`: a judge is
    not measured against 1.0, it is measured against how much two people doing the same
    task agree. It is reported whether or not the gate is being evaluated, because every
    judge figure in the file is uninterpretable without it.

    Every figure carries `n_dispatched` beside `n_scored` and, while `strict`, the report
    **refuses to compute a kappa over a run that is short** *(fix F3)*. A judging pass that
    lost pairs to an empty balance did not sample a smaller version of the population; it
    sampled whichever pairs answered fastest, which correlates with document length. The
    resulting kappa is not the figure it appears to be, and nothing downstream would say
    so. Pass `strict=False` only to inspect a partial run, never to quote one.
    """
    out: dict = {"model": MODEL, "seed": seed,
                 "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "strata": {}}
    for draw, stratum in sorted(A2_DRAW_STRATUM.items()):
        frames = a2_frames(stratum)
        out["current_agent_sha256"] = frames["_current"]
        dispatched_runs = sorted({r["run"] for r in read_jsonl(PROMPTS)
                                  if r.get("draw") == draw})
        coverage = {str(run): run_coverage(run) for run in dispatched_runs}
        short = [run for run, cov in coverage.items() if not cov["complete"]]

        humans: dict[str, pd.Series] = frames["humans"]
        section: dict = {"draw": draw, "file": str(frames["_path"]),
                         "runs": dispatched_runs, "coverage": coverage,
                         "annotators": {who: len(s) for who, s in sorted(humans.items())},
                         # Which judge every figure below is against. `llm_model` is not
                         # decoration: the committed guide under two models is two judges
                         # sharing one agent_sha256 (D38), and a `vs_llm` that did not
                         # name its model would be the pooled field this report already
                         # refuses one axis along.
                         "llm_model": frames["_model"],
                         "models_on_file": {m: len(s) for m, s
                                            in sorted(frames["per_model"].items())},
                         "n_llm": len(frames["llm"])}
        if not dispatched_runs:
            section["status"] = "not dispatched"
            out["strata"][stratum] = section
            continue
        if short and strict:
            section["status"] = "refused"
            section["note"] = (
                f"run(s) {short} are short: pairs were dispatched that never answered. A "
                "kappa over the survivors is a kappa over whichever pairs answered "
                "fastest, not over this stratum. Re-run `--judge` for those runs.")
            out["strata"][stratum] = section
            continue

        def model_block(series: pd.Series, agent: str = "", name: str = "") -> dict:
            """Every judge figure, for **one** model *(D38, move 2)*.

            Called once per model on file so `haiku` and `sonnet` are quoted side by
            side and neither is the unqualified `llm`. The three references of D37 and
            the gate of Q39 are all computed inside it, because each is a statement
            about one judge and the model is part of which judge that is.
            """
            block: dict = {"n_llm": len(series)}
            block["vs_llm"] = {
                who: fig for who, h in sorted(humans.items())
                if (fig := _pairwise(h, series, seed)) is not None}
            consensus_fig = _pairwise(frames["consensus"], series, seed)
            cblock = dict(frames["consensus_coverage"])
            cblock["label_distribution"] = {
                label: int((frames["consensus"] == label).sum())
                for label in session.LABELS}
            cblock["note_selection"] = CONSENSUS_SELECTION_NOTE
            if consensus_fig is not None:
                cblock["vs_llm"] = {
                    "n": consensus_fig["n"],
                    "three_class": consensus_fig["three_class"],
                    "binary": consensus_fig["binary"],
                    "per_class": consensus_fig["per_class"],
                    "confusion_rows_consensus": consensus_fig["confusion_rows_first"],
                    "confusion_rows_llm": consensus_fig["confusion_rows_second"]}
            assert_consensus_carries_coverage(cblock)
            block["consensus"] = cblock
            block["interchangeability"] = interchangeability_gate(humans, series, seed)
            block["label_distribution"] = {
                label: int((series == label).sum()) for label in session.LABELS}
            # Only the graded instrument has one. Every figure in here is fitted against
            # consensus and says so in a field, not in prose (work-plan task 3.4).
            graded = frames["scores"].get(agent, {}).get(name)
            if graded is not None:
                analysis = score_analysis(graded, frames["consensus"])
                if analysis is not None:
                    analysis.update(frames["consensus_coverage"])
                    analysis["note_selection"] = CONSENSUS_SELECTION_NOTE
                    block["score"] = analysis
            return block

        # One entry per annotator, keyed by name. Never a pooled "human".
        vs_llm = {who: fig for who, s in sorted(humans.items())
                  if (fig := _pairwise(s, frames["llm"], seed)) is not None}
        # The gate's denominator: how much two people doing this task agree, on the pairs
        # they both saw. Ordered pairs of names so `first`/`second` in the confusions and
        # in `per_class` are readable without guessing which way round they went.
        ceiling = {f"{x}_vs_{y}": fig
                   for i, x in enumerate(sorted(humans))
                   for y in sorted(humans)[i + 1:]
                   if (fig := _pairwise(humans[x], humans[y], seed)) is not None}
        section["vs_llm"] = vs_llm
        section["human_ceiling"] = ceiling

        # Every instrument on file, each split by model, each with its own complete
        # figure set. Two nested axes and no field crossing either: a guide edit makes a
        # new judge (move 3) and a second model makes another (D38), so runs 5-8 stay
        # quotable under their own hash after the committed instrument moves on rather
        # than vanishing from the report. `by_model` is `by_instrument[current]`, kept
        # under its old name because that is what move 2's finding cites.
        by_instrument = {
            agent: {name: model_block(series, agent, name)
                    for name, series in sorted(models.items())}
            for agent, models in sorted(frames["per_instrument"].items())}
        section["by_instrument"] = by_instrument
        section["instruments_on_file"] = {
            agent: {"committed": agent == frames["_current"],
                    "models": {name: len(series) for name, series
                               in sorted(models.items())}}
            for agent, models in sorted(frames["per_instrument"].items())}
        by_model = by_instrument.get(frames["_current"], {})
        section["by_model"] = by_model
        committed = by_model.get(frames["_model"], {})
        section["consensus"] = committed.get("consensus", {})
        section["interchangeability"] = committed.get("interchangeability", {})
        section["n_scored"] = max((f["n"] for f in vs_llm.values()), default=0)
        section["n_dispatched"] = sum(c["n_dispatched"] for c in coverage.values())
        section["status"] = "scored" if vs_llm else "no overlap"
        if len(humans) > 1 and not ceiling:
            section["note_no_ceiling"] = (
                f"{sorted(humans)} both labelled in this stratum but share no pair, so "
                "there is no human ceiling here and no judge figure in it can be read as "
                "a fraction of one.")
        # Self-consistency is a property of one judge over one draw, and the judge is
        # the guide **and** the model *(D38)*. Runs 5 and 7 are the same guide under two
        # models: that is the model control, and reporting it here would put a
        # between-judge comparison under a field every reader takes for test-retest
        # reliability — the number D33 used to bound how high kappa(A1, llm) could have
        # gone. Pairs that differ on either axis are listed, not scored.
        meta = frames["_run_meta"]
        same = lambda x, y: (meta.get(x, {}).get("models") ==                # noqa: E731
                             meta.get(y, {}).get("models") and
                             meta.get(x, {}).get("agents") ==
                             meta.get(y, {}).get("agents"))
        section["self_consistency"] = {
            f"run{x}_vs_run{y}": kappa_ci(
                frames["_runs"][x].loc[idx].tolist(),
                frames["_runs"][y].loc[idx].tolist(), seed=seed)
            for i, x in enumerate(sorted(frames["_runs"]))
            for y in sorted(frames["_runs"])[i + 1:]
            if same(x, y) and len(idx := frames["_runs"][x].index.intersection(
                frames["_runs"][y].index))}
        section["not_self_consistency"] = {
            f"run{x}_vs_run{y}": {
                "run_x": meta.get(x, {}), "run_y": meta.get(y, {}),
                "why": ("these two runs differ on instrument or model, so their "
                        "agreement is a comparison between judges and not this judge "
                        "repeating itself (D38)")}
            for i, x in enumerate(sorted(frames["_runs"]))
            for y in sorted(frames["_runs"])[i + 1:]
            if not same(x, y)}
        section["run_provenance"] = {str(r): m for r, m in sorted(meta.items())}
        out["strata"][stratum] = section

    out["note_never_pooled"] = (
        "`generic` and `targeted` are two populations (D29). Both figures are true "
        "statements; their average is not, and this report has no field holding one.")
    out["note_never_pooled_references"] = (
        "`yc`, `van` and `consensus` are three references, not three samples of one, and "
        "a figure names which it is against. They are computed over different sets of "
        "pairs — 400 / 126 / 60 on `generic` — so kappa(consensus, llm) = 0.401 beside "
        "kappa(yc, llm) = 0.249 is not the judge improving; it is a different question "
        "asked of a different half of the corpus. Every consensus figure carries its "
        "coverage and its selection-bias note, enforced by "
        "`assert_consensus_carries_coverage`.")
    out["note_never_pooled_instruments"] = (
        "A guide edit is a new judge. `by_instrument` is keyed by agent_sha256 and no "
        "figure spans two of its keys; `by_model` and every section-level block are the "
        "**committed** instrument's, named in `current_agent_sha256`. Runs made under a "
        "superseded instrument keep their own key rather than being dropped — they were "
        "not withdrawn, they describe a judge this repository no longer commits.")
    out["note_never_pooled_models"] = (
        "The committed guide under two models is two judges that share one "
        "agent_sha256. Every stratum names the model its `vs_llm` block is against in "
        "`llm_model`, other models on file are counted in `models_on_file`, and no field "
        "holds a series built from two. Before this fix a second model would have been "
        "folded into the same per-pair majority, where a two-element tie is dropped: the "
        "surviving pairs would have been exactly those the two models agreed on (D38).")
    out["note_never_pooled_annotators"] = (
        "Every judge figure is keyed by the annotator it is against, and there is no "
        "pooled `human`. Two annotators are two references, not two samples of one: on "
        "`generic` the same judge scores 0.380 against yc and 0.142 against van over the "
        "same 126 pairs. Read each against `human_ceiling`, never against 1.0.")
    out["note_separate_from_a1"] = (
        "These figures are over the in-domain corpus and share no code path with "
        "llm-recheck-report.json, which is the A1 recheck. `judge_frames` does not read "
        "the A2 files.")
    A2_REPORT.parent.mkdir(parents=True, exist_ok=True)
    A2_REPORT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def report(seed: int = 0) -> dict:
    frames = judge_frames(seed)
    runs = frames.pop("_runs", {})
    instruments = frames.pop("_instruments", {})
    current = frames.pop("_current", "")
    names = ["a1", "human", "llm"]

    def pairwise_over(restrict: set[str] | None) -> dict:
        out = {}
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                common = frames[x].index.intersection(frames[y].index)
                if restrict is not None:
                    common = common.intersection(pd.Index(sorted(restrict)))
                # A kappa over no pairs is not a figure. A guide edit leaves the recheck
                # draw unjudged by the committed instrument until it is re-judged, and
                # emitting NaN here would read as a measurement that came out empty
                # rather than one never made — the same absent-vs-withdrawn ambiguity
                # `per_instrument` exists to remove. The runs on file keep their figures
                # there; `status` says which case the section level is in.
                if not len(common):
                    continue
                a, b = frames[x].loc[common].tolist(), frames[y].loc[common].tolist()
                out[f"{x}_vs_{y}"] = kappa_ci(a, b, seed=seed)
                out[f"{x}_vs_{y}_binary"] = kappa_ci(
                    collapse(a), collapse(b), seed=seed, labels=BINARY)
        return out

    narrow = judging.a1_recheck(judging.RECHECK_STRATA, seed)
    human_50 = set(narrow.query_id.astype(str) + "__" + narrow.doc_id.astype(str))

    pairwise = pairwise_over(None)
    # The human 50 are a strict subset of the LLM draw (`assert_recheck_nests`), so the
    # figures already published against n=50 stay quotable rather than being silently
    # replaced by a wider number computed over pairs no human ever saw.
    pairwise_50 = pairwise_over(human_50)

    # Self-consistency is a property of *one* judge. Comparing runs of two instruments
    # would report a guide edit as unreliability, which is the opposite of what it is.
    run_agent = {r: a for a, meta in instruments.items() for r in meta["runs"]}
    self_consistency = {}
    run_ids = sorted(runs)
    for i, x in enumerate(run_ids):
        for y in run_ids[i + 1:]:
            if run_agent.get(x) != run_agent.get(y):
                continue
            common = runs[x].index.intersection(runs[y].index)
            self_consistency[f"run{x}_vs_run{y}"] = kappa_ci(
                runs[x].loc[common].tolist(), runs[y].loc[common].tolist(), seed=seed)

    # Every instrument on file, against both human legs, so a guide change is readable as
    # a change rather than as movement in a pooled number.
    per_instrument = {}
    for agent, meta in instruments.items():
        labels = meta["labels"]
        block = {"runs": meta["runs"], "models": meta["models"], "rows": meta["rows"],
                 "is_current": meta["is_current"]}
        for other in ("a1", "human"):
            common = labels.index.intersection(frames[other].index)
            a, b = frames[other].loc[common].tolist(), labels.loc[common].tolist()
            block[f"vs_{other}"] = kappa_ci(a, b, seed=seed)
            block[f"vs_{other}_binary"] = kappa_ci(
                collapse(a), collapse(b), seed=seed, labels=BINARY)
        # Over the recheck draw only, so the marginals belong to the same pairs the
        # kappas above were computed on. An instrument may carry other rows (option D
        # judges all of A1); mixing them in would make the two blocks incomparable.
        drawn = labels.loc[labels.index.intersection(frames["a1"].index)]
        block["marginals"] = {c: int((drawn == c).sum()) for c in session.LABELS}
        block["marginals_n"] = int(len(drawn))
        per_instrument[agent] = block

    llm_covered = frames["a1"].index.intersection(frames["llm"].index)
    all_three = frames["a1"].index
    for name in names[1:]:
        all_three = all_three.intersection(frames[name].index)
    disagreements = [
        {"pair_id": p, "a1": frames["a1"][p], "human": frames["human"][p],
         "llm": frames["llm"][p]}
        for p in sorted(all_three)
        if len({frames[n][p] for n in names}) > 1]

    out = {
        "model": MODEL, "seed": seed,
        "current_agent_sha256": current,
        "status": "scored" if len(frames["llm"]) else "no overlap",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage": {n: int(len(frames[n])) for n in names} | {
            "all_three": int(len(all_three)),
            "human_50_draw": len(human_50),
            "llm_draw": int(len(frames["a1"]))},
        "pairwise_kappa": pairwise,
        "pairwise_kappa_over_the_human_50": pairwise_50,
        "per_instrument": per_instrument,
        "self_consistency_kappa": self_consistency,
        "per_class_agreement": {
            cls: {
                "n": int((frames["a1"].loc[all_three] == cls).sum()),
                "human_agrees": float((frames["human"].loc[all_three][
                    frames["a1"].loc[all_three] == cls] == cls).mean())
                if (frames["a1"].loc[all_three] == cls).any() else float("nan"),
                "llm_agrees": float((frames["llm"].loc[all_three][
                    frames["a1"].loc[all_three] == cls] == cls).mean())
                if (frames["a1"].loc[all_three] == cls).any() else float("nan"),
            } for cls in (session.LABELS if len(all_three) else ())},
        "per_class_llm_vs_a1": {
            cls: {"n": int((frames["a1"].loc[llm_covered] == cls).sum()),
                  "llm_agrees": float(
                      (frames["llm"].loc[llm_covered][
                          frames["a1"].loc[llm_covered] == cls] == cls).mean())
                  if (frames["a1"].loc[llm_covered] == cls).any() else float("nan")}
            for cls in (session.LABELS if len(llm_covered) else ())},
        "disagreements": disagreements,
        "note_not_ground_truth": (
            "The LLM is a third judge with its own bias, known to exist and not measured "
            "here. It breaks a tie between two accounts of the A1 labels; it does not "
            "adjudicate one correct (D33)."),
        "note_not_reproducible": (
            "No seed reproduces a subagent run. llm-recheck.csv is collected data: "
            "append-only, provenance-stamped by agent_sha256 and prompt_sha256, and "
            "never subject to a determinism check."),
        "note_instruments_are_not_pooled": (
            "`llm` above is the instrument named by current_agent_sha256 and nothing "
            "else. A guide edit makes a new judge; its labels are a separate "
            "measurement. Every instrument on file is reported under per_instrument, "
            "and self_consistency_kappa compares only runs that share one."),
        "note_no_overlap": (
            "The committed instrument has judged none of the A1 recheck draw, so every "
            "section-level figure involving `llm` is absent rather than zero: it was "
            "never measured, not measured empty. The instruments that did judge this "
            "draw are under per_instrument, unwithdrawn, with their runs, models and n. "
            "Re-judging the recheck under the committed instrument is what would fill "
            "the section level back in."
            if not len(frames["llm"]) else
            f"The committed instrument covers {len(frames['llm'])} pairs of the draw."),
        "note_human_leg": (
            f"The human leg covers {len(frames['human'])} of the {len(human_50)} pairs "
            "drawn for it. Every kappa involving `human` is provisional until that "
            "sitting finishes; re-run `llm_recheck --report`. The A1-vs-llm leg has no "
            "human dependency and is final at its stated n."
            if len(frames["human"]) < len(human_50) else
            f"The human leg is complete at {len(human_50)} pairs."),
        "note_instrument_difference": (
            "The human saw A1 candidates 1-4 at a time; the judge sees one pair in a "
            "fresh context. Nothing in the A1 instrument spans candidates "
            "(asks_shortlist is False for corpus a1), so no question is lost, but the "
            "two judges are not identical on the co-presence axis."),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


# --- cli -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", action="store_true",
                    help="render .claude/agents/a1-judge.md from the guide")
    ap.add_argument("--dispatch", action="store_true")
    ap.add_argument("--all-a1", action="store_true",
                    help="dispatch all 659 A1 pairs (option D), not just the recheck")
    ap.add_argument("--a2", choices=sorted(A2_DRAW_STRATUM.values()), default=None,
                    help="dispatch one in-domain stratum (D29 — never both at once)")
    ap.add_argument("--a2-report", action="store_true",
                    help="kappa(yc, llm) per in-domain stratum, written to "
                         "output/annotation/llm-a2-report.json")
    ap.add_argument("--allow-short-run", action="store_true",
                    help="compute A2 figures over an incomplete run (F3). For inspection "
                         "only — never quote the result")
    ap.add_argument("--judge", action="store_true",
                    help="run the committed judge over the dispatched prompts")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--limit", type=int, default=None,
                    help="judge at most N pairs this invocation (operator-side pacing)")
    ap.add_argument("--backfill-draw", action="store_true",
                    help="stamp `draw` on prompt records dispatched before D35 (one time)")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--coverage", action="store_true",
                    help="dispatched vs answered vs quarantined, for --run (F3)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--model", default=None,
                    help="the model to dispatch under, and the model --judge is willing "
                         f"to run (default {MODEL}). Model is an axis a figure may not be "
                         "pooled across (D38): --judge refuses a run whose dispatch names "
                         "a different one")
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.agent:
        AGENT_DEF.parent.mkdir(parents=True, exist_ok=True)
        AGENT_DEF.write_text(render_agent_definition(), encoding="utf-8")
        print(f"wrote {AGENT_DEF}  sha256={sha(render_agent_definition())[:12]}")
    dispatch_model = args.model or MODEL
    if args.dispatch and args.all_a1:
        print(json.dumps(dispatch_all_a1(args.run, args.seed, dispatch_model), indent=2))
    elif args.dispatch and args.a2:
        print(json.dumps(dispatch_a2(args.a2, args.run, args.seed, dispatch_model),
                         indent=2))
    elif args.dispatch:
        print(json.dumps(dispatch(args.run, args.seed, args.subsample, dispatch_model),
                         indent=2))
    if args.backfill_draw:
        print(json.dumps(backfill_draw(args.seed), indent=2))
    if args.judge:
        try:
            print(json.dumps(judge(args.run, args.concurrency, args.limit, args.model),
                             indent=2))
        except JudgeFailures as exc:
            # Exit non-zero and say what to do. A failed judging pass that printed a
            # summary and returned 0 is how a short run reaches `--collect` (F2).
            print(json.dumps(exc.summary, indent=2))
            print(f"\n{exc}")
            return 1
    if args.coverage:
        print(json.dumps(run_coverage(args.run), indent=2))
    if args.collect:
        print(json.dumps(collect(), indent=2))
    if args.a2_report:
        out = a2_report(args.seed, strict=not args.allow_short_run)
        print(json.dumps({k: v for k, v in out.items()
                          if not k.startswith("note_")}, indent=2))
        print(f"-> {A2_REPORT}")
    if args.report:
        out = report(args.seed)
        print(json.dumps({k: v for k, v in out.items()
                          if not k.startswith("note_") and k != "disagreements"},
                         indent=2))
        print(f"{len(out['disagreements'])} pairs where the three judges differ "
              f"-> {REPORT}")
    if not any((args.agent, args.dispatch, args.backfill_draw, args.judge, args.collect,
                args.coverage, args.report, args.a2_report)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
