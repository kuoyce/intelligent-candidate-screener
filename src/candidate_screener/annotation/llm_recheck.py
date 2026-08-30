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

Usage:
    uv run python -m candidate_screener.annotation.llm_recheck --agent      # (re)write the judge
    uv run python -m candidate_screener.annotation.llm_recheck --dispatch --run 1
    uv run python -m candidate_screener.annotation.llm_recheck --dispatch --run 2 --subsample 20
    uv run python -m candidate_screener.annotation.llm_recheck --judge --run 1
    uv run python -m candidate_screener.annotation.llm_recheck --collect
    uv run python -m candidate_screener.annotation.llm_recheck --report
"""
from __future__ import annotations

import argparse
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
RECHECK_CSV = MANIFESTS / "llm-recheck.csv"
REPORT = OUTPUT / "annotation" / "llm-recheck-report.json"

#: The judge. Named in every output row, because a different model is a different judge.
MODEL = "claude-sonnet-5"

#: Sections of the guide that describe the *human workflow* and are dropped from the
#: judge's system prompt. The shortlist question goes with them: `asks_shortlist` is
#: already `False` for `corpus == "a1"`, so it is a question this judge is never asked,
#: and leaving it in would invite a single-pair judge to compare against a field it
#: cannot see. Everything that defines the instrument — the three class
#: definitions, all nine worked examples, the tie-breaking rule, the what-not-to-consider
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
#: attributable to an instrument, since no seed will reproduce it.
LLM_COLUMNS = judging.JUDGEMENT_COLUMNS + (
    "model", "agent_sha256", "prompt_sha256", "run", "chars_sent", "parsed_ok",
    "judged_at")


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
description: Judges one job-description/CV pair against the annotation guide's three classes. Returns one JSON object and nothing else. Tool-free by design.
model: sonnet
tools: []
---

You are one annotator in a labelling session. You will be shown **one** job description
and **one** candidate CV, and you return **one** label.

Two rules override anything else:

1. **Reply with exactly one JSON object and no other text.** No prose before it, no
   markdown fence around it, no explanation after it.
   `{{"label":"Good Fit"|"Potential Fit"|"No Fit","reason":"<={REASON_CHARS} chars"}}`
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

def render_prompt(jd_text: str, cv_text: str) -> str:
    """The whole user message. No preamble — the guide is in the agent definition."""
    return (f"JOB DESCRIPTION\n{jd_text}\n\nCANDIDATE\n{cv_text}\n\n"
            "Reply with one JSON object, no prose, no markdown fence:\n"
            f'{{"label":"Good Fit"|"Potential Fit"|"No Fit",'
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


def dispatch(run: int, seed: int = 0, subsample: int | None = None) -> dict:
    from candidate_screener.annotation import ui

    agent_hash = assert_agent_definition_current()
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
                            "prompt": prompt, "prompt_sha256": sha(prompt),
                            "chars_sent": len(prompt),
                            "agent_sha256": agent_hash, "model": MODEL})

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

    return {"run": run, "seed": seed, "pairs": len(records),
            "already_dispatched": len(skipped),
            "jds": len({r["query_id"] for r in records}),
            "subsample": subsample, "agent_sha256": agent_hash, "model": MODEL,
            "chars_sent_median": int(np.median([r["chars_sent"] for r in records]))
            if records else 0,
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
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


#: Files whose presence in an ancestor of the judge's working directory would put this
#: repository's own prose into its context. `AGENTS.md` names the recheck, the kappa it is
#: chasing and the parquet that holds `a1_label`; a judge that has read it is not the
#: instrument `yc` was.
CONTEXT_FILES = ("CLAUDE.md", "AGENTS.md", ".claude")


class JudgeNotIsolated(RuntimeError):
    pass


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


def judge_one(record: dict, cwd: Path) -> dict:
    """One pair, one process, one label. The process holds no tools and no repository."""
    proc = subprocess.run(
        [shutil.which("claude") or "claude", "-p", "--agent", "a1-judge"],
        input=record["prompt"], cwd=str(cwd), capture_output=True, text=True,
        timeout=JUDGE_TIMEOUT_S, check=False)
    raw = proc.stdout.strip()
    if proc.returncode != 0 and not raw:
        raw = f"<judge exited {proc.returncode}: {proc.stderr.strip()[:300]}>"
    return {"pair_id": record["pair_id"], "run": record["run"],
            "returned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw": raw}


def judge(run: int, concurrency: int = CONCURRENCY) -> dict:
    """Run every dispatched pair of `run` that has no raw return yet.

    Idempotent per `(pair_id, run)` for the reason dispatch is: a second return matched
    against the first return's prompt hash would let the provenance columns name a prompt
    that did not produce the label.
    """
    import contextlib

    dispatched = [r for r in read_jsonl(PROMPTS) if r["run"] == run]
    if not dispatched:
        raise FileNotFoundError(f"no run {run} in {PROMPTS} — `--dispatch --run {run}` first")

    on_disk = {r["agent_sha256"] for r in dispatched}
    if on_disk != {sha(AGENT_DEF.read_text(encoding="utf-8"))}:
        raise JudgeNotIsolated(
            "the committed judge is not the one these prompts were dispatched against; "
            "re-dispatch rather than judging with a different instrument")

    done = {(r["pair_id"], r["run"]) for r in read_jsonl(RAW)}
    todo = [r for r in dispatched if (r["pair_id"], r["run"]) not in done]

    returned = 0
    with contextlib.ExitStack() as stack:
        cwd = judge_sandbox(stack)
        RAW.parent.mkdir(parents=True, exist_ok=True)
        with RAW.open("a", encoding="utf-8") as fh:
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                futures = {pool.submit(judge_one, rec, cwd): rec for rec in todo}
                for future in concurrent.futures.as_completed(futures):
                    rec = futures[future]
                    try:
                        got = future.result()
                    except Exception as exc:                    # noqa: BLE001
                        got = {"pair_id": rec["pair_id"], "run": rec["run"],
                               "returned_at": datetime.now(timezone.utc)
                                              .isoformat(timespec="seconds"),
                               "raw": f"<judge failed: {type(exc).__name__}: {exc}>"}
                    fh.write(json.dumps(got, ensure_ascii=False) + "\n")
                    fh.flush()
                    returned += 1
    return {"run": run, "dispatched": len(dispatched), "already_judged": len(done),
            "judged_now": returned, "concurrency": concurrency, "raw": str(RAW)}


# --- collect ---------------------------------------------------------------

class BadReturn(ValueError):
    pass


def parse_return(raw: str) -> dict:
    """One JSON object, one in-scheme label. Anything else is a `BadReturn`.

    Deliberately strict. A judge that returned prose, or a fourth class, did not do the
    task the guide describes, and averaging its output into a kappa would hide that.
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
    reason = str(obj.get("reason", "")).strip()
    return {"label": label, "reason": reason[:REASON_CHARS],
            "reason_overlong": len(reason) > REASON_CHARS}


def collect() -> dict:
    """Validate the raw returns and write **only** `llm-recheck.csv`."""
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
        try:
            parsed = parse_return(got["raw"])
            ok, label, reason = True, parsed["label"], parsed["reason"]
        except BadReturn as exc:
            ok, label, reason = False, "", f"unparsed: {exc}"
            bad.append(key)
        rows.append({
            "pair_id": sent["pair_id"], "batch": 0, "stratum": "a1_recheck",
            "corpus": "a1", "query_id": sent["query_id"], "doc_id": sent["doc_id"],
            "selection_reason": "a1_recheck",
            "annotator": f"llm:{MODEL}:run{got['run']}",
            "label": label, "shortlist_pick": "", "notes": reason,
            "model": MODEL, "agent_sha256": sent["agent_sha256"],
            "prompt_sha256": sent["prompt_sha256"], "run": got["run"],
            "chars_sent": sent["chars_sent"], "parsed_ok": ok,
            "judged_at": got.get("returned_at", "")})

    frame = pd.DataFrame(rows, columns=list(LLM_COLUMNS))
    RECHECK_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RECHECK_CSV, index=False)
    return {"rows": len(frame), "parsed_ok": int(frame.parsed_ok.sum()),
            "unparsed": len(bad), "unparsed_pairs": bad[:10],
            "runs": sorted(frame.run.unique().tolist()) if len(frame) else [],
            "written": str(RECHECK_CSV)}


# --- agreement -------------------------------------------------------------

def cohen_kappa(a: list[str], b: list[str]) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b, labels=list(session.LABELS)))


def kappa_ci(a: list[str], b: list[str], n_boot: int = 2000,
             seed: int = 0) -> dict:
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
            draws.append(cohen_kappa(sa, sb))
    lo, hi = (np.nanpercentile(draws, [2.5, 97.5]) if draws
              else (float("nan"), float("nan")))
    return {"kappa": cohen_kappa(a, b), "lo": float(lo), "hi": float(hi),
            "n": len(a),
            "agreement": float(np.mean([x == y for x, y in zip(a, b)]))}


def majority(labels: list[str]) -> str | None:
    """The per-pair label across runs. A tie is excluded, and the count is reported."""
    counts = pd.Series(labels).value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return None
    return str(counts.index[0])


def judge_frames(seed: int = 0) -> dict[str, pd.Series]:
    """A1's own label, the human's, and the LLM's — one series each, indexed by pair_id."""
    recheck = judging.a1_recheck(judging.LLM_RECHECK_STRATA, seed)
    recheck["pair_id"] = (recheck.query_id.astype(str) + "__"
                          + recheck.doc_id.astype(str))
    a1 = recheck.set_index("pair_id").a1_label

    judged = session.load_judgements()
    human = judged[(judged.corpus == "a1") & judged.label.notna()]
    human = human.drop_duplicates("pair_id").set_index("pair_id").label

    llm = pd.Series(dtype=str)
    runs: dict[int, pd.Series] = {}
    if RECHECK_CSV.exists():
        frame = pd.read_csv(RECHECK_CSV)
        frame = frame[frame.parsed_ok.astype(str).str.lower().isin(("true", "1"))]
        for run, chunk in frame.groupby("run"):
            runs[int(run)] = chunk.drop_duplicates("pair_id").set_index("pair_id").label
        if len(frame):
            grouped = frame.groupby("pair_id").label.apply(lambda s: majority(list(s)))
            llm = grouped.dropna()
    return {"a1": a1, "human": human, "llm": llm, "_runs": runs}


def report(seed: int = 0) -> dict:
    frames = judge_frames(seed)
    runs = frames.pop("_runs")
    names = ["a1", "human", "llm"]

    def pairwise_over(restrict: set[str] | None) -> dict:
        out = {}
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                common = frames[x].index.intersection(frames[y].index)
                if restrict is not None:
                    common = common.intersection(pd.Index(sorted(restrict)))
                out[f"{x}_vs_{y}"] = kappa_ci(
                    frames[x].loc[common].tolist(), frames[y].loc[common].tolist(),
                    seed=seed)
        return out

    narrow = judging.a1_recheck(judging.RECHECK_STRATA, seed)
    human_50 = set(narrow.query_id.astype(str) + "__" + narrow.doc_id.astype(str))

    pairwise = pairwise_over(None)
    # The human 50 are a strict subset of the LLM draw (`assert_recheck_nests`), so the
    # figures already published against n=50 stay quotable rather than being silently
    # replaced by a wider number computed over pairs no human ever saw.
    pairwise_50 = pairwise_over(human_50)

    self_consistency = {}
    run_ids = sorted(runs)
    for i, x in enumerate(run_ids):
        for y in run_ids[i + 1:]:
            common = runs[x].index.intersection(runs[y].index)
            self_consistency[f"run{x}_vs_run{y}"] = kappa_ci(
                runs[x].loc[common].tolist(), runs[y].loc[common].tolist(), seed=seed)

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
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage": {n: int(len(frames[n])) for n in names} | {
            "all_three": int(len(all_three)),
            "human_50_draw": len(human_50),
            "llm_draw": int(len(frames["a1"]))},
        "pairwise_kappa": pairwise,
        "pairwise_kappa_over_the_human_50": pairwise_50,
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
            } for cls in session.LABELS},
        "per_class_llm_vs_a1": {
            cls: {"n": int((frames["a1"].loc[llm_covered] == cls).sum()),
                  "llm_agrees": float(
                      (frames["llm"].loc[llm_covered][
                          frames["a1"].loc[llm_covered] == cls] == cls).mean())
                  if (frames["a1"].loc[llm_covered] == cls).any() else float("nan")}
            for cls in session.LABELS},
        "disagreements": disagreements,
        "note_not_ground_truth": (
            "The LLM is a third judge with its own bias, known to exist and not measured "
            "here. It breaks a tie between two accounts of the A1 labels; it does not "
            "adjudicate one correct (D33)."),
        "note_not_reproducible": (
            "No seed reproduces a subagent run. llm-recheck.csv is collected data: "
            "append-only, provenance-stamped by agent_sha256 and prompt_sha256, and "
            "never subject to a determinism check."),
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
    ap.add_argument("--judge", action="store_true",
                    help="run the committed judge over the dispatched prompts")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.agent:
        AGENT_DEF.parent.mkdir(parents=True, exist_ok=True)
        AGENT_DEF.write_text(render_agent_definition(), encoding="utf-8")
        print(f"wrote {AGENT_DEF}  sha256={sha(render_agent_definition())[:12]}")
    if args.dispatch:
        print(json.dumps(dispatch(args.run, args.seed, args.subsample), indent=2))
    if args.judge:
        print(json.dumps(judge(args.run, args.concurrency), indent=2))
    if args.collect:
        print(json.dumps(collect(), indent=2))
    if args.report:
        out = report(args.seed)
        print(json.dumps({k: v for k, v in out.items()
                          if not k.startswith("note_") and k != "disagreements"},
                         indent=2))
        print(f"{len(out['disagreements'])} pairs where the three judges differ "
              f"-> {REPORT}")
    if not any((args.agent, args.dispatch, args.judge, args.collect, args.report)):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
