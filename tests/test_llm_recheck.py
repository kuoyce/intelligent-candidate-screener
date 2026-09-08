"""Regression tests for the LLM recheck *(D33)*.

Each is one named invariant, and each was shown to fail against the code it guards — the
mutation is named in the docstring, per AGENTS.md testing rule 2. Everything here runs on
inline synthetic frames: no `data/`, no network, and **no subagent is ever spawned**.
"""
from __future__ import annotations

import json
import subprocess

import pandas as pd
import pytest

from candidate_screener.annotation import llm_recheck as llm
from candidate_screener.annotation import session

GUIDE = """# Guide

## What you are being asked
Pick one of three.

## Worked examples
A sketch.

## How the pairs reach you
Run the UI with --annotator.

### Screens with fewer candidates
Second opinion.

## What not to consider
Company prestige.

## Double labelling and adjudication
30% double-labelled.
"""


def units(n_jds: int = 2, per_jd: int = 3) -> pd.DataFrame:
    rows = []
    for j in range(n_jds):
        for c in range(per_jd):
            rows.append({"pair_id": f"j_{j}__r_{j}{c}", "jd_id": f"j_{j}",
                         "cv_id": f"r_{j}{c}", "query_id": f"j_{j}",
                         "doc_id": f"r_{j}{c}", "batch": 0, "stratum": "a1_recheck",
                         "corpus": "a1", "selection_reason": "a1_recheck"})
    return pd.DataFrame(rows)


def texts(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    jd = pd.Series({j: f"job description {j}" for j in frame.jd_id.unique()})
    cv = pd.Series({c: f"candidate resume {c}" for c in frame.cv_id.unique()})
    return jd, cv


# --- blinding --------------------------------------------------------------

def test_prompt_carries_nothing_that_names_the_answer():
    """Mutation: append `a1_label: Good Fit` to `render_prompt`'s output — this fails."""
    prompt = llm.render_prompt("a job description", "a resume")
    llm.assert_not_anchoring(prompt, "synthetic")
    for field in llm.ANCHORING_FIELDS:
        assert field not in prompt


def test_anchoring_assertion_actually_catches_a_leak():
    with pytest.raises(AssertionError, match="anchors"):
        llm.assert_not_anchoring("JOB\nx\n\na1_label: Good Fit", "planted")


def test_agent_definition_is_tool_free():
    """The blinding control itself.

    `judging-queue-full.parquet` carries `a1_label` for all 50 recheck pairs and
    `fit/test.parquet` is its source, so a judge holding `Read` or `Bash` reaches the
    answer key in one command. Mutation: change the frontmatter to `tools: Read` — this
    fails.
    """
    front = llm.AGENT_DEF.read_text(encoding="utf-8").split("---")[1]
    tools = [ln for ln in front.splitlines() if ln.startswith("tools:")]
    assert tools == ["tools: []"], (
        f"the judge must hold no tools; frontmatter says {tools}")


def test_committed_judge_is_what_the_guide_renders():
    """Mutation: edit `docs/annotation-guide.md` without re-running `--agent`."""
    assert llm.agent_definition_is_current(), (
        "`.claude/agents/a1-judge.md` is stale against the guide — re-render with "
        "`--agent`, or a row's agent_sha256 names an instrument that never judged it")


def test_operator_sections_are_dropped_and_the_instrument_is_kept():
    stripped = llm.strip_operator_sections(GUIDE)
    assert "Run the UI with --annotator" not in stripped
    assert "Second opinion" not in stripped, "a `###` under a dropped `##` must go too"
    assert "30% double-labelled" not in stripped
    assert "Company prestige" in stripped, "the what-not-to-consider list is the instrument"
    assert "Pick one of three." in stripped


# --- dispatch coverage -----------------------------------------------------

def test_every_pair_is_served_regardless_of_existing_judgements():
    """Mutation: drop the `judgements` argument and read `session.load_judgements()`.

    A fresh annotator is owed only the pairs `outstanding()` allows — 20 of the 50, once
    one person has labelled. The judge must see all of them or the comparison is computed
    on a different set from the human's.
    """
    frame = units()
    jd, cv = texts(frame)
    served = llm.a1_groups(frame, jd, cv)
    assert sum(len(g.candidates) for g in served) == len(frame)

    already = pd.DataFrame([
        {"pair_id": p, "annotator": "yc", "label": "No Fit", "corpus": "a1"}
        for p in frame.pair_id[:4]],
        columns=list(llm.judging.JUDGEMENT_COLUMNS))
    fewer = llm.a1_groups(frame, jd, cv, judgements=already)
    assert sum(len(g.candidates) for g in fewer) < len(frame), (
        "with a real judgements frame `serve_group` withholds pairs — which is exactly "
        "what passing an empty one is for")


def test_served_groups_never_ask_the_shortlist_question():
    frame = units()
    jd, cv = texts(frame)
    assert all(not g.asks_shortlist for g in llm.a1_groups(frame, jd, cv))


def test_subsample_is_seeded_and_stable():
    ids = [f"j_0__r_{i}" for i in range(50)]
    assert llm.subsample_ids(ids, 20, 0) == llm.subsample_ids(ids, 20, 0)
    assert len(llm.subsample_ids(ids, 20, 0)) == 20
    assert llm.subsample_ids(ids, 20, 0) != llm.subsample_ids(ids, 20, 1)


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "not json at all",
    '{"reason": "no label key"}',
    '{"label": "Great Fit", "reason": "a fourth class"}',
    '{"label": "", "reason": "empty"}',
])
def test_bad_returns_are_refused(raw):
    """Mutation: widen `parse_return` to accept any string — this fails.

    A judge that returned prose, or a fourth class, did not do the task the guide
    describes, and averaging its output into a kappa would hide that.
    """
    with pytest.raises(llm.BadReturn):
        llm.parse_return(raw)


def test_good_return_parses_fenced_or_bare():
    for raw in ('{"label": "Good Fit", "reason": "stack matches"}',
                '```json\n{"label": "Good Fit", "reason": "stack matches"}\n```'):
        assert llm.parse_return(raw)["label"] == "Good Fit"


def test_overlong_reason_is_truncated_and_flagged():
    parsed = llm.parse_return(json.dumps({"label": "No Fit", "reason": "x" * 400}))
    assert len(parsed["reason"]) == llm.REASON_CHARS
    assert parsed["reason_overlong"]


# --- collect ---------------------------------------------------------------

def _stage(tmp_path, monkeypatch, raw_rows, dispatched=None):
    prompts, raw = tmp_path / "p.jsonl", tmp_path / "r.jsonl"
    out, full = tmp_path / "llm-recheck.csv", tmp_path / "llm-recheck-full-a1.csv"
    dispatched = dispatched or [{
        "pair_id": "j_0__r_00", "query_id": "j_0", "doc_id": "r_00", "run": 1,
        "draw": llm.RECHECK_DRAW,
        "prompt": "JOB DESCRIPTION\nx\n\nCANDIDATE\ny", "prompt_sha256": "abc",
        "chars_sent": 32, "agent_sha256": "def", "model": llm.MODEL}]
    prompts.write_text("\n".join(json.dumps(d) for d in dispatched) + "\n")
    raw.write_text("\n".join(json.dumps(r) for r in raw_rows) + "\n")
    monkeypatch.setattr(llm, "PROMPTS", prompts)
    monkeypatch.setattr(llm, "RAW", raw)
    monkeypatch.setattr(llm, "RECHECK_CSV", out)
    monkeypatch.setattr(llm, "FULL_A1_CSV", full)
    monkeypatch.setattr(llm, "DRAW_FILES",
                        {llm.RECHECK_DRAW: out, llm.FULL_A1_DRAW: full})
    monkeypatch.setattr(llm.judging, "JUDGEMENTS", tmp_path / "judgements.csv")
    return out, tmp_path / "judgements.csv"


def test_collect_never_writes_the_human_judgement_file(tmp_path, monkeypatch):
    """Mutation: point the writer at `queue.JUDGEMENTS` — this fails.

    `annotator` is a free string, so an `llm` row in `judgements.csv` would pass every
    existing check and then be silently pooled into the human kappa.
    """
    out, human = _stage(tmp_path, monkeypatch, [
        {"pair_id": "j_0__r_00", "run": 1,
         "raw": '{"label": "No Fit", "reason": "wrong field"}',
         "returned_at": "2026-08-30T00:00:00Z"}])
    result = llm.collect()
    assert out.exists() and result["rows"] == 1
    assert not human.exists(), "the human judgement file must not be touched"
    row = pd.read_csv(out).iloc[0]
    assert row.annotator == f"llm:{llm.MODEL}:run1"
    assert row.corpus == "a1" and row.stratum == "a1_recheck"


def test_collect_carries_the_provenance_of_the_prompt_that_was_sent(tmp_path, monkeypatch):
    """Mutation: write a fresh sha of the *response* instead of the dispatched prompt."""
    out, _ = _stage(tmp_path, monkeypatch, [
        {"pair_id": "j_0__r_00", "run": 1, "raw": '{"label": "Good Fit", "reason": "ok"}',
         "returned_at": "2026-08-30T00:00:00Z"}])
    llm.collect()
    row = pd.read_csv(out).iloc[0]
    assert row.prompt_sha256 == "abc" and row.agent_sha256 == "def"
    assert row.chars_sent == 32


def test_collect_refuses_a_pair_that_was_never_dispatched(tmp_path, monkeypatch):
    _stage(tmp_path, monkeypatch, [
        {"pair_id": "j_9__r_99", "run": 1, "raw": '{"label": "No Fit", "reason": "x"}'}])
    with pytest.raises(llm.BadReturn, match="never dispatched"):
        llm.collect()


def test_unparsed_return_is_excluded_and_counted(tmp_path, monkeypatch):
    out, _ = _stage(tmp_path, monkeypatch, [
        {"pair_id": "j_0__r_00", "run": 1, "raw": "I would say this is a good fit."}])
    result = llm.collect()
    assert result["unparsed"] == 1 and result["parsed_ok"] == 0
    assert pd.read_csv(out).iloc[0].parsed_ok in (False, "False")


def test_collect_routes_each_draw_to_its_own_file(tmp_path, monkeypatch):
    """Mutation: write every row to `RECHECK_CSV` — this fails *(D35)*.

    `llm-recheck.csv` is the file the instrument is validated from: `judge_frames` reads
    it and nothing else. Option D judges a different population with the same judge, so a
    row of its 659 in that file is not a wider sample of the recheck, it is a different
    measurement wearing the recheck's provenance. The scope check caught it only because
    559 of the pairs happened to fall outside the draw; the 100 that fell inside it were
    invisible.
    """
    dispatched = [
        {"pair_id": "j_0__r_00", "query_id": "j_0", "doc_id": "r_00", "run": 1,
         "draw": llm.RECHECK_DRAW, "prompt": "p", "prompt_sha256": "abc",
         "chars_sent": 1, "agent_sha256": "def", "model": llm.MODEL},
        {"pair_id": "j_9__r_99", "query_id": "j_9", "doc_id": "r_99", "run": 2,
         "draw": llm.FULL_A1_DRAW, "prompt": "p", "prompt_sha256": "ghi",
         "chars_sent": 1, "agent_sha256": "def", "model": llm.MODEL}]
    out, _ = _stage(tmp_path, monkeypatch, [
        {"pair_id": "j_0__r_00", "run": 1, "raw": '{"label": "No Fit", "reason": "a"}'},
        {"pair_id": "j_9__r_99", "run": 2, "raw": '{"label": "Good Fit", "reason": "b"}'},
    ], dispatched=dispatched)
    full = llm.FULL_A1_CSV

    result = llm.collect()
    assert result["rows"] == 2
    recheck_rows, full_rows = pd.read_csv(out), pd.read_csv(full)
    assert list(recheck_rows.pair_id) == ["j_0__r_00"], \
        "an option-D row reached the file judge_frames reads"
    assert list(full_rows.pair_id) == ["j_9__r_99"]
    assert set(recheck_rows.draw) == {llm.RECHECK_DRAW}
    assert set(full_rows.draw) == {llm.FULL_A1_DRAW}


def test_collect_refuses_a_prompt_record_with_no_draw(tmp_path, monkeypatch):
    """Mutation: default `draw` to `RECHECK_DRAW` when the record lacks one — this fails.

    Pre-D35 prompts carry no draw and a run number does not imply one: run 2 held both.
    A default would refile option D as a recheck on the first `--collect` after a clone.
    """
    _stage(tmp_path, monkeypatch,
           [{"pair_id": "j_0__r_00", "run": 1,
             "raw": '{"label": "No Fit", "reason": "a"}'}],
           dispatched=[{"pair_id": "j_0__r_00", "query_id": "j_0", "doc_id": "r_00",
                        "run": 1, "prompt": "p", "prompt_sha256": "abc", "chars_sent": 1,
                        "agent_sha256": "def", "model": llm.MODEL}])
    with pytest.raises(llm.BadReturn, match="backfill-draw"):
        llm.collect()


def test_dispatch_refuses_a_run_number_already_used_by_the_other_draw(
        tmp_path, monkeypatch):
    """Mutation: drop `assert_run_holds_one_draw` — this fails silently, which is the point.

    Dedup keys on `(pair_id, run)`. When option D was dispatched as run 2, the 20 pairs
    D33 had already sent under that number were skipped as "already dispatched" and their
    labels became part of what reads as one 659-pair run. Nothing raised and no count
    looked wrong; `run1_vs_run2` widened from n=20 to n=100 in the committed report.
    """
    frame = units()
    jd, cv = texts(frame)
    prompts = tmp_path / "p.jsonl"
    monkeypatch.setattr(llm, "PROMPTS", prompts)
    monkeypatch.setattr(llm, "assert_agent_definition_current", lambda: "sha")
    monkeypatch.setattr(llm, "assert_recheck_nests", lambda seed=0: None)
    monkeypatch.setattr(llm.session, "load_units", lambda seed=0, strata=None: frame)
    monkeypatch.setattr("candidate_screener.annotation.ui.corpus_text",
                        lambda strata_key=None: (jd, cv))

    llm.dispatch(run=2)
    assert {r["draw"] for r in llm.read_jsonl(prompts)} == {llm.RECHECK_DRAW}

    a1 = pd.DataFrame({"jd_id": ["j_0"], "resume_id": ["r_zz"],
                       "job_description_text": ["a job"], "resume_text": ["a cv"]})
    monkeypatch.setattr(llm.pd, "read_parquet", lambda *a, **k: a1)
    with pytest.raises(llm.MixedDraw, match="one draw"):
        llm.dispatch_all_a1(run=2)

    # And the other direction: a recheck may not join a run option D holds.
    monkeypatch.setattr(llm, "PROMPTS", tmp_path / "q.jsonl")
    llm.dispatch_all_a1(run=3)
    with pytest.raises(llm.MixedDraw, match="one draw"):
        llm.dispatch(run=3)


def test_dispatch_refuses_a_pre_d35_run_rather_than_guessing_its_draw(
        tmp_path, monkeypatch):
    """Mutation: treat a missing `draw` as the dispatching draw — this fails.

    A legacy run carries no draw, and run 2 proves the run number does not supply one.
    Dispatching onto it would silently adopt whichever draw asked first.
    """
    frame = units()
    jd, cv = texts(frame)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps(
        {"pair_id": "j_0__r_00", "query_id": "j_0", "doc_id": "r_00", "run": 2,
         "prompt": "p", "prompt_sha256": "abc", "chars_sent": 1,
         "agent_sha256": "sha", "model": llm.MODEL}) + "\n")
    monkeypatch.setattr(llm, "PROMPTS", prompts)
    monkeypatch.setattr(llm, "assert_agent_definition_current", lambda: "sha")
    monkeypatch.setattr(llm, "assert_recheck_nests", lambda seed=0: None)
    monkeypatch.setattr(llm.session, "load_units", lambda seed=0, strata=None: frame)
    monkeypatch.setattr("candidate_screener.annotation.ui.corpus_text",
                        lambda strata_key=None: (jd, cv))
    with pytest.raises(llm.MixedDraw, match="backfill-draw"):
        llm.dispatch(run=2)


def test_judge_frames_never_reads_the_full_a1_file(tmp_path, monkeypatch):
    """Mutation: union the two CSVs in `judge_frames` — this fails *(D35)*.

    Option D judged the recheck's own 100 pairs too, so a union does not merely add rows:
    it turns self-consistency into a comparison between one draw and another, and moves
    the `llm` series with labels no recheck run produced. The synthetic below makes the
    option-D labels the opposite of the recheck's, so a union is visible in the figure.
    """
    pairs = ["j1__r1", "j2__r2", "j3__r3"]
    agent = "a" * 64

    def rows(run, label, draw):
        return [{"pair_id": p, "query_id": p.split("__")[0], "doc_id": p.split("__")[1],
                 "run": run, "label": label, "agent_sha256": agent, "model": "m",
                 "parsed_ok": True, "draw": draw} for p in pairs]

    recheck_csv = tmp_path / "llm-recheck.csv"
    full_csv = tmp_path / "llm-recheck-full-a1.csv"
    pd.DataFrame(rows(1, "Good Fit", llm.RECHECK_DRAW)).to_csv(recheck_csv, index=False)
    pd.DataFrame(rows(2, "No Fit", llm.FULL_A1_DRAW)).to_csv(full_csv, index=False)

    agent_def = tmp_path / "a1-judge.md"
    agent_def.write_text("the judge", encoding="utf-8")
    monkeypatch.setattr(llm, "RECHECK_CSV", recheck_csv)
    monkeypatch.setattr(llm, "FULL_A1_CSV", full_csv)
    monkeypatch.setattr(llm, "AGENT_DEF", agent_def)
    monkeypatch.setattr(llm, "sha", lambda text: agent)
    monkeypatch.setattr(llm.judging, "a1_recheck", lambda strata, seed=0: pd.DataFrame(
        {"query_id": [p.split("__")[0] for p in pairs],
         "doc_id": [p.split("__")[1] for p in pairs],
         "a1_label": ["Good Fit"] * len(pairs)}))
    monkeypatch.setattr(llm.session, "load_judgements", lambda: pd.DataFrame(
        columns=["pair_id", "corpus", "label"]))

    frames = llm.judge_frames(0)
    assert frames["llm"].tolist() == ["Good Fit"] * 3, \
        "option-D labels reached the `llm` series"
    assert sorted(frames["_runs"]) == [1], "run 2 is option D and is not a recheck sitting"
    assert frames["_instruments"][agent]["rows"] == len(pairs)


# --- agreement -------------------------------------------------------------

def test_kappa_is_zero_at_chance_and_one_at_identity():
    labels = list(session.LABELS) * 10
    assert llm.kappa_ci(labels, labels)["kappa"] == pytest.approx(1.0)
    shifted = labels[1:] + labels[:1]
    assert llm.kappa_ci(labels, shifted)["kappa"] == pytest.approx(-0.5)


def test_kappa_reports_its_n_and_its_agreement():
    a = ["Good Fit"] * 6 + ["No Fit"] * 4
    b = ["Good Fit"] * 5 + ["No Fit"] * 5
    out = llm.kappa_ci(a, b)
    assert out["n"] == 10 and out["agreement"] == pytest.approx(0.9)
    assert out["lo"] <= out["kappa"] <= out["hi"]


def test_a_tied_majority_is_excluded_rather_than_broken():
    """Mutation: return `counts.index[0]` unconditionally — this fails.

    Breaking a 1-1 tie by value-count order would silently prefer whichever label sorts
    first, which is a bias with no reason behind it.
    """
    assert llm.majority(["Good Fit", "No Fit"]) is None
    assert llm.majority(["Good Fit", "No Fit", "No Fit"]) == "No Fit"


def test_dispatch_is_idempotent_per_pair_and_run(tmp_path, monkeypatch):
    """Mutation: drop the `already` filter — the second call re-appends all 6.

    Appending the same (pair, run) twice would let one spawn's answer be matched against
    the other's prompt hash, and the provenance columns would then name a prompt that did
    not produce the label.
    """
    frame = units()
    jd, cv = texts(frame)
    monkeypatch.setattr(llm, "PROMPTS", tmp_path / "p.jsonl")
    monkeypatch.setattr(llm, "assert_agent_definition_current", lambda: "sha")
    monkeypatch.setattr(llm, "assert_recheck_nests", lambda seed=0: None)
    monkeypatch.setattr(llm.session, "load_units",
                        lambda seed=0, strata=None: frame)
    monkeypatch.setattr("candidate_screener.annotation.ui.corpus_text",
                        lambda strata_key=None: (jd, cv))

    first = llm.dispatch(run=1)
    second = llm.dispatch(run=1)
    assert first["pairs"] == len(frame) and first["already_dispatched"] == 0
    assert second["pairs"] == 0 and second["already_dispatched"] == len(frame)
    assert len(llm.read_jsonl(llm.PROMPTS)) == len(frame)


# --- judge -----------------------------------------------------------------

def _sandbox(tmp_path):
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / llm.AGENT_DEF.name).write_text(
        llm.AGENT_DEF.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_the_judge_refuses_a_working_directory_that_carries_project_context(tmp_path):
    """Mutation: make `assert_isolated` a no-op — this fails.

    Measured, not assumed. Probing the committed judge from the repository root answers
    *yes* to "is AGENTS.md in your context"; from an isolated directory it answers *no*.
    `AGENTS.md` names the recheck, the kappa it is chasing and the parquet that carries
    `a1_label`, so a judge that has read it is not the instrument `yc` was.
    """
    cwd = _sandbox(tmp_path / "clean")
    llm.assert_isolated(cwd)                      # the sandbox itself is fine

    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\n")
    with pytest.raises(llm.JudgeNotIsolated, match="CLAUDE.md"):
        llm.assert_isolated(cwd)                  # ... until an ancestor carries context


def test_the_judge_refuses_a_sandbox_carrying_any_second_agent(tmp_path):
    """A `.claude/` holding anything but the committed judge is not a sandbox."""
    cwd = _sandbox(tmp_path)
    (cwd / ".claude" / "agents" / "helper.md").write_text("---\nname: helper\n---\n")
    with pytest.raises(llm.JudgeNotIsolated, match="helper.md"):
        llm.assert_isolated(cwd)


def test_the_repository_root_is_never_a_legal_judge_cwd():
    """The one directory this must refuse is the one it is most convenient to use."""
    with pytest.raises(llm.JudgeNotIsolated):
        llm.assert_isolated(llm.PROJECT_ROOT)


def _staged_judge(tmp_path, monkeypatch, returns):
    """`judge()` with the subprocess replaced; nothing spawns, nothing leaves the box."""
    agent_sha = llm.sha(llm.AGENT_DEF.read_text(encoding="utf-8"))
    prompts, raw = tmp_path / "p.jsonl", tmp_path / "r.jsonl"
    prompts.write_text("\n".join(json.dumps({
        "pair_id": f"j_0__r_{i:02d}", "query_id": "j_0", "doc_id": f"r_{i:02d}",
        "run": 1, "prompt": "JOB DESCRIPTION\nx\n\nCANDIDATE\ny",
        "prompt_sha256": "abc", "chars_sent": 32, "agent_sha256": agent_sha,
        "model": llm.MODEL}) for i in range(3)) + "\n")
    monkeypatch.setattr(llm, "PROMPTS", prompts)
    monkeypatch.setattr(llm, "RAW", raw)
    # F1 gave `judge()` a second sink. Unpatched it defaults to the real data directory,
    # where a test's quarantine rows would then be counted by `run_coverage`.
    monkeypatch.setattr(llm, "FAILED", tmp_path / "f.jsonl")
    calls = []

    def fake(record, cwd):
        llm.assert_isolated(cwd)        # every spawn is checked, not just the first
        calls.append(record["pair_id"])
        return {"pair_id": record["pair_id"], "run": record["run"],
                "returned_at": "2026-08-30T00:00:00Z", "raw": returns}

    monkeypatch.setattr(llm, "judge_one", fake)
    return raw, calls


def test_judging_is_idempotent_per_pair_and_run(tmp_path, monkeypatch):
    """Mutation: drop the `done` filter — this fails.

    A second return matched against the first return's `prompt_sha256` would let the
    provenance columns name a prompt that did not produce the label. Dispatch is
    idempotent for this reason; so is judging.
    """
    raw, calls = _staged_judge(tmp_path, monkeypatch, '{"label":"No Fit","reason":"x"}')
    first = llm.judge(run=1, concurrency=2)
    second = llm.judge(run=1, concurrency=2)
    assert first["judged_now"] == 3 and second["judged_now"] == 0
    assert second["already_judged"] == 3
    assert len(calls) == 3 and len(llm.read_jsonl(raw)) == 3


def test_judging_refuses_prompts_dispatched_against_a_different_judge(
        tmp_path, monkeypatch):
    """Mutation: drop the `agent_sha256` comparison — this fails.

    The prompts carry the hash of the judge they were rendered for. Judging them with a
    re-rendered agent would stamp every row with provenance that never held.
    """
    raw, _ = _staged_judge(tmp_path, monkeypatch, '{"label":"No Fit","reason":"x"}')
    rows = llm.read_jsonl(llm.PROMPTS)
    for row in rows:
        row["agent_sha256"] = "0" * 64
    llm.PROMPTS.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(llm.JudgeNotIsolated, match="not the one these prompts"):
        llm.judge(run=1, concurrency=2)
    assert not raw.exists()


def test_a_judge_that_crashes_is_recorded_but_not_as_a_return(tmp_path, monkeypatch):
    """A missing pair and a failed pair are different findings; neither is silent.

    Rewritten by fix F1 (plan/2026-09-06-a2-llm-prelabel). This test previously asserted
    the bug: it required the crash to land in `RAW` as `<judge failed: ...>` and counted
    it in `judged_now`. `judge()` resumes from `RAW`, so that made a crashed pair
    permanently un-retryable while the run reported itself complete — and the figure was
    then computed over whatever survived. The finding is still recorded, in the quarantine
    file, where it does not masquerade as an answer.
    """
    raw, _ = _staged_judge(tmp_path, monkeypatch, '{"label":"No Fit","reason":"x"}')

    def boom(record, cwd):
        raise TimeoutError("judge took too long")

    monkeypatch.setattr(llm, "judge_one", boom)
    with pytest.raises(llm.JudgeFailures) as caught:
        llm.judge(run=1, concurrency=2)

    assert caught.value.summary["judged_now"] == 0
    assert caught.value.summary["failed"] == 3
    assert llm.read_jsonl(raw) == [], "a crash was filed as a return"
    quarantined = llm.read_jsonl(tmp_path / "f.jsonl")
    assert len(quarantined) == 3
    assert all("judge took too long" in q["detail"] for q in quarantined)


# --- the widened LLM leg ---------------------------------------------------

#: The two draws are a property of A1's test split, so the nesting test needs it on disk.
#: The assertion that *guards* the property is exercised without it, just below.
needs_fit_split = pytest.mark.skipif(
    not (llm.PROCESSED / "fit" / "test.parquet").exists(),
    reason="data/processed/fit/test.parquet not built — run `python -m "
           "candidate_screener.data.build --task fit-split --seed 0`.")


@needs_fit_split
def test_the_llm_draw_contains_the_human_draw():
    """Human and machine legs draw the same 100 pairs (raised from 50 after the
    first sitting completed, 30 Aug 2026). The nesting assertion is trivially true."""
    llm.assert_recheck_nests(0)

    human = llm.judging.a1_recheck(llm.judging.RECHECK_STRATA, 0)
    machine = llm.judging.a1_recheck(llm.judging.LLM_RECHECK_STRATA, 0)
    ids = lambda f: set(f.query_id.astype(str) + "__" + f.doc_id.astype(str))  # noqa: E731
    assert len(ids(human)) == 100 and len(ids(machine)) == 100
    assert ids(human) == ids(machine)


def test_a_diverged_draw_is_refused_rather_than_quoted(monkeypatch):
    """The assertion earns its place: give it two draws that do not nest.

    With both legs at 100, the divergence is a *different* 100 rather than a
    size mismatch — achieved by making the fake return different ids depending
    on a hash of the strata dict.
    """
    import pandas as pd

    def fake(strata, seed=0):
        offset = hash(tuple(sorted(strata.items()))) % 1000   # different draw
        return pd.DataFrame([{"query_id": f"j_{offset + i}", "doc_id": f"r_{offset + i}",
                              "a1_label": "No Fit"}
                             for i in range(sum(strata.values()))])

    # Make the two strata dicts differ so they produce different draws
    monkeypatch.setattr(llm.judging, "RECHECK_STRATA",
                        {"Good Fit": 30, "Potential Fit": 30, "No Fit": 40})
    monkeypatch.setattr(llm.judging, "LLM_RECHECK_STRATA",
                        {"Good Fit": 31, "Potential Fit": 30, "No Fit": 40})
    monkeypatch.setattr(llm.judging, "a1_recheck", fake)
    with pytest.raises(AssertionError, match="absent from the LLM"):
        llm.assert_recheck_nests(0)


@needs_fit_split
def test_both_legs_draw_the_same_100_pairs():
    """Human and machine legs are now equal (both 100). `load_units` defaults to
    RECHECK_STRATA; passing LLM_RECHECK_STRATA explicitly must produce the same draw."""
    units_default = llm.session.load_units(0)
    units_explicit = llm.session.load_units(0, llm.judging.LLM_RECHECK_STRATA)
    assert (units_default.corpus == "a1").sum() == 100
    assert (units_explicit.corpus == "a1").sum() == 100
    default_ids = set(units_default[units_default.corpus == "a1"].pair_id)
    explicit_ids = set(units_explicit[units_explicit.corpus == "a1"].pair_id)
    assert default_ids == explicit_ids


def test_the_report_still_quotes_the_human_50_alongside_the_wider_draw(
        tmp_path, monkeypatch):
    """Mutation: drop `pairwise_kappa_over_the_human_50` — this fails.

    A wider kappa computed over pairs no human ever saw must not silently replace the
    figure the card, the catalog and AGENTS.md already quote. Synthetic: 4 pairs in the
    narrow draw, 6 in the wide one, and only the narrow ones carry a human label.
    """
    import pandas as pd

    wide = [f"j{i}__r{i}" for i in range(6)]
    narrow = wide[:4]
    a1 = pd.Series(["Good Fit", "No Fit", "Good Fit", "No Fit", "Good Fit", "No Fit"],
                   index=wide)
    llm_labels = pd.Series(["Good Fit", "No Fit", "No Fit", "No Fit", "No Fit", "Good Fit"],
                           index=wide)
    human = pd.Series(["Good Fit", "No Fit", "Good Fit", "No Fit"], index=narrow)

    monkeypatch.setattr(llm, "judge_frames", lambda seed=0: {
        "a1": a1, "human": human, "llm": llm_labels, "_runs": {}})
    monkeypatch.setattr(llm.judging, "a1_recheck", lambda strata, seed=0: pd.DataFrame(
        {"query_id": [p.split("__")[0] for p in narrow],
         "doc_id": [p.split("__")[1] for p in narrow]}))
    monkeypatch.setattr(llm, "REPORT", tmp_path / "report.json")

    out = llm.report(0)
    assert "pairwise_kappa_over_the_human_50" in out
    assert out["coverage"]["human_50_draw"] == len(narrow)
    assert out["coverage"]["llm_draw"] == len(wide)
    assert out["pairwise_kappa"]["a1_vs_llm"]["n"] == len(wide)
    assert out["pairwise_kappa_over_the_human_50"]["a1_vs_llm"]["n"] == len(narrow)
    # the wider per-class block sees every pair the judge covered, not the human subset
    assert out["per_class_llm_vs_a1"]["Good Fit"]["n"] == 3
    assert out["per_class_agreement"]["Good Fit"]["n"] == 2


@needs_fit_split
def test_the_text_lookup_covers_every_pair_the_llm_draw_serves():
    """Both legs are now 100 and the default strata match, so `corpus_text()` with
    no argument covers every pair in the draw."""
    from candidate_screener.annotation import ui

    strata = llm.judging.LLM_RECHECK_STRATA
    units = llm.session.load_units(0, strata)
    jd_text, cv_text = ui.corpus_text(tuple(sorted(strata.items())))
    a1 = units[units.corpus == "a1"]
    assert not set(a1.jd_id) - set(jd_text.index), "JDs in the draw with no text"
    assert not set(a1.cv_id) - set(cv_text.index), "CVs in the draw with no text"

    default_jd, _ = ui.corpus_text()
    assert not set(a1.jd_id) - set(default_jd.index), \
        "default draw must cover the same pairs now that both legs are 100"


def test_judge_frames_never_pools_two_instruments(tmp_path, monkeypatch):
    """Mutation: take the majority across every run rather than the current agent's —
    this fails.

    A guide edit is a new judge. Its `agent_sha256` changes, and `report()` attributes
    the `llm` series to whatever agent is committed. Pooling runs across instruments
    would blend the old judge's labels into that series and call the blend the new
    judge's measurement — the same failure as an `llm` row in `judgements.csv`, one
    level up, and just as invisible: no exception, no row-count change, a plausible
    kappa.

    Synthetic: two instruments, opposite answers on the same 3 pairs. The committed
    agent is the second one, so the series must be the second one's labels alone.
    """
    import pandas as pd

    agent_old, agent_new = "a" * 64, "b" * 64
    pairs = ["j1__r1", "j2__r2", "j3__r3"]
    rows = []
    for agent, run, label in ((agent_old, 1, "No Fit"), (agent_old, 2, "No Fit"),
                              (agent_new, 3, "Good Fit")):
        for pair in pairs:
            rows.append({"pair_id": pair, "query_id": pair.split("__")[0],
                         "doc_id": pair.split("__")[1], "run": run, "label": label,
                         "agent_sha256": agent, "model": "m", "parsed_ok": True})
    csv = tmp_path / "llm-recheck.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    agent_def = tmp_path / "a1-judge.md"
    agent_def.write_text("new judge", encoding="utf-8")
    monkeypatch.setattr(llm, "RECHECK_CSV", csv)
    monkeypatch.setattr(llm, "AGENT_DEF", agent_def)
    monkeypatch.setattr(llm, "sha", lambda text: agent_new if text == "new judge"
                        else agent_old)
    monkeypatch.setattr(llm.judging, "a1_recheck", lambda strata, seed=0: pd.DataFrame(
        {"query_id": [p.split("__")[0] for p in pairs],
         "doc_id": [p.split("__")[1] for p in pairs],
         "a1_label": ["Good Fit"] * len(pairs)}))
    monkeypatch.setattr(llm.session, "load_judgements", lambda: pd.DataFrame(
        columns=["pair_id", "corpus", "label"]))

    frames = llm.judge_frames(0)
    assert frames["llm"].tolist() == ["Good Fit"] * 3, (
        "the `llm` series carried the old instrument's labels — 2 of the 3 runs on file "
        "belong to a judge that is no longer committed")
    assert set(frames["_instruments"]) == {agent_old, agent_new}
    assert frames["_instruments"][agent_new]["is_current"] is True
    assert frames["_instruments"][agent_old]["is_current"] is False


# --- interruption safety (F1-F3) -------------------------------------------
#
# A credit exhaustion mid-run used to be recorded as a finished pair: `judge_one` turned a
# non-zero exit into a `<judge exited 1: ...>` string, wrote it to `RAW`, and `judge()`
# builds its resume set from `RAW`. The pair was never retried, the run reported
# `remaining: 0`, and the report quoted a kappa over whichever pairs answered before the
# wall — completion-latency order, not a random sample.


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _stage_judge(tmp_path, monkeypatch, n_pairs, outcomes):
    """`n_pairs` dispatched on run 1; `outcomes` maps pair_id -> _Proc or Exception."""
    prompts = tmp_path / "p.jsonl"
    records = [{"pair_id": f"j_0__r_{i:02d}", "query_id": "j_0", "doc_id": f"r_{i:02d}",
                "run": 1, "draw": llm.RECHECK_DRAW, "prompt": f"pair {i}",
                "prompt_sha256": "abc", "chars_sent": 10,
                "agent_sha256": "sha", "model": llm.MODEL} for i in range(n_pairs)]
    prompts.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    monkeypatch.setattr(llm, "PROMPTS", prompts)
    monkeypatch.setattr(llm, "RAW", tmp_path / "r.jsonl")
    monkeypatch.setattr(llm, "FAILED", tmp_path / "f.jsonl")
    monkeypatch.setattr(llm, "AGENT_DEF", tmp_path / "agent.md")
    (tmp_path / "agent.md").write_text("judge", encoding="utf-8")
    monkeypatch.setattr(llm, "sha", lambda text: "sha")
    monkeypatch.setattr(llm, "judge_sandbox", lambda stack: tmp_path)

    def fake_run(cmd, **kw):
        pair = f"j_0__r_{kw['input'].split()[1]:0>2}"
        outcome = outcomes(pair) if callable(outcomes) else outcomes[pair]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    return prompts


def test_a_spawn_that_never_answered_is_not_recorded_as_a_return(tmp_path, monkeypatch):
    """Mutation: file the failure in `RAW` instead of `FAILED` — this fails.

    This is fix F1, and the assertion that matters is the second one: the pair must still
    be in `todo` on the next pass. Against the pre-F1 code the failure lands in `RAW`,
    `done` absorbs it, and the re-run judges nothing.
    """
    good = _Proc(0, '{"label":"Good Fit","reason":"ok"}')
    dead = _Proc(1, "", "Credit balance is too low")
    _stage_judge(tmp_path, monkeypatch, 4,
                 lambda p: good if p.endswith(("00", "01")) else dead)

    with pytest.raises(llm.JudgeFailures) as caught:
        llm.judge(1, concurrency=2)
    assert caught.value.summary["failed"] == 2
    assert caught.value.summary["kinds"] == {"quota": 2}

    raw = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert len(raw) == 2, "an unanswered spawn was filed as a return"
    assert all("<judge exited" not in r["raw"] for r in raw)

    # The point of the quarantine: the failed pairs are still owed.
    cov = llm.run_coverage(1)
    assert cov == {"run": 1, "n_dispatched": 4, "n_answered": 2, "n_quarantined": 2,
                   "n_missing": 2, "complete": False}


def test_the_retry_after_topping_up_judges_exactly_the_failed_pairs(
        tmp_path, monkeypatch):
    """The recovery procedure works: top up, re-run `--judge`, nothing else."""
    good = _Proc(0, '{"label":"No Fit","reason":"ok"}')
    dead = _Proc(1, "", "Credit balance is too low")
    _stage_judge(tmp_path, monkeypatch, 4,
                 lambda p: good if p.endswith(("00", "01")) else dead)
    with pytest.raises(llm.JudgeFailures):
        llm.judge(1, concurrency=2)

    monkeypatch.setattr(llm.subprocess, "run", lambda cmd, **kw: good)
    summary = llm.judge(1, concurrency=2)
    assert summary["attempted"] == 2, (
        "the retry did not pick up exactly the two pairs that failed")
    assert summary["failed"] == 0
    assert llm.run_coverage(1)["complete"] is True


def test_a_timeout_is_infrastructure_not_a_bad_answer(tmp_path, monkeypatch):
    _stage_judge(tmp_path, monkeypatch, 1, lambda p: subprocess.TimeoutExpired("claude", 1))
    with pytest.raises(llm.JudgeFailures) as caught:
        llm.judge(1, concurrency=1)
    assert caught.value.summary["kinds"] == {"timeout": 1}
    assert not (tmp_path / "r.jsonl").exists() or not (
        tmp_path / "r.jsonl").read_text().strip()


def test_a_clean_exit_with_junk_on_stdout_is_still_data(tmp_path, monkeypatch):
    """F1 must not swallow a judge that answered badly — that is a real finding.

    `parse_return` files it `parsed_ok=False`, which is what a misbehaving judge should
    look like. Only spawns that never ran are quarantined.
    """
    _stage_judge(tmp_path, monkeypatch, 1, lambda p: _Proc(0, "I cannot judge this."))
    summary = llm.judge(1, concurrency=1)
    assert summary["failed"] == 0 and summary["judged_now"] == 1
    raw = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert raw[0]["raw"] == "I cannot judge this."
    with pytest.raises(llm.BadReturn):
        llm.parse_return(raw[0]["raw"])


def test_judge_raises_rather_than_returning_a_summary_that_reads_as_success(
        tmp_path, monkeypatch):
    """Mutation: return the summary instead of raising — this fails.

    Fix F2. Every downstream step reads a short run as success, so the operator has to
    learn about it here.
    """
    _stage_judge(tmp_path, monkeypatch, 1, lambda p: _Proc(1, "", "boom"))
    with pytest.raises(llm.JudgeFailures) as caught:
        llm.judge(1, concurrency=1)
    assert "re-running" in str(caught.value)
    assert str(tmp_path / "f.jsonl") in str(caught.value)


def test_no_figure_may_be_quoted_over_a_run_that_is_short(tmp_path, monkeypatch):
    """Fix F3's refusal: a kappa over the survivors is a kappa over the fastest."""
    _stage_judge(tmp_path, monkeypatch, 3,
                 lambda p: _Proc(0, '{"label":"Good Fit","reason":"ok"}')
                 if p.endswith("00") else _Proc(1, "", "usage limit reached"))
    with pytest.raises(llm.JudgeFailures):
        llm.judge(1, concurrency=1)
    with pytest.raises(llm.JudgeFailures):
        llm.assert_run_complete(1)


# --- the in-domain draws (plan/2026-09-06-a2-llm-prelabel) ------------------

def _indomain(monkeypatch, n_generic=2, n_targeted=2):
    """A tiny two-stratum in-domain frame, in `indomain_rows()`'s shape."""
    rows = []
    for i in range(n_generic):
        rows.append({"corpus": "a2", "batch": 1, "stratum": "generic",
                     "query_id": f"g_jd_{i}", "doc_id": f"g_cv_{i}",
                     "selection_reason": "indomain_banded",
                     "query_text": f"generic jd {i}", "candidate_text": f"generic cv {i}",
                     "a1_label": None})
    for i in range(n_targeted):
        rows.append({"corpus": "a2", "batch": 2, "stratum": "targeted",
                     "query_id": f"t_jd_{i}", "doc_id": f"t_cv_{i}",
                     "selection_reason": "indomain_banded",
                     "query_text": f"targeted jd {i}", "candidate_text": f"targeted cv {i}",
                     "a1_label": None})
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(llm.judging, "indomain_rows", lambda: frame.copy())
    return frame


def _stage_a2_dispatch(tmp_path, monkeypatch):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("")
    monkeypatch.setattr(llm, "PROMPTS", prompts)
    monkeypatch.setattr(llm, "assert_agent_definition_current", lambda: "sha")
    _indomain(monkeypatch)
    return prompts


def test_each_a2_stratum_goes_to_its_own_draw_and_file(tmp_path, monkeypatch):
    """D29 composed with D35: two populations, so two draws and two files.

    The average of `generic` and `targeted` describes nothing, and after this the
    pooled figure is not merely forbidden — there is no file holding it.
    """
    prompts = _stage_a2_dispatch(tmp_path, monkeypatch)

    llm.dispatch_a2("generic", run=5)
    llm.dispatch_a2("targeted", run=6)
    sent = llm.read_jsonl(prompts)

    by_draw = {}
    for r in sent:
        by_draw.setdefault(r["draw"], set()).add(r["pair_id"])
    assert set(by_draw) == {llm.A2_GENERIC_DRAW, llm.A2_TARGETED_DRAW}
    assert all(p.startswith("g_") for p in by_draw[llm.A2_GENERIC_DRAW])
    assert all(p.startswith("t_") for p in by_draw[llm.A2_TARGETED_DRAW])
    assert llm.DRAW_FILES[llm.A2_GENERIC_DRAW] != llm.DRAW_FILES[llm.A2_TARGETED_DRAW]
    assert all(r["corpus"] == "a2" for r in sent)


def test_an_a2_draw_may_not_land_on_a_run_another_draw_holds(tmp_path, monkeypatch):
    """D35's guard has to cover the new draws in both directions."""
    prompts = _stage_a2_dispatch(tmp_path, monkeypatch)
    llm.dispatch_a2("generic", run=5)

    with pytest.raises(llm.MixedDraw):
        llm.dispatch_a2("targeted", run=5)

    prompts.write_text(json.dumps({
        "pair_id": "j_0__r_00", "query_id": "j_0", "doc_id": "r_00", "run": 7,
        "draw": llm.RECHECK_DRAW, "prompt": "x", "prompt_sha256": "a",
        "chars_sent": 1, "agent_sha256": "sha", "model": llm.MODEL}) + "\n")
    with pytest.raises(llm.MixedDraw):
        llm.dispatch_a2("generic", run=7)


def test_the_a2_prompt_cannot_carry_the_label_the_human_gave(tmp_path, monkeypatch):
    """The 400 `generic` pairs already have `yc` labels; the judge must not see one.

    Structural, not instructed: `indomain_rows()` joins no judgement, so there is no
    column to leak. This asserts the frame's shape rather than the prompt's text, because
    a prompt-text check passes for the wrong reason once the column exists.
    """
    _stage_a2_dispatch(tmp_path, monkeypatch)
    rows = llm.judging.indomain_rows()
    assert "label" not in rows.columns, "a joined human label would be one line from here"
    assert rows.a1_label.isna().all()

    # The operational form of the same property: the prompt does not move when the
    # judgements file does. The three class names appear in every prompt — they are the
    # output schema the judge is told to use — so their presence proves nothing either
    # way, and this is the assertion that does.
    llm.dispatch_a2("generic", run=5)
    without = [r["prompt_sha256"] for r in llm.read_jsonl(tmp_path / "p.jsonl")]

    judgements = tmp_path / "judgements.csv"
    pd.DataFrame([{
        "pair_id": "g_jd_0__g_cv_0", "batch": 1, "stratum": "generic", "corpus": "a2",
        "query_id": "g_jd_0", "doc_id": "g_cv_0",
        "selection_reason": "indomain_banded", "annotator": "yc", "label": "Good Fit",
        "shortlist_pick": 0, "notes": ""}]).to_csv(judgements, index=False)
    monkeypatch.setattr(llm.judging, "JUDGEMENTS", judgements)
    monkeypatch.setattr(llm, "PROMPTS", tmp_path / "p2.jsonl")

    llm.dispatch_a2("generic", run=5)
    assert [r["prompt_sha256"] for r in llm.read_jsonl(tmp_path / "p2.jsonl")] == without, (
        "the prompt changed when a human label existed — the judge is being anchored")


def test_a2_dispatch_refuses_a_stratum_d29_does_not_define(tmp_path, monkeypatch):
    _stage_a2_dispatch(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not one of"):
        llm.dispatch_a2("everything", run=5)


def test_collect_reproduces_the_a1_identity_for_records_that_predate_a2(
        tmp_path, monkeypatch):
    """`collect` reads identity from the dispatch record now. Runs 1-4 carry none.

    Mutation: drop the defaults — this fails. The two committed A1 files would be
    rewritten with empty corpus/stratum columns on the next `--collect`.
    """
    out, _ = _stage(tmp_path, monkeypatch, [
        {"pair_id": "j_0__r_00", "run": 1,
         "raw": '{"label":"Good Fit","reason":"ok"}'}])
    llm.collect()
    row = pd.read_csv(out).iloc[0]
    assert (row.corpus, row.stratum, row.batch, row.selection_reason) == (
        "a1", "a1_recheck", 0, "a1_recheck")


def test_collect_files_an_a2_row_as_a2(tmp_path, monkeypatch):
    generic = tmp_path / "llm-a2-generic.csv"
    _stage(tmp_path, monkeypatch, [
        {"pair_id": "g_jd_0__g_cv_0", "run": 5,
         "raw": '{"label":"No Fit","reason":"ok"}'}],
        dispatched=[{
            "pair_id": "g_jd_0__g_cv_0", "query_id": "g_jd_0", "doc_id": "g_cv_0",
            "run": 5, "draw": llm.A2_GENERIC_DRAW, "corpus": "a2", "batch": 1,
            "stratum": "generic", "selection_reason": "indomain_banded",
            "prompt": "x", "prompt_sha256": "abc", "chars_sent": 1,
            "agent_sha256": "def", "model": llm.MODEL}])
    monkeypatch.setattr(llm, "DRAW_FILES", {llm.A2_GENERIC_DRAW: generic})
    llm.collect()
    row = pd.read_csv(generic).iloc[0]
    assert (row.corpus, row.stratum, row.batch, row.draw) == (
        "a2", "generic", 1, llm.A2_GENERIC_DRAW)


def test_no_a2_figure_may_be_asked_for_over_both_strata(tmp_path, monkeypatch):
    """D29's refusal, expressed in the API rather than in a comment."""
    with pytest.raises(ValueError, match="never pooled"):
        llm.a2_frames("all")
    with pytest.raises(ValueError, match="never pooled"):
        llm.a2_frames("generic+targeted")


def test_judge_frames_is_untouched_by_the_existence_of_a2_files(tmp_path, monkeypatch):
    """The test that protects the committed D33/D34 figures.

    `judge_frames` reads `RECHECK_CSV` and nothing else. If an A2 file could reach it,
    every published kappa would move the first time Phase 0 collected.
    """
    agent = "a" * 64
    recheck = tmp_path / "llm-recheck.csv"
    pd.DataFrame([{
        "pair_id": "j_0__r_00", "batch": 0, "stratum": "a1_recheck", "corpus": "a1",
        "query_id": "j_0", "doc_id": "r_00", "selection_reason": "a1_recheck",
        "annotator": "llm:m:run1", "label": "Good Fit", "shortlist_pick": "",
        "notes": "", "model": "m", "agent_sha256": agent, "prompt_sha256": "p",
        "run": 1, "chars_sent": 1, "parsed_ok": True, "judged_at": "",
        "draw": llm.RECHECK_DRAW}]).to_csv(recheck, index=False)

    agent_def = tmp_path / "a1-judge.md"
    agent_def.write_text("judge", encoding="utf-8")
    monkeypatch.setattr(llm, "RECHECK_CSV", recheck)
    monkeypatch.setattr(llm, "AGENT_DEF", agent_def)
    monkeypatch.setattr(llm, "sha", lambda text: agent)
    monkeypatch.setattr(llm.judging, "a1_recheck", lambda strata, seed=0: pd.DataFrame(
        {"query_id": ["j_0"], "doc_id": ["r_00"], "a1_label": ["Good Fit"]}))
    monkeypatch.setattr(llm.session, "load_judgements", lambda: pd.DataFrame(
        columns=["pair_id", "corpus", "label"]))

    before = llm.judge_frames(0)["llm"].to_dict()

    # Now put A2 rows on disk, in both in-domain files.
    for draw in (llm.A2_GENERIC_DRAW, llm.A2_TARGETED_DRAW):
        path = tmp_path / f"{draw}.csv"
        pd.DataFrame([{
            "pair_id": f"{draw}_jd__cv", "batch": 1,
            "stratum": llm.A2_DRAW_STRATUM[draw], "corpus": "a2",
            "query_id": f"{draw}_jd", "doc_id": "cv",
            "selection_reason": "indomain_banded", "annotator": "llm:m:run5",
            "label": "No Fit", "shortlist_pick": "", "notes": "", "model": "m",
            "agent_sha256": agent, "prompt_sha256": "p", "run": 5, "chars_sent": 1,
            "parsed_ok": True, "judged_at": "", "draw": draw}]).to_csv(path, index=False)
        monkeypatch.setitem(llm.DRAW_FILES, draw, path)

    assert llm.judge_frames(0)["llm"].to_dict() == before, (
        "an A2 row reached judge_frames — every published A1 kappa would move")


def test_per_class_agreement_catches_a_kappa_carried_by_one_class(tmp_path):
    """A1-vs-llm scored 0.90 / 0.27 / 0.067 per class and still came out positive.

    The gate's second clause exists because of that, and this is what evaluates it.
    """
    human = ["No Fit"] * 8 + ["Good Fit"] * 2
    judge = ["No Fit"] * 8 + ["No Fit"] * 2
    out = llm.per_class_agreement(human, judge)
    assert out["No Fit"] == {"n": 8, "agreement": 1.0}
    assert out["Good Fit"] == {"n": 2, "agreement": 0.0}
    assert out["Potential Fit"]["n"] == 0


def test_a_record_is_torn_only_by_a_newline(tmp_path):
    """Mutation: read with `str.splitlines()` — this fails.

    Prompts are written `ensure_ascii=False`, so a document carrying U+2028 (Djinni CVs
    carry 16) puts a `splitlines()` boundary *inside* a JSON string. Both callers that
    matter read from here: dispatch dedups on `(pair_id, run)`, and `judge`'s resume set
    decides which pairs are still owed.
    """
    path = tmp_path / "p.jsonl"
    records = [{"pair_id": "a__b", "run": 5, "prompt": "line one line two"},
               {"pair_id": "c__d", "run": 5, "prompt": "plain"}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                    encoding="utf-8")

    assert len(path.read_text(encoding="utf-8").splitlines()) == 3, (
        "the fixture must actually contain a splitlines-only separator")
    assert llm.read_jsonl(path) == records


def test_a_pair_the_judge_has_labelled_is_still_owed_by_the_human(tmp_path, monkeypatch):
    """Task 1.2's guard. Mutation: union the A2 files into `judged_pair_ids()` — fails.

    Phase 1 dispatches the judge over the 200 `targeted` pairs *before* their human
    session, and the whole value of that ordering is that the two legs are independent.
    If the judge's output ever suppressed human work, the pairs the human labelled would
    be exactly the ones the judge had not reached — and the prospective comparison would
    be over a set selected by the judge's own throughput.

    It holds structurally today: `outstanding()` counts rows in `judgements.csv`, and the
    judge never writes there. This locks that in.
    """
    frame = _indomain(monkeypatch, n_generic=0, n_targeted=6)
    targeted = frame[frame.stratum == "targeted"]
    ids = [f"{r.query_id}__{r.doc_id}" for r in targeted.itertuples()]

    # The human file must EXIST and be non-empty, or `judged_pair_ids` returns early and
    # a test over it proves nothing. One unrelated pair is enough to get past that.
    judgements = tmp_path / "judgements.csv"
    pd.DataFrame([{
        "pair_id": "someone__else", "batch": 1, "stratum": "generic", "corpus": "a2",
        "query_id": "someone", "doc_id": "else", "selection_reason": "indomain_banded",
        "annotator": "yc", "label": "No Fit", "shortlist_pick": 0, "notes": ""}
    ]).to_csv(judgements, index=False)
    monkeypatch.setattr(llm.judging, "JUDGEMENTS", judgements)

    machine_file = tmp_path / "llm-a2-targeted.csv"
    pd.DataFrame([{
        "pair_id": p, "batch": 2, "stratum": "targeted", "corpus": "a2",
        "query_id": p.split("__")[0], "doc_id": p.split("__")[1],
        "selection_reason": "indomain_banded", "annotator": "llm:m:run6",
        "label": "No Fit", "shortlist_pick": "", "notes": "", "model": "m",
        "agent_sha256": "sha", "prompt_sha256": "p", "run": 6, "chars_sent": 1,
        "parsed_ok": True, "judged_at": "", "draw": llm.A2_TARGETED_DRAW}
        for p in ids]).to_csv(machine_file, index=False)
    monkeypatch.setitem(llm.DRAW_FILES, llm.A2_TARGETED_DRAW, machine_file)
    monkeypatch.setattr(llm.judging, "a1_recheck",
                        lambda strata, seed=0: pd.DataFrame(columns=list(frame.columns)))

    assert llm.judging.judged_pair_ids() == {"someone__else"}, (
        "the judge's labels reached the human resume mechanism")

    # `build_queue` is the consumer that matters: it drops judged pairs, and it is what
    # the annotator's UI is served from.
    queued = set(llm.judging.build_queue(seed=0).pair_id)
    assert queued == set(ids), (
        "the judge's labels suppressed human work — the prospective leg would then be "
        "measured over whichever pairs the judge happened not to reach")


# --- two annotators (Phase 2) ----------------------------------------------

def _stage_a2_scoring(tmp_path, monkeypatch, judgements, llm_labels,
                      stratum="generic", run=5, model="m", extra_llm=()):
    """One A2 stratum on disk: human rows, judge rows, and a complete dispatch record.

    `extra_llm` adds further judging legs as `(run, model, {pair_id: label})`, which is
    how a second model of the same committed guide is staged. A leg may carry two more
    elements — `agent_sha256` and `{pair_id: score}` — which is how a *second instrument*
    is staged beside the committed one, the axis above model.
    """
    draw = next(d for d, s in llm.A2_DRAW_STRATUM.items() if s == stratum)
    agent = "a" * 64
    agent_def = tmp_path / "a1-judge.md"
    agent_def.write_text("judge", encoding="utf-8")
    monkeypatch.setattr(llm, "AGENT_DEF", agent_def)
    monkeypatch.setattr(llm, "sha", lambda text: agent)

    legs = [(leg + (agent, {}))[:5] if len(leg) == 3 else (leg + ({},))[:5]
            for leg in [(run, model, llm_labels), *extra_llm]]
    path = tmp_path / f"{draw}.csv"
    pd.DataFrame([{
        "pair_id": p, "batch": 1, "stratum": stratum, "corpus": "a2",
        "query_id": p.split("__")[0], "doc_id": p.split("__")[1],
        "selection_reason": "indomain_banded", "annotator": f"llm:{mod}:run{r}",
        "label": lab, "shortlist_pick": "", "notes": "", "model": mod,
        "agent_sha256": ag, "prompt_sha256": "p", "run": r, "chars_sent": 1,
        "parsed_ok": True, "judged_at": "", "draw": draw, "score": sc.get(p)}
        for r, mod, labels, ag, sc in legs
        for p, lab in labels.items()]).to_csv(path, index=False)
    monkeypatch.setitem(llm.DRAW_FILES, draw, path)

    prompts = tmp_path / "p.jsonl"
    prompts.write_text("\n".join(json.dumps({
        "pair_id": p, "run": r, "draw": draw, "model": mod})
        for r, mod, labels, _, _ in legs for p in labels) + "\n", encoding="utf-8")
    monkeypatch.setattr(llm, "PROMPTS", prompts)
    raw = tmp_path / "r.jsonl"
    raw.write_text("\n".join(json.dumps({"pair_id": p, "run": r})
                             for r, _, labels, _, _ in legs for p in labels) + "\n",
                   encoding="utf-8")
    monkeypatch.setattr(llm, "RAW", raw)

    monkeypatch.setattr(llm.session, "load_judgements", lambda: pd.DataFrame(judgements))
    monkeypatch.setattr(llm, "A2_REPORT", tmp_path / "a2.json")
    return path


def test_a2_frames_keeps_the_two_annotators_apart(tmp_path, monkeypatch):
    """Mutation: `drop_duplicates("pair_id")` over all annotators — the code before this.

    That version returned one series called `human`, built by keeping whichever row came
    first in `judgements.csv`. On the real file it resolved to all-`yc` on `generic` and
    all-`van` on `targeted`, so the same field meant a different person per stratum and
    said so nowhere. Row order is not a rule, and there is no reading of that series that
    is a statement about a population.
    """
    _stage_a2_scoring(tmp_path, monkeypatch, judgements=[
        {"pair_id": "j__c1", "corpus": "a2", "stratum": "generic",
         "annotator": "yc", "label": "Good Fit"},
        {"pair_id": "j__c1", "corpus": "a2", "stratum": "generic",
         "annotator": "van", "label": "No Fit"},
        {"pair_id": "j__c2", "corpus": "a2", "stratum": "generic",
         "annotator": "yc", "label": "No Fit"},
    ], llm_labels={"j__c1": "Good Fit", "j__c2": "No Fit"})

    frames = llm.a2_frames("generic")
    assert "human" not in frames, "a pooled human series is the bug, not a convenience"
    assert set(frames["humans"]) == {"yc", "van"}
    assert frames["humans"]["yc"]["j__c1"] == "Good Fit"
    assert frames["humans"]["van"]["j__c1"] == "No Fit"
    assert "j__c2" not in frames["humans"]["van"].index


def test_a2_report_names_the_annotator_every_judge_figure_is_against(
        tmp_path, monkeypatch):
    """Mutation: emit one `human_vs_llm` key, as the code did before 7 Sep 2026.

    On the real data that single field is 0.380 or 0.204 depending on which annotator's
    rows happen to come first, under a name AGENTS.md and the plan both read as
    kappa(yc, llm). Here `yc` agrees with the judge on everything and `van` on nothing,
    so a pooled field cannot be right for both — and a report that emits one is claiming
    a figure for a reference it did not use.
    """
    pairs = {f"j__c{i}": ("Good Fit" if i % 2 else "No Fit") for i in range(10)}
    judgements = (
        [{"pair_id": p, "corpus": "a2", "stratum": "generic", "annotator": "yc",
          "label": lab} for p, lab in pairs.items()]
        + [{"pair_id": p, "corpus": "a2", "stratum": "generic", "annotator": "van",
            "label": "Potential Fit"} for p in pairs])
    _stage_a2_scoring(tmp_path, monkeypatch, judgements, llm_labels=pairs)

    section = llm.a2_report(strict=False)["strata"]["generic"]
    assert "human_vs_llm" not in section
    assert set(section["vs_llm"]) == {"yc", "van"}
    assert section["vs_llm"]["yc"]["three_class"]["kappa"] == 1.0
    assert section["vs_llm"]["van"]["three_class"]["kappa"] < 1.0
    assert section["annotators"] == {"van": 10, "yc": 10}


def test_a2_report_carries_the_human_ceiling_beside_every_judge_figure(
        tmp_path, monkeypatch):
    """Mutation: drop the `human_ceiling` block.

    Without it a reader compares kappa(yc, llm) to 1.0, which is the comparison the
    design spec refuses: the gate is *relative*, and its denominator is how much two
    people doing this task agree. On the real `generic` stratum that denominator is
    0.250, below every judge figure in the file — a fact no absolute reading reaches.
    """
    pairs = {f"j__c{i}": ("Good Fit" if i % 2 else "No Fit") for i in range(10)}
    judgements = (
        # `yc` and the judge agree everywhere; `van` agrees with `yc` on half. So the
        # ceiling must come out strictly below the judge figure it sits beside.
        [{"pair_id": p, "corpus": "a2", "stratum": "generic", "annotator": "yc",
          "label": lab} for p, lab in pairs.items()]
        + [{"pair_id": p, "corpus": "a2", "stratum": "generic", "annotator": "van",
            "label": lab if i < 5 else "Potential Fit"}
           for i, (p, lab) in enumerate(pairs.items())])
    _stage_a2_scoring(tmp_path, monkeypatch, judgements, llm_labels=pairs)

    section = llm.a2_report(strict=False)["strata"]["generic"]
    ceiling = section["human_ceiling"]["van_vs_yc"]
    assert ceiling["n"] == 10
    assert "three_class" in ceiling and "binary" in ceiling and "per_class" in ceiling
    assert ceiling["three_class"]["kappa"] < section["vs_llm"]["yc"]["three_class"]["kappa"]


def test_a2_report_says_so_when_two_annotators_share_no_pair(tmp_path, monkeypatch):
    """Mutation: return silently. A stratum with two annotators and no overlap has no
    ceiling, and a judge figure in it cannot be read as a fraction of one. Saying
    nothing leaves the reader to assume the ceiling elsewhere in the file applies.
    """
    judgements = [
        {"pair_id": "j__c0", "corpus": "a2", "stratum": "generic", "annotator": "yc",
         "label": "No Fit"},
        {"pair_id": "j__c1", "corpus": "a2", "stratum": "generic", "annotator": "van",
         "label": "No Fit"}]
    _stage_a2_scoring(tmp_path, monkeypatch, judgements,
                      llm_labels={"j__c0": "No Fit", "j__c1": "No Fit"})

    section = llm.a2_report(strict=False)["strata"]["generic"]
    assert section["human_ceiling"] == {}
    assert "note_no_ceiling" in section


# --- M1/M2: model is an axis (plan/2026-09-08-judge-calibration) -------------

def test_two_models_of_one_guide_are_never_one_series(tmp_path, monkeypatch):
    """Mutation: build `llm` as a majority over every run sharing `agent_sha256`.

    That is the code as it stood on 8 Sep 2026, and it is correct only while there is
    one model on file. A guide edit changes `agent_sha256`; a *model* change does not,
    so run 5 (`haiku`) and run 7 (`sonnet`) of the same committed guide are a
    two-element majority. `majority` returns `None` on a tie and `.dropna()` deletes it,
    so every pair the two models disagreed on vanishes — and what survives is exactly
    the subset two models agreed on. Here they disagree on half the pairs, so the old
    selection returns n=5 of 10: a smaller, easier, non-random sample nobody ordered,
    with no exception and no row count that looks wrong.
    """
    pairs = {f"j__c{i}": "Good Fit" for i in range(10)}
    sonnet = {p: ("Good Fit" if i < 5 else "No Fit")
              for i, p in enumerate(pairs)}
    judgements = [{"pair_id": p, "corpus": "a2", "stratum": "generic",
                   "annotator": "yc", "label": lab} for p, lab in pairs.items()]
    _stage_a2_scoring(tmp_path, monkeypatch, judgements, llm_labels=pairs,
                      run=5, model="claude-haiku-4-5",
                      extra_llm=[(7, "claude-sonnet-5", sonnet)])

    frames = llm.a2_frames("generic")
    assert frames["_model"] == llm.MODEL, (
        "the default is the COMMITTED model, not the most recent run — otherwise "
        "dispatching a control run silently changes what `vs_llm` refers to")
    assert len(frames["llm"]) == 10, (
        "the live series lost pairs — that is the two-model majority dropping every "
        "tie, and the survivors are the pairs the models agreed on")
    assert set(frames["per_model"]) == {"claude-haiku-4-5", "claude-sonnet-5"}
    assert len(frames["per_model"]["claude-haiku-4-5"]) == 10
    assert frames["llm"].tolist() == ["Good Fit"] * 10          # the haiku leg

    # And the caller may ask for the other one by name, rather than by re-deriving it.
    other = llm.a2_frames("generic", model="claude-sonnet-5")["llm"]
    assert other.tolist() == [sonnet[p] for p in other.index]
    with pytest.raises(ValueError, match="no rows under model"):
        llm.a2_frames("generic", model="claude-opus-5")


def test_a_model_comparison_is_not_reported_as_self_consistency(tmp_path, monkeypatch):
    """Mutation: score every pair of runs that share a draw.

    `self_consistency` is the figure D33 used to bound how high kappa(A1, llm) could
    have reached — it means *this judge repeating itself*. Runs 5 and 7 are the same
    guide under two models, which is the model control, and putting it in that field
    would report a between-judge comparison under a name every reader takes for
    test-retest reliability. It is listed instead, with its provenance.
    """
    pairs = {f"j__c{i}": "Good Fit" for i in range(10)}
    sonnet = {p: ("Good Fit" if i < 5 else "No Fit") for i, p in enumerate(pairs)}
    judgements = [{"pair_id": p, "corpus": "a2", "stratum": "generic",
                   "annotator": "yc", "label": lab} for p, lab in pairs.items()]
    _stage_a2_scoring(tmp_path, monkeypatch, judgements, llm_labels=pairs,
                      run=5, model="claude-haiku-4-5",
                      extra_llm=[(7, "claude-sonnet-5", sonnet)])

    section = llm.a2_report(strict=False)["strata"]["generic"]
    assert section["self_consistency"] == {}
    assert "run5_vs_run7" in section["not_self_consistency"]
    assert section["llm_model"] == llm.MODEL
    assert section["models_on_file"] == {"claude-haiku-4-5": 10, "claude-sonnet-5": 10}
    # Each model carries its own complete figure set, and the section-level block is the
    # committed one repeated — never a series built from both.
    assert set(section["by_model"]) == {"claude-haiku-4-5", "claude-sonnet-5"}
    for block in section["by_model"].values():
        assert {"vs_llm", "consensus", "interchangeability"} <= set(block)
    assert (section["consensus"]
            == section["by_model"][llm.MODEL]["consensus"])


def test_a_run_is_judged_by_the_model_its_dispatch_names(tmp_path, monkeypatch):
    """Mutation: `claude -p --agent a1-judge` with no `--model`, as the argv stood.

    The model column was written from the `MODEL` constant at dispatch and never passed
    to the subprocess, so it recorded an assertion rather than a measurement. With one
    model on file that is invisible; with two it is a model control whose second arm
    never ran, wearing the label of the model that was dispatched.
    """
    assert "--model" in llm.judge_argv({"model": "claude-sonnet-5"})
    assert llm.judge_argv({"model": "claude-sonnet-5"})[-1] == "claude-sonnet-5"
    # A pre-M2 record names no model and falls back to the constant it was written under.
    assert llm.judge_argv({})[-1] == llm.MODEL

    prompts = tmp_path / "p.jsonl"
    prompts.write_text("\n".join(json.dumps(
        {"pair_id": f"j__c{i}", "run": 7, "draw": llm.A2_GENERIC_DRAW,
         "model": "claude-sonnet-5"}) for i in range(3)) + "\n", encoding="utf-8")
    monkeypatch.setattr(llm, "PROMPTS", prompts)

    assert llm.assert_run_holds_one_model(7) == "claude-sonnet-5"
    assert llm.assert_run_holds_one_model(7, "claude-sonnet-5") == "claude-sonnet-5"
    with pytest.raises(llm.MixedModel, match="asked for"):
        llm.assert_run_holds_one_model(7, "claude-haiku-4-5")

    # One run, one model: rows carrying two would name a population no figure can.
    with prompts.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pair_id": "j__c9", "run": 7,
                             "draw": llm.A2_GENERIC_DRAW,
                             "model": "claude-haiku-4-5"}) + "\n")
    with pytest.raises(llm.MixedModel, match="dispatched under"):
        llm.assert_run_holds_one_model(7)


# --- Move 1: consensus is a third reference, and an axis (design spec §1) ----

def _consensus_fixture(tmp_path, monkeypatch, yc, van, llm_labels):
    judgements = (
        [{"pair_id": p, "corpus": "a2", "stratum": "generic", "annotator": "yc",
          "label": lab} for p, lab in yc.items()]
        + [{"pair_id": p, "corpus": "a2", "stratum": "generic", "annotator": "van",
            "label": lab} for p, lab in van.items()])
    return _stage_a2_scoring(tmp_path, monkeypatch, judgements, llm_labels)


def test_a_pair_two_annotators_read_differently_is_not_in_the_consensus(
        tmp_path, monkeypatch):
    """Mutation: resolve a disagreement by order, or by a majority of two.

    Neither is a rule. `drop_duplicates` keeps whichever row is first in
    `judgements.csv`, which is file order; `majority` over two raters is a coin toss
    dressed as arithmetic. A pair the two humans read differently has no consensus
    label, and its absence is the whole content of the coverage figure — it is the
    52% of the corpus a substituting judge would be labelling with no reference at
    all (design spec §3).
    """
    yc = {"j__c0": "Good Fit", "j__c1": "No Fit", "j__c2": "Good Fit"}
    van = {"j__c0": "Good Fit", "j__c1": "Potential Fit"}     # c2: single-labelled
    _consensus_fixture(tmp_path, monkeypatch, yc, van,
                       {p: "No Fit" for p in yc})

    frames = llm.a2_frames("generic")
    consensus, coverage = frames["consensus"], frames["consensus_coverage"]
    assert list(consensus.index) == ["j__c0"], "a disagreement is not a consensus label"
    assert consensus["j__c0"] == "Good Fit"
    # A pair only one person saw had no opportunity to disagree and is in neither term.
    assert coverage["n_multi_labelled"] == 2
    assert coverage["n_agreed"] == 1
    assert coverage["coverage"] == 0.5
    assert coverage["annotators"] == ["van", "yc"]


def test_a_consensus_figure_never_travels_without_its_coverage(tmp_path, monkeypatch):
    """Mutation: emit the consensus kappa and drop the coverage beside it.

    kappa(consensus, llm) = 0.401 on `targeted` reads as the judge doing better than
    the 0.249 above it. It is not the same measurement: it is over the 48.5% of pairs
    two annotators agreed on, which is the easy half by construction. A reader shown
    the figure without the fraction has been told something false by omission, so the
    report refuses to build rather than emitting one.
    """
    with pytest.raises(ValueError, match="false by omission"):
        llm.assert_consensus_carries_coverage(
            {"n_agreed": 60, "vs_llm": {}, "note_selection": "x"})
    with pytest.raises(ValueError, match="false by omission"):
        llm.assert_consensus_carries_coverage(
            {"n_agreed": 60, "n_multi_labelled": 126, "coverage": 0.476})

    yc = {f"j__c{i}": "Good Fit" for i in range(6)}
    van = dict(yc)
    _consensus_fixture(tmp_path, monkeypatch, yc, van, dict(yc))

    real = llm.a2_frames

    def stripped(stratum, model=None):
        frames = real(stratum, model)
        frames["consensus_coverage"] = {"annotators": ["van", "yc"]}
        return frames

    monkeypatch.setattr(llm, "a2_frames", stripped)
    with pytest.raises(ValueError, match="false by omission"):
        llm.a2_report(strict=False)


def test_consensus_and_single_annotator_figures_are_not_the_same_field(
        tmp_path, monkeypatch):
    """Mutation: fold consensus into `vs_llm`, or let it overwrite an annotator's row.

    They are computed over different sets of pairs — 400 / 126 / 60 on the real
    `generic` — so one is not an improvement on the other and no field may hold both.
    The reference is a fifth axis beside agent_sha256, draw, stratum and annotator
    (D37), and, like the others, the API cannot express the pooled version.
    """
    yc = {f"j__c{i}": ("Good Fit" if i < 4 else "No Fit") for i in range(8)}
    van = {p: ("Good Fit" if i < 2 else "Potential Fit")
           for i, p in enumerate(yc)}
    _consensus_fixture(tmp_path, monkeypatch, yc, van, dict(yc))

    section = llm.a2_report(strict=False)["strata"]["generic"]
    assert set(section["vs_llm"]) == {"yc", "van"}, "consensus is not an annotator"
    assert "consensus" not in section["vs_llm"]
    consensus = section["consensus"]
    assert consensus["n_agreed"] == 2 and consensus["n_multi_labelled"] == 8
    assert consensus["vs_llm"]["n"] == 2
    assert consensus["vs_llm"]["n"] != section["vs_llm"]["yc"]["n"], (
        "a consensus figure and a single-annotator figure are over different pairs, "
        "and the report has to say so by carrying both n")
    assert "upper" in consensus["note_selection"].lower()
    assert "note_never_pooled_references" in llm.a2_report(strict=False)


def test_the_gate_compares_on_the_pairs_the_three_series_share(tmp_path, monkeypatch):
    """Mutation: read kappa(yc, llm) from the whole-stratum block.

    That is the n=400 figure, 0.380, and the ceiling beside it is measured on the 126
    pairs `van` also labelled. Reading one against the other compares two populations
    — and it is not a small effect: restricted to those same 126 pairs the judge
    scores 0.455, not 0.380. Here `yc` labels every pair and `van` only half, so a
    gate that did not restrict would put an n=8 judge figure beside an n=4 ceiling.
    """
    yc = {f"j__c{i}": ("Good Fit" if i % 2 else "No Fit") for i in range(8)}
    van = {p: yc[p] for i, p in enumerate(yc) if i < 4}
    _consensus_fixture(tmp_path, monkeypatch, yc, van, dict(yc))

    gate = llm.a2_report(strict=False)["strata"]["generic"]["interchangeability"]
    assert gate["n"] == 4, "the gate is over the pairs all three series share"
    for row in gate["rows"].values():
        assert row["n"] == 4
        assert row["human_pair"]["three_class"]["n"] == 4
        assert row["judge"]["three_class"]["n"] == 4, (
            "the judge column is measured on a different population from the human "
            "column it is read against")
    assert gate["rows"]["yc"]["replaced"] == "van"
    assert gate["band"] == llm.INTERCHANGEABILITY_BAND


def test_a_class_the_humans_miss_is_unidentified_not_failed_by_the_judge(
        tmp_path, monkeypatch):
    """Mutation: apply the 0.40 per-class bar to the judge alone, as the old gate did.

    That gate refuses the judge on `Good Fit` at 0.119 — while the two humans hold
    that class between themselves at 0.338 / 0.300, which fails the same bar. A bar no
    human pair can clear is not measuring the judge, and reporting it as a judge
    failure attributes to the model what belongs to the guide (design spec §4).
    """
    yc = {f"j__c{i}": ("Good Fit" if i < 4 else "No Fit") for i in range(8)}
    # `van` calls three of `yc`'s four Good Fits Potential: the class is drawn in two
    # places, and `van` recovers 25% of `yc`'s instances of it — below the 0.40 bar the
    # old gate applied to the judge alone.
    van = {p: ("Potential Fit" if i in (0, 1, 2) else yc[p]) for i, p in enumerate(yc)}
    _consensus_fixture(tmp_path, monkeypatch, yc, van, dict(yc))

    per_class = (llm.a2_report(strict=False)["strata"]["generic"]
                 ["interchangeability"]["per_class"])
    assert per_class["Good Fit"]["unidentified_by_instrument"] is True
    assert per_class["No Fit"]["unidentified_by_instrument"] is False
    assert per_class["Good Fit"]["bar"] == llm.PER_CLASS_BAR
    assert set(per_class["Good Fit"]["human_pair"]) == {"yc", "van"}, (
        "per-class agreement is conditioned on a reference, so the human pair gives "
        "two numbers and the class fails if either does")


# --- Move 3: the graded instrument (design spec §6) --------------------------

def test_the_prompt_asks_for_the_scheme_the_agent_definition_asks_for():
    """Mutation: leave `render_prompt` naming `label` and `reason` alone — this fails.

    The guide and the per-pair message are two places the scheme is written and they are
    one edit. A prompt asking for a label alone under a guide that asks for a score
    leaves the judge to pick which instruction to obey, and `parse_return` refuses
    whichever it did not pick — 600 spawns spent on a contradiction this module wrote.
    """
    prompt = llm.render_prompt("jd", "cv")
    agent = llm.render_agent_definition(GUIDE)
    assert '"score"' in prompt, "the per-pair message does not ask for the score"
    assert f"<{llm.SCORE_MIN}-{llm.SCORE_MAX} integer>" in prompt
    assert '"score"' in agent, "the agent definition does not ask for the score"


def test_a_score_outside_the_scale_is_not_a_score():
    """Mutation: `int(obj["score"])` — this fails on every case below.

    A cut point fitted over a column holding `120`, `"high"` or `85.5` describes no
    instrument, and each of those coerces or truncates into something plausible.
    """
    for bad in ("high", 120, -1, 85.5, None, True, ""):
        with pytest.raises(llm.BadReturn):
            llm.parse_score(bad)
    assert llm.parse_score(85) == 85
    assert llm.parse_score("85") == 85
    assert llm.parse_score(85.0) == 85, "JSON has one number type; 85.0 is a whole number"


def test_a_graded_run_has_no_parsed_row_without_a_score(tmp_path, monkeypatch):
    """Mutation: store `score = obj.get("score")` and keep `parsed_ok = True`.

    That version files a row whose class is usable and whose score is null, and every
    later mean skips it in silence over a sample nobody chose — the F1 shape one column
    along. A return that did not answer the question it was asked is not a partial
    answer, it is an unparsed return.
    """
    dispatched = [
        {"pair_id": "j_0__r_00", "query_id": "j_0", "doc_id": "r_00", "run": 9,
         "draw": llm.RECHECK_DRAW, "prompt": "p", "prompt_sha256": "abc",
         "chars_sent": 1, "agent_sha256": "def", "model": llm.MODEL,
         "response_scheme": llm.SCHEME_SCORED},
        {"pair_id": "j_0__r_01", "query_id": "j_0", "doc_id": "r_01", "run": 9,
         "draw": llm.RECHECK_DRAW, "prompt": "p", "prompt_sha256": "abc",
         "chars_sent": 1, "agent_sha256": "def", "model": llm.MODEL,
         "response_scheme": llm.SCHEME_SCORED}]
    out, _ = _stage(tmp_path, monkeypatch, [
        {"pair_id": "j_0__r_00", "run": 9,
         "raw": '{"label": "Good Fit", "reason": "a"}'},
        {"pair_id": "j_0__r_01", "run": 9,
         "raw": '{"label": "Good Fit", "score": 88, "reason": "b"}'},
    ], dispatched=dispatched)

    result = llm.collect()
    assert result["unparsed"] == 1 and result["scored"] == 1
    rows = pd.read_csv(out).set_index("pair_id")
    assert rows.loc["j_0__r_00"].parsed_ok in (False, "False"), (
        "a scored run's return with no score was filed as a usable label with a null "
        "score — which no later figure would report as missing")
    assert rows.loc["j_0__r_01"].score == 88


def test_a_label_only_run_is_not_failed_for_a_field_it_was_never_asked_for(
        tmp_path, monkeypatch):
    """Mutation: require a score of every return, from this module's constant.

    Runs 1-8 were dispatched under `SCHEME_LABEL` and answered it completely. Reading
    the requirement off today's constant rather than off the dispatch record would mark
    all 1,679 of their rows `parsed_ok=False` on the next `--collect`, and the frame
    builders drop unparsed rows — so every committed D33/D34/A2 figure would recompute
    over nothing at all. A row is parsed under the scheme it was *asked* for.
    """
    dispatched = [{"pair_id": "j_0__r_00", "query_id": "j_0", "doc_id": "r_00", "run": 4,
                   "draw": llm.RECHECK_DRAW, "prompt": "p", "prompt_sha256": "abc",
                   "chars_sent": 1, "agent_sha256": "def", "model": llm.MODEL}]
    out, _ = _stage(tmp_path, monkeypatch, [
        {"pair_id": "j_0__r_00", "run": 4,
         "raw": '{"label": "Good Fit", "reason": "a"}'}], dispatched=dispatched)

    result = llm.collect()
    assert result["unparsed"] == 0 and result["parsed_ok"] == 1
    row = pd.read_csv(out).iloc[0]
    assert row.parsed_ok in (True, "True")
    assert pd.isna(row.score), "a pre-move-3 row gained a score no judge produced"


def test_a_superseded_instrument_keeps_its_own_key_rather_than_vanishing(
        tmp_path, monkeypatch):
    """Mutation: report only rows whose `agent_sha256` is the committed one.

    That is the builder as it stood before move 3, and it is correct only while there
    has never been a guide edit on this corpus. Move 3's edit gives the judge a new
    hash, so on the commit that changes the guide runs 5-8 would drop out of
    `llm-a2-report.json` entirely — not superseded, *absent*, which a reader cannot tell
    from withdrawn. `by_instrument` keys them separately and no figure spans two keys.
    """
    old_agent = "b" * 64
    pairs = {f"j__c{i}": "Good Fit" for i in range(10)}
    graded = {p: ("Good Fit" if i < 5 else "No Fit") for i, p in enumerate(pairs)}
    scores = {p: 90 if i < 5 else 10 for i, p in enumerate(pairs)}
    judgements = [{"pair_id": p, "corpus": "a2", "stratum": "generic",
                   "annotator": "yc", "label": lab} for p, lab in pairs.items()]
    # Leg one is the superseded instrument; leg two is the committed one, graded.
    path = _stage_a2_scoring(tmp_path, monkeypatch, judgements, llm_labels=pairs,
                             run=5, model=llm.MODEL,
                             extra_llm=[(9, llm.MODEL, graded, "a" * 64, scores)])
    # `_stage_a2_scoring` stamps a leg with no agent of its own with the committed hash;
    # run 5 is the superseded instrument, so move it back.
    frame = pd.read_csv(path)
    frame.loc[frame.run == 5, "agent_sha256"] = old_agent
    frame.to_csv(path, index=False)

    frames = llm.a2_frames("generic")
    assert set(frames["per_instrument"]) == {old_agent, "a" * 64}, (
        "an instrument on file is missing from the report — a superseded judge is not "
        "a withdrawn one")
    assert frames["_current"] == "a" * 64
    assert frames["llm"].tolist() == [graded[p] for p in frames["llm"].index], (
        "the live series is not the committed instrument's")
    assert len(frames["per_instrument"][old_agent][llm.MODEL]) == 10
    assert frames["scores"]["a" * 64][llm.MODEL].tolist() == [
        float(scores[p]) for p in frames["scores"]["a" * 64][llm.MODEL].index]
    assert old_agent not in frames["scores"], "a pre-move-3 instrument has no scores"

    report = llm.a2_report(strict=False)
    section = report["strata"]["generic"]
    assert set(section["by_instrument"]) == {old_agent, "a" * 64}
    assert set(section["by_model"]) == {llm.MODEL}, (
        "`by_model` is the committed instrument's models, never a pool of two guides")
    assert section["instruments_on_file"][old_agent]["committed"] is False
    assert section["instruments_on_file"]["a" * 64]["committed"] is True


def test_a_fitted_cut_point_names_the_pairs_it_was_fitted_on():
    """Mutation: return `{"cut_points": ..., "kappa": ...}` with no `fitted_on`.

    A cut point is a fitted parameter and its kappa on the pairs it was fitted on cannot
    miss. Quoted without saying what it was fitted on, it reads as a measurement of the
    judge — the one thing it is not — and it would sit in the same report beside the
    emitted label's genuinely out-of-sample kappa with nothing to tell them apart. The
    rule is `assert_consensus_carries_coverage`'s, one parameter along.
    """
    scores = pd.Series({f"p{i}": i * 10 for i in range(10)})
    reference = pd.Series({f"p{i}": ("No Fit" if i < 4 else
                                     "Potential Fit" if i < 7 else "Good Fit")
                           for i in range(10)})
    fitted = llm.fit_cut_points(scores, reference)
    assert fitted["fitted_on"] == sorted(scores.index), (
        "a fitted cut point does not name the pairs it was fitted on")
    assert fitted["kappa"]["kappa"] == pytest.approx(1.0), (
        "a monotone score should be reproducible by two thresholds exactly")
    assert "note_fitted" in fitted

    with pytest.raises(AssertionError, match="fitted_on"):
        llm.assert_score_figure_names_its_fit(
            {"cut_points": {}, "kappa": {}, "n": 10})
    with pytest.raises(AssertionError, match="fitted_on"):
        llm.assert_score_figure_names_its_fit(
            {"cut_points": {}, "kappa": {}, "n": 10, "fitted_on": []})


def test_a_held_out_score_figure_is_not_fitted_on_the_pairs_it_is_quoted_on():
    """Mutation: fit on every pair and quote the same kappa as `held_out`.

    S3 of the registration is the only out-of-sample number move 3 produces, and it is
    registered in advance precisely so a wide interval on half the pairs cannot
    afterwards be replaced by the full-data fit, which is the number that cannot miss.
    The two halves are disjoint by construction, and the code raises rather than emits if
    they ever stop being.
    """
    scores = pd.Series({f"p{i:02d}": (i * 5) % 101 for i in range(40)})
    reference = pd.Series({p: ("No Fit" if v < 35 else
                               "Potential Fit" if v < 70 else "Good Fit")
                           for p, v in scores.items()})
    analysis = llm.score_analysis(scores, reference)
    held = analysis["held_out"]
    assert not set(held["fitted_on"]) & set(held["quoted_on"]), (
        "the held-out half shares pairs with the fitted half")
    assert held["n_fit"] + held["n"] == len(scores)
    assert analysis["fitted_full"]["n"] == len(scores), (
        "the full-data fit and the held-out figure are two different figures and the "
        "report carries both")
    assert analysis["separation"]["good_vs_potential"]["auc"] == pytest.approx(1.0), (
        "a score that orders the classes perfectly should separate them perfectly")


def test_the_score_block_is_reported_for_the_graded_instrument_alone(
        tmp_path, monkeypatch):
    """Mutation: run `score_analysis` over the committed model's series unconditionally.

    Runs 5-8 carry no score. A block computed for them would be an empty or a
    part-populated figure under an instrument that never emitted one — and `by_model`'s
    reader has no way to tell "this judge scored badly" from "this judge was never asked".
    """
    old_agent = "b" * 64
    pairs = {f"j__c{i}": "Good Fit" for i in range(12)}
    graded = {p: ("Good Fit" if i < 6 else "No Fit") for i, p in enumerate(pairs)}
    scores = {p: 90 if i < 6 else 10 for i, p in enumerate(pairs)}
    judgements = [{"pair_id": p, "corpus": "a2", "stratum": "generic",
                   "annotator": who, "label": lab}
                  for who in ("yc", "van") for p, lab in graded.items()]
    path = _stage_a2_scoring(tmp_path, monkeypatch, judgements, llm_labels=pairs,
                             run=5, model=llm.MODEL,
                             extra_llm=[(9, llm.MODEL, graded, "a" * 64, scores)])
    frame = pd.read_csv(path)
    frame.loc[frame.run == 5, "agent_sha256"] = old_agent
    frame.to_csv(path, index=False)

    section = llm.a2_report(strict=False)["strata"]["generic"]
    assert "score" not in section["by_instrument"][old_agent][llm.MODEL], (
        "an instrument that never emitted a score has a score block")
    graded_block = section["by_instrument"]["a" * 64][llm.MODEL]["score"]
    assert graded_block["n"] > 0 and "fitted_full" in graded_block
    assert "coverage" in graded_block and "note_selection" in graded_block, (
        "a consensus-fitted figure travels with its coverage and its selection note")
