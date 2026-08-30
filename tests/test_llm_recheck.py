"""Regression tests for the LLM recheck *(D33)*.

Each is one named invariant, and each was shown to fail against the code it guards — the
mutation is named in the docstring, per AGENTS.md testing rule 2. Everything here runs on
inline synthetic frames: no `data/`, no network, and **no subagent is ever spawned**.
"""
from __future__ import annotations

import json

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
    prompts, raw, out = (tmp_path / "p.jsonl", tmp_path / "r.jsonl",
                         tmp_path / "llm-recheck.csv")
    dispatched = dispatched or [{
        "pair_id": "j_0__r_00", "query_id": "j_0", "doc_id": "r_00", "run": 1,
        "prompt": "JOB DESCRIPTION\nx\n\nCANDIDATE\ny", "prompt_sha256": "abc",
        "chars_sent": 32, "agent_sha256": "def", "model": llm.MODEL}]
    prompts.write_text("\n".join(json.dumps(d) for d in dispatched) + "\n")
    raw.write_text("\n".join(json.dumps(r) for r in raw_rows) + "\n")
    monkeypatch.setattr(llm, "PROMPTS", prompts)
    monkeypatch.setattr(llm, "RAW", raw)
    monkeypatch.setattr(llm, "RECHECK_CSV", out)
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
    monkeypatch.setattr(llm.session, "load_units", lambda seed=0: frame)
    monkeypatch.setattr("candidate_screener.annotation.ui.corpus_text", lambda: (jd, cv))

    first = llm.dispatch(run=1)
    second = llm.dispatch(run=1)
    assert first["pairs"] == len(frame) and first["already_dispatched"] == 0
    assert second["pairs"] == 0 and second["already_dispatched"] == len(frame)
    assert len(llm.read_jsonl(llm.PROMPTS)) == len(frame)
