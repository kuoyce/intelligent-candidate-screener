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


def test_a_judge_that_crashes_is_recorded_rather_than_dropped(tmp_path, monkeypatch):
    """A missing pair and a failed pair are different findings; only one is silent."""
    raw, _ = _staged_judge(tmp_path, monkeypatch, '{"label":"No Fit","reason":"x"}')

    def boom(record, cwd):
        raise TimeoutError("judge took too long")

    monkeypatch.setattr(llm, "judge_one", boom)
    result = llm.judge(run=1, concurrency=2)
    assert result["judged_now"] == 3
    assert all("judge failed" in r["raw"] for r in llm.read_jsonl(raw))


# --- the widened LLM leg ---------------------------------------------------

#: The two draws are a property of A1's test split, so the nesting test needs it on disk.
#: The assertion that *guards* the property is exercised without it, just below.
needs_fit_split = pytest.mark.skipif(
    not (llm.PROCESSED / "fit" / "test.parquet").exists(),
    reason="data/processed/fit/test.parquet not built — run `python -m "
           "candidate_screener.data.build --task fit-split --seed 0`.")


@needs_fit_split
def test_the_llm_draw_contains_the_human_draw():
    """Mutation: draw the wide set at a different seed — this fails.

    Real data, not a stub: `a1_recheck` takes `order[:n]` from one permutation per
    stratum, so raising `n` appends. Every published n=50 figure is quoted beside a
    figure over the wider draw, which is honest only while the narrow draw nests.
    """
    llm.assert_recheck_nests(0)          # the committed pair of strata

    narrow = llm.judging.a1_recheck(llm.judging.RECHECK_STRATA, 0)
    wide = llm.judging.a1_recheck(llm.judging.LLM_RECHECK_STRATA, 0)
    ids = lambda f: set(f.query_id.astype(str) + "__" + f.doc_id.astype(str))  # noqa: E731
    assert len(ids(narrow)) == 50 and len(ids(wide)) == 100
    assert ids(narrow) < ids(wide), "the human 50 must be a strict subset of the LLM 100"


def test_a_diverged_draw_is_refused_rather_than_quoted(monkeypatch):
    """The assertion earns its place: give it two draws that do not nest."""
    import pandas as pd

    def fake(strata, seed=0):
        offset = 0 if sum(strata.values()) == 50 else 500      # the divergence
        return pd.DataFrame([{"query_id": f"j_{offset + i}", "doc_id": f"r_{offset + i}",
                              "a1_label": "No Fit"}
                             for i in range(sum(strata.values()))])

    monkeypatch.setattr(llm.judging, "a1_recheck", fake)
    with pytest.raises(AssertionError, match="absent from the LLM"):
        llm.assert_recheck_nests(0)


@needs_fit_split
def test_the_human_queue_is_not_widened_by_the_llm_leg():
    """Mutation: point `load_units`' default at `LLM_RECHECK_STRATA` — this fails.

    Widening the machine leg costs no annotator time; widening the human one silently
    adds 50 pairs to a session that is 9 short of finished. `load_units` defaults to the
    human 50 and the LLM path passes its own strata explicitly.
    """
    import inspect
    default = inspect.signature(llm.session.load_units).parameters["strata"].default
    assert default is None, "load_units must default to the human strata, not the LLM's"

    units = llm.session.load_units(0)
    wide = llm.session.load_units(0, llm.judging.LLM_RECHECK_STRATA)
    assert (units.corpus == "a1").sum() == 50
    assert (wide.corpus == "a1").sum() == 100


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
    """Mutation: let `dispatch` call `ui.corpus_text()` with no strata — this fails.

    `corpus_text` is `lru_cache`d and loads A1 text for one draw. Built from the human 50
    while the units come from the LLM 100, half the pairs have no document — caught only
    at `serve_group`, and only because that assertion exists. The two consumers take the
    same strata, and here that is checked rather than remembered.
    """
    from candidate_screener.annotation import ui

    strata = llm.judging.LLM_RECHECK_STRATA
    units = llm.session.load_units(0, strata)
    jd_text, cv_text = ui.corpus_text(tuple(sorted(strata.items())))
    a1 = units[units.corpus == "a1"]
    assert not set(a1.jd_id) - set(jd_text.index), "JDs in the draw with no text"
    assert not set(a1.cv_id) - set(cv_text.index), "CVs in the draw with no text"

    narrow_jd, _ = ui.corpus_text()      # the default draw cannot cover the wider one
    assert set(a1.jd_id) - set(narrow_jd.index)
