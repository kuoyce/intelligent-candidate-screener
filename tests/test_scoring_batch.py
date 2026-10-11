"""The D50 scoring batch: the scoped exemption to D42, the batch, and its bank.

Every test runs on inline synthetic frames with `subprocess.run` replaced: **nothing is
ever spawned** and no LLM is called. Each guard test names the check it guards; each was
shown to fail with that check removed (record: plan/2026-10-06-extraction-and-rerank/
15-p6-batch.md). The tests over the committed manifest read `docs/` only, so they run on a
fresh clone with no `data/`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess

import pandas as pd
import pytest

from candidate_screener.annotation import llm_recheck as llm
from candidate_screener.rerank import scoring_batch as sb

AGENT = sb.INSTRUMENT_PREFIX + "0" * 56
OTHER_AGENT = "47fc2e48" + "0" * 56


def scope(extra_pairs=(), replicate=("j1__c1",)) -> sb.Scope:
    """Train: j1 j2 jF c1-c4 cF. Eval: jE cE. Final set: jF, cF (put in *train* here so
    the final-set clause is tested apart from the region clause)."""
    pairs = [("j1", "c1"), ("j1", "c2"), ("j2", "c3"), ("j2", "c4"), *extra_pairs]
    manifest = pd.DataFrame({"pair_id": [f"{j}__{c}" for j, c in pairs],
                             "jd_id": [j for j, _ in pairs], "cv_id": [c for _, c in pairs]})
    manifest["split"] = "train"
    manifest["replicate"] = manifest.pair_id.isin(replicate)
    partition = pd.DataFrame(
        [(d, "jd", "train") for d in ("j1", "j2", "jF")]
        + [(d, "cv", "train") for d in ("c1", "c2", "c3", "c4", "cF")]
        + [("jE", "jd", "eval"), ("cE", "cv", "eval")],
        columns=["doc_id", "doc_type", "region"])
    final = pd.DataFrame({"jd_id": ["jF"], "cv_id": ["cF"]})
    return sb.Scope.from_frames(manifest, partition, final)


def record(pair="j1__c1", **over) -> dict:
    jd, cv = pair.split("__")
    rec = {"pair_id": pair, "query_id": jd, "doc_id": cv, "run": sb.RUN, "draw": sb.DRAW,
           "agent_sha256": AGENT, "model": sb.MODEL, "prompt": "JOB DESCRIPTION\nx",
           "prompt_sha256": "h", "chars_sent": 10, "split": "train"}
    rec.update(over)
    return rec


def ok(records, sc=None, allow_replicate=False):
    sb.assert_in_scope(records, sc or scope(), agent_sha256=AGENT,
                       allow_replicate=allow_replicate)


@pytest.fixture
def no_spawn(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a subprocess was started")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(sb.subprocess, "run", refuse)
    monkeypatch.setattr(llm.subprocess, "run", refuse)


@pytest.fixture
def bank(tmp_path, monkeypatch):
    """Every file the batch reads or writes, moved to tmp. The frozen bank is empty."""
    monkeypatch.setattr(sb, "PROMPTS", tmp_path / "p.jsonl")
    monkeypatch.setattr(sb, "RAW", tmp_path / "r.jsonl")
    monkeypatch.setattr(sb, "FAILED", tmp_path / "f.jsonl")
    monkeypatch.setattr(sb, "BATCH_MANIFEST", tmp_path / "absent.csv")
    monkeypatch.setattr(llm, "PROMPTS", tmp_path / "frozen.jsonl")
    monkeypatch.setattr(llm, "DRAW_FILES", {})
    return tmp_path


def texts(jd_ids, cv_ids):
    return (pd.Series({j: f"Job {j} needs Python." for j in jd_ids}),
            pd.Series({c: f"Engineer {c}, Python, five years." for c in cv_ids}))


# --- the guard: each clause of D50 -------------------------------------------

def test_an_in_scope_record_passes():
    ok([record(), record("j2__c3")])


def test_the_guard_refuses_a_different_instrument():
    """Guards `check_instrument`'s prefix clause."""
    with pytest.raises(sb.OutOfScope, match="is not 5d986c82"):
        sb.assert_in_scope([record(agent_sha256=OTHER_AGENT)], scope(),
                           agent_sha256=OTHER_AGENT)


def test_the_guard_refuses_a_record_dispatched_against_another_judge():
    """Guards `check_instrument`'s equality clause: same prefix, different bytes."""
    with pytest.raises(sb.OutOfScope, match="committed judge"):
        ok([record(agent_sha256=sb.INSTRUMENT_PREFIX + "f" * 56)])


def test_the_guard_refuses_a_different_model():
    """Guards `check_model`."""
    with pytest.raises(sb.OutOfScope, match="model 'claude-sonnet-5'"):
        ok([record(model="claude-sonnet-5")])


def test_the_guard_refuses_a_final_set_document():
    """Guards the final-set clause of `check_documents` (cF is in the train region)."""
    sc = scope(extra_pairs=[("j1", "cF")])
    with pytest.raises(sb.OutOfScope, match="carries a final-set document"):
        sb.assert_manifest_in_scope(sc)
    with pytest.raises(sb.OutOfScope, match="carries a final-set document"):
        sb.check_documents("jF", "c1", scope())


def test_the_guard_refuses_an_eval_region_document():
    """Guards the eval-region clause; the match keeps the train clause from passing it."""
    with pytest.raises(sb.OutOfScope, match="carries an eval-region document"):
        sb.assert_manifest_in_scope(scope(extra_pairs=[("j1", "cE")]))
    with pytest.raises(sb.OutOfScope, match="carries an eval-region document"):
        sb.check_documents("jE", "c1", scope())


def test_the_guard_refuses_a_document_outside_the_train_region():
    with pytest.raises(sb.OutOfScope, match="outside the A2 train region"):
        sb.check_documents("j1", "c_unknown", scope())


def test_the_guard_refuses_a_pair_not_in_the_manifest():
    """Guards `check_manifest`: j1/c3 are both allowed documents, but not this pair."""
    with pytest.raises(sb.OutOfScope, match="not in the committed batch manifest"):
        ok([record("j1__c3")])


def test_the_guard_refuses_ids_that_disagree_with_the_manifest():
    with pytest.raises(sb.OutOfScope, match="do not match the manifest"):
        ok([record(doc_id="c2")])


def test_the_guard_refuses_a_manifest_over_the_cap(monkeypatch):
    """Guards `assert_manifest_in_scope`'s cap."""
    monkeypatch.setattr(sb, "PAIR_CAP", 3)
    with pytest.raises(sb.OutOfScope, match="over the cap of 3"):
        ok([record()])


def test_the_guard_refuses_a_run_over_its_cap(monkeypatch):
    """Guards `check_cap`."""
    monkeypatch.setattr(sb, "PAIR_CAP", 1)
    with pytest.raises(sb.OutOfScope, match="over its cap of 1"):
        sb.check_cap([record(), record("j1__c2")])


def test_the_guard_refuses_the_replicate_without_its_flag():
    """Guards `check_run`: run 12 is legal only with `replicate=True` (Q-A1b, off)."""
    with pytest.raises(sb.OutOfScope, match="explicit --replicate flag"):
        ok([record(run=sb.REPLICATE_RUN)])
    ok([record(run=sb.REPLICATE_RUN)], allow_replicate=True)


def test_the_replicate_may_only_hold_the_manifests_replicate_pairs():
    with pytest.raises(sb.OutOfScope, match="replicate subset"):
        ok([record("j1__c2", run=sb.REPLICATE_RUN)], allow_replicate=True)


def test_the_replicate_subset_has_its_own_cap(monkeypatch):
    monkeypatch.setattr(sb, "REPLICATE_CAP", 1)
    with pytest.raises(sb.OutOfScope, match="replicate subset holds 2"):
        ok([record()], sc=scope(replicate=("j1__c1", "j1__c2")))


@pytest.mark.parametrize("over", [{"run": 9}, {"run": 13}, {"draw": "a2_generic"}])
def test_the_guard_refuses_any_other_run_or_draw(over):
    with pytest.raises(sb.OutOfScope):
        ok([record(**over)], allow_replicate=True)


def test_out_of_scope_is_a_dispatch_closed():
    """Anything that catches D42's refusal catches this one too."""
    assert issubclass(sb.OutOfScope, llm.DispatchClosed)


def test_the_committed_judge_must_be_the_d50_instrument(monkeypatch):
    monkeypatch.setattr(llm, "assert_agent_definition_current", lambda: OTHER_AGENT)
    with pytest.raises(sb.OutOfScope, match="not 5d986c82"):
        sb.current_instrument()


def test_the_committed_judge_is_the_d50_instrument():
    assert sb.current_instrument().startswith(sb.INSTRUMENT_PREFIX)


# --- D42 still holds everywhere else -----------------------------------------

@pytest.mark.parametrize("call", [
    lambda: llm.dispatch(run=sb.RUN),
    lambda: llm.dispatch_a2("generic", run=sb.RUN),
    lambda: llm.dispatch_all_a1(run=sb.RUN),
    lambda: llm.judge(run=sb.RUN),
    lambda: llm.judge_one(record(), cwd=None),
])
def test_the_frozen_entry_points_still_refuse_after_this_module_loads(call, no_spawn):
    assert llm.DISPATCH_CLOSED is True
    with pytest.raises(llm.DispatchClosed, match="D42"):
        call()


def test_dispatch_refuses_a_run_number_the_frozen_bank_holds(bank, no_spawn):
    """Guards `assert_runs_unused_elsewhere`."""
    llm.PROMPTS.write_text(json.dumps({"run": sb.RUN, "pair_id": "x"}) + "\n")
    with pytest.raises(sb.OutOfScope, match="already used"):
        sb.dispatch(scope=scope(), texts=texts, agent=AGENT)
    assert not sb.PROMPTS.exists()


def test_dispatch_refuses_a_wrong_model_and_writes_nothing(bank, no_spawn):
    with pytest.raises(sb.OutOfScope, match="model"):
        sb.dispatch(model="claude-sonnet-5", scope=scope(), texts=texts, agent=AGENT)
    assert not sb.PROMPTS.exists()


def test_the_spawn_checks_the_scope_itself(no_spawn, tmp_path):
    """Guards the guard inside `spawn_one`: no record reaches `subprocess.run` unchecked."""
    with pytest.raises(sb.OutOfScope):
        sb.spawn_one(record(model="claude-sonnet-5"), tmp_path, scope(), AGENT, False)


# --- dispatch, judge, coverage, collect --------------------------------------

def test_dispatch_is_append_only_and_idempotent(bank, no_spawn):
    first = sb.dispatch(scope=scope(), texts=texts, agent=AGENT)
    again = sb.dispatch(scope=scope(), texts=texts, agent=AGENT)
    assert (first["pairs"], again["pairs"], again["already_dispatched"]) == (4, 0, 4)
    rows = llm.read_jsonl(sb.PROMPTS)
    assert len(rows) == 4 and {r["run"] for r in rows} == {sb.RUN}
    assert {r["model"] for r in rows} == {sb.MODEL}
    assert rows[0]["prompt"].startswith("JOB DESCRIPTION\n")


def test_the_replicate_dispatches_only_behind_its_flag(bank, no_spawn):
    sb.dispatch(scope=scope(), texts=texts, agent=AGENT)
    out = sb.dispatch(replicate=True, scope=scope(), texts=texts, agent=AGENT)
    assert out["run"] == sb.REPLICATE_RUN and out["pairs"] == 1


def test_dry_run_writes_nothing_and_spawns_nothing(bank, no_spawn):
    out = sb.dry_run(scope=scope(), texts=texts, agent=AGENT)
    assert out["pairs_in_scope"] == 4 and out["owed"] == 4 and out["spawned"] == 0
    assert out["sittings_estimate"] == 1 and out["cap"] == sb.PAIR_CAP
    assert not sb.PROMPTS.exists() and not sb.RAW.exists()


class Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


GOOD = '{"label":"Good Fit","score":81,"reason":"python"}'


def judge(monkeypatch, answer):
    seen = []

    def fake_run(argv, **kw):
        seen.append(argv)
        return answer(kw["input"])
    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def test_judge_banks_answers_and_names_the_model(bank, monkeypatch):
    sb.dispatch(scope=scope(), texts=texts, agent=AGENT)
    seen = judge(monkeypatch, lambda p: Proc(stdout=GOOD))
    out = sb.judge(scope=scope(), agent=AGENT)
    assert out["judged_now"] == 4 and out["remaining"] == 0
    assert all(argv[-2:] == ["--model", sb.MODEL] for argv in seen)
    assert sb.run_coverage(sb.RUN)["complete"]
    seen.clear()
    assert sb.judge(scope=scope(), agent=AGENT)["attempted"] == 0 and not seen


def test_a_lost_spawn_is_quarantined_and_retried_exactly(bank, monkeypatch):
    sb.dispatch(scope=scope(), texts=texts, agent=AGENT)
    judge(monkeypatch, lambda p: Proc(stdout=GOOD) if "j1" in p else Proc(returncode=1))
    with pytest.raises(llm.JudgeFailures):
        sb.judge(scope=scope(), agent=AGENT)
    cov = sb.run_coverage(sb.RUN)
    assert (cov["n_answered"], cov["n_missing"], cov["complete"]) == (2, 2, False)
    seen = judge(monkeypatch, lambda p: Proc(stdout=GOOD))
    assert sb.judge(scope=scope(), agent=AGENT)["judged_now"] == 2 and len(seen) == 2


def test_judge_refuses_a_tampered_bank_before_spawning(bank, no_spawn):
    sb.PROMPTS.write_text(json.dumps(record(model="claude-sonnet-5")) + "\n")
    with pytest.raises(sb.OutOfScope):
        sb.judge(scope=scope(), agent=AGENT)


def test_judge_ignores_the_replicate_run_without_its_flag(bank, monkeypatch):
    sb.dispatch(replicate=True, scope=scope(), texts=texts, agent=AGENT)
    judge(monkeypatch, lambda p: Proc(stdout=GOOD))
    with pytest.raises(FileNotFoundError, match="no run 11"):
        sb.judge(scope=scope(), agent=AGENT)
    assert sb.judge(replicate=True, scope=scope(), agent=AGENT)["judged_now"] == 1


def test_collect_refuses_a_short_run(bank, monkeypatch):
    """Guards the coverage check in `collect`."""
    sb.dispatch(scope=scope(), texts=texts, agent=AGENT)
    judge(monkeypatch, lambda p: Proc(stdout=GOOD) if "j1" in p else Proc(returncode=1))
    with pytest.raises(llm.JudgeFailures):
        sb.judge(scope=scope(), agent=AGENT)
    out = bank / "scores.csv"
    with pytest.raises(llm.JudgeFailures, match="Do not `--collect`"):
        sb.collect(out)
    assert not out.exists()


def test_collect_writes_ids_and_scores_keyed_by_the_four_axes(bank, monkeypatch):
    sb.dispatch(scope=scope(), texts=texts, agent=AGENT)
    judge(monkeypatch, lambda p: Proc(stdout=GOOD) if "j1" in p
          else Proc(stdout="I think it fits"))
    sb.judge(scope=scope(), agent=AGENT)
    out = bank / "scores.csv"
    summary = sb.collect(out)
    frame = pd.read_csv(out)
    assert list(frame.columns) == list(sb.SCORE_COLUMNS)
    assert {"agent_sha256", "model", "run", "draw"} <= set(frame.columns)
    assert "notes" not in frame and "prompt" not in frame and "reason" not in frame
    assert (summary["parsed_ok"], summary["unparsed"], summary["all_parsed"]) == (2, 2, False)
    assert frame.score.dropna().between(0, 100).all()


def test_collect_never_writes_the_human_judgement_file(bank, monkeypatch):
    """Guards `collect`'s target check. The human file is a tmp stand-in, so a mutant
    with the check removed overwrites the stand-in and never the committed file."""
    human = bank / "human" / "judgements.csv"
    human.parent.mkdir()
    human.write_text("pair_id,annotator,label\nx,yc,No Fit\n")
    monkeypatch.setattr(sb, "JUDGEMENTS", human)
    before = human.read_bytes()
    sb.dispatch(scope=scope(), texts=texts, agent=AGENT)
    judge(monkeypatch, lambda p: Proc(stdout=GOOD))
    sb.judge(scope=scope(), agent=AGENT)
    with pytest.raises(ValueError, match="never go to judgements.csv"):
        sb.collect(human)
    with pytest.raises(ValueError, match="never go to judgements.csv"):
        sb.collect(bank / "judgements.csv")
    sb.collect(bank / "scores.csv")
    assert human.read_bytes() == before
    assert sb.SCORES_CSV.name == "llm-train-scores.csv"


def test_the_default_collect_target_is_not_the_human_file():
    from candidate_screener.annotation.queue import JUDGEMENTS
    assert sb.SCORES_CSV.resolve() != JUDGEMENTS.resolve()
    assert sb.JUDGEMENTS.resolve() == JUDGEMENTS.resolve()


# --- the batch ---------------------------------------------------------------

def synthetic_corpus():
    jd = pd.DataFrame({"id": [f"j{i}" for i in range(6)],
                       "jd_text": ["python django api"] * 3 + ["java spring"] * 3,
                       "exp_band": ["0-1", "2-3", "4-6"] * 2,
                       "Primary Keyword": ["Python"] * 3 + ["Java"] * 3})
    rows = []
    for kw, words in (("Python", "python django"), ("Java", "java spring")):
        for i in range(60):
            rows.append({"id": f"{kw}{i:03d}", "cv_text": f"{words} {'api ' * (i % 7)}",
                         "Primary Keyword": kw})
    return jd, pd.DataFrame(rows)


def test_the_batch_copies_the_generic_design():
    jd, cv = synthetic_corpus()
    manifest, summary = sb.build_batch(jd, cv, seed=0)
    assert list(manifest.columns) == list(sb.MANIFEST_COLUMNS)
    per_jd = manifest.groupby("jd_id").selection_reason.apply(
        lambda s: s.str.startswith("same_keyword").sum())
    assert (manifest.groupby("jd_id").size() == sb.PER_JD).all()
    assert (per_jd == 8).all()
    assert manifest.cv_id.is_unique
    assert summary["cvs_in_both_splits"] == 0


def test_the_split_is_by_jd_and_drops_validation_cvs_seen_in_train():
    pairs = pd.DataFrame({"pair_id": [f"j{j}__c{c}" for j, c in
                                      [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3)]],
                          "jd_id": ["j0", "j1", "j2", "j3", "j4"],
                          "cv_id": ["c0", "c0", "c1", "c2", "c3"]})
    for seed in range(10):
        out, info = sb.split_by_jd(pairs, seed)
        train, val = out[out.split == "train"], out[out.split == "validation"]
        assert not set(train.jd_id) & set(val.jd_id)
        assert not set(train.cv_id) & set(val.cv_id)
        assert len(out) + info["validation_pairs_dropped_for_cv_overlap"] == len(pairs)


# --- the committed manifest (docs/ only; no data/ needed) --------------------

def test_the_committed_batch_is_in_scope_and_matches_its_digest():
    record_ = json.loads(sb.BATCH_RECORD.read_text(encoding="utf-8"))
    body = sb.BATCH_MANIFEST.read_bytes()
    assert hashlib.sha256(body).hexdigest() == record_["manifest_sha256"]
    committed = sb.Scope.committed()
    sb.assert_manifest_in_scope(committed)
    m = pd.read_csv(sb.BATCH_MANIFEST)
    assert len(m) == sb.PAIR_CAP and m.pair_id.is_unique
    assert m.groupby("split").jd_id.nunique().to_dict() == {"train": 80, "validation": 20}
    train, val = m[m.split == "train"], m[m.split == "validation"]
    assert not set(train.jd_id) & set(val.jd_id)
    assert not set(train.cv_id) & set(val.cv_id)
    assert int(m.replicate.sum()) == sb.REPLICATE_CAP


def test_no_committed_batch_document_is_in_the_final_set_or_the_eval_region():
    m = pd.read_csv(sb.BATCH_MANIFEST)
    final = sb.final_set_pairs()
    part = pd.read_csv(sb.PARTITION_MANIFEST)
    evals = set(part[part.region == "eval"].doc_id)
    docs = set(m.jd_id) | set(m.cv_id)
    assert not docs & (set(final.jd_id) | set(final.cv_id))
    assert not docs & evals


@pytest.mark.skipif(not llm.PROMPTS.exists(), reason="needs data/ (the frozen run-9/10 bank)")
def test_the_text_sent_is_the_text_runs_9_and_10_sent():
    out = sb.verify_text_form()
    assert out["ok"] and out["checked"] == 600
