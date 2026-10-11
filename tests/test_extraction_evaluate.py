"""`extraction.evaluate`: the `--check` diff applies tolerance only to torch figures."""
from __future__ import annotations

import json

import pytest

from candidate_screener.extraction import evaluate as ev

OLD = {"vocab": {"rows": 10}, "skillspan": {"test": {"rules": {"f1": 0.5}, "ner": {"f1": 0.5}}},
       "ner": {"weights_sha256": "aa", "best_epoch": 3},
       "normalisation": {"x": {"methods": {"sbert": {"top1": 0.6}, "string": {"top1": 0.5}}}}}


def _with(path, value):
    new = json.loads(json.dumps(OLD))
    node = new
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return new


def test_identical_records_pass():
    assert ev.diff(OLD, OLD, tol=0.01) == ([], [])


def test_torch_figure_within_tolerance_is_a_note():
    fails, notes = ev.diff(OLD, _with(["skillspan", "test", "ner", "f1"], 0.505), tol=0.01)
    assert not fails and len(notes) == 1


def test_torch_figure_beyond_tolerance_fails():
    fails, _ = ev.diff(OLD, _with(["skillspan", "test", "ner", "f1"], 0.52), tol=0.01)
    assert fails


def test_non_torch_figure_must_be_exact():
    fails, _ = ev.diff(OLD, _with(["skillspan", "test", "rules", "f1"], 0.5001), tol=0.01)
    assert fails
    fails, _ = ev.diff(OLD, _with(["normalisation", "x", "methods", "string", "top1"], 0.5001),
                       tol=0.01)
    assert fails


def test_sbert_is_torch_derived():
    fails, notes = ev.diff(OLD, _with(["normalisation", "x", "methods", "sbert", "top1"], 0.605),
                           tol=0.01)
    assert not fails and notes


def test_weights_digest_is_informational_but_other_strings_are_not():
    fails, notes = ev.diff(OLD, _with(["ner", "weights_sha256"], "bb"), tol=0.01)
    assert not fails and notes
    fails, _ = ev.diff(OLD, _with(["ner", "best_epoch"], "3"), tol=0.01)
    assert fails


def test_missing_key_fails():
    new = json.loads(json.dumps(OLD))
    del new["vocab"]["rows"]
    assert ev.diff(OLD, new)[0]


@pytest.mark.skipif(not ev.RECORD.is_file(), reason="extraction record not built yet")
def test_committed_record_names_n_and_ci_beside_every_f1():
    rec = json.loads(ev.RECORD.read_text())
    blocks = [rec["skillspan"]["test"][sys_][scope] for sys_ in ("rules", "ner")
              for scope in ("union", "skill", "knowledge")]
    blocks += [rec["dataturks"][part][sys_] for part in ("test", "all") for sys_ in ("rules", "ner")]
    for b in blocks:
        assert b["n_units"] > 0 and b["n_gold"] > 0
        for kind in ("strict", "partial"):
            lo, hi = b[kind]["f1_ci"]
            assert lo <= b[kind]["f1"] <= hi
    assert "decide nothing" in rec["dataturks"]["note"]
    assert "timestamp" not in json.dumps(rec).lower()


def test_tagger_config_is_always_exact():
    old = {"ner": {"config": {"lr": 0.0001}}}
    new = {"ner": {"config": {"lr": 0.00011}}}
    assert ev.diff(old, new, tol=0.01)[0]
