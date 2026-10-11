"""No-data tests for registered pilot selection and honest blocked reports."""
import numpy as np
import pandas as pd
import pytest

from candidate_screener.evaluation import entity_weighted as pilot


def test_alpha_choice_uses_primary_metric_and_earlier_registered_tie():
    values = {a: {"recall@50": .7} for a in pilot.ALPHAS}
    values[.25]["recall@50"] = .80000001
    values[.5]["recall@50"] = .80000002
    assert pilot.choose_alpha(values) == .25
    with pytest.raises(ValueError, match="complete"):
        pilot.choose_alpha({0.: {"recall@50": 1}})


def test_fixed_shortlist_cannot_recover_omitted_relevant_cv_or_shrink_denominator():
    frame = pd.DataFrame({"jd_id": ["j"] * 60, "pair_id": [f"p{i:02}" for i in range(60)]})
    baseline = -np.arange(60)
    candidate = np.arange(60)
    rel = np.zeros(60)
    rel[[45, 59]] = 1
    assert pilot.fixed_shortlist_recall(frame, baseline, candidate, rel).iloc[0] == .5


def test_pilot_queries_selected_by_sorted_id_without_dropping_zero_entity_candidates():
    frame = pd.DataFrame([{"jd_id": jd, "cv_id": str(i), "pair_id": f"{jd}_{i}", "split": split}
                          for split, ids in [("dev-train", ["z", "a", "b"]),
                                             ("dev-select", ["y", "c", "d"])]
                          for jd in ids for i in range(500)])
    chosen = pilot.select_queries(frame)
    assert sorted(chosen.jd_id.unique()) == ["a", "b", "c", "d"]
    assert len(chosen) == 2000
    with pytest.raises(ValueError, match="500-CV"):
        pilot.select_queries(frame.drop(frame.index[500]))


def test_blocked_record_contains_no_fabricated_metrics_or_adoption(monkeypatch):
    monkeypatch.setattr(pilot, "prerequisites", lambda: ({}, ["role_weights"], {"code": "digest"}))
    report = pilot.report()
    assert report["status"] == "blocked"
    assert report["metrics"] == {}
    assert report["adopted"] is False
    assert "role_weights" in report["missing_prerequisites"]


def test_integrity_check_fails_for_changed_record_and_accepts_exact_blocked_record(tmp_path, monkeypatch):
    import json
    record = tmp_path / "record.json"
    monkeypatch.setattr(pilot, "RECORD", record)
    expected = {"status": "blocked", "metrics": {}, "adopted": False}
    monkeypatch.setattr(pilot, "report", lambda: expected)
    record.write_text(json.dumps(expected))
    pilot.main(["--check"])
    record.write_text(json.dumps({**expected, "adopted": True}))
    with pytest.raises(SystemExit, match="differs"):
        pilot.main(["--check"])
