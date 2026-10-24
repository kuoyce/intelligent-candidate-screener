import json
from dataclasses import FrozenInstanceError
import pandas as pd
import pytest
from candidate_screener.app.portal_data import PortalDataset, CV_COLUMNS, JD_COLUMNS, RANK_COLUMNS, parse_mentions


def fixture_files(tmp_path):
    cvs = [{key: None for key in CV_COLUMNS} for _ in range(10)]
    for i, row in enumerate(cvs):
        row.update(cv_id=f"cv-{i}", role="predicted software family", role_prob=.8, top_roles="software:.8;data:.2", skills="Python;C++, language", years=3)
    jds = [{key: None for key in JD_COLUMNS}]
    jds[0].update(jd_id="jd-one", stratum="generic", jd_title="Engineer contact private@example.com", role="predicted family", skills="Python", min_years=2)
    ranks = [dict(pair_id=f"pair-{i}", stratum="generic", jd_id="jd-one", cv_id=f"cv-{i}", x2_score=10-i, x2_rank=i+1, final_label="DO_NOT_LOAD_SECRET_LABEL") for i in range(10)]
    pd.DataFrame(cvs).to_csv(tmp_path / "a2-cv-extraction.csv", index=False)
    pd.DataFrame(jds).to_csv(tmp_path / "a2-jd-extraction.csv", index=False)
    pd.DataFrame(ranks).to_csv(tmp_path / "a2-x2-scores.csv", index=False)
    return tmp_path


def load(folder, sources=None):
    return PortalDataset.from_folder(folder, source_documents=sources or {"jd": {}, "cv": {}})


def test_existing_ranks_preserved_and_no_labels_read(tmp_path, monkeypatch):
    folder = fixture_files(tmp_path)
    old = pd.read_csv
    seen = []
    def tracked(*args, **kwargs):
        assert "final_label" not in kwargs["usecols"]
        seen.append(kwargs["usecols"])
        return old(*args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", tracked)
    dataset = load(folder)
    snapshot = dataset.snapshot("jd-one")
    assert len(seen) == 3
    assert [c["rank"] for c in snapshot.candidates] == list(range(1, 11))
    assert [c["cv_id"] for c in snapshot.candidates] == [f"cv-{i}" for i in range(10)]
    assert "DO_NOT_LOAD_SECRET_LABEL" not in snapshot._payload_json
    assert "private@example.com" not in str(dataset.jobs())


@pytest.mark.parametrize("mutation", ["missing", "duplicate_rank", "nan_score", "unknown_cv", "stratum"])
def test_invalid_run_blocks_without_skip_or_backfill(tmp_path, mutation):
    folder = fixture_files(tmp_path)
    path = folder / "a2-x2-scores.csv"
    frame = pd.read_csv(path)
    if mutation == "missing": frame = frame.iloc[:-1]
    elif mutation == "duplicate_rank": frame.loc[1, "x2_rank"] = 1
    elif mutation == "nan_score": frame.loc[0, "x2_score"] = float("nan")
    elif mutation == "unknown_cv": frame.loc[0, "cv_id"] = "unknown"
    else: frame.loc[0, "stratum"] = "targeted"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load(folder)


def test_categories_explicitly_unverified_no_predicted_role_as_title(tmp_path):
    snapshot = load(fixture_files(tmp_path)).snapshot("jd-one")
    candidate = snapshot.candidates[0]
    assert set(candidate["entities"]) == {"KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS"}
    assert candidate["entities"]["TITLE"]["status"] == "unavailable"
    assert candidate["entities"]["DEGREE"]["status"] == "unavailable"
    assert candidate["entities"]["SKILL"]["mentions"] == ["Python", "C++, language"]
    assert all(not v["verified_spans"] for v in candidate["entities"].values())
    assert candidate["preview_text"] is None and candidate["preview_status"] == "unavailable"
    assert "not an extracted job title" in candidate["table_aggregates"]["role_semantics"]


def test_source_previews_redacted_and_snapshots_immutable_and_source_bound(tmp_path):
    folder = fixture_files(tmp_path)
    sources = {"jd": {"jd-one": "Role contact first@example.com"}, "cv": {"cv-0": "😀 Python private@example.com"}}
    first = load(folder, sources).snapshot("jd-one")
    assert first.candidates[0]["preview_text"] == "😀 Python [EMAIL]"
    changed = first.candidates[0]
    changed["rank"] = 99
    assert first.candidates[0]["rank"] == 1
    with pytest.raises(FrozenInstanceError):
        first.jd_id = "different"
    sources["jd"]["jd-one"] = "Role contact second@example.com"
    second = load(folder, sources).snapshot("jd-one")
    assert first.jd_text == second.jd_text
    assert first.snapshot_id != second.snapshot_id


def test_safe_json_delimited_parser_never_evaluates_code():
    assert parse_mentions('["Python", "SQL"]') == ["Python", "SQL"]
    assert parse_mentions("Python;SQL") == ["Python", "SQL"]
    assert parse_mentions("__import__('os').system('touch /tmp/never-run')") == ["__import__('os').system('touch /tmp/never-run')"]
    with pytest.raises(ValueError, match="Malformed"):
        parse_mentions("[not JSON]")


def test_synthetic_demo_never_opens_real_sources_and_supports_job_switch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic demo must not read any source file")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    dataset = PortalDataset.synthetic_demo()
    assert dataset.is_demo and dataset.run_id.startswith("demo-")
    assert len(dataset.jobs()) == 2
    snapshots = [dataset.snapshot(job["jd_id"]) for job in dataset.jobs()]
    assert snapshots[0].snapshot_id != snapshots[1].snapshot_id
    for snapshot in snapshots:
        assert snapshot.model_id == "synthetic-demo-ranks"
        assert snapshot.source_flags["is_demo"] is True
        assert len(snapshot.candidates) == 10
        assert all(c["preview_status"] == "available_redacted_text" for c in snapshot.candidates)


def test_table_change_during_read_rejects_mixed_snapshot(tmp_path, monkeypatch):
    import os
    folder = fixture_files(tmp_path)
    original = pd.read_csv
    def changed(path, **kwargs):
        frame = original(path, **kwargs)
        if str(path).endswith("a2-x2-scores.csv"):
            target = folder / "a2-cv-extraction.csv"
            stat = target.stat()
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return frame
    monkeypatch.setattr(pd, "read_csv", changed)
    with pytest.raises(ValueError, match="changed while loading"):
        load(folder)
