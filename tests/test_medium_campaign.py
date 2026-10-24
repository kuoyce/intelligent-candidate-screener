"""Medium campaign isolation and checkpoint publication invariants."""
import hashlib
import json

import pandas as pd
import pytest

from candidate_screener.extraction import medium_run as run
from candidate_screener.extraction import medium_runtime
from candidate_screener.extraction.run import _json


def test_missing_latest_explains_partial_campaign_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match="explicit campaign ID"):
        run.load_campaign(campaign_root=tmp_path)


def test_medium_root_is_distinct_from_baseline():
    from candidate_screener.extraction.fit_run import ROOT
    assert run.ROOT != ROOT


def test_missing_medium_runtime_manifest_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="runtime manifest"):
        run._identity({"provenance": {}}, tmp_path, 120)


def test_empty_medium_runtime_manifest_rejected(tmp_path):
    (tmp_path / "runtime-versions.json").write_text("{}")
    with pytest.raises(ValueError, match="all pinned package versions"):
        run._identity({"provenance": {}}, tmp_path, 120)


def test_partial_pilot_cannot_publish_finished_pointer(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    sources.mkdir()
    frame = pd.DataFrame([{"resume_id": "r", "jd_id": "j", "resume_text": "Python",
                           "job_description_text": "Python", "label": "Fit"}])
    frame.to_parquet(sources / "train.parquet", index=False)
    frame.iloc[:0].to_parquet(sources / "test.parquet", index=False)
    monkeypatch.setattr(run, "_identity", lambda payload, model, budget:
                        {**payload["provenance"], "runtime": {"manifest": {}}})
    class Proxy:
        runtime_versions = {"gliner": "synthetic"}
        def __init__(self, *args, **kwargs):
            pass
        def extract_document(self, doc, **kwargs):
            return [], {"text": {"complete": True, "intervals": [[0, 6]], "chunks": 1}}
        def chunk_parameters(self):
            return {"token_limit": 64}
        def close(self):
            pass
    monkeypatch.setattr(medium_runtime, "MediumRuntimeProxy", Proxy)
    campaign_root = tmp_path / "campaigns"
    result = run.run_campaign(1, source_root=sources, campaign_root=campaign_root)
    assert result["summary"]["states"] == {"complete_zero": 1, "pending": 1}
    assert not (campaign_root / "latest.json").exists()
    finished = run.run_campaign(source_root=sources, campaign_root=campaign_root)
    assert finished["campaign_id"] == result["campaign_id"]
    assert finished["summary"]["states"] == {"complete_zero": 2}
    assert json.loads((campaign_root / "latest.json").read_text())["campaign_id"] == result["campaign_id"]


def test_checkpoint_cannot_resume_another_model_identity(tmp_path):
    record = {"campaign_id": "old", "source": "fit-cv", "doc_id": "r"}
    record["sha256"] = hashlib.sha256(_json(record).encode()).hexdigest()
    path = tmp_path / "checkpoint.json"
    path.write_text(_json(record))
    with pytest.raises(ValueError, match="identity mismatch"):
        run._load_checkpoint(path, {"source": "fit-cv", "doc_id": "r"}, "new")
