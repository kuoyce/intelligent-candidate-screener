"""The evaluation harness must preserve deadlines, provenance and study boundaries."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from candidate_screener.evaluation.cui.fixtures import load_fixtures
from candidate_screener.evaluation.cui.runner import FrozenQA, run, RecordingComposer
from candidate_screener.evaluation.cui.protocol import digest, write_new, freeze, verify_selected_runtime
from candidate_screener.evaluation.cui.__main__ import main


def small_fixture():
    fixtures = load_fixtures()
    fixtures["families"] = [deepcopy(next(f for f in fixtures["families"] if f["split"] == "dev"))]
    family = fixtures["families"][0]
    family["questions"] = family["questions"][:1]
    family["dialogues"] = []
    return fixtures


def test_frozen_qa_forwards_absolute_deadline_and_replays_without_new_inference(monkeypatch):
    calls = []
    class Client:
        def answer(self, *args):
            pytest.fail("Must preserve bounded answer_before contract")
        def answer_before(self, question, passages, deadline):
            calls.append(deadline)
            return {"status": "not_found", "spans": []}
    monkeypatch.setattr("candidate_screener.evaluation.cui.runner.time.monotonic", lambda: 10.)
    client = FrozenQA(Client())
    first = client.answer_before("question", [], 12.)
    client.read_only = True
    assert client.answer_before("question", [], 11.) == first
    assert calls == [12.]
    with pytest.raises(ValueError, match="differs"):
        client.answer_before("changed", [], 11.)
    with pytest.raises(TimeoutError):
        client.answer_before("question", [], 9.)


def test_stub_run_is_explicitly_development_and_paired():
    result = run(small_fixture(), run_id="test-offline")
    assert result["kind"] == "stub_development"
    assert result["metadata"]["provider"] == "stub"
    assert len(result["records"]) == 2
    first, second = result["records"]
    assert {first["arm"], second["arm"]} == {"controlled", "v3"}
    assert first["prepared_sha256"] == second["prepared_sha256"]
    assert first["source_documents"] == second["source_documents"]
    assert all(r["error"] is None for r in result["records"])


def test_eval_not_exposed_during_stub_debugging():
    with pytest.raises(ValueError, match="development"):
        run(load_fixtures(), split="eval", run_id="not-allowed")


def test_live_requires_real_explicit_clients():
    with pytest.raises(ValueError, match="explicit generator"):
        run(small_fixture(), provider="live", run_id="not-live")


def test_draft_cannot_be_frozen(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("test protocol")
    with pytest.raises(ValueError, match="reviewed fixtures"):
        freeze(small_fixture(), protocol)


def test_artifacts_never_overwrite_existing_results(tmp_path):
    path = tmp_path / "result.json"
    write_new(path, {"original": True})
    with pytest.raises(FileExistsError):
        write_new(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"original": True}


def test_cli_live_requires_approval_before_model_configuration(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Model setup cannot precede approval")
    monkeypatch.setattr("candidate_screener.evaluation.cui.protocol.configured_generator", forbidden)
    with pytest.raises(SystemExit) as error:
        main(["run", "--provider", "live", "--run-id", "blocked", "--output", str(tmp_path / "run.json")])
    assert error.value.code == 2
    assert not (tmp_path / "run.json").exists()


def test_selected_runtime_must_match_frozen_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr("candidate_screener.evaluation.cui.protocol.runtime_identity", lambda composer: {"model": composer.model})
    qa = SimpleNamespace(manifest_path=str(tmp_path / "qa.json"), timeout=30.)
    manifest = {"runtime": {"model": "pinned"}, "qa_manifest": {"path": qa.manifest_path}}
    verify_selected_runtime(manifest, SimpleNamespace(model="pinned"), qa)
    with pytest.raises(ValueError, match="generation configuration"):
        verify_selected_runtime(manifest, SimpleNamespace(model="different"), qa)
    qa.manifest_path = str(tmp_path / "other.json")
    with pytest.raises(ValueError, match="QA manifest"):
        verify_selected_runtime(manifest, SimpleNamespace(model="pinned"), qa)


def test_attempted_output_preserved_without_changing_provider_response():
    output = {"text": "unsupported synthetic claim", "model": "stub"}
    composer = RecordingComposer(SimpleNamespace(compose_before=lambda *a: output))
    assert composer.compose_before("q", {}, {}, 100.) is output
    output["text"] = "mutated"
    assert composer.attempted["text"] == "unsupported synthetic claim"
    assert composer.seconds >= 0


def test_fixture_digest_changes_with_expectations():
    fixtures = small_fixture()
    original = digest(fixtures)
    fixtures["families"][0]["questions"][0]["expected"]["uncertainty"] = not fixtures["families"][0]["questions"][0]["expected"]["uncertainty"]
    assert digest(fixtures) != original


def test_cli_pilot_runs_one_development_pair(tmp_path):
    output = tmp_path / "pilot.json"
    main(["run", "--pilot", "--run-id", "test-pilot", "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["metadata"]["pilot"] is True
    assert len(result["records"]) == 2


def test_cli_pilot_cannot_select_evaluation_outputs(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["run", "--pilot", "--split", "eval", "--run-id", "blocked", "--output", str(tmp_path / "run.json")])
    assert error.value.code == 2
    assert not (tmp_path / "run.json").exists()
