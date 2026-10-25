"""Synthetic callable adapters exercise the proposed boundary, never a model."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import sqlite3
import sys
from threading import Event
import time
from types import ModuleType
import zipfile

import pytest

from candidate_screener.app.portal_data import PortalSnapshot
from candidate_screener.app.portal_jobs import JobService
from candidate_screener.app.portal_ranking import RankingService
from candidate_screener.app.portal_store import AuthorizationError, PortalStore, Principal, RevisionConflict


def result(payload, count=2, *, status="ready"):
    """Documented fixture for a teammate implementing adapter(op, payload)."""
    response = {key: payload[key] for key in (
        "request_id", "jd_id", "jd_revision", "jd_digest", "workspace",
        "authorization_scope", "candidate_pool_ref")}
    response.update(status=status, handle="fixture-handle")
    if status == "ready":
        response.update(run_id="fixture-run", model_id="fixture-model",
                        candidate_pool_version="fixture-pool-v1",
                        score_semantics="Synthetic adapter ordering score; not a probability",
                        eligible_candidate_ids=[f"Applicant-{i}" for i in range(1, count + 1)],
                        candidates=[{"candidate_id": f"Applicant-{i}", "rank": i,
                                     # Deliberately ascending: services must not sort scores.
                                     "score": i / 10, "source_ref": f"fixture:document-{i}",
                                     "cv_text": f"Invented applicant {i}. Python private@example.com",
                                     "entities": {"SKILL": {"mentions": ["Python"]}},
                                     "evidence_provenance": {"extractor": "fixture", "version": "v1"}}
                                    for i in range(1, count + 1)])
    return response


@pytest.fixture
def principal():
    return Principal("ranking-author", ("benchmark",), time.time() + 3600, "local_token")


@pytest.fixture
def jobs(tmp_path, principal):
    return JobService(tmp_path / "workspace.sqlite", principal)


@pytest.fixture
def job(jobs):
    job = jobs.create({"title": "Python Engineer", "department": "Tech",
                       "responsibilities": ["Build services"], "skills": ["Python"]})
    jobs.review(job["job_id"], 1)
    return jobs.publish(job["job_id"], 1)


def service(jobs, adapter=None, **kwargs):
    return RankingService(jobs.path, jobs.principal, adapter=adapter, pool_ref="fixture-pool", **kwargs)


def test_submit_refresh_and_alert_idempotency_survive_reload(jobs, job):
    calls = []

    def adapter(operation, payload):
        calls.append((operation, deepcopy(payload)))
        return result(payload, status="queued" if operation == "submit" else "ready")

    ranking = service(jobs, adapter)
    request = ranking.rank(job["job_id"], request_id="submit-one")
    assert request["status"] == "queued"
    assert ranking.rank(job["job_id"], request_id="submit-one")["request_id"] == "submit-one"
    reopened = service(jobs, adapter)
    assert reopened.rank(job["job_id"])["request_id"] == "submit-one"
    assert reopened.ready_snapshot("submit-one") is None
    assert reopened.refresh("submit-one")["status"] == "ready"
    assert reopened.refresh("submit-one")["status"] == "ready"
    assert [operation for operation, _ in calls] == ["submit", "status"]
    assert calls[-1][1]["handle"] == "fixture-handle"
    assert len([e for e in jobs.events() if e["kind"] == "ranking_ready"]) == 1
    assert jobs.get(job["job_id"])["ranking_status"] == "ready"
    snapshot = reopened.snapshot(job["job_id"])
    assert isinstance(snapshot, PortalSnapshot)
    assert [c["rank"] for c in snapshot.candidates] == [1, 2]
    assert [c["score"] for c in snapshot.candidates] == [.1, .2]
    assert snapshot.source_flags["operational"]
    assert snapshot.source_flags["source_type"] == "operational_live"
    assert "not a probability" in snapshot.source_flags["score_semantics"]
    assert snapshot.source_flags["jd_revision"] == 1
    assert snapshot.model_id == "fixture-model" and snapshot.run_id == "fixture-run"
    assert not snapshot.source_flags["is_demo"]
    assert "private@example.com" not in snapshot._payload_json
    assert "[EMAIL]" in snapshot.candidates[0]["cv_text"]
    assert snapshot.candidates[0]["entities"]["SKILL"]["status"] == "unverified_backend_mentions"
    assert not snapshot.candidates[0]["entities"]["SKILL"]["verified_spans"]
    modified = snapshot.candidates[0]
    modified["rank"] = 99
    assert reopened.ready_snapshot("submit-one").candidates[0]["rank"] == 1
    assert b"private@example.com" not in jobs.path.read_bytes()
    assert "skills" not in calls[0][1]  # Structured requirements are not extra features.


@pytest.mark.parametrize("count", [0, 1, 9, 10])
def test_operational_zero_to_ten_results_preserve_real_count(jobs, job, count):
    ranking = service(jobs, lambda op, p: result(p, count))
    request = ranking.rank(job["job_id"])
    assert request["status"] == "ready"
    snapshot = ranking.snapshot(job["job_id"])
    assert len(snapshot.candidates) == count
    assert snapshot.source_flags["candidate_count"] == count
    assert [c["cv_id"] for c in snapshot.candidates] == [f"Applicant-{i}" for i in range(1, count + 1)]


def test_unconfigured_preserves_published_job_and_never_borrows_csv(jobs, job, monkeypatch):
    monkeypatch.delenv("HR_PORTAL_RANKING_ADAPTER", raising=False)
    monkeypatch.delenv("HR_PORTAL_CANDIDATE_POOL", raising=False)
    ranking = RankingService(jobs.path, jobs.principal)
    assert not ranking.availability["available"]
    request = ranking.rank(job["job_id"])
    assert request["status"] == "service_unavailable"
    assert "not configured" in request["error"]
    assert jobs.get(job["job_id"])["status"] == "published"
    assert ranking.snapshot(job["job_id"]) is None
    assert ranking.rank(job["job_id"])["request_id"] == request["request_id"]
    assert ranking.refresh(request["request_id"])["status"] == "service_unavailable"
    assert [e["kind"] for e in jobs.events()] == ["new_jd"]
    configured = service(jobs, lambda op, p: result(p))
    assert configured.rank(job["job_id"])["status"] == "service_unavailable"
    assert configured.rank(job["job_id"], rerank=True)["status"] == "ready"


def test_trusted_module_function_adapter_configuration(jobs, job, monkeypatch):
    fixture_module = ModuleType("_portal_fixture_adapter")
    fixture_module.rank = lambda op, p: result(p)
    monkeypatch.setitem(sys.modules, fixture_module.__name__, fixture_module)
    monkeypatch.setenv("HR_PORTAL_RANKING_ADAPTER", "_portal_fixture_adapter:rank")
    monkeypatch.setenv("HR_PORTAL_CANDIDATE_POOL", "fixture-pool")
    ranking = RankingService(jobs.path, jobs.principal)
    assert ranking.availability["available"]
    assert ranking.rank(job["job_id"])["status"] == "ready"
    for setting in ("does_not_exist:rank", "not a module", "_portal_fixture_adapter:missing"):
        monkeypatch.setenv("HR_PORTAL_RANKING_ADAPTER", setting)
        assert not RankingService(jobs.path, jobs.principal).availability["available"]


def test_pool_is_required_and_not_chosen_from_benchmark(jobs, job, monkeypatch):
    monkeypatch.delenv("HR_PORTAL_CANDIDATE_POOL", raising=False)
    ranking = RankingService(jobs.path, jobs.principal, adapter=lambda *args: pytest.fail("Must not infer"))
    assert not ranking.availability["available"]
    assert "pool" in ranking.availability["reason"]
    assert ranking.rank(job["job_id"])["status"] == "service_unavailable"


def test_invalid_configuration_is_unavailable_and_never_persisted_as_prose(jobs, job, monkeypatch):
    monkeypatch.setenv("HR_PORTAL_RANKING_ADAPTER", "private@example.com Bearer private-token")
    monkeypatch.setenv("HR_PORTAL_CANDIDATE_POOL", "private@example.com")
    ranking = RankingService(jobs.path, jobs.principal)
    assert not ranking.availability["available"]
    request = ranking.rank(job["job_id"])
    assert request["status"] == "service_unavailable"
    assert "private@example.com" not in json.dumps(request)
    assert b"private-token" not in jobs.path.read_bytes()


def test_backend_candidate_outside_pool_is_rejected(jobs, job):
    def adapter(op, payload):
        response = result(payload)
        response["eligible_candidate_ids"] = ["Applicant-1"]
        return response

    ranking = service(jobs, adapter)
    request = ranking.rank(job["job_id"])
    assert request["status"] == "failed"
    assert ranking.ready_snapshot(request["request_id"]) is None


def test_unpersistable_candidate_id_is_rejected_before_ready(jobs, job):
    def adapter(op, payload):
        response = result(payload)
        response["eligible_candidate_ids"][0] = "foo-1"
        response["candidates"][0]["candidate_id"] = "foo-1"
        return response

    ranking = service(jobs, adapter)
    request = ranking.rank(job["job_id"])
    assert request["status"] == "failed"
    assert ranking.snapshot(job["job_id"]) is None


@pytest.mark.parametrize("identity", [
    "Applicant-1", "1" * 64, "cv-" + "a" * 64,
    "12345678-1234-1234-1234-123456789012", "synthetic-cv-1-1",
])
def test_live_candidate_ids_survive_selection_chat_report_and_package(jobs, job, identity):
    from candidate_screener.app.portal_qa import answer
    from candidate_screener.app.portal_reports import build_package, build_report

    def adapter(op, payload):
        response = result(payload, 10)
        response["eligible_candidate_ids"][0] = identity
        response["candidates"][0]["candidate_id"] = identity
        return response

    ranking = service(jobs, adapter)
    assert ranking.rank(job["job_id"])["status"] == "ready"
    snapshot = ranking.snapshot(job["job_id"])
    turn = answer(snapshot, "What Python skills does candidate 1 have?")
    assert turn["context"]["selected_candidates"] == [identity]
    store = PortalStore(jobs.path)
    record = store.create_session(jobs.principal, "benchmark", job["job_id"], snapshot.digest,
                                  {"snapshot_id": snapshot.snapshot_id, "selected_candidates": [identity],
                                   "qa_turns": [{"question": "Candidate 1 Python skills", "answer": turn}]})
    persisted = store.load_session(jobs.principal, record["session_id"])["state"]
    assert persisted["selected_candidates"] == [identity]
    assert persisted["qa_turns"][0]["answer"]["context"]["selected_candidates"] == [identity]
    report = build_report(snapshot)
    assert report["candidate_ids"][0] == identity
    with zipfile.ZipFile(io.BytesIO(build_package(snapshot, report))) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["attachments"][0]["candidate_id"] == identity
        assert len(manifest["attachments"]) == 10


@pytest.mark.parametrize("mutation", [
    "wrong_request", "wrong_jd", "wrong_revision", "wrong_digest", "wrong_pool", "wrong_workspace",
    "wrong_authorization", "duplicate", "gap", "boolean_rank", "nan", "boolean_score", "too_many",
    "missing_source", "bad_digest", "fabricated_fit", "missing_model", "missing_semantics",
    "bad_handle", "credentials", "unknown_category", "bad_provenance", "bad_status",
])
def test_invalid_adapter_response_is_failed_and_never_persisted_raw(jobs, job, mutation):
    def adapter(op, payload):
        response = result(payload)
        if mutation == "wrong_request": response["request_id"] = "other"
        elif mutation == "wrong_jd": response["jd_id"] = "other"
        elif mutation == "wrong_revision": response["jd_revision"] = 2
        elif mutation == "wrong_digest": response["jd_digest"] = "a" * 64
        elif mutation == "wrong_pool": response["candidate_pool_ref"] = "other"
        elif mutation == "wrong_workspace": response["workspace"] = "other"
        elif mutation == "wrong_authorization": response["authorization_scope"] = "other"
        elif mutation == "duplicate": response["candidates"][1]["candidate_id"] = "Applicant-1"
        elif mutation == "gap": response["candidates"][1]["rank"] = 3
        elif mutation == "boolean_rank": response["candidates"][0]["rank"] = True
        elif mutation == "nan": response["candidates"][0]["score"] = float("nan")
        elif mutation == "boolean_score": response["candidates"][0]["score"] = True
        elif mutation == "too_many": response = result(payload, 11)
        elif mutation == "missing_source": response["candidates"][0].pop("source_ref")
        elif mutation == "bad_digest": response["candidates"][0]["source_digest"] = "a" * 64
        elif mutation == "fabricated_fit": response["candidates"][0]["fit"] = "Fit"
        elif mutation == "missing_model": response.pop("model_id")
        elif mutation == "missing_semantics": response.pop("score_semantics")
        elif mutation == "bad_handle": response["handle"] = "private@example.com"
        elif mutation == "credentials": response["candidates"][0]["evidence_provenance"] = {"token": "secret"}
        elif mutation == "unknown_category": response["candidates"][0]["entities"] = {"UNKNOWN": []}
        elif mutation == "bad_provenance": response["candidates"][0]["evidence_provenance"] = []
        elif mutation == "bad_status": response["status"] = "surprise"
        return response

    ranking = service(jobs, adapter)
    request = ranking.rank(job["job_id"])
    assert request["status"] == "failed"
    assert "private@example.com" not in json.dumps(request)
    assert request["handle"] is None
    assert ranking.snapshot(job["job_id"]) is None
    assert len([e for e in ranking.events() if e["kind"] == "ranking_failed"]) == 1


def test_missing_candidate_text_and_evidence_are_honest_and_no_fit_inference(jobs, job):
    def adapter(op, payload):
        response = result(payload)
        response["candidates"][0]["cv_text"] = None
        response["candidates"][0].pop("entities")
        response["candidates"][1].update(fit="No Fit", decision_provenance={"rule_version": "backend-v1"})
        return response

    ranking = service(jobs, adapter)
    ranking.rank(job["job_id"])
    first, second = ranking.snapshot(job["job_id"]).candidates
    assert first["cv_text"] is None and first["preview_status"] == "unavailable"
    assert all(v["status"] == "unavailable" for v in first["entities"].values())
    assert "fit" not in first
    assert second["fit"] == "No Fit" and second["decision_provenance"] == {"rule_version": "backend-v1"}


def test_new_revision_and_new_rerank_cannot_be_overwritten_by_old_completion(jobs, job):
    pending = set()

    def adapter(op, payload):
        if op == "submit":
            pending.add(payload["request_id"])
            return result(payload, status="running")
        return result(payload)

    ranking = service(jobs, adapter)
    old = ranking.rank(job["job_id"])
    updated = jobs.update(job["job_id"], 1, {"title": "New Engineer"})
    ranking.refresh(old["request_id"])
    assert ranking.ready_snapshot(old["request_id"]).jd_title == "Python Engineer"
    assert ranking.snapshot(job["job_id"]) is None
    assert jobs.get(job["job_id"])["ranking_status"] == "not_requested"
    with pytest.raises(RevisionConflict):
        ranking.rank(job["job_id"])
    jobs.review(job["job_id"], updated["revision"])
    jobs.publish(job["job_id"], updated["revision"])
    next_run = ranking.rank(job["job_id"])
    fresh = ranking.rank(job["job_id"], rerank=True, request_id="explicit-rerank")
    assert fresh["request_id"] != next_run["request_id"]
    assert ranking.rank(job["job_id"], rerank=True, request_id="explicit-rerank")["request_id"] == "explicit-rerank"
    ranking.refresh(next_run["request_id"])
    assert ranking.snapshot(job["job_id"]) is None
    assert jobs.get(job["job_id"])["active_request_id"] == fresh["request_id"]
    ranking.refresh(fresh["request_id"])
    assert ranking.snapshot(job["job_id"]).jd_title == "New Engineer"
    assert len(pending) == 3


def test_failed_rerank_preserves_historical_snapshot_without_calling_it_current(jobs, job):
    failing = [False]

    def adapter(op, payload):
        if failing[0]:
            raise RuntimeError("private@example.com Bearer private-token")
        return result(payload)

    ranking = service(jobs, adapter)
    first = ranking.rank(job["job_id"])
    old = ranking.ready_snapshot(first["request_id"])
    failing[0] = True
    failed = ranking.rank(job["job_id"], rerank=True)
    assert failed["status"] == "failed"
    assert ranking.snapshot(job["job_id"]) is None
    assert ranking.ready_snapshot(first["request_id"]).digest == old.digest
    assert "private-token" not in jobs.path.read_bytes().decode(errors="ignore")


def test_request_owner_workspace_revocation_and_id_reuse(jobs, job, principal):
    ranking = service(jobs, lambda op, p: result(p))
    request = ranking.rank(job["job_id"], request_id="owned-request")
    for other in (replace(principal, subject="another"), replace(principal, departments=("other",))):
        outsider = RankingService(jobs.path, other)
        assert outsider.requests() == []
        for action in (lambda: outsider.get(request["request_id"]),
                       lambda: outsider.refresh(request["request_id"]),
                       lambda: outsider.ready_snapshot(request["request_id"]),
                       lambda: outsider.rank(job["job_id"])):
            with pytest.raises(AuthorizationError):
                action()
        if "benchmark" in other.departments:
            assert outsider.snapshot(job["job_id"]).digest == ranking.snapshot(job["job_id"]).digest
        else:
            with pytest.raises(AuthorizationError):
                outsider.snapshot(job["job_id"])
    another = jobs.create({"title": "Another", "department": "Tech", "responsibilities": ["Other work"]})
    jobs.review(another["job_id"], 1)
    jobs.publish(another["job_id"], 1)
    with pytest.raises(RevisionConflict, match="another JD"):
        ranking.rank(another["job_id"], request_id="owned-request")
    assert len(ranking.requests(job["job_id"])) == 1
    jobs.delete(job["job_id"], 1)
    assert ranking.ready_snapshot(request["request_id"]) is not None  # Authorized history.
    with pytest.raises(AuthorizationError):
        ranking.rank(job["job_id"])


def test_shared_ready_snapshot_without_private_request_access_or_stale_publication(jobs, job, principal):
    ranking = service(jobs, lambda op, p: result(p))
    request = ranking.rank(job["job_id"])
    reviewer = RankingService(jobs.path, replace(principal, subject="reviewer-two"))
    viewer_jobs = JobService(jobs.path, reviewer.principal)
    original = ranking.snapshot(job["job_id"])
    assert reviewer.snapshot(job["job_id"]).digest == original.digest
    assert viewer_jobs.get(job["job_id"])["ranking_status"] == "ready"
    assert viewer_jobs.get(job["job_id"])["can_edit"] is False
    assert {e["kind"] for e in reviewer.events()} == {"new_jd"}
    assert reviewer.requests() == []
    for action in (lambda: reviewer.get(request["request_id"]),
                   lambda: reviewer.ready_snapshot(request["request_id"]),
                   lambda: reviewer.refresh(request["request_id"]),
                   lambda: reviewer.requests(job["job_id"]),
                   lambda: reviewer.rank(job["job_id"]),
                   lambda: reviewer.rank(job["job_id"], rerank=True)):
        with pytest.raises(AuthorizationError):
            action()
    # Readers keep the published result during a private edit; authors see their
    # new draft without calling an older ranking current for that draft.
    jobs.update(job["job_id"], 1, {"title": "Private new title"})
    assert ranking.snapshot(job["job_id"]) is None
    assert reviewer.snapshot(job["job_id"]).digest == original.digest
    with pytest.raises(AuthorizationError):
        reviewer.snapshot(job["job_id"], 2)
    jobs.review(job["job_id"], 2)
    jobs.publish(job["job_id"], 2)
    assert reviewer.snapshot(job["job_id"]) is None
    assert reviewer.snapshot(job["job_id"], 1) is None
    assert viewer_jobs.get(job["job_id"])["ranking_status"] == "not_requested"
    current = ranking.rank(job["job_id"])
    assert current["request_id"] != request["request_id"]
    assert reviewer.snapshot(job["job_id"]).jd_title == "Private new title"
    assert reviewer.snapshot(job["job_id"]).source_flags["jd_revision"] == 2
    jobs.delete(job["job_id"], 2)
    with pytest.raises(AuthorizationError):
        reviewer.snapshot(job["job_id"])
    assert ranking.ready_snapshot(request["request_id"]).digest == original.digest


def test_revocation_during_adapter_call_blocks_result_persistence(jobs, job):
    live = [True]

    def adapter(op, payload):
        live[0] = False
        return result(payload)

    ranking = service(jobs, adapter, guard=lambda: live[0])
    with pytest.raises(AuthorizationError, match="revoked"):
        ranking.rank(job["job_id"], request_id="revoked-request")
    reopened = service(jobs, adapter)
    assert reopened.get("revoked-request")["status"] == "queued"
    assert reopened.ready_snapshot("revoked-request") is None


def test_concurrent_double_submit_and_refresh_invoke_once(jobs, job):
    entered, release = Event(), Event()
    calls = []

    def adapter(op, payload):
        calls.append(op)
        entered.set()
        assert release.wait(3)
        return result(payload, status="queued" if op == "submit" else "ready")

    ranking = service(jobs, adapter)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(ranking.rank, job["job_id"], request_id="concurrent-request")
        assert entered.wait(3)
        assert ranking.rank(job["job_id"], request_id="concurrent-request")["status"] == "queued"
        assert ranking.refresh("concurrent-request")["status"] == "queued"
        release.set()
        assert future.result(timeout=3)["status"] == "queued"
    entered.clear()
    release.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(ranking.refresh, "concurrent-request")
        assert entered.wait(3)
        assert ranking.refresh("concurrent-request")["status"] == "queued"
        release.set()
        assert future.result(timeout=3)["status"] == "ready"
    assert calls == ["submit", "status"]


def test_interrupted_submission_recovers_by_status_without_resubmitting(jobs, job):
    calls = []

    def adapter(op, payload):
        calls.append(op)
        if op == "submit":
            raise KeyboardInterrupt("synthetic process interruption")
        assert payload["handle"] is None
        return result(payload)

    ranking = service(jobs, adapter)
    with pytest.raises(KeyboardInterrupt):
        ranking.rank(job["job_id"], request_id="interrupted-request")
    with sqlite3.connect(jobs.path) as db:
        db.execute("UPDATE operational_ranking_requests SET processing_until=0 WHERE request_id='interrupted-request'")
    assert ranking.rank(job["job_id"])["status"] == "queued"
    assert ranking.refresh("interrupted-request")["status"] == "ready"
    assert calls == ["submit", "status"]


def test_refresh_unavailable_adapter_preserves_live_handle(jobs, job, monkeypatch):
    ranking = service(jobs, lambda op, p: result(p, status="running" if op == "submit" else "ready"))
    request = ranking.rank(job["job_id"])
    monkeypatch.delenv("HR_PORTAL_RANKING_ADAPTER", raising=False)
    unavailable = service(jobs)
    response = unavailable.refresh(request["request_id"])
    assert response["refresh_unavailable"]
    assert response["status"] == "running" and response["handle"] == "fixture-handle"
    assert ranking.refresh(request["request_id"])["status"] == "ready"


def test_source_digest_and_ready_snapshot_provenance_persist(jobs, job):
    def adapter(op, payload):
        response = result(payload)
        response["evidence_provenance"] = {"extractor": "fixture-extractor"}
        response["jd_entities"] = {"SKILL": ["Python"]}
        row = response["candidates"][0]
        row["source_digest"] = hashlib.sha256(row["cv_text"].encode()).hexdigest()
        return response

    ranking = service(jobs, adapter)
    request = ranking.rank(job["job_id"])
    snapshot = ranking.ready_snapshot(request["request_id"])
    assert snapshot.source_flags["candidate_pool_version"] == "fixture-pool-v1"
    assert snapshot.source_flags["evidence_provenance"] == {"extractor": "fixture-extractor"}
    assert snapshot.jd_entities["SKILL"]["mentions"] == ["Python"]
    assert snapshot.candidates[0]["source_digest"] == hashlib.sha256(
        b"Invented applicant 1. Python private@example.com").hexdigest()
    with sqlite3.connect(jobs.path) as db:
        assert db.execute("SELECT version FROM operational_schema WHERE component='ranking'").fetchone() == (1,)


@pytest.mark.parametrize("system", ["D63", "D64"])
def test_ranking_system_selects_the_model_in_the_payload(jobs, job, system):
    calls = []
    ranking = service(jobs, lambda op, p: calls.append(deepcopy(p)) or result(p))
    assert ranking.rank(job["job_id"], system=system)["status"] == "ready"
    assert calls[0]["ranking_system"] == system
    assert ranking.get(calls[0]["request_id"])["payload"]["ranking_system"] == system


@pytest.mark.parametrize("system", ["X2", "d64", "", None, 64, ["D64"]])
def test_unknown_ranking_system_is_refused_before_any_write(jobs, job, system):
    ranking = service(jobs, lambda op, p: pytest.fail("Must not infer"))
    with pytest.raises(ValueError):
        ranking.rank(job["job_id"], system=system)
    assert ranking.requests() == []
    assert jobs.get(job["job_id"])["ranking_status"] == "not_requested"
    calls = []
    ranking.adapter = lambda op, p: calls.append(p) or result(p)
    ranking.rank(job["job_id"])
    assert calls[0]["ranking_system"] == "D63"
