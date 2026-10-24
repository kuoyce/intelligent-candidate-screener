"""Live ranking cannot replace CSV scores or disable existing export reviews."""
from dataclasses import replace
import time

from candidate_screener.app.portal_controller import PortalController
from candidate_screener.app.portal_data import synthetic_demo
from candidate_screener.app.portal_ranking import RankingService
from candidate_screener.app.portal_store import PortalStore, Principal
from candidate_screener.app.portal_workspace import attach_services


def ready_adapter(operation, payload):
    response = {key: payload[key] for key in ("request_id", "jd_id", "jd_revision", "jd_digest", "workspace", "authorization_scope", "candidate_pool_ref")}
    response.update(status="ready", run_id="fixture-run", model_id="fixture-live-model",
                    candidate_pool_version="fixture-v1", score_semantics="Fixture ordering score, not a probability",
                    eligible_candidate_ids=["Applicant-1"], candidates=[{
                        "candidate_id": "Applicant-1", "rank": 1, "score": 0.3,
                        "source_ref": "fixture:one", "cv_text": "Invented applicant. Python services."}])
    return response


def test_csv_and_live_snapshots_remain_independent_with_model_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("HR_PORTAL_RANKING_ADAPTER", raising=False)
    monkeypatch.delenv("HR_PORTAL_CANDIDATE_POOL", raising=False)
    exports = replace(synthetic_demo(), is_demo=False)
    principal = Principal("reviewer", ("benchmark",), time.time()+3600, "local_token")
    controller = attach_services(PortalController(exports, PortalStore(tmp_path/"workspace.sqlite"), principal, auth_guard=lambda: True))
    jd_id = exports.jobs()[0]["jd_id"]
    original = exports.snapshot(jd_id)
    assert not controller.ranking_service.availability["available"]
    record = controller.open_job(jd_id)
    assert controller.snapshot(record["session_id"]).digest == original.digest
    job = controller.job_service.create({"title": "New Python role", "department": "Tech", "responsibilities": ["Build APIs"]})
    controller.job_service.review(job["job_id"], job["revision"])
    controller.job_service.publish(job["job_id"], job["revision"])
    ranking = RankingService(controller.store.path, principal, guard=controller._guard, adapter=ready_adapter, pool_ref="fixture-pool")
    controller.ranking_service = controller.dataset.ranking_service = ranking
    ranking.rank(job["job_id"])
    live = controller.open_job(job["job_id"])
    assert len(controller.snapshot(live["session_id"]).candidates) == 1
    assert controller.snapshot(record["session_id"]).candidates == original.candidates
    ranking.adapter = None
    assert controller.snapshot(record["session_id"]).digest == original.digest
    assert controller.dataset.snapshot(job["job_id"]).model_id == "fixture-live-model"
