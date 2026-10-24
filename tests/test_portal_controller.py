"""Guarded end-to-end operations on synthetic ranking and source fixtures."""
from dataclasses import replace
import io
import time
import zipfile

import pytest

from candidate_screener.app.portal_controller import PortalController, StaleReview
from candidate_screener.app.portal_data import synthetic_demo
from candidate_screener.app.portal_store import PortalStore, Principal, AuthorizationError


@pytest.fixture
def controller(tmp_path):
    principal = Principal("reviewer", ("demo",), time.time()+3600, "demo")
    return PortalController(synthetic_demo(), PortalStore(tmp_path / "store.sqlite"), principal)


def test_report_and_ten_cv_package_use_same_backend_snapshot(controller):
    record = controller.open_job(controller.jobs()[0]["jd_id"])
    report, content = controller.package(record["session_id"])
    assert report["complete"] and len(report["candidate_ids"]) == 10
    assert report["top8_ids"] == report["candidate_ids"][:8]
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert len([n for n in archive.namelist() if n.startswith("cvs/")]) == 10
    saved = controller.historical_report(record["session_id"])
    assert saved["report_sha256"] == report["report_sha256"]


def test_demo_identity_cannot_open_real_mode_even_in_shared_store(controller):
    with pytest.raises(PermissionError, match="Demonstration"):
        PortalController(replace(controller.dataset, is_demo=False), controller.store,
                         replace(controller.principal, departments=("benchmark",)))


def test_revocation_is_rechecked_before_every_preview_and_session_read(controller):
    access = {"allowed": True}
    principal = Principal("local", ("benchmark",), time.time()+3600, "local_token")
    service = PortalController(replace(controller.dataset, is_demo=False), controller.store, principal,
                               auth_guard=lambda: access["allowed"])
    record = service.open_job(service.jobs()[0]["jd_id"])
    candidate = service.snapshot(record["session_id"]).candidates[0]["cv_id"]
    access["allowed"] = False
    for action in (lambda: service.jobs(), lambda: service.load_session(record["session_id"]),
                   lambda: service.preview(record["session_id"], candidate),
                   lambda: service.report(record["session_id"]), lambda: service.events()):
        with pytest.raises(PermissionError):
            action()


def test_changed_source_cannot_generate_new_report_or_preview(controller):
    record = controller.open_job(controller.jobs()[0]["jd_id"])
    controller.freshness_guard = lambda: False
    with pytest.raises(StaleReview):
        controller.report(record["session_id"])
    with pytest.raises(StaleReview):
        controller.preview(record["session_id"], "untrusted-id")


def test_foreign_candidate_and_session_are_not_accepted(controller):
    record = controller.open_job(controller.jobs()[0]["jd_id"])
    with pytest.raises(PermissionError):
        controller.preview(record["session_id"], "candidate-outside-this-job")
    other = replace(controller.principal, subject="different-reviewer")
    service = PortalController(controller.dataset, controller.store, other)
    with pytest.raises(AuthorizationError):
        service.load_session(record["session_id"])


def test_unconfigured_mail_never_becomes_sent(controller):
    record = controller.open_job(controller.jobs()[0]["jd_id"])
    directory = {record["jd_id"]: {"dept": {"department": "demo"}}}
    draft = controller.prepare_handoff(record["session_id"], "dept", directory)
    result = controller.confirm_handoff(record["session_id"], draft["package_id"], "confirmation1")
    assert result["status"] == "draft" and not result["gateway_available"]
    assert result["receipt"] is None
