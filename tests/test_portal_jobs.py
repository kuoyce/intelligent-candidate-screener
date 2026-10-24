"""Synthetic operational authoring invariants; never opens a source corpus."""
from dataclasses import replace
import json
import sqlite3
import time

import pytest

from candidate_screener.app.portal_jobs import (
    GUIDED_FIELDS, JobService, guided_question, render_jd, validate_fields,
)
from candidate_screener.app.portal_store import (
    AuthorizationError, PortalStore, Principal, RevisionConflict,
)


@pytest.fixture
def principal():
    return Principal("author-one", ("benchmark",), time.time() + 3600, "local_token")


@pytest.fixture
def fields():
    return {"title": "Software Engineer", "department": "Technology",
            "responsibilities": {"required": ["Build reliable services"], "preferred": []},
            "skills": {"required": ["Python"], "preferred": ["SQL"]},
            "knowledge": {"required": ["Distributed systems"], "preferred": []},
            "min_years": 0, "max_years": None,
            "qualifications": {"required": [], "preferred": ["Relevant degree"]},
            "location": "Singapore", "work_arrangement": "Hybrid"}


@pytest.fixture
def service(tmp_path, principal):
    return JobService(tmp_path / "workspace.sqlite", principal)


def test_owner_persisted_crud_and_immutable_revision_history(service, fields, principal):
    original = service.create(fields)
    assert original["revision"] == 1 and original["status"] == "draft"
    fields["skills"]["required"].append("Should not mutate saved requirements")
    assert service.get(original["job_id"])["fields"]["skills"]["required"] == ["Python"]
    new = service.update(original["job_id"], 1, {"title": "Senior Engineer"})
    assert new["revision"] == 2 and new["status"] == "draft"
    reopened = JobService(service.path, principal)
    assert reopened.list()[0]["title"] == "Senior Engineer"
    assert reopened.get(original["job_id"], 1)["title"] == "Software Engineer"
    assert reopened.get(original["job_id"], 1)["is_current"] is False
    assert reopened.update(new["job_id"], 2, {"title": "Senior Engineer"})["revision"] == 2
    with pytest.raises(RevisionConflict):
        reopened.update(original["job_id"], 1, {"title": "Lost update"})
    reopened.delete(original["job_id"], 2)
    assert reopened.list() == []
    assert reopened.delete(original["job_id"], 2)["deleted"]
    with pytest.raises(AuthorizationError):
        reopened.get(original["job_id"])


def test_publication_requires_explicit_review_of_current_revision(service, fields):
    job = service.create(fields)
    with pytest.raises(ValueError, match="Review the final"):
        service.publish(job["job_id"], 1)
    service.review(job["job_id"], 1)
    assert service.publish(job["job_id"], 1)["status"] == "published"
    assert service.publish(job["job_id"], 1)["status"] == "published"
    changed = service.update(job["job_id"], 1, {"skills": {"required": ["SQL"]}})
    assert changed["status"] == "draft"
    assert service.get(job["job_id"], 1)["status"] == "published"
    with pytest.raises(ValueError, match="Review the final"):
        service.publish(job["job_id"], 2)
    with pytest.raises(RevisionConflict):
        service.publish(job["job_id"], 1)


def test_publication_alert_atomic_deduplicated_and_reads_explicit(service, fields, monkeypatch):
    job = service.create(fields)
    service.review(job["job_id"], 1)
    original = service._event

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic transaction failure")

    monkeypatch.setattr(service, "_event", fail)
    with pytest.raises(sqlite3.OperationalError):
        service.publish(job["job_id"], 1)
    assert service.get(job["job_id"])["status"] == "reviewed"
    monkeypatch.setattr(service, "_event", original)
    service.publish(job["job_id"], 1)
    service.publish(job["job_id"], 1)
    events = service.events()
    assert len(events) == 1 and events[0]["kind"] == "new_jd"
    assert len(service.events(unread_only=True)) == 1
    event_id = events[0]["event_id"]
    service.mark_read(event_id)
    assert service.events(unread_only=True) == []
    service.mark_read(event_id, read=False)
    assert len(service.events(unread_only=True)) == 1


def test_owner_workspace_and_auth_realm_isolation(service, fields, principal):
    job = service.create(fields)
    service.review(job["job_id"], 1)
    service.publish(job["job_id"], 1)
    event_id = service.events()[0]["event_id"]
    for other in (replace(principal, departments=("other",)),
                  replace(principal, auth_mode="demo")):
        outsider = JobService(service.path, other)
        assert outsider.list() == []
        assert outsider.events() == []
        for action in (lambda: outsider.get(job["job_id"]),
                       lambda: outsider.update(job["job_id"], 1, {"title": "Takeover"}),
                       lambda: outsider.review(job["job_id"], 1),
                       lambda: outsider.publish(job["job_id"], 1),
                       lambda: outsider.delete(job["job_id"], 1),
                       lambda: outsider.mark_read(event_id)):
            with pytest.raises(AuthorizationError):
                action()
    with pytest.raises(AuthorizationError):
        service.create(fields, workspace="other")


def test_publication_shared_between_reviewers_with_private_drafts_and_independent_reads(service, fields, principal):
    reviewer = JobService(service.path, replace(principal, subject="reviewer-two"))
    draft = service.create(fields)
    assert reviewer.list() == [] and reviewer.events() == []
    with pytest.raises(AuthorizationError):
        reviewer.get(draft["job_id"])
    service.review(draft["job_id"], 1)
    with pytest.raises(AuthorizationError):
        reviewer.get(draft["job_id"])
    published = service.publish(draft["job_id"], 1)
    shared = reviewer.get(draft["job_id"])
    assert shared["title"] == fields["title"] and shared["can_edit"] is False
    assert shared["status"] == "published" and shared["is_current"]
    assert service.get(draft["job_id"])["can_edit"] is True
    assert [j["job_id"] for j in reviewer.list()] == [draft["job_id"]]
    assert reviewer.list(owned_only=True) == []
    assert [j["job_id"] for j in service.list(owned_only=True)] == [draft["job_id"]]
    event_id = service.events()[0]["event_id"]
    assert reviewer.events()[0]["event_id"] == event_id
    assert reviewer.events()[0]["audience"] == "workspace"
    service.mark_read(event_id)
    assert service.events(unread_only=True) == []
    assert reviewer.events()[0]["read_at"] is None
    reviewer.mark_read(event_id)
    service.mark_read(event_id, read=False)
    assert reviewer.events(unread_only=True) == []
    assert len(service.events(unread_only=True)) == 1
    assert JobService(service.path, reviewer.principal).events(unread_only=True) == []

    # A newer draft must not leak content, revision numbers or private edit time.
    private = service.update(draft["job_id"], 1, {"title": "PRIVATE DRAFT TITLE"})
    service.review(draft["job_id"], 2)
    assert private["revision"] == 2
    assert reviewer.get(draft["job_id"]) == shared
    assert reviewer.list()[0]["revision"] == 1
    with pytest.raises(AuthorizationError):
        reviewer.get(draft["job_id"], 2)
    for action in (lambda: reviewer.update(draft["job_id"], 1, {"title": "Takeover"}),
                   lambda: reviewer.review(draft["job_id"], 1),
                   lambda: reviewer.publish(draft["job_id"], 1),
                   lambda: reviewer.delete(draft["job_id"], 1),
                   lambda: reviewer.next_question(draft["job_id"])):
        with pytest.raises(AuthorizationError):
            action()
    # All non-publication event types default to private, including deliveries.
    with service._db() as db:
        for kind in ("ranking_ready", "ranking_failed", "email_captured", "email_failed"):
            service._event(db, f"private:{kind}", published, kind, {"message": "Private update"})
    assert {e["kind"] for e in reviewer.events()} == {"new_jd"}
    for event in service.events():
        if event["kind"] != "new_jd":
            with pytest.raises(AuthorizationError):
                reviewer.mark_read(event["event_id"])
    service.publish(draft["job_id"], 2)
    assert reviewer.get(draft["job_id"])["revision"] == 2
    assert len(reviewer.events()) == 2
    assert reviewer.get(draft["job_id"], 1)["is_current"] is False
    service.delete(draft["job_id"], 2)
    assert reviewer.list() == []
    with pytest.raises(AuthorizationError):
        reviewer.get(draft["job_id"])


def test_revocation_guard_rechecked_on_every_operation(tmp_path, principal, fields):
    live = [True]
    service = JobService(tmp_path / "workspace.sqlite", principal, lambda: live[0])
    job = service.create(fields)
    live[0] = False
    for action in (service.list, service.events, lambda: service.get(job["job_id"]),
                   lambda: service.create(fields), lambda: service.delete(job["job_id"], 1)):
        with pytest.raises(AuthorizationError, match="revoked"):
            action()
    with pytest.raises(AuthorizationError):
        JobService(service.path, replace(principal, expires_at=0))


def test_no_return_guard_is_success(tmp_path, principal, fields):
    service = JobService(tmp_path / "workspace.sqlite", principal, lambda: None)
    assert service.create(fields)["status"] == "draft"


@pytest.mark.parametrize("change", [
    {"min_years": -1}, {"max_years": 101}, {"min_years": True},
    {"min_years": float("nan")}, {"max_years": float("inf")},
    {"min_years": 5, "max_years": 2}, {"min_years": "3"},
    {"skills": {"required": ["Python"], "preferred": ["Python"]}},
    {"skills": {"mandatory": ["Python"]}}, {"title": "x" * 201},
    {"skills": {"required": [""]}}, {"owner": "another-user"},
    {"location": "Bearer sensitive-key"}, {"skills": ["x"] * 41},
])
def test_invalid_fields_rejected_without_persistence(service, fields, change):
    with pytest.raises(ValueError):
        service.create({**fields, **change})
    assert service.list() == []


def test_guided_flow_skip_and_full_deterministic_render(service, fields):
    draft = service.create()
    assert service.next_question(draft["job_id"])["field"] == "title"
    for field in GUIDED_FIELDS:
        assert service.next_question(draft["job_id"])["field"] == field
        draft = service.update(draft["job_id"], draft["revision"], {field: fields[field]})
    assert service.next_question(draft["job_id"]) is None
    assert guided_question(fields, GUIDED_FIELDS) is None
    text = render_jd(fields)
    assert text == render_jd(dict(reversed(list(fields.items()))))
    assert "Required:\n- Python\nPreferred:\n- SQL" in text
    assert "Experience: at least 0 years" in text and "Minimum years" not in text
    assert "Relevant degree" in text and "Work arrangement: Hybrid" in text
    assert validate_fields({"skills": ["Python"]})["skills"]["required"] == ["Python"]
    assert "Not specified" in render_jd({})
    with pytest.raises(ValueError, match="title"):
        service.review(service.create()["job_id"], 1)


def test_text_redacted_before_persistence_and_migrations_do_not_touch_portal_store(service, fields, principal):
    store = PortalStore(service.path)
    legacy = store.create_session(principal, "benchmark", "synthetic-jd-1", "source", {})
    job = service.create({**fields, "title": "Engineer private@example.com",
                          "skills": {"required": ["Python https://example.com/private"]}})
    service.review(job["job_id"], 1)
    service.publish(job["job_id"], 1)
    assert "private@example.com" not in json.dumps(service.get(job["job_id"]))
    assert "https://example.com" not in json.dumps(service.events())
    assert "[EMAIL]" in job["jd_text"]
    assert b"private@example.com" not in service.path.read_bytes()
    assert store.load_session(principal, legacy["session_id"])["revision"] == 0
    assert store.list_events(principal) == []
    with sqlite3.connect(service.path) as db:
        assert db.execute("SELECT version FROM operational_schema WHERE component='jobs'").fetchone() == (2,)
        db.execute("UPDATE operational_schema SET version=999 WHERE component='jobs'")
    with pytest.raises(ValueError, match="Unsupported"):
        JobService(service.path, principal)


def test_jobs_v2_additive_migration_preserves_original_reads_and_legacy_store(service, fields, principal):
    store = PortalStore(service.path)
    legacy = store.create_session(principal, "benchmark", "synthetic-jd-1", "source", {})
    store.ingest_event(principal, "legacy-portal-event", "benchmark", "new_jd", "synthetic-jd-1", {})
    store.mark_event_read(principal, "legacy-portal-event")
    job = service.create(fields)
    service.review(job["job_id"], 1)
    published = service.publish(job["job_id"], 1)
    publication = service.events()[0]["event_id"]
    with service._db() as db:
        service._event(db, "old-private-ranking", published, "ranking_ready", {"message": "Private update"})
    old_read_at = time.time() - 60
    with sqlite3.connect(service.path) as db:
        # Reconstruct the exact pre-v2 additive boundary in this temporary DB.
        db.execute("DROP TABLE operational_event_reads")
        db.execute("DROP TABLE operational_event_audiences")
        db.execute("UPDATE operational_schema SET version=1 WHERE component='jobs'")
        db.execute("UPDATE operational_events SET read_at=?", (old_read_at,))
    migrated = JobService(service.path, principal)
    assert all(e["read_at"] == old_read_at for e in migrated.events())
    reviewer = JobService(service.path, replace(principal, subject="reviewer-two"))
    assert len(reviewer.events()) == 1
    assert reviewer.events()[0]["event_id"] == publication
    assert reviewer.events()[0]["read_at"] is None
    migrated.mark_read(publication, read=False)
    reopened = JobService(service.path, principal)
    assert reopened.events(unread_only=True)[0]["event_id"] == publication
    with sqlite3.connect(service.path) as db:
        assert db.execute("SELECT version FROM operational_schema WHERE component='jobs'").fetchone() == (2,)
        assert db.execute("SELECT read_at FROM operational_events WHERE event_id=?", (publication,)).fetchone() == (old_read_at,)
    assert store.load_session(principal, legacy["session_id"])["revision"] == 0
    assert store.list_events(principal)[0]["read_at"] is not None


@pytest.mark.parametrize("low", [None, 0, 0.5, 1, 1.5, 2, 3, 10, 20, 100])
@pytest.mark.parametrize("high", [None, 0, 0.5, 1, 1.5, 2, 3, 10, 20, 100])
def test_rendered_experience_reads_back_through_the_ranking_extractor(fields, low, high):
    """The model reads the JD text only (D53), so the entered years must survive as text.
    Its extractor ignores requirements above JD_MAX_YEARS and reads "up to" as from zero."""
    from candidate_screener.extraction.experience import JD_MAX_YEARS, jd_experience
    if low is not None and high is not None and low > high:
        pytest.skip("validate_fields refuses a minimum above the maximum")
    found = jd_experience(render_jd({**fields, "min_years": low, "max_years": high}))
    if any(value is not None and value > JD_MAX_YEARS for value in (low, high)):
        expected = (None, None)
    else:
        expected = (0 if low is None and high is not None else low, high)
    assert (found.min_years, found.max_years) == expected
