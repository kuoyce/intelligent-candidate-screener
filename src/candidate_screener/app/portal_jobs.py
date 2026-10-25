"""Private JD authoring and published workspace views, independent of exports.

Public records are dictionaries. ``revision`` is an optimistic concurrency token:
update creates a new immutable content revision in draft state; review/publish
operate only on the current revision. Delete archives the job, retaining history.
get/list include published jobs from the same authorized workspace/auth realm;
can_edit is true only for the author. Other reviewers see the latest published
revision even during a newer private draft. list(owned_only=True) is for authoring.
Publication events have a workspace audience and independent recipient read
states. All other operational events and all mutations remain owner-only.
Requirements use {"required": [str], "preferred": [str]}; strings/lists are also
accepted as required requirements. None/empty optional fields mean unspecified.
The optional guard is the controller's zero-argument ``_guard`` (None is success).
No method publishes externally, runs a model, or edits a research artifact.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid

from candidate_screener.annotation.redact import redact_text
from .portal_store import (
    DEFAULT_PATH, AuthorizationError, PortalStore, RevisionConflict, _subject,
)

REQUIREMENT_FIELDS = ("responsibilities", "skills", "knowledge", "qualifications")
GUIDED_FIELDS = ("title", "department", "responsibilities", "skills", "knowledge",
                 "min_years", "max_years", "qualifications", "location", "work_arrangement")
_QUESTIONS = (
    "What is the job title?", "Which hiring department owns this role?",
    "What are the required and preferred responsibilities?",
    "Which skills are required, and which are preferred?",
    "What domain knowledge is required or preferred?",
    "What is the minimum experience in years, or is it unspecified?",
    "What is the maximum experience in years, or is it unspecified?",
    "Which qualifications are required or preferred?",
    "What is the location, or is it unspecified?",
    "What is the work arrangement, or is it unspecified?",
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _text(value, name, limit=2000, *, empty=True):
    if not isinstance(value, str) or len(value) > limit or any(
        ord(c) < 32 and c not in "\n\t" for c in value
    ):
        raise ValueError(f"Invalid {name}")
    if re.search(r"\bBearer\s+\S+", value, re.I):
        raise ValueError("Credentials cannot be persisted")
    value = redact_text(value.strip())
    if not empty and not value:
        raise ValueError(f"{name} is required")
    return value


def _identity(value, name="identity"):
    # IDs are typed routing values, not prose: reject emails/URLs/control chars.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", value):
        raise ValueError(f"Invalid {name}")
    return value


def _revision(value):
    if type(value) is not int or value < 1:
        raise ValueError("Invalid JD revision")
    return value


def validate_fields(fields, *, complete=False):
    """Validate/redact a full or partial field map, without inferring criteria."""
    if not isinstance(fields, dict) or set(fields) - set(GUIDED_FIELDS):
        raise ValueError("Unknown JD fields")
    result = {}
    for name in GUIDED_FIELDS:
        value = fields.get(name)
        if name in REQUIREMENT_FIELDS:
            if value is None:
                value = {}
            if isinstance(value, (str, list)):
                value = {"required": [value] if isinstance(value, str) else value}
            if not isinstance(value, dict) or set(value) - {"required", "preferred"}:
                raise ValueError(f"Invalid {name} requirements")
            result[name] = {}
            for level in ("required", "preferred"):
                items = value.get(level, [])
                if not isinstance(items, list) or len(items) > 40:
                    raise ValueError(f"Invalid {name} {level} list")
                cleaned = [_text(v, name, 1000, empty=False) for v in items]
                result[name][level] = list(dict.fromkeys(cleaned))
            if set(result[name]["required"]) & set(result[name]["preferred"]):
                raise ValueError(f"{name} cannot be both required and preferred")
        elif name in ("min_years", "max_years"):
            if value is not None and (type(value) not in (int, float)
                    or not math.isfinite(value) or not 0 <= value <= 100):
                raise ValueError(f"{name} must be between 0 and 100 or unspecified")
            result[name] = value
        else:
            result[name] = _text("" if value is None else value, name, 200)
    low, high = result["min_years"], result["max_years"]
    if low is not None and high is not None and low > high:
        raise ValueError("Minimum experience cannot exceed maximum experience")
    if complete:
        for name in ("title", "department"):
            if not result[name]:
                raise ValueError(f"{name} is required before review")
        if not any(result["responsibilities"].values()):
            raise ValueError("At least one responsibility is required before review")
    if len(_json(result).encode()) > 60_000:
        raise ValueError("JD exceeds storage limit")
    return result


def _experience(low, high):
    """One phrase the ranking model's experience rules read back from the text (D53):
    "2-6 years", "at least 2 years", "up to 6 years" (read as 0-6) or "Not specified"."""
    unit = lambda value: "year" if value == 1 else "years"
    if low is not None and high is not None:
        return f"{low:g}-{high:g} years"
    if low is not None:
        return f"at least {low:g} {unit(low)}"
    return f"up to {high:g} {unit(high)}" if high is not None else "Not specified"


def render_jd(fields):
    """Faithful deterministic full text, including explicit unspecified fields."""
    data = validate_fields(fields)
    lines = [data["title"] or "Title: Not specified",
             "Department: " + (data["department"] or "Not specified")]
    for name, label in (("responsibilities", "Responsibilities"), ("skills", "Skills"),
                        ("knowledge", "Domain knowledge"), ("qualifications", "Qualifications")):
        lines.extend(["", label])
        for level in ("required", "preferred"):
            items = data[name][level]
            lines.append(level.capitalize() + ":")
            lines.extend(["- " + item for item in items] or ["Not specified"])
    lines.extend(["", "Experience: " + _experience(data["min_years"], data["max_years"]),
                  "", "Location: " + (data["location"] or "Not specified"),
                  "Work arrangement: " + (data["work_arrangement"] or "Not specified")])
    return "\n".join(lines)


def guided_question(fields, confirmed_fields=()):
    """One stable next question; explicitly skipped optional fields stay skipped."""
    validate_fields(fields)
    confirmed = set(confirmed_fields)
    if confirmed - set(GUIDED_FIELDS):
        raise ValueError("Unknown confirmed JD field")
    for name, question in zip(GUIDED_FIELDS, _QUESTIONS, strict=True):
        if name not in confirmed:
            return {"field": name, "question": question,
                    "optional": name not in {"title", "department", "responsibilities"}}
    return None


class _OperationalService:
    def __init__(self, path=DEFAULT_PATH, principal=None, guard=None):
        self.path, self.principal, self.guard = Path(path), principal, guard
        self._guard()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        self.path.chmod(0o600)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS operational_schema(component TEXT PRIMARY KEY, version INTEGER NOT NULL)")
            version = db.execute("SELECT version FROM operational_schema WHERE component='jobs'").fetchone()
            if version and version[0] not in {1, 2}:
                raise ValueError("Unsupported operational jobs schema version")
            if not version:
                for statement in (
                    """CREATE TABLE operational_jobs(
                    job_id TEXT PRIMARY KEY,owner TEXT NOT NULL,workspace TEXT NOT NULL,
                    revision INTEGER NOT NULL,created_at REAL NOT NULL,updated_at REAL NOT NULL,
                    deleted_at REAL,active_request_id TEXT,ranking_status TEXT NOT NULL)""",
                    """CREATE TABLE operational_job_revisions(
                    job_id TEXT NOT NULL,revision INTEGER NOT NULL,status TEXT NOT NULL,
                    fields_json TEXT NOT NULL,jd_text TEXT NOT NULL,content_digest TEXT NOT NULL,
                    confirmed_json TEXT NOT NULL,provenance TEXT NOT NULL,created_at REAL NOT NULL,
                    reviewed_at REAL,published_at REAL,PRIMARY KEY(job_id,revision),
                    FOREIGN KEY(job_id) REFERENCES operational_jobs(job_id))""",
                    """CREATE TABLE operational_events(
                    event_id TEXT PRIMARY KEY,owner TEXT NOT NULL,workspace TEXT NOT NULL,
                    kind TEXT NOT NULL,jd_id TEXT NOT NULL,revision INTEGER NOT NULL,
                    request_id TEXT,body_json TEXT NOT NULL,created_at REAL NOT NULL,read_at REAL)""",
                    "CREATE INDEX operational_jobs_owner ON operational_jobs(owner,updated_at)",
                    "CREATE INDEX operational_events_owner ON operational_events(owner,created_at)",
                ):
                    db.execute(statement)
                db.execute("INSERT INTO operational_schema VALUES('jobs',1)")
            if not version or version[0] == 1:
                # Additive v2: retain the old event table and its original reads.
                # The per-recipient table becomes authoritative after migration.
                db.execute("""CREATE TABLE operational_event_audiences(
                    event_id TEXT PRIMARY KEY,audience TEXT NOT NULL
                    CHECK(audience IN ('owner','workspace')),auth_mode TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES operational_events(event_id))""")
                db.execute("""CREATE TABLE operational_event_reads(
                    event_id TEXT NOT NULL,recipient TEXT NOT NULL,read_at REAL NOT NULL,
                    PRIMARY KEY(event_id,recipient),
                    FOREIGN KEY(event_id) REFERENCES operational_events(event_id))""")
                db.execute("""INSERT INTO operational_event_audiences
                    SELECT e.event_id,CASE WHEN e.kind='new_jd' AND EXISTS(
                        SELECT 1 FROM operational_job_revisions j WHERE j.job_id=e.jd_id
                        AND j.revision=e.revision AND j.status='published')
                        THEN 'workspace' ELSE 'owner' END,
                        substr(e.owner,1,instr(e.owner,':')-1) FROM operational_events e""")
                db.execute("""INSERT INTO operational_event_reads
                    SELECT event_id,owner,read_at FROM operational_events WHERE read_at IS NOT NULL""")
                db.execute("UPDATE operational_schema SET version=2 WHERE component='jobs'")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _guard(self, workspace=None):
        PortalStore._authorize(self.principal, workspace)
        if workspace is not None and (self.principal.auth_mode == "demo") != (workspace == "demo"):
            raise AuthorizationError("Demonstration and operational workspaces are separate")
        if self.guard is not None and self.guard() is False:
            raise AuthorizationError("Access revoked; sign in again")

    def _owned(self, db, job_id, *, archived=False):
        self._guard()
        row = db.execute("SELECT * FROM operational_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None or row["owner"] != _subject(self.principal) or (row["deleted_at"] and not archived):
            raise AuthorizationError("Job unavailable or access denied")
        self._guard(row["workspace"])
        return row

    def _job(self, db, job_id, revision=None, *, archived=False):
        head = self._owned(db, job_id, archived=archived)
        revision = head["revision"] if revision is None else _revision(revision)
        row = db.execute("SELECT * FROM operational_job_revisions WHERE job_id=? AND revision=?",
                         (job_id, revision)).fetchone()
        if row is None:
            raise ValueError("Unknown JD revision")
        return self._job_record(head, row, can_edit=True, is_current=revision == head["revision"])

    @staticmethod
    def _job_record(head, row, *, can_edit, is_current):
        result = {**dict(head), **dict(row)}
        result["fields"] = json.loads(result.pop("fields_json"))
        result["confirmed_fields"] = json.loads(result.pop("confirmed_json"))
        result.update(jd_id=head["job_id"], title=result["fields"]["title"], jd_title=result["fields"]["title"],
                      department=result["fields"]["department"], is_current=is_current, can_edit=can_edit)
        if not result["is_current"]:
            result.update(active_request_id=None, ranking_status="historical")
        return result

    def _visible_job(self, db, job_id, revision=None):
        """Read published content without weakening the mutation ownership gate."""
        self._guard()
        head = db.execute("SELECT * FROM operational_jobs WHERE job_id=?", (job_id,)).fetchone()
        if head is None or head["deleted_at"] is not None:
            raise AuthorizationError("Job unavailable or access denied")
        if head["owner"] == _subject(self.principal):
            return self._job(db, job_id, revision)
        if head["owner"].split(":", 1)[0] != self.principal.auth_mode:
            raise AuthorizationError("Job unavailable or access denied")
        self._guard(head["workspace"])
        published = db.execute("""SELECT * FROM operational_job_revisions
            WHERE job_id=? AND status='published' ORDER BY revision DESC LIMIT 1""", (job_id,)).fetchone()
        if published is None:
            raise AuthorizationError("Job unavailable or access denied")
        row = published
        if revision is not None:
            row = db.execute("""SELECT * FROM operational_job_revisions
                WHERE job_id=? AND revision=? AND status='published'""", (job_id, _revision(revision))).fetchone()
            if row is None:
                raise AuthorizationError("Job revision unavailable or access denied")
        # Do not leak the private head revision, its timestamp or its request state.
        public_head = {**dict(head), "revision": row["revision"],
                       "updated_at": row["published_at"], "active_request_id": None,
                       "ranking_status": "not_requested"}
        if db.execute("SELECT 1 FROM operational_schema WHERE component='ranking'").fetchone():
            request = db.execute("""SELECT request_id,status,updated_at FROM operational_ranking_requests
                WHERE job_id=? AND revision=? AND owner=? AND workspace=? ORDER BY rowid DESC LIMIT 1""",
                (job_id, row["revision"], head["owner"], head["workspace"])).fetchone()
            if request:
                public_head.update(active_request_id=request["request_id"], ranking_status=request["status"],
                                   updated_at=max(row["published_at"], request["updated_at"]))
        return self._job_record(public_head, row, can_edit=False,
                                is_current=row["revision"] == published["revision"])

    def _event(self, db, event_id, job, kind, body, request_id=None):
        db.execute("INSERT OR IGNORE INTO operational_events VALUES(?,?,?,?,?,?,?,?,?,NULL)",
                   (event_id, _subject(self.principal), job["workspace"], kind, job["job_id"],
                    job["revision"], request_id, _json(body), time.time()))
        db.execute("INSERT OR IGNORE INTO operational_event_audiences VALUES(?,?,?)",
                   (event_id, "workspace" if kind == "new_jd" else "owner", self.principal.auth_mode))

    def events(self, *, unread_only=False):
        self._guard()
        with self._db() as db:
            rows = db.execute("""SELECT e.*,a.audience,r.read_at AS recipient_read_at
                FROM operational_events e JOIN operational_event_audiences a ON a.event_id=e.event_id
                LEFT JOIN operational_event_reads r ON r.event_id=e.event_id AND r.recipient=?
                WHERE a.auth_mode=? AND (e.owner=? OR a.audience='workspace')
                ORDER BY e.created_at DESC,e.event_id""",
                (_subject(self.principal), self.principal.auth_mode, _subject(self.principal))).fetchall()
            result = []
            for row in rows:
                if row["workspace"] not in self.principal.departments or (unread_only and row["recipient_read_at"] is not None):
                    continue
                self._guard(row["workspace"])
                item = dict(row)
                item["read_at"] = item.pop("recipient_read_at")
                item["body"] = json.loads(item.pop("body_json"))
                item.update(department=item["workspace"], read=item["read_at"] is not None,
                            event_source="operational")
                result.append(item)
            return result

    def mark_read(self, event_id, *, read=True):
        self._guard()
        if type(read) is not bool:
            raise ValueError("Read state must be boolean")
        with self._db() as db:
            row = db.execute("""SELECT e.* FROM operational_events e
                JOIN operational_event_audiences a ON a.event_id=e.event_id
                WHERE e.event_id=? AND a.auth_mode=? AND (e.owner=? OR a.audience='workspace')""",
                (event_id, self.principal.auth_mode, _subject(self.principal))).fetchone()
            if row is None:
                raise AuthorizationError("Event unavailable or access denied")
            self._guard(row["workspace"])
            if read:
                db.execute("INSERT OR IGNORE INTO operational_event_reads VALUES(?,?,?)",
                           (event_id, _subject(self.principal), time.time()))
            else:
                db.execute("DELETE FROM operational_event_reads WHERE event_id=? AND recipient=?",
                           (event_id, _subject(self.principal)))
        return {"event_id": event_id, "read": read}


class JobService(_OperationalService):
    def create(self, fields=None, *, workspace="benchmark"):
        self._guard(workspace)
        fields = {} if fields is None else fields
        clean = validate_fields(fields)
        job_id, now = str(uuid.uuid4()), time.time()
        with self._db() as db:
            db.execute("INSERT INTO operational_jobs VALUES(?,?,?,?,?,?,NULL,NULL,?)",
                       (job_id, _subject(self.principal), workspace, 1, now, now, "not_requested"))
            self._insert_revision(db, job_id, 1, clean, list(fields), now)
            return self._job(db, job_id)

    @staticmethod
    def _insert_revision(db, job_id, revision, fields, confirmed, now):
        text = render_jd(fields)
        db.execute("INSERT INTO operational_job_revisions VALUES(?,?,?,?,?,?,?,?,?,NULL,NULL)",
                   (job_id, revision, "draft", _json(fields), text, _digest({"fields": fields, "text": text}),
                    _json(sorted(confirmed)), "deterministic-guided-v1", now))

    def get(self, job_id, revision=None):
        with self._db() as db:
            return self._visible_job(db, job_id, revision)

    def list(self, *, owned_only=False):
        self._guard()
        if type(owned_only) is not bool:
            raise ValueError("owned_only must be boolean")
        with self._db() as db:
            rows = db.execute("""SELECT job_id,workspace,owner FROM operational_jobs j
                WHERE deleted_at IS NULL AND (owner=? OR EXISTS(
                    SELECT 1 FROM operational_job_revisions r
                    WHERE r.job_id=j.job_id AND r.status='published'))""",
                              (_subject(self.principal),)).fetchall()
            jobs = [self._visible_job(db, row["job_id"]) for row in rows
                    if row["workspace"] in self.principal.departments
                    and row["owner"].split(":", 1)[0] == self.principal.auth_mode
                    and (not owned_only or row["owner"] == _subject(self.principal))]
            return sorted(jobs, key=lambda job: (-job["updated_at"], job["job_id"]))

    def update(self, job_id, revision, fields):
        _revision(revision)
        if not isinstance(fields, dict) or not fields:
            raise ValueError("Supply changed JD fields")
        with self._db() as db:
            current = self._job(db, job_id)
            if current["revision"] != revision:
                raise RevisionConflict("JD changed; reload before editing")
            clean = validate_fields({**current["fields"], **fields})
            confirmed = set(current["confirmed_fields"]) | set(fields)
            if clean == current["fields"] and confirmed == set(current["confirmed_fields"]):
                return current
            now = time.time()
            self._insert_revision(db, job_id, revision + 1, clean, confirmed, now)
            db.execute("UPDATE operational_jobs SET revision=?,updated_at=?,active_request_id=NULL,ranking_status='not_requested' WHERE job_id=?",
                       (revision + 1, now, job_id))
            return self._job(db, job_id)

    def next_question(self, job_id):
        with self._db() as db:
            job = self._job(db, job_id)
        return guided_question(job["fields"], job["confirmed_fields"])

    def review(self, job_id, revision):
        return self._transition(job_id, revision, publish=False)

    def publish(self, job_id, revision):
        return self._transition(job_id, revision, publish=True)

    def _transition(self, job_id, revision, *, publish):
        _revision(revision)
        with self._db() as db:
            job = self._job(db, job_id)
            if job["revision"] != revision:
                raise RevisionConflict("JD changed; review the current revision")
            validate_fields(job["fields"], complete=True)
            if job["status"] == "published" or (not publish and job["status"] == "reviewed"):
                return job
            if publish and job["status"] != "reviewed":
                raise ValueError("Review the final JD preview before publication")
            status, timestamp = ("published", "published_at") if publish else ("reviewed", "reviewed_at")
            now = time.time()
            db.execute(f"UPDATE operational_job_revisions SET status=?,{timestamp}=? WHERE job_id=? AND revision=?",
                       (status, now, job_id, revision))
            db.execute("UPDATE operational_jobs SET updated_at=? WHERE job_id=?", (now, job_id))
            if publish:
                self._event(db, f"operational:published:{job_id}:{revision}", job, "new_jd",
                            {"message": f"Published {job['title']} (revision {revision})",
                             "title": job["title"], "hiring_department": job["department"]})
            return self._job(db, job_id)

    def delete(self, job_id, revision):
        _revision(revision)
        with self._db() as db:
            job = self._job(db, job_id, archived=True)
            if job["revision"] != revision:
                raise RevisionConflict("JD changed; reload before archiving")
            if job["deleted_at"] is None:
                db.execute("UPDATE operational_jobs SET deleted_at=?,updated_at=? WHERE job_id=?",
                           (time.time(), time.time(), job_id))
        return {"job_id": job_id, "deleted": True}
