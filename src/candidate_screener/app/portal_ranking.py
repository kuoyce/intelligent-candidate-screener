"""Persisted live-ranking boundary; no bundled model or benchmark fallback.

Configure trusted server environment HR_PORTAL_RANKING_ADAPTER=module:function
and HR_PORTAL_CANDIDATE_POOL=<opaque pool reference>, or inject adapter/pool_ref.
The callable ``adapter(operation, payload) -> dict`` must return promptly; use
queued/running plus an opaque handle for long inference. It must deduplicate
submit by request_id and permit status recovery by request_id without a handle.
Only a new explicit rank/rerank invokes submit; refresh invokes status.

Payload (schema_version=1): request_id, jd_id, jd_revision, jd_digest, jd_title,
jd_text, workspace, authorization_scope (opaque owner digest), candidate_pool_ref,
top_k=10, ranking_system ('D63' or 'D64'; optional, a payload without it means D63).
Only title/text are inference inputs; routing/scope are not features, and
ranking_system selects the model rather than describing the JD.
Status payload adds handle. EVERY response echoes request_id, jd_id, jd_revision,
jd_digest, workspace, authorization_scope and candidate_pool_ref. Status is
queued/running/ready/failed/service_unavailable; optional handle is an opaque ID.

A ready response additionally requires run_id, model_id, candidate_pool_version,
score_semantics, eligible_candidate_ids (the adapter's authorized pool manifest),
and candidates (0..10 in backend order). Each candidate requires candidate_id,
rank (contiguous from 1), finite score, source_ref; cv_text may be None.
Candidate IDs MUST satisfy portal_store._opaque_identity: UUIDs, 16..128 hex
digits, cv-/candidate-/a2-cv- plus 16..128 hex digits, or Applicant-<1..6 digits>
(the store also accepts its existing generated synthetic IDs). Arbitrary labels
such as foo-1 are rejected before ready, so selection/chat/package saves work.
Optional entities map category to {mentions:[str]} (or a list); these remain UNVERIFIED
backend mentions. Optional jd_entities follows the same schema. Optional
evidence_provenance is a JSON object on the result or candidate. Fit/No Fit may be supplied
only with nonempty decision_provenance. A source_digest, if supplied with text,
must equal SHA256 of that exact text. Never persist raw adapter responses/errors.

Snapshots retain the existing PortalSnapshot fields; source_flags contains
source_type='operational_live', operational=True, request_id and jd_revision.
snapshot(job_id) returns the author's current revision's active ready request;
other authorized workspace reviewers can read the latest published revision's
active ready snapshot, including while a newer private draft exists. Requests,
refresh/submit and ready_snapshot(request_id) remain strictly owner-only; the
latter permits the author's immutable historical results as well.
Python cannot forcibly time out a callable: adapters own their inference timeout.
There is intentionally no HTTP endpoint discovery or guessed candidate pool.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
import re
import time
import uuid

from .portal_data import CATEGORIES, PortalSnapshot
from .portal_jobs import (
    _OperationalService, _digest, _identity, _json, _revision, _text,
)
from .portal_store import AuthorizationError, RevisionConflict, _opaque_identity, _subject

_TERMINAL = {"ready", "failed", "service_unavailable"}
_SYSTEMS = ("D63", "D64")
_ECHO = ("request_id", "jd_id", "jd_revision", "jd_digest", "workspace",
         "authorization_scope", "candidate_pool_ref")


def _provenance(value, depth=0):
    """Bounded redacted metadata, with no credentials or arbitrary objects."""
    if depth > 6:
        raise ValueError("Evidence provenance nesting exceeds limit")
    if isinstance(value, dict):
        if len(value) > 40:
            raise ValueError("Evidence provenance exceeds limit")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", key):
                raise ValueError("Invalid provenance key")
            if re.search(r"token|password|secret|credential|authorization|cookie", key, re.I):
                raise ValueError("Credentials cannot be persisted")
            result[key] = _provenance(item, depth + 1)
        return result
    if isinstance(value, list):
        if len(value) > 50:
            raise ValueError("Evidence provenance list exceeds limit")
        return [_provenance(v, depth + 1) for v in value]
    if isinstance(value, str):
        return _text(value, "evidence provenance", 2000)
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Invalid evidence provenance")


def _entities(value=None):
    if value is not None and (not isinstance(value, dict) or set(value) - set(CATEGORIES)):
        raise ValueError("Invalid extraction categories")
    result = {}
    for category in CATEGORIES:
        item = (value or {}).get(category, [])
        mentions = item.get("mentions", []) if isinstance(item, dict) else item
        if not isinstance(mentions, list) or len(mentions) > 50:
            raise ValueError("Invalid backend extraction mentions")
        mentions = [_text(v, "extracted mention", 1000, empty=False) for v in mentions]
        result[category] = {"status": "unverified_backend_mentions" if mentions else "unavailable",
                            "mentions": mentions, "verified_spans": [],
                            "source": "backend extraction; verified source offsets unavailable"}
    return result


def _ready_snapshot(request, response):
    """Validate backend identity/order/membership before constructing any view."""
    expected = request["payload"]
    for key in _ECHO:
        if type(response.get(key)) is not type(expected[key]) or response.get(key) != expected[key]:
            raise ValueError("Ranking response identity does not match request")
    run_id = _identity(response.get("run_id"), "backend run ID")
    model_id = _text(response.get("model_id"), "backend model ID", 200, empty=False)
    pool_version = _text(response.get("candidate_pool_version"), "pool version", 200, empty=False)
    semantics = _text(response.get("score_semantics"), "score semantics", 500, empty=False)
    result_provenance = _provenance(response.get("evidence_provenance", {}))
    if not isinstance(result_provenance, dict):
        raise ValueError("Result evidence provenance must be an object")
    members = response.get("eligible_candidate_ids")
    if not isinstance(members, list) or len(members) > 10_000:
        raise ValueError("An authorized candidate pool manifest is required")
    members = [_opaque_identity(v) for v in members]
    if len(members) != len(set(members)):
        raise ValueError("Duplicate candidate pool member")
    allowed = set(members)
    raw_candidates = response.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) > 10:
        raise ValueError("Operational rankings require zero to ten candidates")
    candidates, seen = [], set()
    for rank, item in enumerate(raw_candidates, 1):
        if not isinstance(item, dict):
            raise ValueError("Invalid candidate result")
        identity = _opaque_identity(item.get("candidate_id"))
        if identity not in allowed or identity in seen:
            raise ValueError("Candidate is duplicate or outside the authorized pool")
        seen.add(identity)
        score = item.get("score")
        if type(item.get("rank")) is not int or item["rank"] != rank:
            raise ValueError("Backend ranks must be contiguous and ordered")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Backend scores must be finite numbers")
        source_ref = _text(item.get("source_ref"), "candidate source reference", 500, empty=False)
        raw_text = item.get("cv_text")
        preview = None if raw_text is None else _text(raw_text, "candidate source text", 60_000)
        source_digest = item.get("source_digest")
        if source_digest is not None and (not isinstance(source_digest, str)
                or not re.fullmatch(r"[a-f0-9]{64}", source_digest)):
            raise ValueError("Invalid source digest")
        if raw_text is not None:
            digest = hashlib.sha256(raw_text.encode()).hexdigest()
            if source_digest is not None and source_digest != digest:
                raise ValueError("Candidate source digest does not match text")
            source_digest = digest
        entities = _entities(item.get("entities"))
        provenance = _provenance(item.get("evidence_provenance", {}))
        if not isinstance(provenance, dict):
            raise ValueError("Evidence provenance must be an object")
        flags = {"preview_status": "available_redacted_text" if preview else "unavailable",
                 "verified_spans_available": False, "table_mentions_only": False,
                 "source_type": "operational_live", "is_demo": expected["workspace"] == "demo",
                 "page_metadata": "unavailable", "pdf_preview": "unavailable"}
        candidate = {"candidate_id": identity, "cv_id": identity, "rank": rank, "score": float(score),
                     "score_semantics": semantics, "source_ref": source_ref,
                     "source_digest": source_digest, "preview_text": preview, "cv_text": preview,
                     "preview_status": flags["preview_status"], "entities": entities, "profile": entities,
                     "table_aggregates": {}, "source_flags": flags, "coverage": flags,
                     "evidence_provenance": provenance}
        if "fit" in item:
            decision = _provenance(item.get("decision_provenance"))
            if item["fit"] not in ("Fit", "No Fit") or not isinstance(decision, dict) or not decision:
                raise ValueError("Backend Fit calls require decision provenance")
            candidate.update(fit=item["fit"], decision_provenance=decision)
        candidates.append(candidate)
    payload = {"jd_text": expected["jd_text"], "jd_entities": _entities(response.get("jd_entities")), "candidates": candidates,
               "source_flags": {"source_type": "operational_live", "operational": True,
                   "is_demo": expected["workspace"] == "demo", "request_id": request["request_id"],
                   "jd_revision": expected["jd_revision"], "jd_digest": expected["jd_digest"],
                   "preview_status": "available_redacted_text", "verified_spans_available": False,
                   "table_mentions_only": False, "candidate_count": len(candidates),
                   "candidate_pool_ref": expected["candidate_pool_ref"],
                   "candidate_pool_version": pool_version, "score_semantics": semantics,
                   "evidence_provenance": result_provenance,
                   "pool_membership_semantics": "authorized manifest supplied by trusted ranking adapter"}}
    digest = _digest({"run_id": run_id, "model_id": model_id, "jd_id": expected["jd_id"],
                      "jd_title": expected["jd_title"], **payload})
    return PortalSnapshot(digest[:24], run_id, expected["jd_id"], expected["jd_title"],
                          digest, _json(payload), model_id)


class RankingService(_OperationalService):
    def __init__(self, path, principal, guard=None, *, adapter=None, pool_ref=None):
        super().__init__(path, principal, guard)
        self.adapter = adapter
        self.adapter_id = "injected" if adapter is not None else os.environ.get("HR_PORTAL_RANKING_ADAPTER", "").strip()
        self.pool_ref = pool_ref if pool_ref is not None else os.environ.get("HR_PORTAL_CANDIDATE_POOL", "").strip()
        self.configuration_error = None
        if self.pool_ref:
            try:
                _identity(self.pool_ref, "candidate pool reference")
            except ValueError:
                self.configuration_error = "Invalid authorized candidate pool configuration"
                self.pool_ref = ""
        if adapter is None and self.adapter_id:
            if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", self.adapter_id):
                self.configuration_error = "Invalid ranking adapter configuration"
                self.adapter_id = ""
            else:
                try:
                    module, function = self.adapter_id.split(":")
                    self.adapter = getattr(importlib.import_module(module), function)
                except Exception:
                    self.configuration_error = "Configured ranking adapter could not be loaded"
        if self.adapter is not None and not callable(self.adapter):
            self.configuration_error = "Configured ranking adapter is not callable"
            self.adapter = None
        with self._db() as db:
            version = db.execute("SELECT version FROM operational_schema WHERE component='ranking'").fetchone()
            if version and version[0] != 1:
                raise ValueError("Unsupported operational ranking schema version")
            if not version:
                db.execute("""CREATE TABLE operational_ranking_requests(
                    request_id TEXT PRIMARY KEY,job_id TEXT NOT NULL,revision INTEGER NOT NULL,
                    owner TEXT NOT NULL,workspace TEXT NOT NULL,status TEXT NOT NULL,
                    adapter_id TEXT NOT NULL,payload_json TEXT NOT NULL,handle TEXT,
                    snapshot_json TEXT,error TEXT,created_at REAL NOT NULL,updated_at REAL NOT NULL,
                    processing_token TEXT,processing_until REAL NOT NULL,
                    FOREIGN KEY(job_id,revision) REFERENCES operational_job_revisions(job_id,revision))""")
                db.execute("CREATE INDEX operational_ranking_owner ON operational_ranking_requests(owner,created_at)")
                db.execute("INSERT INTO operational_schema VALUES('ranking',1)")

    @property
    def availability(self):
        self._guard()
        reason = self.configuration_error
        if not reason and self.adapter is None:
            reason = "Live ranking adapter is not configured"
        if not reason and not self.pool_ref:
            reason = "An authorized candidate pool is not configured"
        return {"available": reason is None, "reason": reason}

    def _request(self, db, request_id):
        self._guard()
        row = db.execute("SELECT * FROM operational_ranking_requests WHERE request_id=?", (request_id,)).fetchone()
        if row is None or row["owner"] != _subject(self.principal):
            raise AuthorizationError("Ranking request unavailable or access denied")
        self._owned(db, row["job_id"], archived=True)
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        snapshot = result.pop("snapshot_json")
        result["snapshot_id"] = json.loads(snapshot)["snapshot_id"] if snapshot else None
        result.pop("processing_token")
        result.pop("processing_until")
        return result

    def get(self, request_id):
        with self._db() as db:
            return self._request(db, request_id)

    def requests(self, job_id=None):
        self._guard()
        with self._db() as db:
            if job_id is not None:
                self._owned(db, job_id, archived=True)
            rows = db.execute("SELECT request_id,job_id,workspace FROM operational_ranking_requests WHERE owner=? ORDER BY created_at DESC,request_id",
                              (_subject(self.principal),)).fetchall()
            return [self._request(db, row["request_id"]) for row in rows
                    if (job_id is None or row["job_id"] == job_id)
                    and row["workspace"] in self.principal.departments]

    def rank(self, job_id, revision=None, *, request_id=None, rerank=False, system="D63"):
        """Submit at most once. Explicit rerank creates another historical run;
        system picks the model of a new request (D63, or D64's C2 on the top 50)."""
        self._guard()
        if type(rerank) is not bool:
            raise ValueError("rerank must be boolean")
        if type(system) is not str or system not in _SYSTEMS:
            raise ValueError("Unknown ranking system")
        if request_id is not None:
            _identity(request_id, "ranking request ID")
        with self._db() as db:
            job = self._job(db, job_id, revision)
            if not job["is_current"] or job["status"] != "published":
                raise RevisionConflict("Rank only the current published JD revision")
            if request_id is not None:
                prior = db.execute("SELECT request_id FROM operational_ranking_requests WHERE request_id=?", (request_id,)).fetchone()
                if prior:
                    old = self._request(db, request_id)
                    if (old["job_id"], old["revision"]) != (job_id, job["revision"]):
                        raise RevisionConflict("Request ID belongs to another JD revision")
                    return old
            if job["active_request_id"] and not rerank:
                return self._request(db, job["active_request_id"])
            request_id = request_id or str(uuid.uuid4())
            payload = {"schema_version": 1, "request_id": request_id, "jd_id": job_id,
                       "jd_revision": job["revision"], "jd_digest": job["content_digest"],
                       "jd_title": job["title"], "jd_text": job["jd_text"], "workspace": job["workspace"],
                       "authorization_scope": hashlib.sha256(_subject(self.principal).encode()).hexdigest(),
                       "candidate_pool_ref": self.pool_ref, "top_k": 10, "ranking_system": system}
            available = self.availability
            status = "queued" if available["available"] else "service_unavailable"
            now, token = time.time(), str(uuid.uuid4())
            db.execute("INSERT INTO operational_ranking_requests VALUES(?,?,?,?,?,?,?,?,NULL,NULL,?,?,?,?,?)",
                       (request_id, job_id, job["revision"], _subject(self.principal), job["workspace"],
                        status, self.adapter_id, _json(payload), available["reason"], now, now,
                        token if available["available"] else None, now + 90 if available["available"] else 0))
            db.execute("UPDATE operational_jobs SET active_request_id=?,ranking_status=?,updated_at=? WHERE job_id=?",
                       (request_id, status, now, job_id))
            request = self._request(db, request_id)
        if not available["available"]:
            return request
        return self._invoke(request, "submit", token)

    submit = rank

    def refresh(self, request_id):
        """Explicit bounded status refresh; never repeats submit, even after crash."""
        with self._db() as db:
            request = self._request(db, request_id)
            if request["status"] in _TERMINAL:
                return request
            job = self._owned(db, request["job_id"], archived=True)
            if job["deleted_at"]:
                raise RevisionConflict("Archived jobs cannot refresh inference")
            availability = self.availability
            if not availability["available"] or self.adapter_id != request["adapter_id"]:
                # A temporary missing/changed adapter must not destroy a live handle.
                return {**request, "refresh_unavailable": True,
                        "error": availability["reason"] or "Original ranking adapter is not configured"}
            lease = db.execute("SELECT processing_until FROM operational_ranking_requests WHERE request_id=?", (request_id,)).fetchone()[0]
            if lease > time.time():
                return request
            token = str(uuid.uuid4())
            db.execute("UPDATE operational_ranking_requests SET processing_token=?,processing_until=? WHERE request_id=?",
                       (token, time.time() + 90, request_id))
        return self._invoke(request, "status", token)

    def _invoke(self, request, operation, token):
        payload = dict(request["payload"])
        if operation == "status":
            payload["handle"] = request["handle"]
        snapshot, handle, error = None, None, None
        try:
            response = self.adapter(operation, payload)
            if not isinstance(response, dict) or len(_json(response).encode()) > 2_000_000:
                raise ValueError("Invalid ranking response")
            for key in _ECHO:
                if type(response.get(key)) is not type(request["payload"][key]) or response.get(key) != request["payload"][key]:
                    raise ValueError("Ranking response identity does not match request")
            status = response.get("status")
            if status not in {"queued", "running", *_TERMINAL}:
                raise ValueError("Invalid ranking status")
            handle = response.get("handle", request["handle"])
            if handle is not None:
                _identity(handle, "ranking handle")
            if status == "ready":
                snapshot = _ready_snapshot(request, response)
            elif status in {"failed", "service_unavailable"}:
                error = "Ranking adapter reported failure; request a new run" if status == "failed" else "Ranking adapter is unavailable; request a new run"
        except Exception:
            # Exception/response prose may contain raw CVs, secrets or remote URLs.
            status, error = "failed", "Ranking adapter failed or returned an invalid result; request a new run"
            handle = request["handle"]
        self._guard(request["workspace"])  # Recheck revocation after external work.
        with self._db() as db:
            current = self._request(db, request["request_id"])
            stored = db.execute("SELECT processing_token FROM operational_ranking_requests WHERE request_id=?", (request["request_id"],)).fetchone()
            if current["status"] in _TERMINAL or stored[0] != token:
                return current
            now = time.time()
            db.execute("UPDATE operational_ranking_requests SET status=?,handle=?,snapshot_json=?,error=?,updated_at=?,processing_token=NULL,processing_until=0 WHERE request_id=?",
                       (status, handle, _json(asdict(snapshot)) if snapshot else None, error, now, request["request_id"]))
            # An old revision/run may finish, but cannot replace the active run.
            db.execute("UPDATE operational_jobs SET ranking_status=?,updated_at=? WHERE job_id=? AND revision=? AND active_request_id=? AND deleted_at IS NULL",
                       (status, now, request["job_id"], request["revision"], request["request_id"]))
            if status in {"ready", "failed"}:
                job = self._job(db, request["job_id"], request["revision"], archived=True)
                self._event(db, f"operational:ranking:{request['request_id']}:{status}", job,
                            "ranking_ready" if status == "ready" else "ranking_failed",
                            {"message": f"Ranking {'ready' if status == 'ready' else 'failed'} for {job['title']}",
                             "snapshot_id": snapshot.snapshot_id if snapshot else None,
                             "candidate_count": len(snapshot.candidates) if snapshot else None}, request["request_id"])
            return self._request(db, request["request_id"])

    def ready_snapshot(self, request_id):
        with self._db() as db:
            return self._snapshot(db, request_id)

    def _snapshot(self, db, request_id):
        request = self._request(db, request_id)
        if request["status"] != "ready":
            return None
        row = db.execute("SELECT snapshot_json FROM operational_ranking_requests WHERE request_id=?", (request_id,)).fetchone()
        return PortalSnapshot(**json.loads(row[0]))

    get_ready_snapshot = ready_snapshot

    def snapshot(self, job_id, revision=None):
        with self._db() as db:
            job = self._visible_job(db, job_id, revision)
            if not job["is_current"] or not job["active_request_id"]:
                return None
            if job["can_edit"]:
                return self._snapshot(db, job["active_request_id"])
            # _visible_job chooses the public revision's latest request. Bypass
            # only the owner-only request view, never expose its payload/handle.
            row = db.execute("""SELECT snapshot_json FROM operational_ranking_requests
                WHERE request_id=? AND job_id=? AND revision=? AND owner=? AND workspace=? AND status='ready'""",
                (job["active_request_id"], job_id, job["revision"], job["owner"], job["workspace"])).fetchone()
            return PortalSnapshot(**json.loads(row[0])) if row and row[0] else None
