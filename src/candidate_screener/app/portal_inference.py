"""Hiring Desk ranking adapter over the inference API (plan 2026-10-24-hiring-desk-inference-api).

Configure the trusted server environment:

    HR_PORTAL_RANKING_ADAPTER=candidate_screener.app.portal_inference:adapter
    HR_PORTAL_CANDIDATE_POOL=a2-final-cvs

The Hiring Desk entry point applies these as defaults (``use_as_default``): an
exported value wins, and an empty one leaves live ranking unconfigured.

``adapter(operation, payload)`` implements the ``portal_ranking`` contract. One
``inference.Screener`` (loaded once, never fitted) ranks the configured pool on one
worker thread, in request order. D63 (the default) waits up to ``wait_seconds`` when
nothing is ahead of it and usually returns ready; D64 (``ranking_system="D64"``: C2
reorders L1's top 50, a few minutes for a new JD) returns at once and is collected by
status refresh.
Jobs are kept in memory by request_id; a status for an unknown request (the process
restarted) reruns it from the payload, which carries the full request: both systems
are deterministic for a fixed pool and date. Only ``jd_text``/``jd_title`` and the
pool's ``cv_id``/``cv_text`` reach the model (D53, D54); routing and authorization
never do. Errors are logged without document text and returned as failed or
service_unavailable, never raised for a model or pool problem.
"""
from __future__ import annotations

from collections import OrderedDict
import datetime as dt
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import threading
import time

from candidate_screener.config import PROJECT_ROOT

SYSTEMS = ("D63", "D64")
# Mirrors portal_ranking._ECHO (a test keeps them equal): every response echoes these.
ECHO = ("request_id", "jd_id", "jd_revision", "jd_digest", "workspace",
        "authorization_scope", "candidate_pool_ref")
TOP_K, KEEP, MENTIONS, KEYS, CHARS = 10, 200, 50, 40, 2000
MAX_RESPONSE = 2_000_000
LOG = PROJECT_ROOT / "logs/hiring-desk-adapter.log"
DEFAULT_ENVIRONMENT = {
    "HR_PORTAL_RANKING_ADAPTER": "candidate_screener.app.portal_inference:adapter",
    "HR_PORTAL_CANDIDATE_POOL": "a2-final-cvs",
}
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f]")
_ACTIVE = ("queued", "running")
_log = logging.getLogger(__name__)
_handlers = set()


def use_as_default(environ=os.environ):
    """Point the Hiring Desk at this adapter and pool unless the environment says otherwise."""
    for name, value in DEFAULT_ENVIRONMENT.items():
        environ.setdefault(name, value)


def _digest(value):
    return hashlib.sha256(str(value).encode("utf-8", "surrogatepass")).hexdigest()


def _source_text(text):
    """Line endings to \\n and other C0 controls to spaces: the app accepts only \\n/\\t."""
    return _CONTROL.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))


def _plain(value):
    """Plain JSON types the app's provenance validator accepts: numpy scalars unwrapped,
    tuples to lists, non-finite floats to None, collections bounded. Dates read "1 Sep 2023":
    the app's redaction takes an ISO date for a phone number."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in list(value.items())[:KEYS]}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in list(value)[:MENTIONS]]
    if isinstance(value, (dt.date, dt.datetime)):
        return f"{value.day} {value:%b %Y}"
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, str):
        return str(value)[:CHARS]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int)):
        return value
    return str(value)[:CHARS]


def _mentions(values):
    out = []
    for value in values:
        text = str(value).strip()
        if text and len(text) <= 1000 and text not in out:
            out.append(text)
    return out[:MENTIONS]


def _number(value):
    """A finite float, or None (numpy scalars and NaN included)."""
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _years(low, high):
    low, high = _number(low), _number(high)
    if low is not None and high is not None:
        return f"{low:g}-{high:g} years"
    if low is None and high is None:
        return None
    return f"at least {low:g} years" if low is not None else f"up to {high:g} years"


@lru_cache(maxsize=1)
def a2_final_cvs():
    """The 600 A2 final-set CVs as (`cv_id`/`cv_text` frame, reference date). `a2_pairs` holds no
    labels (D46); its metadata columns are dropped here and never reach the model (D53)."""
    from candidate_screener.extraction.experience import REFERENCE_DATE
    from candidate_screener.matching import corpus
    pool = corpus.a2_pairs("v2")[["cv_id", "cv_text"]].drop_duplicates().reset_index(drop=True)
    if pool.cv_id.duplicated().any() or not pool.cv_text.map(lambda t: isinstance(t, str) and bool(t.strip())).all():
        raise ValueError("The A2 pool needs one non-empty text per CV id")
    return pool, REFERENCE_DATE


class _Job:
    def __init__(self, payload, system):
        self.payload, self.system = payload, system
        self.handle = "job-" + _digest(payload["request_id"])[:32]
        self.status, self.body, self.done = "queued", None, threading.Event()


class InferenceAdapter:
    """``adapter(operation, payload)`` over ``screener_factory()`` and named pools; see the module doc.

    ``pools`` maps a candidate_pool_ref to a zero-argument loader returning (frame of
    ``cv_id``/``cv_text``, reference date). ``model_digest`` names the frozen L1 record
    (default: SHA-256 of ``inference.model.RECORD``)."""

    def __init__(self, screener_factory, pools, *, wait_seconds=45.0, model_digest=None, log_path=None):
        self.screener_factory, self.pools = screener_factory, dict(pools)
        self.wait_seconds, self.model_digest = float(wait_seconds), model_digest
        self.screener, self._pools = None, {}
        self._jobs, self._lock, self._queue, self._worker = OrderedDict(), threading.Lock(), queue.Queue(), None
        if log_path is not None:
            _log_to(Path(log_path))

    def __call__(self, operation, payload):
        if operation not in ("submit", "status") or not isinstance(payload, dict):
            raise ValueError("Unsupported ranking operation")
        echo = {key: payload.get(key) for key in ECHO}
        request_id, system = payload.get("request_id"), payload.get("ranking_system", "D63")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("A ranking request ID is required")
        wait = False
        with self._lock:
            job = self._jobs.get(request_id)
            if job is not None and any(type(job.payload.get(k)) is not type(payload.get(k))
                                       or job.payload.get(k) != payload.get(k) for k in ECHO):
                _log.warning("request %s: identity differs from the stored job", request_id)
                return {**echo, "status": "failed", "handle": job.handle}
            if job is None:
                if system not in SYSTEMS:
                    _log.warning("request %s: unknown ranking system", request_id)
                    return {**echo, "status": "failed", "handle": None}
                if payload.get("candidate_pool_ref") not in self.pools:
                    _log.warning("request %s: unknown candidate pool", request_id)
                    return {**echo, "status": "service_unavailable", "handle": None}
                # Status for an unknown request is a restart: rerun it from the full payload.
                wait = system == "D63" and not any(j.status in _ACTIVE for j in self._jobs.values())
                job = _Job({k: v for k, v in payload.items() if k != "handle"}, system)
                self._jobs[request_id] = job
                self._queue.put(job)
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._work, name="hiring-desk-inference", daemon=True)
                    self._worker.start()
                _log.info("request %s: %s queued (%s)", request_id, system, operation)
        if wait:
            job.done.wait(self.wait_seconds)
        with self._lock:
            response = {**echo, "status": job.status, "handle": job.handle}
            if job.status == "ready":
                response.update(job.body)
        return response

    def _work(self):
        while True:
            job = self._queue.get()
            request_id, started = job.payload["request_id"], time.monotonic()
            with self._lock:
                job.status = "running"
            try:
                status, body = "ready", self._rank(job)
            except FileNotFoundError:
                status, body = "service_unavailable", None
                _log.exception("request %s: model parts, weights or pool missing", request_id)
            except Exception:
                status, body = "failed", None
                _log.exception("request %s: ranking failed", request_id)
            with self._lock:
                job.status, job.body = status, body
                finished = [key for key, item in self._jobs.items() if item.status not in _ACTIVE]
                for key in finished[:max(0, len(finished) - KEEP)]:
                    del self._jobs[key]
            job.done.set()
            _log.info("request %s: %s %s in %.1f s", request_id, job.system, status, time.monotonic() - started)

    def _pool(self, ref):
        if ref not in self._pools:
            from .portal_store import _opaque_identity
            frame, reference = self.pools[ref]()
            frame = frame[["cv_id", "cv_text"]].astype({"cv_id": str}).reset_index(drop=True)
            for identity in frame.cv_id:
                _opaque_identity(identity)
            rows = sorted(f"{i}:{_digest(t)}" for i, t in zip(frame.cv_id, frame.cv_text))
            self._pools[ref] = (frame, reference, f"{ref}:" + _digest("\n".join(rows))[:16])
        return self._pools[ref]

    def _rank(self, job):
        payload, system = job.payload, job.system
        if self.model_digest is None:
            from candidate_screener.inference.model import RECORD
            self.model_digest = hashlib.sha256(RECORD.read_bytes()).hexdigest()
        if self.screener is None:
            started = time.monotonic()
            self.screener = self.screener_factory()
            _log.info("screener loaded in %.1f s", time.monotonic() - started)
        ref = payload["candidate_pool_ref"]
        pool, reference, version = self._pool(ref)
        progress = (lambda done, total: _log.info("request %s: C2 %d/%d pairs", payload["request_id"], done, total)
                    ) if system == "D64" else None
        # D53/D54: the JD's text and title, and the pool's ids and texts; nothing else.
        result = self.screener.rank(payload["jd_text"], pool[["cv_id", "cv_text"]], jd_title=payload.get("jd_title") or None,
                                    top_k=TOP_K, system="D63", rerank=system == "D64",
                                    reference_date=reference, progress=progress)
        texts = dict(zip(pool.cv_id, pool.cv_text))
        candidates = []
        for item in result.top(TOP_K):
            profile, why = result.cvs[item.cv_id], self.screener.explain(result, item.cv_id)
            entities = {"SKILL": {"mentions": _mentions(profile.skills)}}
            if _number(profile.years) is not None:
                years = f"{_number(profile.years):g} years" + (f" ({profile.years_source})" if profile.years_source else "")
                entities["EXPERIENCE_YEARS"] = {"mentions": [years]}
            provenance = {key: why.get(key) for key in (
                "system", "rank", "score", "reasons", "jd_role", "cv_role", "role_match", "matched_skills",
                "missing_skills", "years_required", "years_stated", "years_source", "experience_compatible",
                "jd_top_roles", "cv_top_roles", "feature_shares", "l1_intercept", "note")}
            provenance.update(l1_score=item.l1_score, x2_score=item.x2_score, c2_score=item.c2_score)
            # The app redacts source references: a UUID's hyphenated digit groups read as a phone
            # number or [NUMBER] (53 of the 600 A2 ids), so the reference drops the hyphens.
            candidates.append({"candidate_id": str(item.cv_id), "rank": int(item.rank), "score": float(item.score),
                               "source_ref": f"{ref}/{str(item.cv_id).replace('-', '')}",
                               "cv_text": _source_text(texts[item.cv_id]),
                               "entities": entities, "evidence_provenance": _plain(provenance)})
        jd = result.jd
        jd_entities = {"SKILL": {"mentions": _mentions(jd.skills)}}
        if payload.get("jd_title"):
            jd_entities["TITLE"] = {"mentions": _mentions([payload["jd_title"]])}
        if _years(jd.min_years, jd.max_years):
            jd_entities["EXPERIENCE_YEARS"] = {"mentions": [_years(jd.min_years, jd.max_years)]}
        k, w, record = self.screener.c2_k, self.screener.c2_w, self.model_digest[:12]
        if system == "D64":
            model_id = f"D64 frozen L1 + C2 on its top {k}, w {w:g} (l1-chat3-model.json {record})"
            semantics = (f"D64: within L1's top {k}, 10 + min-max L1 + {w:g} x min-max C2; below it, min-max L1. "
                         "Higher ranks first; not a Fit probability or a calibrated score.")
        else:
            model_id = f"D63 frozen L1 (l1-chat3-model.json {record})"
            semantics = ("D63: frozen-L1 ordering score over the configured candidate pool. "
                         "Higher ranks first; not a Fit probability or a calibrated score.")
        result_provenance = {"system": result.system, "scaling": result.scaling, "pool_size": result.pool_size,
                             "reference_date": result.reference_date, "top_k": TOP_K,
                             "l1_record_sha256": self.model_digest, "jd_role": jd.role, "jd_role_prob": jd.role_prob,
                             "jd_years": {"min": jd.min_years, "max": jd.max_years}}
        if system == "D64":
            result_provenance.update(c2_k=k, c2_w=w)
        body = {"run_id": f"{system.lower()}-" + _digest(json.dumps([payload["request_id"], version, self.model_digest, system]))[:24],
                "model_id": model_id, "candidate_pool_version": version, "score_semantics": semantics,
                "eligible_candidate_ids": [str(i) for i in pool.cv_id], "candidates": candidates,
                "jd_entities": jd_entities, "evidence_provenance": _plain(result_provenance)}
        if len(json.dumps(body, ensure_ascii=False, allow_nan=False).encode()) > MAX_RESPONSE:
            raise ValueError("Ranking response exceeds the app's size limit")
        return body


def _log_to(path):
    if path in _handlers:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)
    _handlers.add(path)


def _load_screener():
    from candidate_screener.inference import Screener
    return Screener.load()


_DEFAULT, _DEFAULT_LOCK = None, threading.Lock()


def adapter(operation, payload):
    """The configured entry point: one shared adapter over the real Screener and the A2 pool."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = InferenceAdapter(_load_screener, {"a2-final-cvs": a2_final_cvs}, log_path=LOG)
    return _DEFAULT(operation, payload)
