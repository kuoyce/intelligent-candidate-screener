"""The Hiring Desk's inference adapter (plan 2026-10-24-hiring-desk-inference-api) on a synthetic
Screener: the real `inference.Screener` maths over injected parts, never a model or `data/`."""
import datetime as dt
import json
import logging
import threading
import time

import numpy as np
import pandas as pd
import pytest

from candidate_screener.app import portal_inference
from candidate_screener.app.portal_inference import ECHO, InferenceAdapter
from candidate_screener.app.portal_jobs import JobService, _identity
from candidate_screener.app.portal_ranking import RankingService, _ECHO, _ready_snapshot
from candidate_screener.app.portal_store import Principal
from candidate_screener.extraction import experience
from candidate_screener.inference import CVProfile, JDProfile, L1Model, Screener
from candidate_screener.matching.learned import F12
from candidate_screener.matching.role_filter import per_jd_minmax

POOL, DATE, DIGEST = "test-pool", dt.date(2023, 9, 1), "e" * 64
VOCAB = ("python", "sql", "docker", "java", "kubernetes")
JD = "Python Engineer\nSkills: Python, SQL, Docker\nExperience: at least 2 years"
SKILLS = ["python sql docker", "java", "python", "sql docker", "python sql", "docker kubernetes",
          "python docker sql java", "kubernetes", "python kubernetes", "java sql", "sql", "python java"]
# A leading all-digit group, as 53 of the 600 A2 ids have: hyphenated, the app redacts it.
IDS = [f"12345678-1b80-5212-985c-{i:012x}" for i in range(len(SKILLS))]
# Every pool text carries Windows line endings, as 530 of the 600 A2 final-set CVs do.
TEXTS = [f"Applicant {i}\r\nDeveloper: {s}\r\n{i % 6 + 1} years of experience\r\n" for i, s in enumerate(SKILLS)]


def featurize(pairs, reference):
    rows = []
    for jd, cv in zip(pairs.jd_text, pairs.cv_text):
        j, c = set(jd.lower().replace(",", " ").split()), set(cv.lower().split())
        overlap = len(j & c) / max(len(j), 1)
        rows.append({"role_match": 0.9 if "python" in c else 0.1, "experience_compatible": 1.0,
                     "raw": overlap, "coverage": overlap, "coverage_unweighted": overlap,
                     "log_matched": np.log1p(len(j & c)), "log_jd_concepts": np.log1p(len(j)),
                     "log_cv_concepts": np.log1p(len(c)), "skill_overlap": overlap,
                     "exp_gap": 0.0, "exp_known": 0.0, "sbert": (len(cv) % 7) / 7})
    X = pd.DataFrame(rows, index=pairs.pair_id.to_numpy())
    X["tfidf_scaled"] = per_jd_minmax(pairs.jd_id.to_numpy(), X.raw.to_numpy())
    return X[list(F12)]


def extract_jds(texts, titles):
    return [JDProfile(None, t, "Python", 0.9, (("Python", 0.9), ("Java", 0.1)),
                      tuple(w for w in VOCAB if w in x.lower().replace(",", " ").split()), (), 2.0, None, None)
            for x, t in zip(texts, titles)]


def extract_cvs(texts, reference):
    out = []
    for x in texts:
        e = experience.cv_experience(x, reference)
        role = "Python" if "python" in x.lower().split() else "Java"
        out.append(CVProfile(None, role, 0.8, ((role, 0.8),), tuple(w for w in VOCAB if w in x.lower().split()),
                             (), e.years, e.source, None, None))
    return out


def numpy_cvs(texts, reference):
    """numpy scalars where a real extractor may leave them, and more skills than the app keeps."""
    return [CVProfile(None, np.str_("Python"), np.float64(0.8), ((np.str_("Python"), np.float32(0.8)),),
                      tuple(np.str_(f"skill {k}") for k in range(80)), (), np.float64(3.0), np.str_("statement"),
                      None, None) for _ in texts]


L1 = L1Model(tuple(F12), (0.0,) * len(F12), (1.0,) * len(F12),
             tuple(float(i + 1) for i in range(len(F12))), 10.0, 1.0)
BACKGROUND = pd.DataFrame({"id": [f"b{i}" for i in range(5)], "cv_text": [f"background {i} sql" for i in range(5)]})


def c2_scores(pairs, progress=None):
    return pd.Series([float(len(t) % 11) for t in pairs.cv_text], index=pairs.pair_id.to_numpy())


class Recording:
    """The Screener the adapter sees: real `Screener` maths, every `rank` call recorded."""

    def __init__(self, extract=extract_cvs, c2=c2_scores, explain=None, fail=None):
        self.inner = Screener(featurize, L1, extract_jds, extract, c2=c2, background=lambda: BACKGROUND,
                              c2_k=50, c2_w=0.25)
        self.c2_k, self.c2_w, self.calls, self._explain, self.fail = 50, 0.25, [], explain, fail

    def rank(self, jd_text, cvs, **kwargs):
        self.calls.append({"jd_text": jd_text, "cvs": cvs.copy(), "kwargs": kwargs})
        if self.fail:
            raise self.fail
        return self.inner.rank(jd_text, cvs, **kwargs)

    def explain(self, screening, cv_id):
        why = self.inner.explain(screening, cv_id)
        return self._explain(why) if self._explain else why


def pool():
    return pd.DataFrame({"cv_id": IDS, "cv_text": TEXTS}), DATE


def make(screener=None, pools=None, **kwargs):
    screener = screener or Recording()
    kwargs = {"wait_seconds": 10.0, "model_digest": DIGEST, **kwargs}
    return InferenceAdapter(lambda: screener, {POOL: pool} if pools is None else pools, **kwargs), screener


def payload(request_id="request-1", system=None, **extra):
    """A request as `RankingService.rank` builds it."""
    result = {"schema_version": 1, "request_id": request_id, "jd_id": "route-jd-0001", "jd_revision": 1,
              "jd_digest": "d" * 64, "jd_title": "Python Engineer", "jd_text": JD, "workspace": "route-space",
              "authorization_scope": "a" * 64, "candidate_pool_ref": POOL, "top_k": 10}
    if system:
        result["ranking_system"] = system
    return {**result, **extra}


def settle(adapter, request, timeout=10.0):
    deadline = time.monotonic() + timeout
    while True:
        response = adapter("status", request)
        if response["status"] not in ("queued", "running") or time.monotonic() > deadline:
            return response
        time.sleep(0.01)


@pytest.fixture
def jobs(tmp_path):
    return JobService(tmp_path / "workspace.sqlite", Principal("ranking-author", ("benchmark",), time.time() + 3600, "local_token"))


@pytest.fixture
def job(jobs):
    job = jobs.create({"title": "Python Engineer", "department": "Tech",
                       "responsibilities": ["Build services"], "skills": ["Python", "SQL"], "min_years": 2})
    jobs.review(job["job_id"], 1)
    return jobs.publish(job["job_id"], 1)


def test_echo_fields_are_the_app_contracts():
    assert ECHO == _ECHO


def test_crlf_pool_ranks_ready_through_the_ranking_service(jobs, job):
    adapter, screener = make()
    ranking = RankingService(jobs.path, jobs.principal, adapter=adapter, pool_ref=POOL)
    assert ranking.rank(job["job_id"])["status"] == "ready"
    snapshot = ranking.snapshot(job["job_id"])
    candidates = snapshot.candidates
    assert [c["rank"] for c in candidates] == list(range(1, 11))
    assert {c["cv_id"] for c in candidates} <= set(IDS)
    assert all(c["source_ref"] == f"{POOL}/{c['cv_id'].replace('-', '')}" for c in candidates)
    assert all(c["cv_text"] and "\r" not in c["cv_text"] for c in candidates)
    assert [c["score"] for c in candidates] == sorted((c["score"] for c in candidates), reverse=True)
    assert snapshot.model_id == f"D63 frozen L1 (l1-chat3-model.json {DIGEST[:12]})"
    assert snapshot.source_flags["candidate_pool_version"].startswith(POOL + ":")
    assert "not a Fit probability" in snapshot.source_flags["score_semantics"]
    reasons = candidates[0]["evidence_provenance"]["reasons"]
    assert reasons and all(isinstance(line, str) and line for line in reasons)
    assert snapshot.source_flags["evidence_provenance"]["system"] == "D63"
    assert snapshot.source_flags["evidence_provenance"]["reference_date"] == "1 Sep 2023"
    assert snapshot.jd_entities["TITLE"]["mentions"] == ["Python Engineer"]
    assert any(c["entities"]["SKILL"]["status"] == "unverified_backend_mentions" for c in candidates)
    assert "fit" not in candidates[0]
    assert len(screener.calls) == 1


def test_same_request_ranks_once_and_status_recovers_after_a_restart():
    adapter, screener = make()
    request = payload()
    first = adapter("submit", request)
    assert first["status"] == "ready"
    again = adapter("submit", request)
    assert again == first and len(screener.calls) == 1
    restarted, fresh = make()
    recovered = restarted("status", {**request, "handle": first["handle"]})
    assert recovered["status"] == "ready" and len(fresh.calls) == 1
    assert recovered["run_id"] == first["run_id"]
    assert [(c["candidate_id"], c["score"]) for c in recovered["candidates"]] == \
        [(c["candidate_id"], c["score"]) for c in first["candidates"]]


def test_a_known_request_with_another_identity_fails_without_running():
    adapter, screener = make()
    adapter("submit", payload())
    assert adapter("submit", payload(jd_digest="b" * 64))["status"] == "failed"
    assert len(screener.calls) == 1


def test_d64_returns_at_once_and_a_d63_behind_it_reports_queued():
    started, release = threading.Event(), threading.Event()

    def slow_c2(pairs, progress=None):
        started.set()
        release.wait(10)
        return c2_scores(pairs, progress)

    adapter, _ = make(Recording(c2=slow_c2), wait_seconds=5.0)
    d64, d63 = payload("request-d64", "D64"), payload("request-d63")
    clock = time.monotonic()
    submitted = adapter("submit", d64)
    assert time.monotonic() - clock < 2
    assert submitted["status"] in ("queued", "running")
    assert _identity(submitted["handle"]) == submitted["handle"]
    assert started.wait(5)
    clock = time.monotonic()
    behind = adapter("submit", d63)
    assert time.monotonic() - clock < 2 and behind["status"] == "queued"
    release.set()
    ready = settle(adapter, d64)
    snapshot = _ready_snapshot({"request_id": d64["request_id"], "payload": d64}, ready)
    assert snapshot.model_id.startswith("D64 frozen L1 + C2 on its top 50, w 0.25")
    assert "C2" in snapshot.source_flags["score_semantics"]
    assert all(c["evidence_provenance"]["c2_score"] is not None for c in snapshot.candidates)
    assert snapshot.source_flags["evidence_provenance"]["c2_k"] == 50
    assert settle(adapter, d63)["status"] == "ready"


def test_the_model_sees_only_the_jd_text_title_and_the_pools_ids_and_texts():
    adapter, screener = make()
    request = payload()
    assert adapter("submit", request)["status"] == "ready"
    (call,) = screener.calls
    assert call["jd_text"] == JD
    assert list(call["cvs"].columns) == ["cv_id", "cv_text"]
    assert call["cvs"].cv_id.tolist() == IDS and call["cvs"].cv_text.tolist() == TEXTS
    assert call["kwargs"]["jd_title"] == "Python Engineer"
    assert call["kwargs"]["reference_date"] == DATE and call["kwargs"].get("jd_id") is None
    passed = json.dumps({k: v for k, v in call["kwargs"].items() if k != "progress"}, default=str)
    for key in ("request_id", "jd_id", "jd_digest", "workspace", "authorization_scope"):
        assert request[key] not in passed + call["jd_text"]


def test_unknown_pool_and_missing_parts_are_service_unavailable(jobs, job):
    unknown, screener = make(pools={})
    ranking = RankingService(jobs.path, jobs.principal, adapter=unknown, pool_ref=POOL)
    assert ranking.rank(job["job_id"])["status"] == "service_unavailable"
    assert unknown("submit", payload())["status"] == "service_unavailable" and not screener.calls
    loads = []

    def factory():
        loads.append(1)
        if len(loads) == 1:
            raise FileNotFoundError("no inference parts")
        return Recording()

    missing = InferenceAdapter(factory, {POOL: pool}, wait_seconds=10.0, model_digest=DIGEST)
    assert missing("submit", payload("request-1"))["status"] == "service_unavailable"
    assert missing("submit", payload("request-2"))["status"] == "ready" and len(loads) == 2


def test_numpy_values_and_long_lists_pass_the_app_validator(jobs, job):
    def numpy_explain(why):
        return {**why, "rank": np.int64(why["rank"]), "score": np.float64(why["score"]),
                "role_match": np.float32(0.5), "experience_compatible": np.bool_(True),
                "missing_skills": tuple(f"missing {k}" for k in range(70)), "years_stated": np.float64(3.0)}

    adapter, _ = make(Recording(extract=numpy_cvs, explain=numpy_explain))
    ranking = RankingService(jobs.path, jobs.principal, adapter=adapter, pool_ref=POOL)
    assert ranking.rank(job["job_id"])["status"] == "ready"
    first = ranking.snapshot(job["job_id"]).candidates[0]
    assert len(first["entities"]["SKILL"]["mentions"]) == 50
    assert first["entities"]["EXPERIENCE_YEARS"]["mentions"] == ["3 years (statement)"]
    assert first["evidence_provenance"]["experience_compatible"] is True
    assert len(first["evidence_provenance"]["missing_skills"]) == 50


def test_unknown_system_fails_and_errors_are_logged_without_document_text(tmp_path):
    adapter, screener = make()
    assert adapter("submit", payload(system="X2"))["status"] == "failed" and not screener.calls
    log = tmp_path / "adapter.log"
    broken, _ = make(Recording(fail=RuntimeError("model exploded")), log_path=log)
    try:
        assert broken("submit", payload("request-broken"))["status"] == "failed"
        text = log.read_text(encoding="utf-8")
        assert "request-broken" in text and "Traceback" in text and "model exploded" in text
        assert "Applicant 0" not in text and JD not in text
    finally:
        for handler in list(portal_inference._log.handlers):
            if getattr(handler, "baseFilename", None) == str(log):
                portal_inference._log.removeHandler(handler)
                handler.close()
        portal_inference._handlers.discard(log)


def test_finished_jobs_are_bounded_and_an_evicted_request_reruns(monkeypatch):
    monkeypatch.setattr(portal_inference, "KEEP", 2)
    adapter, screener = make()
    for index in range(3):
        assert adapter("submit", payload(f"request-{index}"))["status"] == "ready"
    assert list(adapter._jobs) == ["request-1", "request-2"]
    assert adapter("status", payload("request-0"))["status"] == "ready" and len(screener.calls) == 4


def test_the_default_entry_point_is_built_lazily(monkeypatch):
    built = []

    class Probe:
        def __init__(self, factory, pools, **kwargs):
            built.append((factory, sorted(pools), kwargs))

        def __call__(self, operation, request):
            return {"status": "probe"}

    monkeypatch.setattr(portal_inference, "_DEFAULT", None)
    monkeypatch.setattr(portal_inference, "InferenceAdapter", Probe)
    assert portal_inference.adapter("submit", payload()) == {"status": "probe"}
    portal_inference.adapter("status", payload())
    assert len(built) == 1 and built[0][1] == ["a2-final-cvs"]
    assert built[0][2]["log_path"] == portal_inference.LOG


def test_onnx_rejects_d64_without_loading_a_model_or_starting_a_job():
    loads = []
    def factory():
        loads.append(1)
        return Recording()
    adapter = InferenceAdapter(factory, {POOL: pool}, model_digest=DIGEST, embedding_backend="onnx")
    response = adapter("submit", payload(system="D64"))
    assert response["status"] == "service_unavailable"
    assert loads == [] and not adapter._jobs and adapter._worker is None


def test_onnx_d63_preserves_rank_math_and_records_backend_provenance():
    current = Recording()
    current.embedding_artifact_sha256 = "a" * 64
    adapter = InferenceAdapter(lambda: current, {POOL: pool}, wait_seconds=10, model_digest=DIGEST,
                               embedding_backend="onnx")
    response = adapter("submit", payload())
    assert response["status"] == "ready"
    assert current.calls[0]["kwargs"]["system"] == "D63"
    assert current.calls[0]["kwargs"]["rerank"] is False
    assert response["evidence_provenance"]["embedding_backend"] == "onnx"
    assert response["evidence_provenance"]["embedding_artifact_sha256"] == "a" * 64
    assert "SBERT onnx" in response["model_id"]
    assert [item["rank"] for item in response["candidates"]] == list(range(1, 11))


def test_onnx_artifact_change_cannot_reuse_the_previous_backend_run_identity():
    results = []
    for digest in ("a" * 64, "b" * 64):
        current = Recording()
        current.embedding_artifact_sha256 = digest
        adapter = InferenceAdapter(lambda: current, {POOL: pool}, wait_seconds=10, model_digest=DIGEST,
                                   embedding_backend="onnx")
        results.append(adapter("submit", payload()))
    assert results[0]["run_id"] != results[1]["run_id"]
    assert results[0]["candidates"] == results[1]["candidates"]


def test_configured_screener_passes_explicit_backend_to_api(monkeypatch):
    calls = []
    monkeypatch.setattr(Screener, "load", lambda **kwargs: calls.append(kwargs) or object())
    monkeypatch.setenv("HR_PORTAL_EMBEDDING_BACKEND", "onnx")
    portal_inference._load_screener()
    assert calls == [{"embedding_backend": "onnx"}]
    portal_inference._load_screener("torch")
    assert calls[-1] == {"embedding_backend": "torch"}
