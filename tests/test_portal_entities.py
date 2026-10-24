"""Medium portal invariants: synthetic clients and workers, no model assets."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import os
import select
import stat
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from candidate_screener.app import portal_entities as entities
from candidate_screener.app.portal_store import AuthorizationError, Principal


@pytest.fixture
def principal():
    return Principal("synthetic-reviewer", ("benchmark",), time.time() + 3600, "local_token")


def snapshot(first="Python skills; no completed degree, studies ongoing."):
    return SimpleNamespace(jd_id="synthetic-jd-1", snapshot_id="same-snapshot-id",
                           digest="unchanged-table-digest", jd_text="Python developer required.",
                           candidates=({"cv_id": "A", "preview_status": "available_redacted_text",
                                        "preview_text": first},
                                       {"cv_id": "B", "preview_status": "available_redacted_text",
                                        "preview_text": "Never submit this other applicant."}))


def response(document, *, zero=False):
    text = document["fields"]["text"]
    start = text.find("Python")
    records = [] if start < 0 or zero else [
        {key: document[key] for key in ("doc_id", "source", "doc_type", "text_version")}
        | {"field": "text", "start": start, "end": start + 6, "surface": "Python", "label": "KNOWLEDGE",
           "canonical_id": None, "method": "gliner-zero-shot", "extractor_version": entities.VERSION,
           "confidence": .9, "context_status": "unqualified"}]
    return records, {"text": {"characters": len(text), "intervals": [[0, len(text)]],
                               "chunks": 1, "planned_chunks": 1, "complete": True,
                               "quarantined_intervals": [], "quarantined_predictions": 0,
                               "interrupted": False}}


class Client:
    def __init__(self, transform=None, release=None):
        self.calls = []
        self.parameters = []
        self.transform, self.release = transform, release
        self.entered = threading.Event()
        self.closed = False
        self.active = self.max_active = 0
        self.lock = threading.Lock()

    def extract_document(self, document, **parameters):
        with self.lock:
            self.active += 1
            self.max_active = max(self.active, self.max_active)
            self.calls.append(deepcopy(document))
            self.parameters.append(parameters)
        try:
            self.entered.set()
            if self.release is not None:
                assert self.release.wait(3), "test did not release inference"
            result = response(document)
            return self.transform(document, result) if self.transform else result
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        self.closed = True
        if self.release is not None:
            self.release.set()


@pytest.fixture(autouse=True)
def cleanup_registry():
    yield
    entities._shutdown()
    with entities._LOCK:
        entities._MANAGERS.clear()
        entities._ENTRIES.clear()


def latest(service, current, selected=("A",), timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = service.bundle(current, selected)
        if not any(item["status"] in {"queued", "running"} for item in result["documents"].values()):
            return result
        threading.Event().wait(.005)
    pytest.fail("background entity work did not finish")


def test_lazy_nonblocking_prepare_and_latest_api(tmp_path, principal):
    release, loading = threading.Event(), threading.Event()
    client = Client()
    loads = []

    def loader():
        loads.append(True)
        loading.set()
        assert release.wait(3)
        return client

    service = entities.PortalEntityService(tmp_path, principal, loader=loader)
    current = snapshot()
    assert not loads
    assert service.bundle(current, ["A"])["status"] == "unavailable"
    before = time.monotonic()
    pending = service.prepare(current, ["A"])
    assert time.monotonic() - before < .3
    assert pending["status"] in {"queued", "running"}
    assert set(pending["documents"]) == {current.jd_id, "A"}
    assert loading.wait(1)
    before = time.monotonic()
    assert service.bundle(current, ["A"])["status"] in {"queued", "running"}
    assert time.monotonic() - before < .3
    release.set()
    result = latest(service, current)
    assert result["status"] == "complete"
    assert len(loads) == 1
    assert (result["model_id"], result["revision"]) == (entities.MODEL_ID, entities.MODEL_REVISION)
    record = result["documents"]["A"]["entities"][0]
    text = current.candidates[0]["preview_text"]
    assert text[record["start"]:record["end"]] == record["surface"]
    context = record["context"]
    assert context["text"] == text[context["start"]:context["end"]]
    assert "no completed degree" in context["text"] and "ongoing" in context["text"]
    assert record["coordinate_space"] == "redacted_preview"
    assert all(parameters["threshold"] == .5 and parameters["overlap_tokens"] == 32
               for parameters in client.parameters)
    json.dumps(result, allow_nan=False)


def test_only_authorized_requested_preview_fields_enter_model(tmp_path, principal):
    class Other(dict):
        def get(self, key, *args):
            assert key != "preview_text", "unrequested preview was accessed"
            return super().get(key, *args)

    current = snapshot()
    current.candidates = (current.candidates[0], Other(current.candidates[1]))
    current.jd_entities = {"SKILL": {"mentions": ["Invented table metadata"]}}
    client = Client()
    service = entities.PortalEntityService(tmp_path, principal, loader=lambda: client)
    service.prepare(current, ["A"])
    assert latest(service, current)["status"] == "complete"
    assert {document["doc_id"] for document in client.calls} == {current.jd_id, "A"}
    assert {document["fields"]["text"] for document in client.calls} == {
        current.jd_text, current.candidates[0]["preview_text"]}
    with pytest.raises(AuthorizationError):
        service.prepare(current, ["foreign"])
    assert len(client.calls) == 2


def test_recreated_services_concurrent_prepares_and_subsets_do_not_duplicate(tmp_path, principal):
    release = threading.Event()
    client, loads = Client(release=release), []

    def loader():
        loads.append(True)
        return client

    current = snapshot()

    def prepare(index):
        service = entities.PortalEntityService(tmp_path, principal, loader=loader)
        return service.prepare(current, ["A", "A"] if index % 2 else ["A"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = list(pool.map(prepare, range(24)))
    assert client.entered.wait(1)
    assert all(set(item["documents"]) == {current.jd_id, "A"} for item in pending)
    release.set()
    service = entities.PortalEntityService(tmp_path, principal, loader=loader)
    assert latest(service, current)["status"] == "complete"
    for _ in range(4):
        assert service.prepare(current, ["A"])["status"] == "complete"
    service.prepare(current, ["B", "A"])
    assert latest(service, current, ["B", "A"])["status"] == "complete"
    assert Counter(document["doc_id"] for document in client.calls) == {current.jd_id: 1, "A": 1, "B": 1}
    assert len(loads) == 1 and client.max_active == 1


def test_disk_cache_is_private_and_survives_memory_loss_without_loading(tmp_path, principal):
    client = Client()
    current = snapshot()
    service = entities.PortalEntityService(tmp_path, principal, loader=lambda: client)
    service.prepare(current, ["A"])
    result = latest(service, current)
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 2
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    for path in files:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert "synthetic-reviewer" not in str(path)
        assert "Never submit" not in path.read_text()
    with entities._LOCK:
        entities._ENTRIES.clear()

    def forbidden():
        pytest.fail("disk cache hit must not load the model")

    reopened = entities.PortalEntityService(tmp_path, principal, loader=forbidden)
    assert reopened.prepare(current, ["A"]) == result
    result["documents"]["A"]["entities"].clear()
    assert reopened.bundle(current, ["A"])["documents"]["A"]["entities"]
    assert len(client.calls) == 2


def test_source_and_configuration_changes_invalidate_only_affected_document(tmp_path, principal, monkeypatch):
    client = Client()
    loader = lambda: client
    service = entities.PortalEntityService(tmp_path, principal, loader=loader)
    before = snapshot()
    service.prepare(before, ["A"])
    original = latest(service, before)
    changed = snapshot("😀 Changed source. Python developer, diploma expired.")
    service.prepare(changed, ["A"])
    updated = latest(service, changed)
    assert original["documents"]["A"]["source_digest"] != updated["documents"]["A"]["source_digest"]
    assert len(client.calls) == 3
    assert updated["documents"]["A"]["entities"][0]["start"] == changed.candidates[0]["preview_text"].index("Python")
    monkeypatch.setattr(entities, "THRESHOLD", .6)
    configured = entities.PortalEntityService(tmp_path, principal, loader=loader)
    configured.prepare(changed, ["A"])
    assert latest(configured, changed)["status"] == "complete"
    assert len(client.calls) == 5
    assert configured.config_digest != service.config_digest


def test_guard_on_memory_disk_hits_and_after_background_work(tmp_path, principal):
    live = [True]
    release = threading.Event()
    client = Client(release=release)
    service = entities.PortalEntityService(tmp_path, principal, guard=lambda: live[0], loader=lambda: client)
    current = snapshot()
    service.prepare(current, ["A"])
    assert client.entered.wait(1)
    live[0] = False
    release.set()
    with pytest.raises(AuthorizationError):
        service.bundle(current, ["A"])
    # Re-enable only to inspect the denied task's public result: no entities or
    # persistent result should have been published after the post-work guard.
    until = time.monotonic() + 2
    while time.monotonic() < until:
        with entities._LOCK:
            if all(item["status"] == "unavailable" for item in entities._ENTRIES.values()):
                break
        threading.Event().wait(.005)
    live[0] = True
    result = latest(service, current)
    assert result["status"] == "unavailable"
    assert not list(tmp_path.rglob("*.json"))
    assert not any(item["entities"] for item in result["documents"].values())


def test_cache_hits_do_not_grant_owner_workspace_or_realm_access(tmp_path, principal):
    client = Client()
    loader = lambda: client
    current = snapshot()
    service = entities.PortalEntityService(tmp_path, principal, loader=loader)
    service.prepare(current, ["A"])
    latest(service, current)
    for other in (replace(principal, subject="someone-else"),
                  replace(principal, departments=("engineering",)),
                  replace(principal, auth_mode="demo", departments=("demo",))):
        scoped = entities.PortalEntityService(tmp_path, other, loader=loader)
        assert scoped.bundle(current, ["A"])["status"] == "unavailable"
    current.workspace = "engineering"
    with pytest.raises(AuthorizationError):
        service.bundle(current, ["A"])
    current.workspace = "demo"
    invalid = replace(principal, departments=("demo",))
    with pytest.raises(AuthorizationError):
        entities.PortalEntityService(tmp_path, invalid).prepare(current, ["A"])
    with pytest.raises(AuthorizationError):
        entities.PortalEntityService(tmp_path, replace(principal, expires_at=0))
    assert len(client.calls) == 2


def test_one_model_is_shared_but_same_principal_workspaces_have_separate_caches(tmp_path, principal):
    principal = replace(principal, departments=("benchmark", "engineering"))
    client, loads = Client(), []

    def loader():
        loads.append(True)
        return client

    current = snapshot()
    current.workspace = "benchmark"
    service = entities.PortalEntityService(tmp_path, principal, loader=loader)
    service.prepare(current, ["A"])
    latest(service, current)
    current.workspace = "engineering"
    recreated = entities.PortalEntityService(tmp_path, principal, loader=loader)
    assert recreated.bundle(current, ["A"])["status"] == "unavailable"
    recreated.prepare(current, ["A"])
    assert latest(recreated, current)["status"] == "complete"
    assert len(client.calls) == 4 and len(loads) == 1


def test_ambiguous_workspace_cannot_share_an_authorized_department_cache(tmp_path, principal):
    principal = replace(principal, departments=("benchmark", "engineering"))
    client = Client()
    service = entities.PortalEntityService(tmp_path, principal, loader=lambda: client)
    with pytest.raises(AuthorizationError, match="workspace"):
        service.prepare(snapshot(), ["A"])
    assert not client.calls


def test_guard_rechecked_for_valid_memory_and_disk_cache(tmp_path, principal):
    live = [True]
    client = Client()
    service = entities.PortalEntityService(tmp_path, principal, guard=lambda: live[0], loader=lambda: client)
    current = snapshot()
    service.prepare(current, ["A"])
    latest(service, current)
    live[0] = False
    with pytest.raises(AuthorizationError):
        service.prepare(current, ["A"])
    with entities._LOCK:
        entities._ENTRIES.clear()
    with pytest.raises(AuthorizationError):
        service.bundle(current, ["A"])


@pytest.mark.parametrize("change", [
    {"start": True}, {"end": 90000}, {"start": -1}, {"surface": "fabricated"},
    {"doc_id": "other-owner"}, {"doc_type": "other-kind"}, {"field": "metadata"},
    {"label": "CERTIFICATION"}, {"confidence": float("nan")}, {"confidence": True},
])
def test_invalid_predictions_are_quarantined_never_complete_zero(tmp_path, principal, change):
    def corrupt(document, result):
        if document["doc_type"] == "cv":
            result[0][0].update(change)
        return result

    current, client = snapshot(), Client(corrupt)
    service = entities.PortalEntityService(tmp_path, principal, loader=lambda: client)
    service.prepare(current, ["A"])
    result = latest(service, current)
    assert result["status"] == "partial"
    cv = result["documents"]["A"]
    assert cv["status"] == "partial" and not cv["entities"]
    assert not cv["coverage"]["text"]["complete"]
    assert cv["coverage"]["text"]["quarantined_predictions"] == 1


def test_unsubmitted_tail_and_worker_quarantine_cannot_claim_complete(tmp_path, principal):
    def omit_tail(document, result):
        records, coverage = result
        if document["doc_type"] == "cv":
            coverage["text"]["intervals"] = [[0, 6]]
        return records, coverage

    current = snapshot()
    client = Client(omit_tail)
    service = entities.PortalEntityService(tmp_path, principal, loader=lambda: client)
    service.prepare(current, ["A"])
    cv = latest(service, current)["documents"]["A"]
    assert cv["status"] == "partial" and len(cv["entities"]) == 1
    assert not cv["coverage"]["text"]["complete"]
    with entities._LOCK:
        entities._ENTRIES.clear()
    assert service.bundle(current, ["A"])["documents"]["A"]["status"] == "partial"


def test_cache_write_failure_preserves_validated_partial_evidence(tmp_path, principal, monkeypatch):
    def partial(document, result):
        result[1]["text"]["interrupted"] = True
        result[1]["text"]["complete"] = False
        return result

    client = Client(partial)
    service = entities.PortalEntityService(tmp_path, principal, loader=lambda: client)

    def unwritable(*args):
        raise OSError("sensitive local path")

    monkeypatch.setattr(service, "_write", unwritable)
    current = snapshot()
    service.prepare(current, ["A"])
    result = latest(service, current)
    assert result["status"] == "partial"
    assert all(doc["entities"] and doc["status"] == "partial" and doc["cache_status"] == "unavailable"
               for doc in result["documents"].values())
    assert "sensitive" not in json.dumps(result)
    service.prepare(current, ["A"])
    assert len(client.calls) == 2


def test_complete_empty_is_distinct_from_missing_partial_unavailable_and_failed(tmp_path, principal):
    current = snapshot()

    def zero(document, result):
        return response(document, zero=True)

    complete = entities.PortalEntityService(tmp_path / "zero", principal, loader=lambda: Client(zero))
    complete.prepare(current, ["A"])
    result = latest(complete, current)
    assert result["status"] == "complete"
    assert all(not doc["entities"] and doc["coverage"]["text"]["complete"]
               for doc in result["documents"].values())

    def fail(document, result):
        raise RuntimeError("SENSITIVE DOCUMENT CONTENT MUST NOT APPEAR")

    failed = entities.PortalEntityService(tmp_path / "failed", principal, loader=lambda: Client(fail))
    failed.prepare(current, ["A"])
    result = latest(failed, current)
    assert result["status"] == "failed"
    assert "SENSITIVE" not in json.dumps(result)
    assert not list((tmp_path / "failed").rglob("*.json"))

    def absent():
        raise FileNotFoundError("SENSITIVE LOCAL PATH")

    unavailable = entities.PortalEntityService(tmp_path / "absent", principal, loader=absent)
    unavailable.prepare(current, ["A"])
    assert latest(unavailable, current)["status"] == "unavailable"
    unavailable.prepare(current, ["A"])
    assert latest(unavailable, current)["status"] == "unavailable"

    def partial(document, result):
        return [], {"text": {"characters": len(document["fields"]["text"]),
                              "intervals": [], "complete": False, "interrupted": True}}

    interrupted = entities.PortalEntityService(tmp_path / "partial", principal, loader=lambda: Client(partial))
    interrupted.prepare(current, ["A"])
    assert latest(interrupted, current)["status"] == "partial"


def test_unavailable_preview_and_unredacted_text_are_never_inferred(tmp_path, principal):
    current = snapshot("Python person@example.org")
    current.jd_text = ""
    current.candidates[1]["preview_status"] = "unavailable"
    client = Client()
    service = entities.PortalEntityService(tmp_path, principal, loader=lambda: client)
    result = service.prepare(current, ["A", "B"])
    assert result["status"] == "unavailable" and not client.calls
    assert result["documents"]["A"]["reason"] == "redaction_required"
    assert result["documents"]["B"]["reason"] == "source_unavailable"
    assert not list(tmp_path.rglob("*.json"))


def test_foreign_copied_disk_cache_is_rejected_without_inference(tmp_path, principal):
    current, client = snapshot(), Client()
    service = entities.PortalEntityService(tmp_path, principal, loader=lambda: client)
    service.prepare(current, ["A"])
    latest(service, current)
    other = entities.PortalEntityService(tmp_path, replace(principal, subject="other-owner"), loader=lambda: client)
    scope, sources = other._request(current, ["A"])
    originals = {json.loads(path.read_text())["document"]["owner_id"]: path for path in tmp_path.rglob("*.json")}
    for path, _, initial in sources:
        path.parent.mkdir(exist_ok=True)
        path.write_text(originals[initial["owner_id"]].read_text())
    result = other.prepare(current, ["A"])
    assert result["status"] == "failed"
    assert all(doc["reason"] == "invalid_cache" and not doc["entities"]
               for doc in result["documents"].values())
    assert len(client.calls) == 2


def test_injected_timeout_releases_manager_and_is_never_a_zero(tmp_path, principal, monkeypatch):
    monkeypatch.setattr(entities, "INFERENCE_TIMEOUT", .03)
    monkeypatch.setattr(entities, "CLEANUP_TIMEOUT", .02)
    release, client = threading.Event(), None
    client = Client(release=release)
    service = entities.PortalEntityService(tmp_path, principal, loader=lambda: client)
    current = snapshot()
    service.prepare(current, ["A"])
    result = latest(service, current)
    assert result["documents"][current.jd_id]["status"] == "failed"
    assert result["documents"][current.jd_id]["reason"] == "timeout"
    assert client.closed
    assert len(client.calls) == 2


def test_process_boundary_hard_timeout_kills_blocked_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(entities, "LOAD_TIMEOUT", .2)
    pid_file = tmp_path / "pid"
    code = "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    before = time.monotonic()
    with pytest.raises(TimeoutError):
        entities._RuntimeClient([sys.executable, "-c", code, str(pid_file)])
    assert time.monotonic() - before < 2
    assert pid_file.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_shutdown_finds_and_kills_a_proxy_still_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(entities, "LOAD_TIMEOUT", 30)
    pid_file, errors = tmp_path / "loading-pid", []
    code = "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"

    def start():
        try:
            entities._RuntimeClient([sys.executable, "-c", code, str(pid_file)])
        except Exception as error:
            errors.append(type(error))

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    until = time.monotonic() + 2
    while not pid_file.exists() and time.monotonic() < until:
        threading.Event().wait(.005)
    assert pid_file.exists() and entities._RUNTIMES
    entities._shutdown()
    thread.join(timeout=1)
    assert not thread.is_alive() and errors
    assert not entities._RUNTIMES
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_process_boundary_hard_timeout_kills_read_and_child_process(tmp_path, monkeypatch):
    monkeypatch.setattr(entities, "INFERENCE_TIMEOUT", .15)
    child_file = tmp_path / "child"
    code = "\n".join([
        "import json,os,pathlib,subprocess,sys,time",
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])",
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))",
        "print('PORTAL_ENTITIES\\t'+json.dumps({'status':'ready'}),flush=True)",
        "sys.stdin.readline()",
        "time.sleep(30)",
    ])
    runtime = entities._RuntimeClient([sys.executable, "-c", code, str(child_file)])
    # Both supervisor and child inherit the pipe's write end. A duplicate read
    # descriptor reaches EOF only after both have closed it or been killed.
    monitor = os.dup(runtime.process.stdout.fileno())
    before = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            runtime.extract_document({"doc_id": "synthetic", "fields": {"text": "Python"}})
        assert time.monotonic() - before < 2
        assert runtime.process.poll() is not None
        assert int(child_file.read_text()) != runtime.process.pid
        assert select.select([monitor], [], [], 1)[0], "model child still holds its inherited pipe open"
        assert os.read(monitor, 1) == b""
    finally:
        os.close(monitor)
        runtime.close()


def test_failed_tasks_are_not_duplicated_on_reruns(tmp_path, principal):
    loads = []

    def fail():
        loads.append(True)
        raise RuntimeError("offline failure")

    service = entities.PortalEntityService(tmp_path, principal, loader=fail)
    current = snapshot()
    service.prepare(current, ["A"])
    assert latest(service, current)["status"] == "failed"
    before = len(loads)
    for _ in range(5):
        recreated = entities.PortalEntityService(tmp_path, principal, loader=fail)
        assert recreated.prepare(current, ["A"])["status"] == "failed"
    assert len(loads) == before
