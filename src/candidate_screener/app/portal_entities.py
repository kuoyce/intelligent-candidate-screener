"""Authorized, asynchronous extraction over portal redacted display coordinates.

``prepare(snapshot, candidate_ids)`` schedules the JD and only the requested CVs;
``bundle`` reads their latest state without scheduling. Both return JSON values.
The controller must supply its current authorization/freshness guard. A loader
test seam returns an ``extract_document(document, **parameters)`` client; normal
operation always uses the installed isolated medium runtime, never downloads.

Only the model is shared between principals. Content and durable results live
under a digest of the cache root, principal, auth realm and workspace. Module
registries survive Streamlit script reruns. No Streamlit or neural imports occur
in the UI thread. Pending and unsuccessful extraction never mean zero mentions.
"""
from __future__ import annotations

import atexit
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time

from candidate_screener.annotation.redact import PATTERNS, redact_text
from candidate_screener.extraction.medium_assets import MODEL_DIR, MODEL_ID, MODEL_REVISION
from candidate_screener.extraction.medium_inference import VERSION
from candidate_screener.extraction.schema import validate_entity
from candidate_screener.extraction.sentence_chunking import HEADROOM_TOKENS, STRATEGY_VERSION
from candidate_screener.extraction.transformer import CONTEXT_VERSION, MODEL_LABELS
from .portal_store import AuthorizationError, PortalStore

LOAD_TIMEOUT = 120.0
INFERENCE_TIMEOUT = 90.0
CLEANUP_TIMEOUT = 1.0
THRESHOLD = .5
OVERLAP_TOKENS = 32
_PREFIX = "PORTAL_ENTITIES\t"
_TERMINAL = frozenset({"complete", "partial", "failed", "unavailable"})
_LOCK = threading.RLock()
_ENTRIES = {}
_MANAGERS = {}
_RUNTIMES = {}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _configuration():
    return {"version": "portal-medium-v1", "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "extractor_version": VERSION, "labels": dict(MODEL_LABELS), "threshold": THRESHOLD,
            "chunking": {"strategy": STRATEGY_VERSION, "overlap_tokens": OVERLAP_TOKENS,
                         "headroom_tokens": HEADROOM_TOKENS},
            "context_version": CONTEXT_VERSION, "coordinate_space": "redacted_preview",
            "redaction": [(name, pattern.pattern, pattern.flags, replacement)
                          for name, pattern, replacement in PATTERNS]}


class _Unavailable(RuntimeError):
    pass


def _kill(process):
    """Kill the supervisor's entire session, including uv and its model child."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=CLEANUP_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            stream.close()


class _RuntimeClient:
    """Hard limits outside the legacy proxy's blocking constructor and _read.

    The supervisor owns a fresh process group. A hung load/read/write cannot
    hold the manager forever: timeout kills the group and reaps the supervisor.
    ``command`` is an internal seam for tests using a tiny synthetic worker.
    """

    def __init__(self, command=None):
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           UV_OFFLINE="1")
        self.process = subprocess.Popen(
            command or [sys.executable, "-m", "candidate_screener.app.portal_entities", "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True, env=environment)
        self._buffer = b""
        self._closed = False
        # Register before waiting for readiness: shutdown must also find a
        # supervisor whose legacy proxy is still blocked in its constructor.
        with _LOCK:
            _RUNTIMES[self.process.pid] = self
        try:
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            message = self._read(time.monotonic() + LOAD_TIMEOUT)
            if message.get("status") == "unavailable":
                raise _Unavailable("Local medium runtime unavailable")
            if message.get("status") != "ready":
                raise RuntimeError("Medium initialization failed")
            self._parameters = message.get("chunk_parameters", {})
        except BaseException:
            self.close()
            raise

    def _read(self, deadline):
        while True:
            if b"\n" in self._buffer:
                line, self._buffer = self._buffer.split(b"\n", 1)
                if line.startswith(_PREFIX.encode()):
                    return json.loads(line[len(_PREFIX):])
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("Medium worker timed out")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError("Medium worker stopped")
            self._buffer += data
            if len(self._buffer) > 32 * 1024 * 1024:
                raise ValueError("Medium response exceeds limit")

    def extract_document(self, document, **parameters):
        deadline = time.monotonic() + INFERENCE_TIMEOUT
        payload = memoryview((json.dumps({"document": document, "parameters": parameters}) + "\n").encode())
        try:
            while payload:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                    raise TimeoutError("Medium worker timed out")
                payload = payload[os.write(self.process.stdin.fileno(), payload):]
            message = self._read(deadline)
            if message.get("status") != "ok":
                raise RuntimeError("Medium inference failed")
            return message["records"], message["coverage"]
        except BaseException:
            self.close()
            raise

    def close(self):
        with _LOCK:
            if self._closed:
                return
            self._closed = True
            _RUNTIMES.pop(self.process.pid, None)
        _kill(self.process)


def _default_loader():
    from candidate_screener.config import PROJECT_ROOT
    runtime = PROJECT_ROOT / "runtime/gliner-medium/.venv/bin/python"
    if not runtime.is_file() or not (MODEL_DIR / "model-manifest.json").is_file():
        raise _Unavailable("Installed local medium assets required")
    return _RuntimeClient()


def _worker_main():
    """Private supervisor; exception messages and document text never reach logs."""
    def emit(payload):
        print(_PREFIX + json.dumps(payload), flush=True)

    from candidate_screener.extraction.medium_runtime import MediumRuntimeProxy
    proxy = None
    try:
        proxy = MediumRuntimeProxy(MODEL_DIR)
        emit({"status": "ready", "chunk_parameters": proxy.chunk_parameters()})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                records, coverage = proxy.extract_document(request["document"], **request["parameters"])
                emit({"status": "ok", "records": records, "coverage": coverage})
            except Exception:
                emit({"status": "failed"})
    except (FileNotFoundError, ImportError):
        emit({"status": "unavailable"})
    except Exception:
        emit({"status": "failed"})
    finally:
        if proxy is not None:
            proxy.close()


def _dispose(client):
    if client is not None and hasattr(client, "close"):
        try:
            client.close()
        except Exception:
            pass


def _bounded(call, seconds, *, cancel=None, dispose_late=False):
    """Also bound injected clients; production additionally kills its OS worker."""
    done, abandoned, result = threading.Event(), threading.Event(), []
    gate = threading.Lock()

    def run():
        try:
            outcome = (True, call())
        except BaseException as exc:
            outcome = (False, exc)
        with gate:
            result.append(outcome)
            done.set()
            late = abandoned.is_set()
        if late and dispose_late and result[0][0]:
            _dispose(result[0][1])

    threading.Thread(target=run, daemon=True, name="portal-medium-io").start()
    if not done.wait(seconds):
        with gate:
            timed_out = not done.is_set()
            if timed_out:
                abandoned.set()
        if timed_out:
            if cancel is not None:
                cancel()
            raise TimeoutError("Medium operation timed out")
    success, value = result[0]
    if not success:
        raise value
    return value


def _intervals(value, length):
    if not isinstance(value, list):
        raise ValueError("Invalid coverage intervals")
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(number) is not int for number in pair)
                or not 0 <= pair[0] < pair[1] <= length):
            raise ValueError("Invalid coverage offsets")
    return [list(pair) for pair in value]


def _covered(intervals, length):
    end = 0
    for start, stop in sorted(intervals):
        if start > end:
            return False
        end = max(end, stop)
    return end == length


def _normalise(document, result, threshold):
    """Revalidate the process/cache boundary, retaining incomplete provenance."""
    records, raw_coverage = result
    if not isinstance(records, list) or not isinstance(raw_coverage, dict):
        raise ValueError("Invalid extraction response")
    text = document["fields"]["text"]
    raw = raw_coverage.get("text", {})
    if not isinstance(raw, dict) or raw.get("characters", len(text)) != len(text):
        raise ValueError("Coverage belongs to a different source")
    coverage = {"characters": len(text), "intervals": _intervals(raw.get("intervals", []), len(text)),
                "quarantined_intervals": _intervals(raw.get("quarantined_intervals", []), len(text))}
    for key in ("sentence_count", "headroom_tokens", "token_limit", "word_limit", "fallback_windows",
                "oversized_sentences", "quarantine_count", "chunks", "planned_chunks", "quarantined_predictions"):
        value = raw.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError("Invalid coverage counter")
        coverage[key] = value
    coverage["strategy"] = STRATEGY_VERSION
    coverage["planned_intervals"] = _intervals(raw.get("planned_intervals", []), len(text))
    coverage["interrupted"] = raw.get("interrupted") is True
    entities, seen, invalid = [], set(), 0
    for raw_record in records:
        try:
            if not isinstance(raw_record, dict):
                raise ValueError("Invalid entity")
            record = {key: raw_record[key] for key in (
                "doc_id", "source", "doc_type", "field", "text_version", "start", "end", "surface",
                "label", "canonical_id", "method", "extractor_version", "confidence", "context_status")}
            if (any(record[key] != document[key] for key in ("doc_id", "source", "doc_type", "text_version"))
                    or record["field"] != "text" or record["method"] != "gliner-zero-shot"
                    or record["extractor_version"] != VERSION
                    or type(record["start"]) is not int or type(record["end"]) is not int
                    or record["canonical_id"] is not None):
                raise ValueError("Entity belongs to a different source or configuration")
            validate_entity(record, text)
            if not any(a <= record["start"] < record["end"] <= b for a, b in coverage["intervals"]):
                raise ValueError("Entity outside processed intervals")
            if record["confidence"] < threshold:
                raise ValueError("Entity below configured threshold")
            key = record["start"], record["end"], record["label"]
            if key in seen:
                continue
            seen.add(key)
            record["entity_id"] = _digest(record)
            record["coordinate_space"] = "redacted_preview"
            # Retain nearby original wording, including negation/ongoing/expiry.
            start = max(0, record["start"] - 240)
            end = min(len(text), record["end"] + 240)
            record["context"] = {"start": start, "end": end, "text": text[start:end]}
            entities.append(record)
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid += 1
    coverage["quarantined_predictions"] += invalid
    coverage["complete"] = (raw.get("complete") is True
                            and _covered(coverage["intervals"], len(text))
                            and not coverage["quarantined_intervals"]
                            and not coverage["quarantined_predictions"] and not coverage["interrupted"])
    return {"status": "complete" if coverage["complete"] else "partial",
            "entities": sorted(entities, key=lambda row: (row["start"], row["end"], row["label"])),
            "coverage": {"text": coverage}}


class _Manager:
    def __init__(self, loader):
        self.loader, self.client = loader, None
        self.jobs = queue.Queue(maxsize=128)
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True, name="portal-medium-manager")
        self.thread.start()

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            service, scope, path, document, initial = job
            try:
                service._guard(scope)
                if self.closed:
                    raise RuntimeError("Entity worker stopped")
                with _LOCK:
                    _ENTRIES[str(path)] = {**initial, "status": "running"}
                if self.client is None:
                    self.client = _bounded(self.loader, LOAD_TIMEOUT + CLEANUP_TIMEOUT,
                                           dispose_late=True)
                if self.closed:
                    raise RuntimeError("Entity worker stopped")
                parameters = {"threshold": service.config["threshold"],
                              "overlap_tokens": service.config["chunking"]["overlap_tokens"],
                              "deadline": time.monotonic() + INFERENCE_TIMEOUT * .9}
                from candidate_screener.extraction.medium_inference import extract_document
                result = _bounded(lambda: extract_document(document, self.client, **parameters),
                                  INFERENCE_TIMEOUT + CLEANUP_TIMEOUT,
                                  cancel=lambda: _dispose(self.client))
                completed = {**initial, **_normalise(document, result, service.config["threshold"])}
                service._guard(scope)
                try:
                    service._write(path, scope, completed)
                except (OSError, ValueError):
                    service._guard(scope)
                    # A cache failure does not erase validated partial results
                    # or turn a successful extraction into a model failure.
                    completed["cache_status"] = "unavailable"
                service._guard(scope)
            except (AuthorizationError, PermissionError):
                completed = {**initial, "status": "unavailable", "reason": "access_revoked"}
            except (_Unavailable, FileNotFoundError, ImportError):
                completed = {**initial, "status": "unavailable", "reason": "local_model_unavailable"}
            except Exception as exc:
                completed = {**initial, "status": "failed",
                             "reason": "timeout" if isinstance(exc, TimeoutError) else "extraction_failed"}
                _dispose(self.client)
                self.client = None
            with _LOCK:
                _ENTRIES[str(path)] = completed

    def close(self):
        self.closed = True
        _dispose(self.client)
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass


def _shutdown():
    with _LOCK:
        managers, runtimes = tuple(_MANAGERS.values()), tuple(_RUNTIMES.values())
        for manager in managers:
            manager.closed = True
    for runtime in runtimes:
        runtime.close()
    for manager in managers:
        manager.close()


atexit.register(_shutdown)


class PortalEntityService:
    """Private per-document cache with a shared, serialized lazy model worker.

    ``guard()`` may raise or return False; a None return is successful. Pass the
    controller's guard so revocation and source freshness are enforced. When the
    snapshot has no workspace/department field, a bound controller department
    or its sole authorized department scopes its workspace. A principal with
    multiple departments needs an explicit workspace or bound controller guard.
    """

    def __init__(self, cache_dir, principal, guard=None, loader=None):
        self.cache_dir = Path(cache_dir).absolute()
        self.principal, self.guard, self.loader = principal, guard, loader
        self.config = _configuration()
        self.config_digest = _digest(self.config)
        self._identity = self._principal_identity()
        self._guard()

    def _principal_identity(self):
        PortalStore._authorize(self.principal)
        return (self.principal.subject, self.principal.auth_mode, self.principal.departments)

    def _guard(self, scope=None):
        if self._principal_identity() != self._identity:
            raise AuthorizationError("Entity service identity changed")
        if scope is not None and scope["workspace"] is not None:
            workspace = scope["workspace"]
            PortalStore._authorize(self.principal, workspace)
            if (self.principal.auth_mode == "demo") != (workspace == "demo"):
                raise AuthorizationError("Demonstration and operational workspaces are separate")
        if self.guard is not None and self.guard() is False:
            raise AuthorizationError("Access revoked; sign in again")

    def _request(self, snapshot, candidate_ids):
        self._guard()
        if isinstance(candidate_ids, (str, bytes)):
            raise ValueError("Candidate IDs must be a collection")
        selected = tuple(dict.fromkeys(candidate_ids))
        if any(not isinstance(identifier, str) or not identifier for identifier in selected):
            raise ValueError("Invalid candidate ID")
        candidates = {}
        # Do not access unrequested preview texts or use table entity summaries.
        for candidate in snapshot.candidates:
            identifier = candidate.get("cv_id")
            if identifier in selected:
                if identifier in candidates:
                    raise AuthorizationError("Ambiguous candidate ownership")
                candidates[identifier] = candidate
        if set(candidates) != set(selected):
            raise AuthorizationError("Candidate outside the current snapshot")
        jd_id = snapshot.jd_id
        if not isinstance(jd_id, str) or not jd_id or jd_id in selected:
            raise ValueError("Invalid or ambiguous JD identity")
        controller = getattr(self.guard, "__self__", None)
        workspace = getattr(snapshot, "workspace", None) or getattr(snapshot, "department", None)
        workspace = workspace or getattr(controller, "department", None)
        if workspace is None and len(self.principal.departments) == 1:
            workspace = self.principal.departments[0]
        if workspace is None:
            raise AuthorizationError("An explicit entity workspace is required for this identity")
        scope = {"owner": self.principal.subject, "realm": self.principal.auth_mode,
                 "departments": list(self.principal.departments), "workspace": workspace}
        self._guard(scope)
        sources = [(jd_id, "jd", snapshot.jd_text, True)]
        for identifier in selected:
            candidate = candidates[identifier]
            available = candidate.get("preview_status") == "available_redacted_text"
            sources.append((identifier, "cv", candidate.get("preview_text") if available else None, available))
        documents = []
        for owner, kind, text, available in sources:
            digest = hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None
            entry = {"owner_id": owner, "doc_type": kind, "status": "unavailable", "entities": [],
                     "coverage": {}, "source_digest": digest, "coordinate_space": "redacted_preview"}
            if not available or not isinstance(text, str) or not text.strip():
                entry["reason"] = "source_unavailable"
            elif redact_text(text) != text:
                entry["reason"] = "redaction_required"
            else:
                entry["reason"] = "not_prepared"
            document = {"doc_id": owner, "source": "portal-redacted-preview", "doc_type": kind,
                        "text_version": "redacted-preview-v1", "fields": {"text": text}}
            key = _digest({"owner": owner, "kind": kind, "source_digest": digest,
                           "config_digest": self.config_digest})
            path = self.cache_dir / _digest(scope) / (key + ".json")
            documents.append((path, document, entry))
        return scope, documents

    @staticmethod
    def _private_directory(path):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Invalid entity cache directory")
        path.chmod(0o700)

    def _read(self, path, scope, document, initial):
        self._guard(scope)
        if not path.exists():
            return initial
        try:
            if path.parent.is_symlink():
                raise ValueError("Invalid entity cache directory")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor) as stream:
                payload = json.load(stream)
            if (payload.get("scope_digest") != _digest(scope)
                    or payload.get("config_digest") != self.config_digest
                    or payload.get("cache_key") != path.stem):
                raise ValueError("Foreign cache identity")
            cached = payload["document"]
            if any(cached.get(key) != initial[key] for key in ("owner_id", "doc_type", "source_digest")):
                raise ValueError("Foreign cached source")
            result = _normalise(document, (cached["entities"], cached["coverage"]), self.config["threshold"])
            if cached["status"] != result["status"]:
                raise ValueError("Invalid cached completion")
            result = {**initial, **result}
            result.pop("reason", None)
        except (OSError, KeyError, TypeError, ValueError):
            result = {**initial, "status": "failed", "reason": "invalid_cache"}
        self._guard(scope)
        return result

    def _write(self, path, scope, completed):
        self._guard(scope)
        self._private_directory(self.cache_dir)
        self._private_directory(path.parent)
        payload = {"scope_digest": _digest(scope), "config_digest": self.config_digest,
                   "cache_key": path.stem, "document": completed}
        descriptor, name = tempfile.mkstemp(prefix=".entity-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            self._guard(scope)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _bundle(self, snapshot, candidate_ids, *, prepare):
        scope, sources = self._request(snapshot, candidate_ids)
        documents = {}
        for path, document, initial in sources:
            self._guard(scope)
            if initial["reason"] != "not_prepared":
                documents[initial["owner_id"]] = initial
                continue
            with _LOCK:
                entry = _ENTRIES.get(str(path))
                if entry is None:
                    entry = self._read(path, scope, document, initial)
                    if entry["status"] in _TERMINAL and entry.get("reason") != "not_prepared":
                        _ENTRIES[str(path)] = entry
                if prepare and entry.get("reason") == "not_prepared":
                    entry = {**initial, "status": "queued"}
                    entry.pop("reason", None)
                    _ENTRIES[str(path)] = entry
                    manager_key = None if self.loader is None else id(self.loader)
                    if manager_key not in _MANAGERS:
                        _MANAGERS[manager_key] = _Manager(self.loader or _default_loader)
                    try:
                        _MANAGERS[manager_key].jobs.put_nowait((self, scope, path, document, entry))
                    except queue.Full:
                        entry = {**initial, "status": "failed", "reason": "queue_full"}
                        _ENTRIES[str(path)] = entry
                documents[initial["owner_id"]] = deepcopy(entry)
        self._guard(scope)
        statuses = {document["status"] for document in documents.values()}
        if "running" in statuses:
            status = "running"
        elif "queued" in statuses:
            status = "queued"
        elif statuses == {"complete"}:
            status = "complete"
        elif "complete" in statuses or "partial" in statuses:
            status = "partial"
        elif "failed" in statuses:
            status = "failed"
        else:
            status = "unavailable"
        return {"status": status, "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "config_digest": self.config_digest, "documents": documents}

    def prepare(self, snapshot, candidate_ids):
        """Queue absent authorized sources once; never wait for model work."""
        return self._bundle(snapshot, candidate_ids, prepare=True)

    def bundle(self, snapshot, candidate_ids):
        """Read the latest result, reauthorizing even a memory/disk cache hit."""
        return self._bundle(snapshot, candidate_ids, prepare=False)


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    _worker_main()
