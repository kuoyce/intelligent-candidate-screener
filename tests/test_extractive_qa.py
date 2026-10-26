"""QA source-boundary and offline asset guards without neural dependencies."""
import json
import subprocess
import sys
import time
import threading
import pytest
from candidate_screener.llm.qa_assets import MODEL_ID, file_digest, verify_manifest
from candidate_screener.llm.extractive_qa import QaClient, best_context_span, answerable


def _before(client, deadline):
    method = getattr(client, "answer_before", None)
    return method("question", [], deadline) if method else client.answer("question", [])


def test_absolute_turn_deadline_includes_waiting_for_shared_reader_lock(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    client = QaClient(manifest, timeout=.5)
    client._lock.acquire()
    release = threading.Timer(.3, client._lock.release)
    release.start()
    started = time.monotonic()
    try:
        result = _before(client, started + .04)
        assert time.monotonic() - started < .2
        assert result["status"] == "unavailable"
    finally:
        release.join()
        client.close()


def test_absolute_turn_deadline_does_not_restart_for_worker_exchange(synthetic_qa_client):
    client, processes = synthetic_qa_client(
        "import sys,time\n"
        "sys.stdin.readline()\n"
        "time.sleep(.3)\n"
        "print('{\\\"status\\\":\\\"not_found\\\",\\\"spans\\\":[]}',flush=True)\n", timeout=.6)
    started = time.monotonic()
    result = _before(client, started + .04)
    assert time.monotonic() - started < .2
    assert result["status"] == "unavailable"
    assert processes[0].poll() is not None


def test_list_qa_maps_two_unicode_mentions_and_preserves_negation_context():
    from candidate_screener.llm.extractive_qa import ExtractiveQA
    qa = ExtractiveQA.__new__(ExtractiveQA)
    qa.threshold = 0.
    qa.manifest = {"model_id": "fake", "revision": "fake"}
    seen = []
    def native(question, context):
        seen.append(context)
        for item in ("AWS", "Azure"):
            if item in context:
                start = context.index(item)
                return (10., start, start+len(item)), 0.
        return None, 0.
    qa._context_answer = native
    text = "😀 Qualifications:\nAWS certificate. It expired in 2020.\nAzure certificate.\n"
    result = qa.answer("List all certificates", [{"quote": text, "passage_id": "p", "candidate_id": "A"}])
    assert {span["text"] for span in result["spans"]} == {"AWS", "Azure"}
    assert all(text[span["start"]:span["end"]] == span["text"] for span in result["spans"])
    assert any("AWS" in context and "expired" in context for context in seen)
    assert not result["exhaustive"]


def test_list_context_limit_is_shared_across_passages():
    from candidate_screener.llm.extractive_qa import ExtractiveQA
    qa = ExtractiveQA.__new__(ExtractiveQA)
    qa.threshold = 0.
    qa.manifest = {"model_id": "fake", "revision": "fake"}
    called = []
    qa._context_answer = lambda question, text: (called.append(text) or None, 0.)
    passages = [{"quote": "\n".join(f"Credential {i}." for i in range(20)), "passage_id": str(j)} for j in range(3)]
    result = qa.answer("List all credentials", passages)
    assert len(called) == 32
    assert result["incomplete"] and not result["exhaustive"]


def test_qa_context_mask_cannot_select_question_or_special_tokens():
    scores = [100., 90., 2., 5., 100.]
    assert best_context_span(scores, scores, [(0,0), (0,8), (0,3), (4,9), (0,0)],
                             [None, 0, 1, 1, None]) == (10., 4, 9)


def test_qa_span_retains_unicode_original_offsets():
    text = "😀 AWS certified"
    best = best_context_span([0, 5, 0], [0, 5, 0], [(0,1),(2,5),(6,15)], [1,1,1])
    assert text[best[1]:best[2]] == "AWS"


def test_qa_unavailable_worker_abstains(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr("candidate_screener.llm.extractive_qa.subprocess.Popen", unavailable)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    assert QaClient(manifest).answer("question", []) == {"status": "unavailable", "spans": [], "exhaustive": False}


@pytest.fixture
def synthetic_qa_client(monkeypatch, tmp_path):
    """Exercise real pipes and process cleanup without importing a model."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    popen = subprocess.Popen
    clients, processes = [], []

    def create(script, *, timeout=.15):
        def launch(*args, **kwargs):
            process = popen([sys.executable, "-u", "-c", script], **kwargs)
            processes.append(process)
            return process
        monkeypatch.setattr("candidate_screener.llm.extractive_qa.subprocess.Popen", launch)
        client = QaClient(manifest, timeout=timeout)
        clients.append(client)
        return client, processes

    yield create
    for client in clients:
        client.close()


@pytest.mark.parametrize("response", ["{", '{"status":"not_found","spans":[]}'])
def test_qa_timeout_bounds_response_without_newline(synthetic_qa_client, response):
    client, processes = synthetic_qa_client(
        "import sys,time\n"
        "sys.stdin.readline()\n"
        f"sys.stdout.write({response!r}); sys.stdout.flush()\n"
        "time.sleep(.8)\n"
        "sys.stdout.write('\\n'); sys.stdout.flush()\n")
    started = time.monotonic()
    result = client.answer("question", [])
    elapsed = time.monotonic() - started
    assert elapsed < .6, f"Partial response exceeded {client.timeout}s deadline: {elapsed:.3f}s"
    assert result == {"status": "unavailable", "spans": [], "exhaustive": False}
    assert client._process is None
    assert processes[0].poll() is not None


def test_qa_timeout_does_not_restart_for_each_response_fragment(synthetic_qa_client):
    client, processes = synthetic_qa_client(
        "import sys,time\n"
        "sys.stdin.readline()\n"
        "for part in '{\\\"status\\\":\\\"not_found\\\"}':\n"
        "    sys.stdout.write(part); sys.stdout.flush(); time.sleep(.04)\n"
        "sys.stdout.write('\\n'); sys.stdout.flush()\n")
    started = time.monotonic()
    result = client.answer("question", [])
    elapsed = time.monotonic() - started
    assert elapsed < .6, f"Trickled response exceeded {client.timeout}s deadline: {elapsed:.3f}s"
    assert result["status"] == "unavailable"
    assert client._process is None
    assert processes[0].poll() is not None


def test_qa_timeout_bounds_request_when_worker_does_not_read(synthetic_qa_client):
    client, processes = synthetic_qa_client("import time; time.sleep(.8)")
    started = time.monotonic()
    result = client.answer("question", [{"quote": "x" * 100_000, "passage_id": "p"}])
    elapsed = time.monotonic() - started
    assert elapsed < .6, f"Blocked request exceeded {client.timeout}s deadline: {elapsed:.3f}s"
    assert result["status"] == "unavailable"
    assert client._process is None
    assert processes[0].poll() is not None


def test_qa_persistent_worker_preserves_fragmented_unicode_responses(synthetic_qa_client):
    client, processes = synthetic_qa_client(
        "import json,sys,time\n"
        "for count,line in enumerate(sys.stdin, 1):\n"
        "    request = json.loads(line)\n"
        "    payload = (json.dumps({'status':'supported','spans':[], 'count':count,\n"
        "        'question':request['question'], 'text':'😀 café'}, ensure_ascii=False) + '\\n').encode()\n"
        "    split = payload.index('😀'.encode()) + 1\n"
        "    sys.stdout.buffer.write(payload[:split]); sys.stdout.buffer.flush(); time.sleep(.01)\n"
        "    sys.stdout.buffer.write(payload[split:]); sys.stdout.buffer.flush()\n",
        timeout=2)
    for count, question in enumerate(("First 😀 question?", "Second café question?"), 1):
        assert client.answer(question, []) == {
            "status": "supported", "spans": [], "count": count, "question": question, "text": "😀 café"}
        assert len(processes) == 1 and client._process is processes[0]
        assert processes[0].poll() is None


def test_qa_timeout_discards_partial_response_before_restarting_worker(synthetic_qa_client):
    client, processes = synthetic_qa_client(
        "import json,sys,time\n"
        "for count,line in enumerate(sys.stdin, 1):\n"
        "    request = json.loads(line)\n"
        "    if request['question'] == 'stall':\n"
        "        sys.stdout.write('{'); sys.stdout.flush(); time.sleep(.8)\n"
        "        sys.stdout.write('}\\n'); sys.stdout.flush()\n"
        "    else:\n"
        "        print(json.dumps({'status':'not_found','spans':[], 'count':count}), flush=True)\n")
    assert client.answer("stall", [])["status"] == "unavailable"
    assert client._process is None and processes[0].poll() is not None
    for count in (1, 2):
        assert client.answer("recover", []) == {"status": "not_found", "spans": [], "count": count}
        assert len(processes) == 2 and client._process is processes[1]


def test_qa_no_answer_wins_over_weak_context_span():
    assert not answerable((2., 0, 3), 5., 0.)
    assert answerable((7., 0, 3), 5., 0.)
    assert not answerable(None, 5., 0.)
    assert not answerable((float("nan"), 0, 3), 5., 0.)


def test_qa_manifest_detects_changed_weights(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    names = ["config.json", "tokenizer_config.json", "model.safetensors"]
    for name in names:
        (checkpoint / name).write_text("synthetic")
    manifest = {"model_id": MODEL_ID, "revision": "a"*40,
                "artifact_sha256": {name: file_digest(checkpoint/name) for name in names}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify_manifest(path) == manifest
    (checkpoint / "model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="artifact mismatch"):
        verify_manifest(path)
