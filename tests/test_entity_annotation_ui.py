"""Synthetic local HTTP transport checks; no source corpus required."""
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from candidate_screener.entity_annotation.ui import create_server


class RevisionConflict(ValueError):
    pass


class Service:
    def session(self, annotator):
        return {"campaign": {"campaign_id": "synthetic", "labels": ["SKILL"], "guideline_version": "1"}, "documents": [], "progress": {}}

    def document(self, doc_id, annotator):
        return {"doc_id": doc_id, "text": "redacted 😀 Python", "revision": 0, "spans": []}

    def save(self, payload):
        if payload.get("revision") == 8:
            raise RevisionConflict("Stale revision")
        return {"revision": 1, "received_unit": payload["offset_unit"]}


@pytest.fixture
def server():
    instance = create_server(Service(), port=0)
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{instance.server_port}"
    instance.shutdown()
    instance.server_close()
    worker.join()


def test_serves_local_client_and_redacted_document(server):
    with urlopen(server + "/") as response:
        assert "Human entity annotation" in response.read().decode()
        assert response.headers["Cache-Control"] == "no-store"
    with urlopen(server + "/api/document?doc_id=d&annotator=a") as response:
        assert json.load(response)["text"] == "redacted 😀 Python"


def post(server, payload, **headers):
    return urlopen(Request(server + "/api/save", data=json.dumps(payload).encode(),
                           headers={"Content-Type": "application/json", **headers}))


def test_save_and_revision_conflict(server):
    with post(server, {"revision": 0, "offset_unit": "utf16"}) as response:
        assert json.load(response)["received_unit"] == "utf16"
    with pytest.raises(HTTPError) as exc:
        post(server, {"revision": 8})
    assert exc.value.code == 409


def test_cross_origin_or_rebinding_write_rejected(server):
    for headers in ({"Origin": "https://attacker.invalid"}, {"Host": "attacker.invalid"}):
        with pytest.raises(HTTPError) as exc:
            post(server, {}, **headers)
        assert exc.value.code == 403
    with pytest.raises(HTTPError) as exc:
        urlopen(Request(server + "/api/session", headers={"Host": "attacker.invalid"}))
    assert exc.value.code == 403


def test_rejects_external_bind():
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server(Service(), host="0.0.0.0", port=0)


def test_client_uses_safe_dom_and_explicit_utf16_offsets():
    from candidate_screener.entity_annotation.ui import HTML
    source = HTML.read_text()
    assert "innerHTML" not in source and "insertAdjacentHTML" not in source
    assert "offset_unit:'utf16'" in source
    assert "textContent=doc.text.slice" in source
    assert "reviewed_all:" in source
    assert "Revision conflict" in source
