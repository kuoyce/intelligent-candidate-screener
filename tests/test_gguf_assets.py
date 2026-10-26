"""Pinned downloads use tiny synthetic bytes and never access the network."""
import hashlib
import io
import tarfile

import pytest

from candidate_screener.llm import gguf_assets as assets


class Response(io.BytesIO):
    def __init__(self, content, status=200, content_range=None):
        super().__init__(content)
        self.status = status
        self.headers = {} if content_range is None else {"Content-Range": content_range}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def serve(monkeypatch, content, *, status=200, content_range=None):
    requests = []

    def open_response(request, timeout):
        requests.append(request)
        assert timeout == 120
        return Response(content, status, content_range)

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_response)
    return requests


def test_verified_download_publishes_only_complete_pinned_bytes(monkeypatch, tmp_path):
    body = b"synthetic frozen artifact"
    serve(monkeypatch, body)
    target = tmp_path / "model.gguf"
    assert assets.fetch("https://example.invalid/model", target, sha(body), len(body)) == target
    assert target.read_bytes() == body
    assert not target.with_suffix(".gguf.partial").exists()


@pytest.mark.parametrize("body, expected, expected_size", [
    (b"wrong", b"right", 5), (b"short", b"shorter", 7), (b"oversized", b"tiny", 4),
])
def test_wrong_digest_or_size_never_publishes(monkeypatch, tmp_path, body, expected, expected_size):
    serve(monkeypatch, body)
    target = tmp_path / "model.gguf"
    with pytest.raises(ValueError):
        assets.fetch("https://example.invalid/model", target, sha(expected), expected_size)
    assert not target.exists()
    assert not (tmp_path / "manifest.json").exists()


def test_resume_requires_exact_offset_and_verifies_full_combined_bytes(monkeypatch, tmp_path):
    body = b"full frozen bytes"
    target = tmp_path / "model.gguf"
    target.with_suffix(".gguf.partial").write_bytes(body[:5])
    requests = serve(monkeypatch, body[5:], status=206, content_range=f"bytes 5-{len(body)-1}/{len(body)}")
    assets.fetch("https://example.invalid/model", target, sha(body), len(body))
    assert requests[0].get_header("Range") == "bytes=5-"
    assert target.read_bytes() == body


@pytest.mark.parametrize("content_range", [None, "bytes 0-15/16", "bytes 50-60/61"])
def test_mismatched_resume_offset_does_not_append_or_publish(monkeypatch, tmp_path, content_range):
    target = tmp_path / "model.gguf"
    partial = target.with_suffix(".gguf.partial")
    partial.write_bytes(b"first")
    serve(monkeypatch, b"remaining", status=206, content_range=content_range)
    with pytest.raises(ValueError, match="resume offset"):
        assets.fetch("https://example.invalid/model", target, sha(b"firstremaining"), 14)
    assert partial.read_bytes() == b"first"
    assert not target.exists()


def test_server_ignoring_range_resets_partial_instead_of_appending(monkeypatch, tmp_path):
    body = b"complete replacement"
    target = tmp_path / "model.gguf"
    target.with_suffix(".gguf.partial").write_bytes(b"stale")
    requests = serve(monkeypatch, body, status=200)
    assets.fetch("https://example.invalid/model", target, sha(body), len(body))
    assert requests[0].get_header("Range") == "bytes=5-"
    assert target.read_bytes() == body


@pytest.mark.parametrize("tampered", [b"tampered", b"bad"])
def test_existing_artifact_tamper_fails_without_network_or_overwrite(monkeypatch, tmp_path, tampered):
    target = tmp_path / "model.gguf"
    target.write_bytes(tampered)
    requests = serve(monkeypatch, b"expected")
    with pytest.raises(ValueError, match="Existing artifact"):
        assets.fetch("https://example.invalid/model", target, sha(b"expected"), 8)
    assert not requests
    assert target.read_bytes() == tampered


def test_verified_existing_artifact_reuses_without_network(monkeypatch, tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"expected")
    requests = serve(monkeypatch, b"unused")
    assert assets.fetch("https://example.invalid/model", target, sha(b"expected"), 8) == target
    assert not requests


def test_complete_partial_verifies_before_publish_without_network(monkeypatch, tmp_path):
    target = tmp_path / "model.gguf"
    target.with_suffix(".gguf.partial").write_bytes(b"expected")
    requests = serve(monkeypatch, b"unused")
    assets.fetch("https://example.invalid/model", target, sha(b"expected"), 8)
    assert not requests
    assert target.read_bytes() == b"expected"


def test_oversized_existing_partial_is_rejected_without_network(monkeypatch, tmp_path):
    target = tmp_path / "model.gguf"
    target.with_suffix(".gguf.partial").write_bytes(b"too many bytes")
    requests = serve(monkeypatch, b"unused")
    with pytest.raises(ValueError, match="Partial artifact exceeds"):
        assets.fetch("https://example.invalid/model", target, sha(b"short"), 5)
    assert not requests and not target.exists()


@pytest.fixture
def extracted_runtime(monkeypatch, tmp_path):
    archive = tmp_path / "trusted.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, content in [("bin/llama-server", b"pinned executable"), ("bin/libggml.dylib", b"pinned library")]:
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o755
            bundle.addfile(info, io.BytesIO(content))
        link = tarfile.TarInfo("bin/libggml-link.dylib")
        link.type, link.linkname = tarfile.SYMTYPE, "libggml.dylib"
        bundle.addfile(link)
    runtime = tmp_path / "runtime"
    with tarfile.open(archive) as bundle:
        bundle.extractall(runtime, filter="data")

    def fake_fetch(url, destination, expected_sha256, expected_size):
        if url == assets.RUNTIME_URL:
            return archive
        destination.write_bytes(b"synthetic model")
        return destination

    monkeypatch.setattr(assets, "fetch", fake_fetch)
    return tmp_path, runtime


@pytest.mark.parametrize("mutation", ["binary", "library", "symlink", "extra", "missing"])
def test_provision_refuses_modified_extracted_runtime_without_overwriting(extracted_runtime, mutation):
    destination, runtime = extracted_runtime
    if mutation == "binary":
        (runtime / "bin/llama-server").write_bytes(b"changed executable")
    elif mutation == "library":
        (runtime / "bin/libggml.dylib").write_bytes(b"changed library")
    elif mutation == "symlink":
        (runtime / "bin/libggml-link.dylib").unlink()
        (runtime / "bin/libggml-link.dylib").symlink_to("llama-server")
    elif mutation == "extra":
        (runtime / "bin/injected.dylib").write_bytes(b"extra library")
    else:
        (runtime / "bin/libggml.dylib").unlink()
    before = {str(path): path.readlink().as_posix() if path.is_symlink() else path.read_bytes()
              for path in runtime.rglob("*") if path.is_symlink() or path.is_file()}
    with pytest.raises(ValueError, match="runtime"):
        assets.provision(destination)
    after = {str(path): path.readlink().as_posix() if path.is_symlink() else path.read_bytes()
             for path in runtime.rglob("*") if path.is_symlink() or path.is_file()}
    assert after == before
    assert not (destination / "manifest.json").exists()


def test_unchanged_extracted_runtime_can_be_reused(extracted_runtime):
    destination, runtime = extracted_runtime
    before = (runtime / "bin/llama-server").stat().st_mtime_ns
    manifest = assets.provision(destination)
    assert manifest["llama_server"] == str(runtime / "bin/llama-server")
    assert (runtime / "bin/llama-server").stat().st_mtime_ns == before
