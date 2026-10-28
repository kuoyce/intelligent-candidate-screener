"""Pinned downloads use tiny synthetic bytes and never access the network."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from candidate_screener.llm import gguf_assets as assets


class Response(io.BytesIO):
    def __init__(self, content, status=200, content_range=None):
        super().__init__(content)
        self.status = status
        self.headers = {} if content_range is None else {"Content-Range": content_range}


def host(monkeypatch, system, machine):
    """The machine `provision` believes it runs on."""
    monkeypatch.setattr(assets.platform, "system", lambda: system)
    monkeypatch.setattr(assets.platform, "machine", lambda: machine)


def starts(monkeypatch, output="version: 0.7.0-dev (build 11896, commit 9125a2ebe)", *,
           returncode=0, error=None):
    """A stand-in for `llama-server --version`; returns the list of commands it was given."""
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if error is not None:
            raise error
        return SimpleNamespace(returncode=returncode, stdout="", stderr=output)

    fake = SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired,
                           SubprocessError=subprocess.SubprocessError)
    monkeypatch.setattr(assets, "subprocess", fake, raising=False)
    return calls


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


def serve_in_turn(monkeypatch, responses):
    """Each request gets the next response; records the requests."""
    requests, queue = [], list(responses)

    def open_response(request, timeout):
        requests.append(request)
        return queue.pop(0) if queue else Response(b"")

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_response)
    return requests


def test_a_stream_that_ends_early_resumes_in_the_same_call(monkeypatch, tmp_path):
    # Under arm64 emulation the HTTPS stream ended silently after ~175 MB (plan 2026-10-28, E5):
    # fetch must resume from the partial bytes, then verify the whole artifact.
    body = b"complete frozen artifact bytes"
    requests = serve_in_turn(monkeypatch, [
        Response(body[:7]),
        Response(body[7:12], 206, f"bytes 7-{len(body)-1}/{len(body)}"),
        Response(body[12:], 206, f"bytes 12-{len(body)-1}/{len(body)}")])
    target = tmp_path / "model.gguf"
    assert assets.fetch("https://example.invalid/model", target, sha(body), len(body)) == target
    assert target.read_bytes() == body
    assert [r.get_header("Range") for r in requests] == [None, "bytes=7-", "bytes=12-"]


@pytest.mark.parametrize("error", [TimeoutError, ConnectionResetError])
def test_a_stalled_or_reset_stream_resumes_in_the_same_call(monkeypatch, tmp_path, error):
    body = b"complete frozen artifact bytes"

    class Fails(Response):
        def read(self, size=-1):
            chunk = super().read(size)
            if not chunk:
                raise error("stream broke")
            return chunk

    requests = serve_in_turn(monkeypatch, [
        Fails(body[:9]), Response(body[9:], 206, f"bytes 9-{len(body)-1}/{len(body)}")])
    target = tmp_path / "model.gguf"
    assert assets.fetch("https://example.invalid/model", target, sha(body), len(body)) == target
    assert [r.get_header("Range") for r in requests] == [None, "bytes=9-"]


def test_resuming_after_an_early_end_is_bounded_and_never_publishes(monkeypatch, tmp_path):
    requests = serve_in_turn(monkeypatch, [Response(b"abc")])   # then empty bodies forever
    target = tmp_path / "model.gguf"
    with pytest.raises(ValueError, match="size or SHA-256"):
        assets.fetch("https://example.invalid/model", target, sha(b"abcdefgh"), 8)
    assert len(requests) == assets.FETCH_ATTEMPTS
    assert not target.exists()


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
    host(monkeypatch, "Darwin", "x86_64")  # the macos-x64 entry and its `runtime/` folder
    starts(monkeypatch)
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


# Plan 2026-10-28-gguf-cross-platform (G1): one pinned runtime per platform.

#: Measured 28 Oct 2026 from our own download (plan 01 §1).
PINS = {
    "macos-x64": ("llama-b11896-bin-macos-x64.tar.gz", 12_038_557,
                  "2ff5c5f3b5ba8a12a4a868c290510799416ad78c61528e9d7754404aa6fd6fb5"),
    "macos-arm64": ("llama-b11896-bin-macos-arm64.tar.gz", 16_297_217,
                    "f7fdf25367d2a403a91ec91729322a3c6a3a80e95da15844aa2c7be9dab3e11b"),
    "linux-x64": ("llama-b11896-bin-ubuntu-x64.tar.gz", 22_497_065,
                  "b02ed815efe497c0e818aadaae1b2538309f25424034555dcc1b7211c783e25c"),
    "linux-arm64": ("llama-b11896-bin-ubuntu-arm64.tar.gz", 18_221_582,
                    "68942cd6067dd632831f12e0bcdb4f8edbd91882e633a4e8593e0690f0624261"),
}
HOSTS = {"macos-x64": ("Darwin", "x86_64"), "macos-arm64": ("Darwin", "arm64"),
         "linux-x64": ("Linux", "x86_64"), "linux-arm64": ("Linux", "aarch64")}


@pytest.mark.parametrize("system, machine, key", [
    ("Darwin", "x86_64", "macos-x64"), ("Darwin", "arm64", "macos-arm64"),
    ("Linux", "x86_64", "linux-x64"), ("Linux", "amd64", "linux-x64"),
    ("Linux", "aarch64", "linux-arm64"), ("Linux", "arm64", "linux-arm64"),
])
def test_platform_key_maps_each_pinned_host(system, machine, key):
    assert assets.platform_key(system, machine) == key


def test_platform_key_defaults_to_the_running_host(monkeypatch):
    host(monkeypatch, "Linux", "aarch64")
    assert assets.platform_key() == "linux-arm64"


@pytest.mark.parametrize("system, machine", [
    ("Windows", "AMD64"), ("Linux", "s390x"), ("Darwin", "aarch64"), ("Linux", "i686"), ("", ""),
])
def test_unknown_host_is_refused_with_the_pinned_keys(system, machine):
    with pytest.raises(ValueError, match="--platform") as error:
        assets.platform_key(system, machine)
    assert all(key in str(error.value) for key in PINS)
    assert "b11896" in str(error.value)


def test_known_but_unpinned_key_is_refused_the_same_way(monkeypatch, tmp_path):
    runtimes = dict(assets.RUNTIMES)
    del runtimes["macos-arm64"]
    monkeypatch.setattr(assets, "RUNTIMES", runtimes)
    with pytest.raises(ValueError, match="--platform"):
        assets.platform_key("Darwin", "arm64")
    with pytest.raises(ValueError, match="--platform"):
        assets.provision_runtime(tmp_path, "macos-arm64")
    assert not any(tmp_path.iterdir())


def test_the_runtime_table_holds_exactly_the_measured_pins():
    assert assets.RELEASE == "b11896"
    assert {key: tuple(value) for key, value in assets.RUNTIMES.items()} == PINS


def test_macos_x64_entry_is_byte_for_byte_unchanged():
    """The teammate's download and extracted runtime must be reused, not fetched again."""
    assert assets.RUNTIME_URL == ("https://github.com/ggml-org/llama.cpp/releases/download/"
                                  "b11896/llama-b11896-bin-macos-x64.tar.gz")
    assert assets.RUNTIME_SHA256 == "2ff5c5f3b5ba8a12a4a868c290510799416ad78c61528e9d7754404aa6fd6fb5"
    assert assets.runtime_url("macos-x64") == assets.RUNTIME_URL
    assert assets.RUNTIMES["macos-x64"].size == 12_038_557
    assert assets.RUNTIMES["macos-x64"].sha256 == assets.RUNTIME_SHA256
    assert assets.local_names("macos-x64") == ("llama-b11896-macos-x64.tar.gz", "runtime")


@pytest.mark.parametrize("key", [key for key in PINS if key != "macos-x64"])
def test_other_platforms_have_their_own_local_names(key):
    assert assets.local_names(key) == (f"llama-b11896-{key}.tar.gz", f"runtime-{key}")


def synthetic_archive(path):
    with tarfile.open(path, "w:gz") as bundle:
        for name, content in [("build/bin/llama-server", b"pinned executable"),
                              ("build/bin/libllama.so", b"pinned library")]:
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o755
            bundle.addfile(info, io.BytesIO(content))
    return path


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    """`fetch` that records its pins and hands back a synthetic archive or model."""
    calls = []
    source = synthetic_archive(tmp_path / "source.tar.gz")

    def fake_fetch(url, destination, expected_sha256, expected_size):
        calls.append((url, Path(destination).name, expected_sha256, expected_size))
        if url.endswith(".gguf"):
            Path(destination).write_bytes(b"synthetic model")
        else:
            Path(destination).write_bytes(source.read_bytes())
        return Path(destination)

    monkeypatch.setattr(assets, "fetch", fake_fetch)
    return calls


@pytest.mark.parametrize("key", list(PINS))
def test_each_platform_fetches_its_own_url_size_and_digest(recorded, monkeypatch, tmp_path, key):
    host(monkeypatch, *HOSTS[key])
    starts(monkeypatch)
    destination = tmp_path / "out"
    manifest = assets.provision(destination)
    asset, size, digest_ = PINS[key]
    archive, folder = assets.local_names(key)
    assert recorded[0] == (f"https://github.com/ggml-org/llama.cpp/releases/download/b11896/{asset}",
                           archive, digest_, size)
    assert recorded[1][0].endswith(f"/{assets.REVISION}/{assets.FILENAME}")
    assert manifest["runtime_platform"] == key
    assert manifest["runtime_archive"] == asset
    assert manifest["runtime_archive_sha256"] == digest_
    assert manifest["llama_release"] == "b11896"
    assert manifest["llama_server"] == str((destination / folder / "build/bin/llama-server").resolve())
    assert Path(manifest["llama_server"]).is_absolute()
    assert manifest["start_check"] == "passed"
    assert json.loads((destination / "manifest.json").read_text()) == manifest


@pytest.mark.parametrize("key", list(PINS))
def test_runtime_download_requests_the_pinned_url(monkeypatch, tmp_path, key):
    """The real `fetch` behind a fake `urlopen`: the request goes to this key's asset."""
    seen = []

    class Stop(Exception):
        pass

    def open_response(request, timeout):
        seen.append(request.full_url)
        raise Stop

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_response)
    with pytest.raises(Stop):
        assets.provision_runtime(tmp_path, key)
    assert seen == [f"https://github.com/ggml-org/llama.cpp/releases/download/b11896/{PINS[key][0]}"]
    assert not (tmp_path / "manifest.json").exists()


def test_manifest_keeps_every_existing_field_and_the_model_pin(recorded, monkeypatch, tmp_path):
    host(monkeypatch, "Linux", "x86_64")
    starts(monkeypatch)
    manifest = assets.provision(tmp_path / "out")
    assert {"model": "Qwen/Qwen3.5-4B", "revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
            "source_repository": "unsloth/Qwen3.5-4B-GGUF", "gguf": "Qwen3.5-4B-Q4_K_M.gguf",
            "sha256": "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
            "size_bytes": 2_740_937_888, "quantization": "Q4_K_M", "max_context_tokens": 4096,
            "max_output_tokens": 256, "enable_thinking": False}.items() <= manifest.items()
    assert recorded[1] == (f"https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/"
                           f"{assets.REVISION}/{assets.FILENAME}", assets.FILENAME,
                           assets.SHA256, assets.SIZE)


def test_start_check_runs_version_with_a_timeout(recorded, monkeypatch, tmp_path):
    host(monkeypatch, "Linux", "x86_64")
    calls = starts(monkeypatch)
    manifest = assets.provision(tmp_path / "out")
    [(command, kwargs)] = calls
    assert command == [manifest["llama_server"], "--version"]
    assert kwargs["timeout"] == 120


@pytest.mark.parametrize("key, glibc", [("linux-x64", "2.34"), ("linux-arm64", "2.38")])
@pytest.mark.parametrize("failure", ["exit", "wrong-build", "timeout", "missing-library"])
def test_failed_start_check_publishes_no_manifest_and_names_the_requirements(
        recorded, monkeypatch, tmp_path, key, glibc, failure):
    host(monkeypatch, *HOSTS[key])
    if failure == "exit":
        starts(monkeypatch, "error while loading shared libraries: libgomp.so.1", returncode=127)
    elif failure == "wrong-build":
        starts(monkeypatch, "version: 0.7.0-dev (build 11895, commit 0000000)")
    elif failure == "timeout":
        starts(monkeypatch, error=subprocess.TimeoutExpired(["llama-server"], 120))
    else:
        starts(monkeypatch, error=OSError(8, "Exec format error"))
    destination = tmp_path / "out"
    with pytest.raises(RuntimeError) as error:
        assets.provision(destination)
    assert f"glibc >= {glibc}" in str(error.value) and "libgomp1" in str(error.value)
    assert not (destination / "manifest.json").exists()
    assert not (destination / "manifest.json.partial").exists()
    assert len(recorded) == 1, "the 2.74 GB model was fetched before the runtime started"


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["gguf_assets.py", *argv])
    assets.main()


def test_platform_override_for_another_host_skips_and_records_the_start_check(
        recorded, monkeypatch, tmp_path, capsys):
    host(monkeypatch, "Linux", "x86_64")
    calls = starts(monkeypatch)
    destination = tmp_path / "out"
    run_main(monkeypatch, "--platform", "macos-x64", "--destination", str(destination))
    assert calls == []
    assert "skipped" in capsys.readouterr().out
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["runtime_platform"] == "macos-x64"
    assert manifest["runtime_archive"] == "llama-b11896-bin-macos-x64.tar.gz"
    assert manifest["runtime_archive_sha256"] == PINS["macos-x64"][2]
    assert manifest["start_check"] == "skipped (platform override)"
    assert recorded[0][0] == assets.RUNTIME_URL


def test_platform_override_works_on_an_unpinned_host(recorded, monkeypatch, tmp_path):
    """A Windows or s390x machine can still prepare a folder for a pinned platform."""
    host(monkeypatch, "Windows", "AMD64")
    calls = starts(monkeypatch)
    run_main(monkeypatch, "--platform", "linux-arm64", "--destination", str(tmp_path / "out"))
    assert calls == []
    assert json.loads((tmp_path / "out/manifest.json").read_text())["start_check"] == \
        "skipped (platform override)"


def test_platform_override_matching_the_host_still_runs_the_start_check(recorded, monkeypatch, tmp_path):
    host(monkeypatch, "Linux", "x86_64")
    calls = starts(monkeypatch)
    run_main(monkeypatch, "--platform", "linux-x64", "--destination", str(tmp_path / "out"))
    assert len(calls) == 1
    assert json.loads((tmp_path / "out/manifest.json").read_text())["start_check"] == "passed"


@pytest.mark.parametrize("value", ["windows-x64", "ubuntu-x64", "linux-amd64", ""])
def test_platform_override_accepts_only_pinned_keys(recorded, monkeypatch, tmp_path, value):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--platform", value, "--destination", str(tmp_path / "out"))
    assert recorded == []


def test_platform_runtimes_share_one_folder_without_overwriting(recorded, monkeypatch, tmp_path):
    host(monkeypatch, "Linux", "x86_64")
    starts(monkeypatch)
    destination = tmp_path / "out"
    run_main(monkeypatch, "--platform", "linux-x64", "--destination", str(destination))
    run_main(monkeypatch, "--platform", "macos-x64", "--destination", str(destination))
    assert {path.name for path in destination.iterdir()} >= {
        "runtime", "runtime-linux-x64", "llama-b11896-macos-x64.tar.gz", "llama-b11896-linux-x64.tar.gz"}


DESTINATION_PROBE = r"""
import importlib.util, json, sys
from pathlib import Path
import candidate_screener.config as config
spec = importlib.util.spec_from_file_location("probe_gguf", Path(config.__file__).parent / "llm/gguf_assets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({"destination": str(module.DESTINATION), "project": str(config.PROJECT_ROOT)}))
"""


def destination_for(cwd, models_root):
    """`DESTINATION` read in a fresh interpreter, as `config` reads the variable at import."""
    import candidate_screener
    package = Path(candidate_screener.__file__).resolve().parent
    env = {key: value for key, value in os.environ.items() if key != "SCREENER_MODELS_ROOT"}
    if models_root is not None:
        env["SCREENER_MODELS_ROOT"] = str(models_root)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(package.parent), env.get("PYTHONPATH")]))
    result = subprocess.run([sys.executable, "-c", DESTINATION_PROBE], cwd=cwd, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_destination_native_default_is_unchanged(tmp_path, value):
    seen = destination_for(tmp_path, value)
    assert seen["destination"] == str(Path(seen["project"]) / "data/processed/chatbot/qwen35")


def test_destination_follows_the_models_root(tmp_path):
    root = tmp_path / "models"
    assert destination_for(tmp_path, root)["destination"] == str(root / "chatbot/qwen35")
