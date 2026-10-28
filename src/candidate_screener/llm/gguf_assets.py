"""Provision the pinned, frozen Qwen3.5-4B GGUF and an isolated llama.cpp runtime.

One llama.cpp release (b11896) is pinned per platform: macOS (x64, arm64) and Linux
(x64, arm64), each by asset name, size and SHA-256 measured from our own download (plan
2026-10-28-gguf-cross-platform, 01 §1). The host picks its own key; `--platform` chooses
one explicitly, for example to prepare a folder for another machine. When the key matches
the host, `llama-server --version` must start before the 2.74 GB model is fetched or a
manifest is written. Standard library and `candidate_screener.config` only.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
import tarfile
from typing import NamedTuple
import urllib.request

from candidate_screener.config import MODELS

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
FILENAME = "Qwen3.5-4B-Q4_K_M.gguf"
SHA256 = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
SIZE = 2_740_937_888

RELEASE = "b11896"  # commit 9125a2ebe8e5a38f0f4232170c134a29ed730b66
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/"


class Runtime(NamedTuple):
    asset: str
    size: int
    sha256: str


#: Measured 28 Oct 2026 from our own download (plan 2026-10-28-gguf-cross-platform, 01 §1).
RUNTIMES = {
    "macos-x64": Runtime("llama-b11896-bin-macos-x64.tar.gz", 12_038_557,
                         "2ff5c5f3b5ba8a12a4a868c290510799416ad78c61528e9d7754404aa6fd6fb5"),
    "macos-arm64": Runtime("llama-b11896-bin-macos-arm64.tar.gz", 16_297_217,
                           "f7fdf25367d2a403a91ec91729322a3c6a3a80e95da15844aa2c7be9dab3e11b"),
    "linux-x64": Runtime("llama-b11896-bin-ubuntu-x64.tar.gz", 22_497_065,
                         "b02ed815efe497c0e818aadaae1b2538309f25424034555dcc1b7211c783e25c"),
    "linux-arm64": Runtime("llama-b11896-bin-ubuntu-arm64.tar.gz", 18_221_582,
                           "68942cd6067dd632831f12e0bcdb4f8edbd91882e633a4e8593e0690f0624261"),
}
#: What a Linux runtime needs from the host (01 §2); shown when its start check fails.
REQUIREMENTS = {"linux-x64": "glibc >= 2.34 and libgomp1 (apt install libgomp1)",
                "linux-arm64": "glibc >= 2.38 and libgomp1 (apt install libgomp1)",
                "macos-x64": "a macOS version the llama.cpp b11896 release supports",
                "macos-arm64": "a macOS version the llama.cpp b11896 release supports"}
#: (system, machine) as `platform` reports them -> runtime key.
HOSTS = {("Darwin", "x86_64"): "macos-x64", ("Darwin", "arm64"): "macos-arm64",
         ("Linux", "x86_64"): "linux-x64", ("Linux", "amd64"): "linux-x64",
         ("Linux", "aarch64"): "linux-arm64", ("Linux", "arm64"): "linux-arm64"}
START_CHECK_TIMEOUT = 120
FETCH_ATTEMPTS = 50  # requests per artifact, counting resumes after an early end
#: The macos-x64 names predate the table; kept so existing downloads are reused.
RUNTIME_SHA256 = RUNTIMES["macos-x64"].sha256
RUNTIME_URL = RELEASE_URL + RUNTIMES["macos-x64"].asset
DESTINATION = MODELS / "chatbot/qwen35"


def unpinned(what):
    return ValueError(f"No pinned llama.cpp {RELEASE} runtime for {what}; pinned: "
                      f"{', '.join(RUNTIMES)}; pass --platform to choose one explicitly")


def platform_key(system=None, machine=None):
    """The pinned runtime key for this host (or the given system/machine)."""
    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    key = HOSTS.get((system, machine))
    if key not in RUNTIMES:
        raise unpinned(f"{system or '?'}/{machine or '?'}")
    return key


def pinned(key):
    if key not in RUNTIMES:
        raise unpinned(repr(key))
    return RUNTIMES[key]


def runtime_url(key):
    return RELEASE_URL + pinned(key).asset


def local_names(key):
    """(archive file, runtime folder) inside the destination; macos-x64 keeps its old names."""
    pinned(key)
    if key == "macos-x64":
        return f"llama-{RELEASE}-macos-x64.tar.gz", "runtime"
    return f"llama-{RELEASE}-{key}.tar.gz", f"runtime-{key}"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(url, destination, expected_sha256, expected_size):
    """Resume partial bytes, verify everything, then publish atomically."""
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size == expected_size and digest(destination) == expected_sha256:
            return destination
        raise ValueError("Existing artifact differs from the pinned source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    # A stream may end early, stall or break (under arm64 emulation it stopped after ~175 MB, and a
    # resumed one later timed out; plan 2026-10-28-gguf-cross-platform E5): resume from the
    # partial bytes, a bounded number of times; the whole artifact is verified below either way.
    for _ in range(FETCH_ATTEMPTS):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError("Partial artifact exceeds its expected size")
        if offset == expected_size:
            break
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Download resume offset differs from request")
            count = offset if resumed else 0
            with partial.open("ab" if resumed else "wb") as stream:
                try:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        count += len(chunk)
                        if count > expected_size:
                            raise ValueError("Artifact exceeds pinned size")
                        stream.write(chunk)
                except (TimeoutError, ConnectionError, http.client.IncompleteRead):
                    pass  # a stalled or broken stream: keep the bytes, resume on the next attempt
    if not partial.exists() or partial.stat().st_size != expected_size or digest(partial) != expected_sha256:
        raise ValueError("Artifact size or SHA-256 mismatch; manifest not written")
    partial.replace(destination)
    return destination


def verify_runtime(archive, runtime):
    """Compare extracted files/links to the verified archive without changing them."""
    runtime = Path(runtime)
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError("Invalid extracted runtime directory")
    root, expected = runtime.resolve(), set()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe runtime archive path")
            path = runtime.joinpath(*name.parts)
            if any(parent.is_symlink() for parent in path.parents if parent != runtime and root in parent.parents):
                raise ValueError("Extracted runtime parent differs from archive")
            if member.isdir():
                if path.is_symlink() or not path.is_dir():
                    raise ValueError("Extracted runtime directory differs from archive")
                continue
            if name.as_posix() in expected:
                raise ValueError("Duplicate runtime archive member")
            expected.add(name.as_posix())
            if member.issym():
                if (not path.is_symlink() or str(path.readlink()) != member.linkname
                        or not path.resolve().is_relative_to(root)):
                    raise ValueError("Extracted runtime symlink differs from archive")
            elif member.isfile() or member.islnk():
                if (path.is_symlink() or not path.is_file()
                        or member.isfile() and path.stat().st_size != member.size):
                    raise ValueError("Extracted runtime file differs from archive")
                with bundle.extractfile(member) as source:
                    expected_digest = hashlib.file_digest(source, "sha256").hexdigest()
                if digest(path) != expected_digest:
                    raise ValueError("Extracted runtime bytes differ from archive")
            else:
                raise ValueError("Unsupported runtime archive member")
    actual = {path.relative_to(runtime).as_posix() for path in runtime.rglob("*")
              if path.is_symlink() or not path.is_dir()}
    if actual != expected:
        raise ValueError("Extracted runtime files differ from archive")


def start_check(binary, key):
    """`llama-server --version` must start and name the pinned build."""
    hint = (f"the {key} runtime of llama.cpp {RELEASE} did not start; it needs "
            f"{REQUIREMENTS[key]}; manifest not written")
    try:
        result = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                                timeout=START_CHECK_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"{hint} ({error})") from error
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 or f"build {RELEASE.lstrip('b')}" not in output:
        detail = " ".join(output.split())[-300:]
        raise RuntimeError(f"{hint} (exit {result.returncode}: {detail})")
    return output


def provision_runtime(destination, key, *, check=True):
    """Fetch, extract and verify one pinned runtime; returns its `llama-server` path."""
    entry = pinned(key)
    archive_name, folder = local_names(key)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = fetch(runtime_url(key), destination / archive_name, entry.sha256, entry.size)
    runtime = destination / folder
    if runtime.is_symlink():
        raise ValueError("Invalid extracted runtime directory")
    if not runtime.exists():
        with tarfile.open(archive) as bundle:
            bundle.extractall(runtime, filter="data")
    verify_runtime(archive, runtime)
    binaries = list(runtime.rglob("llama-server"))
    if len(binaries) != 1:
        raise ValueError("Expected exactly one runtime server")
    if check:
        start_check(binaries[0], key)
    return binaries[0]


def provision(destination=DESTINATION, platform=None):
    """`platform=None` uses the host's key; another key skips the start check, recorded."""
    try:
        host = platform_key()
    except ValueError:
        if platform is None:
            raise
        host = None
    key = host if platform is None else platform
    pinned(key)
    check = key == host
    destination = Path(destination).resolve()
    binary = provision_runtime(destination, key, check=check)
    status = "passed" if check else "skipped (platform override)"
    if check:
        print(f"Pinned {key} runtime started (build {RELEASE}); downloading 2.74GB frozen model", flush=True)
    else:
        print(f"Pinned {key} runtime ready; start check skipped: host is {host or 'unpinned'}, "
              f"not {key}; downloading 2.74GB frozen model", flush=True)
    model = fetch(f"https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/{REVISION}/{FILENAME}",
                  destination / FILENAME, SHA256, SIZE)
    entry = RUNTIMES[key]
    manifest = {"model": MODEL, "revision": REVISION, "source_repository": "unsloth/Qwen3.5-4B-GGUF",
                "gguf": model.name, "sha256": SHA256, "size_bytes": SIZE,
                "llama_server": str(binary), "llama_release": RELEASE,
                "runtime_platform": key, "runtime_archive": entry.asset,
                "runtime_archive_sha256": entry.sha256, "start_check": status,
                "quantization": "Q4_K_M",
                "max_context_tokens": 4096, "max_output_tokens": 256, "enable_thinking": False}
    temporary = destination / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "manifest.json")
    print(json.dumps({"manifest": str(destination / "manifest.json"), "sha256": SHA256}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--platform", choices=sorted(RUNTIMES),
                        help="runtime to provision instead of the host's own; a different "
                             "platform skips the start check")
    args = parser.parse_args()
    provision(args.destination, args.platform)


if __name__ == "__main__":
    main()
