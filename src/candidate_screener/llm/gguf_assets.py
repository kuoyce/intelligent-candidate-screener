"""Provision pinned, frozen Qwen3.5 GGUF and isolated Intel-macOS runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
FILENAME = "Qwen3.5-4B-Q4_K_M.gguf"
SHA256 = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
SIZE = 2_740_937_888
RUNTIME_SHA256 = "2ff5c5f3b5ba8a12a4a868c290510799416ad78c61528e9d7754404aa6fd6fb5"
RUNTIME_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11896/llama-b11896-bin-macos-x64.tar.gz"
DESTINATION = Path(__file__).resolve().parents[3] / "data/processed/chatbot/qwen35"


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
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError("Partial artifact exceeds its expected size")
    if offset < expected_size:
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Download resume offset differs from request")
            count = offset if resumed else 0
            with partial.open("ab" if resumed else "wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(chunk)
                    if count > expected_size:
                        raise ValueError("Artifact exceeds pinned size")
                    stream.write(chunk)
    if partial.stat().st_size != expected_size or digest(partial) != expected_sha256:
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


def provision(destination=DESTINATION):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = fetch(RUNTIME_URL, destination / "llama-b11896-macos-x64.tar.gz", RUNTIME_SHA256, 12_038_557)
    runtime = destination / "runtime"
    if runtime.is_symlink():
        raise ValueError("Invalid extracted runtime directory")
    if not runtime.exists():
        with tarfile.open(archive) as bundle:
            bundle.extractall(runtime, filter="data")
    verify_runtime(archive, runtime)
    binaries = list(runtime.rglob("llama-server"))
    if len(binaries) != 1:
        raise ValueError("Expected exactly one runtime server")
    print("Pinned Intel runtime ready; downloading 2.74GB frozen model", flush=True)
    model = fetch(f"https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/{REVISION}/{FILENAME}",
                  destination / FILENAME, SHA256, SIZE)
    manifest = {"model": MODEL, "revision": REVISION, "source_repository": "unsloth/Qwen3.5-4B-GGUF",
                "gguf": model.name, "sha256": SHA256, "size_bytes": SIZE,
                "llama_server": str(binaries[0]), "llama_release": "b11896",
                "runtime_archive_sha256": RUNTIME_SHA256, "quantization": "Q4_K_M",
                "max_context_tokens": 4096, "max_output_tokens": 256, "enable_thinking": False}
    temporary = destination / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "manifest.json")
    print(json.dumps({"manifest": str(destination / "manifest.json"), "sha256": SHA256}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    provision(args.destination)


if __name__ == "__main__":
    main()
