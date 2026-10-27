"""Provision the workshop's Qwen checkpoint; no applicant data or model training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from candidate_screener.config import MODELS

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RAG_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR = MODELS / "chatbot/qwen"
DEFAULT_MANIFEST = MODEL_DIR / "manifest.json"


def local_settings(manifest_path=DEFAULT_MANIFEST):
    """App defaults for the separately provisioned loopback workshop service."""
    manifest = json.loads(Path(manifest_path).read_text())
    return {"SCREENER_QWEN_BASE_URL": "http://127.0.0.1:8000/v1",
            "SCREENER_QWEN_MODEL": manifest["model"], "SCREENER_QWEN_REVISION": manifest["revision"],
            "SCREENER_QWEN_SHA256": manifest["sha256"],
            "SCREENER_QWEN_TOKENIZER": "service", "SCREENER_QWEN_RUNTIME": "local Transformers service",
            "SCREENER_QWEN_QUANTIZATION": manifest["quantization"],
            "SCREENER_QWEN_CONTEXT_TOKENS": str(manifest["max_context_tokens"]),
            "SCREENER_QWEN_OUTPUT_TOKENS": str(manifest["max_output_tokens"]),
            "SCREENER_QWEN_TIMEOUT": "90", "SCREENER_QWEN_TEMPERATURE": "0",
            "SCREENER_QWEN_JSON_MODE": "false"}


def asset_digest(directory: Path, *, exclude_download_cache=True) -> str:
    """Local checkpoint integrity, excluding downloader bookkeeping only."""
    digest = hashlib.sha256()
    files = sorted(p for p in directory.rglob("*") if p.is_file()
                   and (not exclude_download_cache or ".cache" not in p.parts))
    if not files:
        raise ValueError("Local model checkpoint is empty.")
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Linked model artifacts are not supported.")
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def provision(destination=MODEL_DIR, *, revision=None, model_id=MODEL_ID):
    from huggingface_hub import HfApi, snapshot_download
    destination = Path(destination)
    if model_id not in {MODEL_ID, RAG_MODEL_ID}:
        raise ValueError("Unsupported pinned generator baseline")
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        checkpoint = destination / manifest["checkpoint"]
        if (manifest["model"] == model_id and (revision is None or revision == manifest["revision"])
                and asset_digest(checkpoint) == manifest["sha256"]):
            return manifest
        raise ValueError("Existing Qwen assets do not match their manifest.")
    # Resolve once to an immutable commit. Repeat provisioning retains that revision.
    revision = revision or HfApi().model_info(model_id).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An immutable Hugging Face commit revision is required.")
    checkpoint = destination / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, revision=revision, local_dir=checkpoint,
                      allow_patterns=["config.json", "generation_config.json", "model*.safetensors",
                                      "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json"])
    manifest = {"model": model_id, "revision": revision, "checkpoint": "checkpoint",
                "sha256": asset_digest(checkpoint), "device": "cpu", "dtype": "float32",
                "quantization": "none", "max_context_tokens": 4096, "max_output_tokens": 256,
                "threads": 4, "source": "user-supplied workshop notebook; official Qwen model card"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--model", choices=[MODEL_ID, RAG_MODEL_ID], default=MODEL_ID)
    parser.add_argument("--revision", help="Immutable 40-character model commit; resolved once when omitted")
    args = parser.parse_args()
    destination = args.destination or (MODEL_DIR if args.model == MODEL_ID else MODEL_DIR.parent / "qwen15")
    manifest = provision(destination, revision=args.revision, model_id=args.model)
    print(json.dumps({"model": manifest["model"], "revision": manifest["revision"],
                      "sha256": manifest["sha256"], "manifest": str(destination / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
