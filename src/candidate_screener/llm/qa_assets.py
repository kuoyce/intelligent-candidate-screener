"""Explicit acquisition and offline verification for the extractive QA baseline."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from candidate_screener.config import MODELS

MODEL_ID = "deepset/roberta-base-squad2"
ROOT = MODELS / "chatbot/qa/extractive"


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_manifest(path: Path = ROOT / "manifest.json") -> dict:
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("model_id") != MODEL_ID or re.fullmatch(r"[a-f0-9]{40}", manifest.get("revision", "")) is None:
        raise ValueError("Invalid pinned extractive checkpoint identity")
    if "training" in manifest:
        payload = {key: value for key, value in manifest.items() if key != "adaptation_id"}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
        if manifest.get("adaptation_id") != expected or manifest["training"].get("training_split") != "train":
            raise ValueError("Adapted QA provenance identity mismatch")
    files = manifest.get("artifact_sha256", {})
    if not {"config.json", "tokenizer_config.json"} <= set(files) or not any(x in files for x in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError("Incomplete extractive checkpoint manifest")
    checkpoint = path.parent / "checkpoint"
    actual = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file() and ".cache" not in p.relative_to(checkpoint).parts}
    if actual != set(files):
        raise ValueError("Extractive checkpoint artifact inventory mismatch")
    for name, expected in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid checkpoint artifact path")
        artifact = checkpoint / name
        if artifact.is_symlink() or not artifact.resolve().is_relative_to(checkpoint.resolve()):
            raise ValueError("Linked extractive checkpoint artifacts are unsupported")
        if file_digest(artifact) != expected:
            raise ValueError("Extractive checkpoint artifact mismatch")
    return manifest


def download_assets(destination: Path = ROOT, revision: str | None = None):
    """Acquire the checkpoint at `revision`, an immutable commit; the latest one when omitted."""
    destination = Path(destination)
    if revision is not None and re.fullmatch(r"[a-f0-9]{40}", revision) is None:
        raise ValueError(f"The QA checkpoint revision must be a 40-character hexadecimal commit, not {revision!r}")
    if (destination / "manifest.json").exists():
        manifest = verify_manifest(destination / "manifest.json")
        if revision is not None and manifest["revision"] != revision:
            raise ValueError(f"Existing extractive checkpoint is revision {manifest['revision']}, not {revision}")
        return manifest
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL_ID, revision=revision)
    revision = revision or info.sha
    names = {item.rfilename for item in info.siblings}
    weight = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    selected = [name for name in [weight, "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "README.md"] if name in names]
    snapshot_download(MODEL_ID, revision=revision, local_dir=destination / "checkpoint", allow_patterns=selected)
    manifest = {"model_id": MODEL_ID, "revision": revision,
                "license": (info.card_data or {}).get("license", "unknown"),
                "artifact_sha256": {name: file_digest(destination / "checkpoint" / name) for name in selected},
                "runtime": {"project": "runtime/transformer", "torch": "2.2.2", "transformers": "4.48.3"}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify_manifest(destination / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--revision", help="Immutable 40-character commit to download; the latest is resolved when omitted")
    args = parser.parse_args()
    result = verify_manifest(args.destination / "manifest.json") if args.verify else download_assets(args.destination, args.revision)
    print(json.dumps({"model_id": result["model_id"], "revision": result["revision"]}))
