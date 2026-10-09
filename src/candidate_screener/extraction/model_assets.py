"""Download pinned public GLiNER assets; no document text leaves the machine."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from candidate_screener.config import PROCESSED
from .inputs import sha256

MODEL_ID = "urchade/gliner_small-v2.1"
MODEL_REVISION = "4e091416cf7c3481db542c2a3d26156916f3a47f"
MODEL_DIR = PROCESSED / "extraction/models/gliner-small-v2.1"


def download_assets(destination: Path = MODEL_DIR) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=destination,
                      allow_patterns=["gliner_config.json", "pytorch_model.bin"])
    config = json.loads((destination / "gliner_config.json").read_text())
    backbone = config["model_name"]
    manifest_path = destination / "model-manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = previous.get("encoder", {}).get("revision") or HfApi().model_info(backbone).sha
    encoder = destination / "encoder-assets"
    snapshot_download(backbone, revision=revision, local_dir=encoder,
                      allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json",
                                      "spm.model", "sentencepiece*", "vocab*", "merges.txt"])
    files = [p for p in destination.rglob("*") if p.is_file()
             and ".cache" not in p.parts and p.name not in
             {"model-manifest.json", "runtime-versions.json"}]
    manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "encoder": {"model_id": backbone, "revision": revision},
                "artifact_sha256": {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(files)},
                "downloaded_at": datetime.now(timezone.utc).isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    result = download_assets(args.destination)
    print(json.dumps({"model_id": result["model_id"], "revision": result["revision"],
                      "encoder": result["encoder"], "files": len(result["artifact_sha256"])}))
