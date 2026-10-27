"""Acquire and verify the independently pinned medium checkpoint assets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from candidate_screener.config import MODELS

MODEL_ID = "gliner-community/gliner_medium-v2.5"
MODEL_REVISION = "88c3b98b57ad5e7d66fb209ed61c53f4b1fd05da"
MODEL_DIR = MODELS / "extraction/models/gliner-medium-v2.5"
ASSETS = ["gliner_config.json", "pytorch_model.bin", "README.md",
          "added_tokens.json", "special_tokens_map.json", "spm.model",
          "tokenizer.json", "tokenizer_config.json"]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_assets(destination: Path = MODEL_DIR) -> dict:
    manifest = json.loads((destination / "model-manifest.json").read_text())
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Medium checkpoint manifest identity mismatch")
    required = set(ASSETS) | {"encoder-assets/config.json"}
    if not required <= set(manifest.get("artifact_sha256", {})):
        raise ValueError("Medium checkpoint manifest lacks required artifacts")
    for name, expected in manifest["artifact_sha256"].items():
        path = destination / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid manifest artifact path")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Medium checkpoint artifact mismatch: {name}")
    return manifest


def download_assets(destination: Path = MODEL_DIR, encoder_revision: str | None = None) -> dict:
    """The encoder's assets come at `encoder_revision`, an immutable commit; when omitted, at the
    revision already recorded in `destination`, else at the latest one."""
    if encoder_revision is not None and re.fullmatch(r"[0-9a-f]{40}", encoder_revision) is None:
        raise ValueError("The encoder revision must be a 40-character hexadecimal commit, "
                         f"not {encoder_revision!r}")
    from huggingface_hub import HfApi, snapshot_download
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=destination,
                      allow_patterns=ASSETS)
    config = json.loads((destination / "gliner_config.json").read_text())
    previous_path = destination / "model-manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    backbone = config["model_name"]
    revision = (encoder_revision or previous.get("encoder", {}).get("revision")
                or HfApi().model_info(backbone).sha)
    snapshot_download(backbone, revision=revision, local_dir=destination / "encoder-assets",
                      allow_patterns=["config.json"])
    files = ASSETS + ["encoder-assets/config.json"]
    manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "license": "apache-2.0", "weight_representation": "pytorch_model.bin",
                "encoder": {"model_id": backbone, "revision": revision},
                "tokenizer": {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                              "source": "checkpoint-supplied"},
                "artifact_sha256": {name: digest(destination / name) for name in files},
                "downloaded_at": datetime.now(timezone.utc).isoformat()}
    previous_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify_assets(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=MODEL_DIR)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--encoder-revision", help="Immutable 40-character commit of the encoder's "
                        "assets to download; when omitted, the recorded one, else the latest")
    args = parser.parse_args()
    result = (verify_assets(args.destination) if args.verify
              else download_assets(args.destination, args.encoder_revision))
    print(json.dumps({key: result[key] for key in ["model_id", "revision", "encoder", "tokenizer"]}))
