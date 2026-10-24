"""Asset provenance failures are caught before optional neural dependencies load."""
import json

import pytest

from candidate_screener.extraction.medium_assets import ASSETS, MODEL_ID, MODEL_REVISION, digest, verify_assets


def checkpoint(tmp_path):
    names = ASSETS + ["encoder-assets/config.json"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic")
    manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "artifact_sha256": {name: digest(tmp_path / name) for name in names}}
    (tmp_path / "model-manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_medium_manifest_verifies_every_asset(tmp_path):
    manifest = checkpoint(tmp_path)
    assert verify_assets(tmp_path) == manifest
    (tmp_path / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact mismatch: tokenizer.json"):
        verify_assets(tmp_path)


def test_medium_manifest_rejects_missing_tokenizer_provenance(tmp_path):
    manifest = checkpoint(tmp_path)
    del manifest["artifact_sha256"]["tokenizer.json"]
    (tmp_path / "model-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lacks required"):
        verify_assets(tmp_path)


def test_medium_manifest_rejects_other_revision(tmp_path):
    manifest = checkpoint(tmp_path)
    manifest["revision"] = "moving-main"
    (tmp_path / "model-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_assets(tmp_path)
