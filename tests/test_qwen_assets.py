"""Model provisioning pins one revision; no network downloads in tests."""
import json
from pathlib import Path

import pytest

from candidate_screener.llm import qwen_assets as assets


def test_provision_pins_and_reuses_revision_without_training(tmp_path, monkeypatch):
    import huggingface_hub
    from types import SimpleNamespace
    calls = []
    sha = 'a' * 40
    class API:
        def model_info(self, model):
            calls.append(('info', model))
            return SimpleNamespace(sha=sha)
    def download(model, revision, local_dir, allow_patterns):
        assert model == assets.MODEL_ID and revision == sha
        assert 'model*.safetensors' in allow_patterns
        calls.append(('download', revision))
        (Path(local_dir) / 'config.json').write_text('{}')
        (Path(local_dir) / 'model.safetensors').write_bytes(b'synthetic weights')
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    first = assets.provision(tmp_path)
    assert first['revision'] == sha
    assert assets.provision(tmp_path) == first
    assert len(calls) == 2
    (tmp_path / 'checkpoint' / 'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='manifest'):
        assets.provision(tmp_path)


def test_digest_excludes_download_metadata_but_checks_artifact_bytes(tmp_path):
    (tmp_path / 'model').write_bytes(b'synthetic')
    before = assets.asset_digest(tmp_path)
    (tmp_path / '.cache').mkdir()
    (tmp_path / '.cache' / 'metadata').write_bytes(b'changing download timestamp')
    assert assets.asset_digest(tmp_path) == before
    (tmp_path / 'model').write_bytes(b'changed')
    assert assets.asset_digest(tmp_path) != before


def test_manifest_uses_cpu_bounds_and_no_quantization(tmp_path, monkeypatch):
    import huggingface_hub
    def download(*args, local_dir, **kwargs):
        (Path(local_dir) / 'model').write_bytes(b'weights')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    manifest = assets.provision(tmp_path, revision='b' * 40)
    assert manifest['device'] == 'cpu' and manifest['quantization'] == 'none'
    assert manifest['max_output_tokens'] < manifest['max_context_tokens']
    assert json.loads((tmp_path / 'manifest.json').read_text()) == manifest


def test_model_revision_must_be_immutable(tmp_path):
    with pytest.raises(ValueError, match='immutable'):
        assets.provision(tmp_path, revision='main')


def test_larger_generator_baseline_does_not_overwrite_existing_model(tmp_path, monkeypatch):
    import huggingface_hub
    def download(model, *, local_dir, **kwargs):
        (Path(local_dir) / 'model.safetensors').write_bytes(model.encode())
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    original = assets.provision(tmp_path, revision='a' * 40)
    with pytest.raises(ValueError, match='manifest'):
        assets.provision(tmp_path, revision='b' * 40, model_id=assets.RAG_MODEL_ID)
    assert json.loads((tmp_path / 'manifest.json').read_text()) == original
    larger = assets.provision(tmp_path / 'larger', revision='b' * 40, model_id=assets.RAG_MODEL_ID)
    assert larger['model'] == assets.RAG_MODEL_ID and larger['revision'] == 'b' * 40
