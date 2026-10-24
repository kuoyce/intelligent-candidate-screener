"""Export selected metadata only; unavailable weights never invoke training."""
import json

import pytest

from candidate_screener.app import rerank_assets as assets
from candidate_screener.app.screening_service import checkpoint_digest


def fixture(tmp_path):
    record = tmp_path / 'record.json'
    record.write_text(json.dumps({'batch': {'rungs': {'R2': {'selected': 1, 'grid': [
        {'weights': 'r2-unselected'}, {'weights': 'r2-chosen'}]}}},
        'final_labels': 'must never inspect', 'final_scores': {'R2': -100}}))
    weights = tmp_path / 'weights'
    chosen = weights / 'r2-chosen'
    chosen.mkdir(parents=True)
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (chosen / name).write_text('{}')
    (chosen / 'model.safetensors').write_bytes(b'synthetic trained bytes')
    return record, weights, chosen, tmp_path / 'chat' / 'manifest.json'


def test_selected_record_index_and_export_contract(tmp_path):
    record, weights, chosen, destination = fixture(tmp_path)
    assert assets.selected_checkpoint(record_path=record, weights_root=weights) == chosen
    path = assets.prepare_manifest(record_path=record, weights_root=weights, destination=destination)
    manifest = json.loads(path.read_text())
    assert manifest['selected_weights'] == 'r2-chosen'
    assert manifest['sha256'] == checkpoint_digest(chosen)
    assert manifest['score_semantics'] == 'ordering_logit'
    assert manifest['max_length'] == 384
    assert 'binary_threshold' not in manifest
    assert 'final_scores' not in manifest
    assert (destination.parent / manifest['checkpoint']).resolve() == chosen
    assert assets.prepare_manifest(record_path=record, weights_root=weights, destination=destination) == path


def test_missing_exact_selected_cache_never_substitutes(tmp_path, monkeypatch):
    record, weights, chosen, destination = fixture(tmp_path)
    (chosen / 'model.safetensors').unlink()
    from candidate_screener.rerank import models
    monkeypatch.setattr(models, 'fit_rung', lambda *a, **k: pytest.fail('Training must not run'))
    with pytest.raises(assets.MissingRerankerCache, match='weights'):
        assets.prepare_manifest(record_path=record, weights_root=weights, destination=destination)
    assert not destination.exists()


def test_modified_checkpoint_cannot_replace_frozen_manifest(tmp_path):
    record, weights, chosen, destination = fixture(tmp_path)
    assets.prepare_manifest(record_path=record, weights_root=weights, destination=destination)
    pinned = destination.read_text()
    (chosen / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Existing frozen manifest differs'):
        assets.prepare_manifest(record_path=record, weights_root=weights, destination=destination)
    assert destination.read_text() == pinned


def test_trusted_digest_and_non_cross_encoder_rung(tmp_path):
    record, weights, chosen, destination = fixture(tmp_path)
    with pytest.raises(ValueError, match='trusted'):
        assets.prepare_manifest(record_path=record, weights_root=weights, destination=destination,
                                expected_sha256='0' * 64)
    with pytest.raises(ValueError, match='R1/R2'):
        assets.selected_checkpoint(record_path=record, weights_root=weights, rung='R3')
