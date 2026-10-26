"""Torch-free embedding semantics and integrity on inline synthetic tensors."""
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from candidate_screener.inference.onnx_encoder import (
    CONFIG, OnnxEncoder, chunks, file_digest, pooled_chunks, verify_manifest,
)


@pytest.fixture
def model_dir(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"synthetic graph")
    (root / "tokenizer.json").write_text("{}")
    source = {"model.safetensors": "a" * 64, "tokenizer.json": "b" * 64}
    manifest = {"schema_version": 1, "source_model_key": "sbert-weak-3255167debe9",
                "source_checkpoint_files": source,
                "source_checkpoint_sha256": hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "config": CONFIG, "files": {name: file_digest(root / name) for name in ("model.onnx", "tokenizer.json")},
                "validation": {"passed": True, "reference": "pytorch", "atol": 1e-5, "rtol": 1e-4}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


class Tokenizer:
    def __init__(self):
        self.pieces = []

    def enable_truncation(self, **kwargs):
        assert kwargs == {"max_length": 256}

    def token_to_id(self, name):
        return 0 if name == "[PAD]" else None

    def enable_padding(self, **kwargs):
        assert kwargs == {"pad_id": 0, "pad_token": "[PAD]"}

    def encode_batch(self, pieces):
        self.pieces.extend(pieces)
        return [SimpleNamespace(ids=[2 if p.startswith("B") else 1, 0],
                                attention_mask=[1, 0], type_ids=[0, 0]) for p in pieces]


class Session:
    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, names, feed):
        self.calls += 1
        assert all(value.dtype == np.int64 for value in feed.values())
        ids = feed["input_ids"]
        hidden = np.zeros((*ids.shape, 384), dtype=np.float32)
        hidden[ids == 1, 0] = 3.
        hidden[ids == 2, 1] = 4.
        hidden[ids == 0, :2] = 100.
        return [hidden]


def encoder(root, cache_dir=None, **kwargs):
    return OnnxEncoder(root, cache_dir, session=kwargs.pop("session", Session()),
                       tokenizer=kwargs.pop("tokenizer", Tokenizer()), **kwargs)


def test_pooling_excludes_padding_before_normalization(monkeypatch):
    hidden = np.zeros((1, 2, 384), dtype=np.float32)
    hidden[0, 0, 0] = 3.
    hidden[0, 1, 1] = 100.
    module = importlib.import_module("candidate_screener.inference.onnx_encoder")
    if os.environ.get("ONNX_TEST_UNMASKED"):
        # Mutation proves this named invariant catches silently averaged padding.
        monkeypatch.setattr(module, "pooled_chunks", lambda h, m: module._unit(h.mean(axis=1)))
    result = module.pooled_chunks(hidden, [[1, 0]])
    np.testing.assert_array_equal(result[0, :2], [1., 0.])


def test_document_normalizes_each_chunk_before_averaging(model_dir):
    tokenizer = Tokenizer()
    model = encoder(model_dir, tokenizer=tokenizer)
    result = model.encode(["A" * 800 + "B"])
    assert tokenizer.pieces == ["A" * 800, "B"]
    np.testing.assert_allclose(result[0, :2], np.sqrt(.5), atol=1e-7)
    assert result.dtype == np.float32 and result.shape == (1, 384)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1., atol=1e-7)


def test_chunk_whitespace_and_empty_document_match_existing_semantics():
    from candidate_screener.matching.sbert import chunks as reference
    for text in ("", "  A\n\tB  ", "A" * 1599 + " B", "😀 " * 801):
        assert chunks(text) == reference(text)


def test_repeats_preserve_input_order_and_do_not_infer_twice(model_dir):
    session = Session()
    model = encoder(model_dir, session=session)
    result = model.encode(["B", "A", "B"])
    np.testing.assert_array_equal(result[0], result[2])
    np.testing.assert_array_equal(result[0, :2], [0., 1.])
    np.testing.assert_array_equal(result[1, :2], [1., 0.])
    model.encode(["A", "B"])
    assert session.calls == 1


def test_cache_is_separate_and_reloaded_without_inference(model_dir, tmp_path):
    cache = tmp_path / "onnx-cache"
    model = encoder(model_dir, cache)
    expected = model.encode(["A", "B"])
    assert model.path.parent == cache
    assert model.artifact_sha256 in model.path.name
    session = Session()
    loaded = encoder(model_dir, cache, session=session)
    np.testing.assert_array_equal(loaded.encode(["A", "B"]), expected)
    assert session.calls == 0


@pytest.mark.parametrize("change", ["nan", "shape", "identity", "unnormalized", "duplicates"])
def test_cache_rejects_invalid_values_identity_and_shape(model_dir, tmp_path, change):
    cache = tmp_path / "cache"
    model = encoder(model_dir, cache)
    model.encode(["A", "B"])
    with np.load(model.path, allow_pickle=False) as saved:
        values = {name: saved[name].copy() for name in saved.files}
    if change == "nan": values["vectors"][0, 0] = np.nan
    if change == "shape": values["vectors"] = values["vectors"][:, :10]
    if change == "identity": values["artifact_sha256"] = np.asarray("f" * 64)
    if change == "unnormalized": values["vectors"] *= 2
    if change == "duplicates": values["keys"][1] = values["keys"][0]
    np.savez(model.path, **values)
    with pytest.raises(ValueError, match="cache"):
        encoder(model_dir, cache)


@pytest.mark.parametrize("name", ["model.onnx", "tokenizer.json"])
def test_changed_artifact_bytes_fail_before_loading_runtime(model_dir, name):
    (model_dir / name).write_bytes(b"altered")
    with pytest.raises(ValueError, match="digest"):
        encoder(model_dir)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "x\\file"])
def test_manifest_cannot_load_artifacts_outside_model_root(model_dir, name):
    manifest = json.loads((model_dir / "manifest.json").read_text())
    manifest["files"][name] = "a" * 64
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_manifest(model_dir)


def test_source_checkpoint_aggregate_and_semantics_are_bound(model_dir):
    manifest = json.loads((model_dir / "manifest.json").read_text())
    manifest["source_checkpoint_files"]["model.safetensors"] = "f" * 64
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="aggregate"):
        verify_manifest(model_dir)


def test_unverified_export_is_only_available_to_explicit_bootstrap(model_dir):
    manifest = json.loads((model_dir / "manifest.json").read_text())
    manifest["validation"]["passed"] = False
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="equivalence"):
        encoder(model_dir)
    assert encoder(model_dir, require_verified=False).encode(["A"]).shape == (1, 384)


def test_runtime_does_not_require_optional_neural_imports(model_dir, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "sentence_transformers"}:
            raise AssertionError("Torch-free inference imported a neural Python stack")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert encoder(model_dir).encode(["A"]).shape == (1, 384)


@pytest.mark.parametrize("kind", ["zero", "nan", "wrong_dimension"])
def test_invalid_token_vectors_fail_explicitly(kind):
    hidden = np.zeros((1, 2, 384), dtype=np.float32)
    if kind == "nan": hidden[0, 0, 0] = np.nan
    if kind == "wrong_dimension": hidden = hidden[:, :, :10]
    with pytest.raises(ValueError):
        pooled_chunks(hidden, [[1, 1]])


def test_empty_input_does_not_invoke_runtime(model_dir):
    session = Session()
    assert encoder(model_dir, session=session).encode([]).shape == (0, 384)
    assert session.calls == 0
