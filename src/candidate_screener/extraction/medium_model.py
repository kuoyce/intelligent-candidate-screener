"""Offline medium loading with its supplied tokenizer and strict state loading."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from .medium_assets import MODEL_DIR, verify_assets


def runtime_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in
            ("gliner", "torch", "transformers", "spacy", "numpy", "huggingface-hub",
             "sentencepiece", "protobuf", "tokenizers", "click")}


def load_local_model(path: str | Path = MODEL_DIR, device: str = "cpu"):
    path = Path(path)
    manifest = verify_assets(path)
    from gliner import GLiNER
    from gliner.config import GLiNERConfig
    from transformers import AutoConfig, AutoTokenizer
    import torch

    values = json.loads((path / "gliner_config.json").read_text())
    values["model_name"] = str(path / "encoder-assets")
    config = GLiNERConfig(**values)
    config.encoder_config = AutoConfig.from_pretrained(values["model_name"], local_files_only=True)
    # The published backbone reserves 128100 embeddings; this GLiNER checkpoint
    # was trained with its 128003-token supplied vocabulary instead.
    config.encoder_config.vocab_size = config.vocab_size
    # v2.5 has added entity tokens absent from the backbone tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if len(tokenizer) != config.vocab_size or tokenizer.convert_tokens_to_ids(config.ent_token) != config.class_token_index:
        raise ValueError("Checkpoint tokenizer vocabulary/entity token does not match model config")
    model = GLiNER(config, tokenizer=tokenizer, encoder_from_pretrained=False)
    state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.medium_manifest = manifest
    model.runtime_versions = runtime_versions()
    return model
