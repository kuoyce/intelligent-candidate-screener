"""Verify loader construction without requiring neural dependencies or weights."""
import json
import sys
from types import ModuleType, SimpleNamespace

from candidate_screener.extraction import medium_model


def test_checkpoint_vocabulary_and_supplied_tokenizer_used_before_construction(tmp_path, monkeypatch):
    config = {"vocab_size": 128003, "class_token_index": 128001, "ent_token": "<<ENT>>"}
    (tmp_path / "gliner_config.json").write_text(json.dumps(config))
    class Tokenizer:
        def __len__(self):
            return 128003
        def convert_tokens_to_ids(self, token):
            return 128001
    tokenizer = Tokenizer()
    calls = {}
    def auto_tokenizer(path, **kwargs):
        calls["tokenizer_path"] = path
        assert kwargs["local_files_only"]
        return tokenizer
    def gliner(config, **kwargs):
        assert config.encoder_config.vocab_size == 128003
        assert kwargs["tokenizer"] is tokenizer
        assert kwargs["encoder_from_pretrained"] is False
        def load(state, strict):
            assert strict is True
            calls["state_loaded"] = True
        return SimpleNamespace(model=SimpleNamespace(load_state_dict=load),
                               to=lambda device: None, eval=lambda: None)
    modules = {name: ModuleType(name) for name in ("gliner", "gliner.config", "transformers", "torch")}
    modules["gliner"].GLiNER = gliner
    modules["gliner.config"].GLiNERConfig = lambda **values: SimpleNamespace(**values)
    modules["transformers"].AutoConfig = SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(vocab_size=128100))
    modules["transformers"].AutoTokenizer = SimpleNamespace(from_pretrained=auto_tokenizer)
    modules["torch"].load = lambda *a, **k: {}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(medium_model, "verify_assets", lambda path: {})
    monkeypatch.setattr(medium_model, "runtime_versions", lambda: {})
    medium_model.load_local_model(tmp_path)
    assert calls == {"tokenizer_path": str(tmp_path), "state_loaded": True}
