import pytest
import json
from candidate_screener.llm.qa_train import window_target, training_examples


def test_answer_containing_overflow_is_positive_with_context_only_offsets():
    offsets=[(0,0),(0,6),(0,3),(4,7),(0,0)]
    assert window_target(offsets,[None,0,1,1,None],{"start":4,"end":7},0)==(3,3)
    assert window_target(offsets,[None,0,1,1,None],{"start":20,"end":24},0)==(0,0)
    assert window_target(offsets,[None,0,1,1,None],None,0)==(0,0)


def test_training_never_rounds_internal_answer_boundaries():
    with pytest.raises(ValueError,match="no rounding"):
        window_target([(0,0),(0,6)],[None,1],{"start":1,"end":6},0)


def test_qa_training_excludes_dev_test_and_lists():
    from candidate_screener.app.rag_evaluation import build_scenarios
    examples,skipped=training_examples(build_scenarios())
    assert examples and skipped["list"]>0
    assert all(int(row["family_id"][2:])<24 for row in examples)
    assert any(row["answer"] is None for row in examples)


def test_adapted_checkpoint_has_separate_immutable_training_identity(tmp_path,monkeypatch):
    from candidate_screener.llm import qa_train
    from candidate_screener.llm.qa_assets import verify_manifest,MODEL_ID
    monkeypatch.setattr(qa_train,"PROJECT_ROOT",tmp_path)
    class Model:
        def save_pretrained(self,path,**kwargs):
            path.mkdir()
            (path/"config.json").write_text("{}")
            (path/"model.safetensors").write_text("synthetic")
    class Tokenizer:
        def save_pretrained(self,path):
            (path/"tokenizer_config.json").write_text("{}")
    base={"model_id":MODEL_ID,"revision":"a"*40,"runtime":{},"artifact_sha256":{}}
    destination=tmp_path/"data/adapted"
    manifest=qa_train.save_model(Model(),Tokenizer(),destination,base,{"training_split":"train"})
    assert verify_manifest(destination/"manifest.json")["adaptation_id"]==manifest["adaptation_id"]
    manifest["training"]["training_split"]="test"
    (destination/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="provenance identity"):
        verify_manifest(destination/"manifest.json")
