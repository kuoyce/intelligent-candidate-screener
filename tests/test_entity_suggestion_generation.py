"""Synthetic publication/generation safeguards; no model weights required."""
import hashlib
import json
from pathlib import Path

import pytest

from candidate_screener.entity_annotation import suggestion_generation as generation
from candidate_screener.entity_annotation.suggestion_worker import generate_records, PROMPTS
from candidate_screener.entity_annotation.suggestions import publish_set, load_for_document
from candidate_screener.entity_annotation.text import prepare_text


def document(identity='d', split='train'):
    text = '😀 Python developer contact me@example.org'
    return {'doc_id': identity, 'split': split, 'side': 'cv', 'source': 'real-corpus-fixture',
            'source_text': text, **prepare_text(text)}


def campaign():
    return {'campaign_id': 'campaign', 'guideline_version': 'guide',
            'labels': ['KNOWLEDGE', 'SKILL', 'TITLE', 'DEGREE'],
            'annotators': ['annotator1', 'annotator2'], 'documents': [document()]}


def record(doc, state='complete'):
    a = doc['source_text'].index('Python')
    return {'doc_id': doc['doc_id'], 'source_sha256': doc['source_sha256'], 'state': state,
            'entities': [{'start': a, 'end': a + 6, 'surface': 'Python',
                          'label': 'KNOWLEDGE', 'confidence': .8}]}


def test_atomic_publish_failure_leaves_old_active_set_untouched(tmp_path, monkeypatch):
    c = campaign();r = record(c['documents'][0])
    first = publish_set(c, [r], model_id='model1', training_snapshot_id='snapshot', suggestion_root=tmp_path)
    import os
    def fail_link(*args):
        raise OSError('simulated interrupted publication')
    monkeypatch.setattr(os, 'link', fail_link)
    with pytest.raises(OSError):
        publish_set(c, [r], model_id='model2', training_snapshot_id='snapshot', suggestion_root=tmp_path)
    assert json.loads((tmp_path/'active.json').read_text())['suggestion_set_id'] == first
    assert load_for_document(c, c['documents'][0], tmp_path)['available']
    assert not list(tmp_path.rglob('.manifest-*'))


def test_incomplete_predictions_retained_but_not_available(tmp_path):
    c=campaign();r=record(c['documents'][0], 'incomplete')
    identity=publish_set(c,[r],model_id='model',training_snapshot_id='snapshot',suggestion_root=tmp_path)
    assert json.loads((tmp_path/'sets'/identity/'manifest.json').read_text())['documents'][0]['entities']
    assert load_for_document(c,c['documents'][0],tmp_path)=={'available': False, 'state': 'incomplete'}


def test_generation_four_prompts_exact_source_and_restores_baseline(monkeypatch):
    from candidate_screener.extraction import medium_inference, transformer
    previous = dict(transformer.MODEL_LABELS)
    def extract(doc, model, **kwargs):
        assert transformer.MODEL_LABELS == PROMPTS
        assert len(transformer.MODEL_LABELS)==4
        text=doc['fields']['text'];a=text.index('Python')
        return [{'start':a,'end':a+6,'surface':text[a:a+6], 'label':'KNOWLEDGE','confidence':.8}], {'text': {'complete':True}}
    monkeypatch.setattr(medium_inference,'extract_document',extract)
    result=generate_records(campaign(),['d'],object())
    assert result[0]['state']=='complete'
    assert result[0]['entities'][0]['surface']=='Python'
    assert transformer.MODEL_LABELS==previous


def test_native_generation_cannot_see_heldout_document():
    c=campaign();c['documents'][0]['split']='test'
    with pytest.raises(ValueError,match='training'):
        generate_records(c,['d'],object())


def test_inference_rejects_fallback_clipping_learned_lexical_surface(monkeypatch):
    from candidate_screener.extraction import sentence_chunking
    text = 'HighlightsExcelNotes'
    doc = document()
    doc.update(source_text=text, **prepare_text(text))
    c = campaign();c['documents']=[doc]
    planner = lambda *a,**k: {'windows':[(0,10),(10,len(text))],
        'quarantined_intervals':[], 'diagnostics':{}}
    monkeypatch.setattr(sentence_chunking,'plan_chunks',planner)
    class Model:
        def predict_entities(self,*args,**kwargs):
            pytest.fail('A chunk with changed tokenizer rules must never reach inference')
    result = generate_records(c,['d'],Model())
    assert result[0]['state']=='failed'
    assert result[0]['error']=='TokenizationBoundaryError'
    assert result[0]['entities']==[]
    assert sentence_chunking.plan_chunks is planner


def test_inference_allows_whitespace_boundary_without_lexical_rule_changes():
    from candidate_screener.entity_annotation.suggestion_worker import validate_lexical_windows
    text='HighlightsExcelNotes next'
    plan={'windows':[(0,21),(21,len(text))]}
    assert validate_lexical_windows(text,plan) is plan


def test_latest_any_human_draft_excluded_not_just_reviewed():
    class Service:
        def _assigned(self,c,a): return {'d','d2'}
    c=campaign();c['documents'].append(document('d2'))
    assert generation.eligible_document_ids(Service(),c,[{'doc_id':'d','annotator':'annotator1','status':'draft'}])==['d2']


def real_model_fixture(tmp_path, c):
    docs=[document(str(i)) for i in range(10)];c['documents']=docs
    weight=tmp_path/'representation.pt';weight.write_bytes(b'synthetic fixture bytes')
    import re
    from candidate_screener.entity_annotation.character_tokenization import derive_tokenizer, BASE_PATTERN
    rows = [{'doc_id':d['doc_id'], 'text':d['source_text'], 'source_sha256':d['source_sha256'],
             'split':'train','reviewed_all':True,'spans':[], 'revision':1,'campaign_id':c['campaign_id']}
            for d in docs]
    policy = derive_tokenizer(rows, lambda text: [(m.group(),m.start(),m.end()) for m in re.finditer(BASE_PATTERN,text)])
    manifest={'tokenizer_policy':policy,'tokenizer_sha256':policy['tokenizer_sha256'],
        'snapshot':{'jsonl_sha256':policy['snapshot_sha256'],'source_revisions':[
        {'doc_id':d['doc_id'],'source_sha256':d['source_sha256'],'campaign_id':c['campaign_id'],
         'annotator':'annotator1','guideline_version':'guide','revision':1} for d in docs]},
        'artifact_sha256':{'representation.pt':hashlib.sha256(weight.read_bytes()).hexdigest()}}
    def save():
        manifest.pop('model_id',None)
        manifest['model_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
        (tmp_path/'finetune-manifest.json').write_text(json.dumps(manifest))
    save();return manifest,save


def test_model_publication_refuses_wrong_campaign_source_and_synthetic(tmp_path):
    c=campaign();manifest,save=real_model_fixture(tmp_path,c)
    assert generation.verify_model(tmp_path,c)['model_id']==manifest['model_id']
    manifest['snapshot']['source_revisions'][0]['source_sha256']='wrong';save()
    with pytest.raises(ValueError,match='provenance'):
        generation.verify_model(tmp_path,c)
    manifest['synthetic']=True;save()
    with pytest.raises(ValueError,match='Synthetic'):
        generation.verify_model(tmp_path,c)


def test_model_publication_requires_frozen_tokenizer_snapshot_parity(tmp_path):
    c=campaign();manifest,save=real_model_fixture(tmp_path,c)
    manifest['tokenizer_sha256']='wrong';save()
    with pytest.raises(ValueError,match='tokenizer snapshot'):
        generation.verify_model(tmp_path,c)


def test_publication_rechecks_draft_saved_while_inference_was_running(tmp_path, monkeypatch):
    from contextlib import nullcontext
    c = campaign()
    c['documents'].append(document('d2'))
    class Service:
        path = tmp_path / 'manifest.json'
        def _locked(self): return nullcontext()
        def campaign(self): return c
        def _assigned(self,c,a): return {'d','d2'}
        def records(self):
            return [{'doc_id':'d','annotator':'annotator1','status':'draft'}]
    monkeypatch.setattr(generation, 'verify_model', lambda *args: {
        'model_id': 'model', 'snapshot': {'jsonl_sha256': 'snapshot'}})
    result = generation.publish_results(Service(), tmp_path, {
        'campaign_id': c['campaign_id'], 'model_id': 'model', 'training_snapshot_id': 'snapshot',
        'records': [record(d) for d in c['documents']]})
    assert result['documents'] == 1
    assert result['skipped_since_generation'] == 1
    assert not load_for_document(c,c['documents'][0],tmp_path/'suggestions')['available']
    assert load_for_document(c,c['documents'][1],tmp_path/'suggestions')['available']
