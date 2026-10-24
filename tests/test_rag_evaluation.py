"""Synthetic corpus leakage, exact evidence ownership and clustered uncertainty."""
import copy
import pytest
from candidate_screener.app.rag_evaluation import (aggregate,build_scenarios,evaluation_spec,evaluate_predictions,legacy_baseline,paired_comparison,score_turn,validate_scenarios,write_manifest)

def oracle(f,t):
    refs=t['answers']
    return {'answers':[r['text'] for r in refs],'candidate_ids':t['expected_candidate_ids'],'route':t['route'],
        'abstained':t['is_impossible'],'clarified':t['must_clarify'],'backend_ranks':t['expected_backend_ranks'],
        'citations':[{**r,'snapshot_id':f['family_id'],'quote':r['text']} for r in refs],
        'evidence_refs':t['evidence_refs'],'claims':[{'type':'source_statement','text':r['text'],'relation':r['relation'],'evidence':r} for r in refs]}

def test_frozen_corpus_family_splits_and_task_coverage():
    families=build_scenarios();spec=evaluation_spec(families)
    assert spec['splits']=={'train':{'families':24,'turns':120},'dev':{'families':12,'turns':60},'test':{'families':12,'turns':60}}
    for split in ('train','dev','test'):
        assert {'fact','list','rank','compare','followup','unanswerable','ambiguous','long'}<={t['task'] for f in families if f['split']==split for t in f['turns']}
    validate_scenarios(families)

def test_late_answer_and_outside_top_three_are_explicit_targets():
    families=build_scenarios();f,t=next((f,t) for f in families for t in f['turns'] if t['task']=='long')
    assert t['answers'][0]['start']>3000
    f,t=next((f,t) for f in families for t in f['turns'] if t['task']=='followup')
    assert t['expected_candidate_ids']==[f['candidate_ids'][3]]

def test_changed_source_and_family_document_leak_rejected():
    f=build_scenarios();f[0]['turns'][0]['answers'][0]['start']+=1
    with pytest.raises(ValueError,match='boundary'):validate_scenarios(f)
    f=build_scenarios();f[1]['documents'][0]['text']=f[0]['documents'][0]['text']
    with pytest.raises(ValueError,match='leakage'):validate_scenarios(f)

def test_correct_answer_with_wrong_owner_or_stale_citation_cannot_complete():
    f=build_scenarios()[0];t=f['turns'][0];p=oracle(f,t)
    assert score_turn(f,t,p)['task_completion']['numerator']==1
    p['citations'][0]['candidate_id']=f['candidate_ids'][1]
    assert score_turn(f,t,p)['citation_ownership']['numerator']==0
    assert score_turn(f,t,p)['task_completion']['numerator']==0
    p=oracle(f,t);p['citations'][0]['snapshot_id']='stale'
    assert score_turn(f,t,p)['task_completion']['numerator']==0

def test_valid_citation_does_not_support_wrong_relation_or_fabricated_claim():
    f=build_scenarios()[1];t=next(t for t in f['turns'] if t['task']=='compare');p=oracle(f,t)
    p['claims'][1]['relation']='supported';m=score_turn(f,t,p)
    assert m['claim_support']['numerator']==2 and m['contradiction_rate']['numerator']==1
    p=oracle(f,t);p['claims'][0]['text']='Has every possible qualification'
    assert score_turn(f,t,p)['claim_support']['numerator']==2

def test_undefined_metrics_and_family_clustered_ci():
    rows=[{'family_id':'one','metrics':{'task':{'numerator':n,'denominator':1},'empty':{'numerator':0,'denominator':0}}} for n in (0,1)]
    result=aggregate(rows,resamples=40)
    assert result['metrics']['task']['ci95']==[.5,.5]
    assert result['metrics']['empty']['value'] is None and result['metrics']['empty']['ci95'] is None
    assert result['latency_seconds']['p50'] is None

def test_heldout_freeze_and_complete_denominators():
    with pytest.raises(ValueError,match='frozen'):evaluate_predictions({},split='test')
    with pytest.raises(ValueError,match='exactly'):evaluate_predictions({},split='dev')
    with pytest.raises(ValueError,match='frozen'):legacy_baseline(split='test')

def test_oracle_dev_completion_and_paired_identical_delta():
    predictions={t['turn_id']:oracle(f,t) for f in build_scenarios() if f['split']=='dev' for t in f['turns']}
    report=evaluate_predictions(predictions,resamples=30)
    assert report['report']['metrics']['task_completion']['value']==1
    delta=paired_comparison(report,report,resamples=30)
    assert delta['delta']==0 and delta['ci95']==[0,0] and delta['families']==12
    wrong=copy.deepcopy(report);wrong['rows'].pop()
    with pytest.raises(ValueError,match='identical'):paired_comparison(report,wrong,resamples=10)

def test_spec_cannot_be_silently_replaced(tmp_path):
    path=tmp_path/'spec.json';write_manifest(path);path.write_text('{}')
    with pytest.raises(ValueError,match='Frozen'):write_manifest(path)


def test_changed_backend_score_is_a_contract_failure_even_when_ranks_match():
    f=build_scenarios()[0];t=next(t for t in f['turns'] if t['task']=='rank');p=oracle(f,t)
    p['backend_facts']=[{'candidate_id':f['candidate_ids'][0], 'rank':1,'retrieval_score':.99,
        'rerank_score':None,'classification_status':'unavailable','fit_call':None}]
    result=score_turn(f,t,p)
    assert result['backend_fact_integrity']['numerator']==0
    assert result['task_completion']['numerator']==0


def test_whole_source_passage_covers_exact_oracle_without_matching_its_boundaries():
    f=build_scenarios()[0];t=f['turns'][0];p=oracle(f,t);doc=f['documents'][0]
    wider={'document_id':doc['document_id'],'candidate_id':doc['candidate_id'],
        'start':0,'end':500,'quote':doc['text'][:500],'snapshot_id':f['family_id']}
    p['citations']=[wider];p['evidence_refs']=[wider]
    result=score_turn(f,t,p)
    assert result['evidence_recall_at_k']['numerator']==1
    assert result['citation_ownership']['numerator']==1
    assert result['task_completion']['numerator']==1
