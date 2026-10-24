"""Primary extractive routing preserves snapshot and authorization boundaries."""
from dataclasses import replace
from types import SimpleNamespace
import time

import pytest

from candidate_screener.app.portal_controller import PortalController, StaleReview
from candidate_screener.app.portal_data import synthetic_demo
from candidate_screener.app.portal_store import PortalStore, Principal


@pytest.fixture
def controller(tmp_path):
    return PortalController(synthetic_demo(), PortalStore(tmp_path/'workspace.sqlite'),
                            Principal('reviewer', ('demo',), time.time()+3600, 'demo'))


def test_controller_uses_extractive_engine_without_invoking_qwen(controller):
    record = controller.open_job(controller.jobs()[0]['jd_id'])
    calls = []
    def answer(snapshot, question, context):
        calls.append((snapshot.jd_id,question))
        return {'text':'A source answer.','citations':[], 'mode':'roberta_extractive',
                'status':'answered','model':'deepset/roberta-base-squad2',
                'context':{'snapshot_id':snapshot.snapshot_id,'jd_id':snapshot.jd_id}}
    controller.qa_service = SimpleNamespace(answer=answer)
    provider = SimpleNamespace(complete=lambda *a: pytest.fail('Qwen must not be called'))
    result = controller.chat(record['session_id'],'Compare 1 and 2','extractive-turn',provider=provider)
    assert result['mode']=='roberta_extractive' and len(calls)==1
    assert controller.chat(record['session_id'],'Compare 1 and 2','extractive-turn') == result
    assert len(calls)==1


def test_comparison_rejects_foreign_candidates_before_engine(controller):
    record=controller.open_job(controller.jobs()[0]['jd_id'])
    controller.qa_service=SimpleNamespace(compare=lambda *a: pytest.fail('Unauthorized comparison'))
    with pytest.raises(PermissionError):
        controller.compare(record['session_id'],['synthetic-cv-1-1','foreign'])


def test_comparison_rechecks_source_freshness_after_inference(controller):
    record=controller.open_job(controller.jobs()[0]['jd_id'])
    ids=[item['cv_id'] for item in controller.snapshot(record['session_id']).candidates[:2]]
    def compare(*args):
        controller.freshness_guard=lambda:False
        return {'text':'Source evidence','comparison':[]}
    controller.qa_service=SimpleNamespace(compare=compare)
    with pytest.raises(StaleReview):
        controller.compare(record['session_id'],ids)
