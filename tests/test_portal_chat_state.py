"""Conversation persistence is scoped, idempotent and concurrency protected."""
from dataclasses import replace
import sys
import time
from types import SimpleNamespace

import pytest

from candidate_screener.app.portal_controller import PortalController, StaleReview
from candidate_screener.app.portal_data import synthetic_demo
from candidate_screener.app.portal_store import AuthorizationError, PortalStore, Principal, RevisionConflict


@pytest.fixture
def chat(tmp_path, monkeypatch):
    controller = PortalController(synthetic_demo(), PortalStore(tmp_path/'chat.sqlite'),
                                  Principal('reviewer', ('demo',), time.time()+3600, 'demo'))
    calls = []
    def respond(snapshot, question, *, context, provider):
        calls.append((snapshot.snapshot_id, question, context, provider))
        return {'text': 'Review the supplied evidence.', 'citations': [],
                'context': {'topic': 'skills'}, 'mode': 'fallback', 'status': 'answered', 'model': None}
    service = SimpleNamespace(answer=respond)
    monkeypatch.setitem(sys.modules, 'candidate_screener.app.portal_qa', service)
    session = controller.open_job(controller.jobs()[0]['jd_id'])['session_id']
    return controller, session, service, calls


def test_chat_retry_returns_saved_answer_and_preserves_context(chat):
    controller, session, _, calls = chat
    result = controller.chat(session, 'Which skills?', 'submission-one')
    assert controller.chat(session, 'Which skills?', 'submission-one') == result
    assert len(calls) == 1
    controller.chat(session, 'What about experience?', 'submission-two')
    assert calls[-1][2] == {'topic': 'skills'}
    assert len(controller.chat_history(session)) == 2
    with pytest.raises(RevisionConflict):
        controller.chat(session, 'Different question', 'submission-one')


def test_concurrent_navigation_is_not_overwritten_by_chat(chat):
    controller, session, service, _ = chat
    original = service.answer
    def concurrent(*args, **kwargs):
        controller.remember(session, 'Navigation', 'A selected action', menu='Compare')
        return original(*args, **kwargs)
    service.answer = concurrent
    with pytest.raises(RevisionConflict):
        controller.chat(session, 'Compare them', 'turn-one')
    saved = controller.load_session(session)['state']
    assert saved['menu'] == 'Compare'
    assert not saved.get('qa_turns')


def test_revoked_or_changed_sources_after_inference_do_not_persist(chat):
    controller, session, service, _ = chat
    original = service.answer
    def changed(*args, **kwargs):
        controller.freshness_guard = lambda: False
        return original(*args, **kwargs)
    service.answer = changed
    with pytest.raises(StaleReview):
        controller.chat(session, 'Which skills?', 'turn-one')
    assert not controller.load_session(session)['state'].get('qa_turns')


def test_foreign_identity_cannot_read_or_add_chat(chat):
    controller, session, _, calls = chat
    stranger = PortalController(controller.dataset, controller.store,
                                replace(controller.principal, subject='other'))
    with pytest.raises(AuthorizationError):
        stranger.chat_history(session)
    with pytest.raises(AuthorizationError):
        stranger.chat(session, 'Show candidates', 'turn-one')
    assert not calls


def test_chat_bounds_history_and_redacts_question_before_inference(chat):
    controller, session, _, calls = chat
    controller.chat(session, 'Contact private@example.com?', 'private-turn')
    assert 'private@example.com' not in calls[-1][1]
    for index in range(15):
        controller.chat(session, 'Which skills?', f'turn-{index}')
    assert len(controller.chat_history(session)) == 12
    with pytest.raises(ValueError):
        controller.chat(session, 'x'*2001, 'too-long')


def test_revocation_during_generation_is_checked_again(chat):
    controller, _, service, _ = chat
    allowed = {'value': True}
    real = PortalController(replace(controller.dataset, is_demo=False), controller.store,
                           Principal('local', ('benchmark',), time.time()+3600, 'local_token'),
                           auth_guard=lambda: allowed['value'])
    session = real.open_job(real.jobs()[0]['jd_id'])['session_id']
    original = service.answer
    def revoke(*args, **kwargs):
        allowed['value'] = False
        return original(*args, **kwargs)
    service.answer = revoke
    with pytest.raises(PermissionError):
        real.chat(session, 'Which skills?', 'revoked')
    assert not real.store.load_session(real.principal, session)['state'].get('qa_turns')


def test_new_dropdown_selection_seeds_chat_once_without_overriding_later_chat(chat):
    controller, session, _, calls = chat
    snapshot = controller.snapshot(session)
    first, second = [item['cv_id'] for item in snapshot.candidates[:2]]
    controller.remember(session, 'Requirement evidence', 'Selected',
                        extra={'selected_candidates': [first], 'selected_category': 'DEGREE'})
    controller.chat(session, 'What about them?', 'dropdown-followup')
    assert calls[-1][2]['selected_candidates'] == [first]
    assert calls[-1][2]['category'] == 'DEGREE'
    assert calls[-1][2]['snapshot_id'] == snapshot.snapshot_id
    controller.chat(session, 'Follow up', 'conversation-followup')
    assert calls[-1][2] == {'topic': 'skills'}
    controller.remember(session, 'Open CV', 'Selected', extra={'selected_candidates': [second]})
    controller.chat(session, 'Their experience?', 'new-dropdown')
    assert calls[-1][2]['selected_candidates'] == [second]
    assert calls[-1][2]['category'] == 'GENERAL'


def test_dropdown_candidate_outside_snapshot_is_rejected(chat):
    controller, session, _, calls = chat
    with pytest.raises(PermissionError):
        controller.remember(session, 'Open CV', 'Selected',
                            extra={'selected_candidates': ['synthetic-cv-999999']})
    assert not calls


def test_every_existing_dropdown_category_can_establish_context(chat):
    from candidate_screener.app.portal_data import CATEGORIES
    controller, session, _, calls = chat
    candidate = controller.snapshot(session).candidates[0]['cv_id']
    for index, category in enumerate(CATEGORIES):
        controller.remember(session, 'Requirement evidence', 'Selected',
                            extra={'selected_candidates': [candidate], 'selected_category': category})
        controller.chat(session, 'What about them?', f'category-{index}')
        assert calls[-1][2]['category'] == category
