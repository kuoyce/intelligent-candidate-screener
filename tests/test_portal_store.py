"""Local portal persistence tests use invented identities and temporary databases."""
from dataclasses import replace
import json
from pathlib import Path
import stat
import time

import pytest

from candidate_screener.app.portal_store import AuthorizationError,PortalStore,Principal,RevisionConflict


def principal(subject='hr-one',departments=('engineering',)):
    return Principal(subject,departments,time.time()+120,'local_token')


@pytest.fixture
def store(tmp_path):
    return PortalStore(tmp_path/'private/workspace.sqlite')


def session(store):
    return store.create_session(principal(),'engineering','jd-one','source-v1',{'menu':'job_review'})


def manifest():
    ids=[f'Applicant-{i}' for i in range(10)]
    return {'candidate_ids':ids[:8],'comparison_ids':ids,'attachment_digests':{i:'a'*64 for i in ids[:8]},'report_digest':'b'*64}


def package(store,s):
    return store.prepare_package(principal(),s['session_id'],s['revision'],'package-v1','department-directory-id',manifest())


def test_session_owner_scope_expiry_and_optimistic_concurrency(store):
    s=session(store)
    for forbidden in (principal('another-hr'),principal(departments=('sales',)),replace(principal(),expires_at=time.time()-1)):
        with pytest.raises(AuthorizationError):store.load_session(forbidden,s['session_id'])
    saved=store.save_session(principal(),s['session_id'],0,{'menu':'report'})
    assert saved['revision']==1
    with pytest.raises(RevisionConflict):store.save_session(principal(),s['session_id'],0,{'menu':'old-tab'})
    assert store.load_session(principal(),s['session_id'])['state']=={'menu':'report'}
    assert store.list_sessions(principal('another-hr'))==[]


def test_credentials_rejected_and_persisted_strings_redacted(store):
    with pytest.raises(ValueError,match='Credentials'):
        store.create_session(principal(),'engineering','j','v',{'access_token':'sensitive'})
    s=store.create_session(principal(),'engineering','j','v',{'quote':'Contact invented@example.org'})
    assert 'invented@example.org' not in s['state']['quote']
    assert stat.S_IMODE(store.path.stat().st_mode)==0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode)==0o700
    with pytest.raises(AuthorizationError):
        store.list_sessions(replace(principal(),auth_mode='login_id_only'))


def test_chat_citation_identity_survives_prose_redaction(store):
    identifier = 'qa:' + '1234567890' * 2 + '1234'
    owner = 'a2-cv-' + '1234567890' * 2
    citation = {'evidence_id': identifier, 'owner': owner, 'quote': 'Contact invented@example.org'}
    saved = store.create_session(principal(), 'engineering', 'j', 'v', {'citations': [citation]})
    restored = saved['state']['citations'][0]
    assert restored['evidence_id'] == identifier
    assert restored['owner'] == owner
    assert restored['quote'] == 'Contact [EMAIL]'


def test_real_event_ingest_deduplicates_and_read_state_is_per_recipient(store):
    assert store.list_events(principal())==[]
    a=store.ingest_event(principal(),'portal-event-123','engineering','new_applicant','jd-one',{'summary':'New applicant'})
    b=store.ingest_event(principal(),'portal-event-123','engineering','new_applicant','jd-one',{'summary':'New applicant'})
    assert not a['deduplicated'] and b['deduplicated']
    store.mark_event_read(principal(),'portal-event-123')
    assert store.list_events(principal(),unread_only=True)==[]
    assert len(store.list_events(principal('hr-two'),unread_only=True))==1
    assert store.list_events(principal(departments=('sales',)))==[]
    with pytest.raises(ValueError,match='reused'):
        store.ingest_event(principal(),'portal-event-123','engineering','new_jd','jd-two',{})


def test_unconfigured_gateway_never_queues_or_claims_sent(store):
    p=package(store,session(store))
    result=store.confirm_queue(principal(),p['package_id'],'action-one')
    assert result['status']=='draft' and not result['gateway_available']
    assert result['request_id'] is None
    with pytest.raises(ValueError,match='queued'):
        store.record_gateway_receipt(principal(),p['package_id'],'action-one','sent',{},gateway_id='fake')


def test_exact_eight_top_ranked_attachments_and_all_ten_required(store):
    s=session(store);m=manifest();m['candidate_ids'][-1]=m['comparison_ids'][9]
    with pytest.raises(ValueError,match='all10'):
        store.prepare_package(principal(),s['session_id'],0,'pkg','directory',m)
    m=manifest();m['attachment_digests'].pop(m['candidate_ids'][0])
    with pytest.raises(ValueError,match='all10'):
        store.prepare_package(principal(),s['session_id'],0,'pkg','directory',m)


def test_duplicate_confirmation_one_delivery_request_and_unknown_not_sent(store):
    p=package(store,session(store))
    queued=store.confirm_queue(principal(),p['package_id'],'action-one',gateway_configured=True)
    duplicate=store.confirm_queue(principal(),p['package_id'],'action-two',gateway_configured=True)
    assert queued['status']=='queued' and duplicate['request_id']=='action-one'
    unknown=store.record_gateway_receipt(principal(),p['package_id'],'action-one','unknown',{'reference':'fake-timeout'},gateway_id='fake')
    assert unknown['status']=='unknown'
    again=store.confirm_queue(principal(),p['package_id'],'action-three',gateway_configured=True)
    assert again['status']=='unknown' and again['request_id']=='action-one'
    sent=store.record_gateway_receipt(principal(),p['package_id'],'action-one','sent',{'reference':'fake-accepted'},gateway_id='fake')
    assert sent['status']=='sent'
    with pytest.raises(ValueError,match='regress'):
        store.record_gateway_receipt(principal(),p['package_id'],'action-one','unknown',{},gateway_id='fake')


def test_package_confirmation_invalidated_by_changed_session(store):
    s=session(store);p=package(store,s)
    store.save_session(principal(),s['session_id'],0,{'menu':'updated'},source_fingerprint='new-source')
    with pytest.raises(RevisionConflict,match='stale'):
        store.confirm_queue(principal(),p['package_id'],'action-one',gateway_configured=True)
    with pytest.raises(AuthorizationError):store.load_package(principal('another-hr'),p['package_id'])


def test_authentication_realms_isolate_same_subject_department_in_shared_database(store):
    real=principal();demo=replace(real,auth_mode='demo')
    real_session=session(store);p=package(store,real_session)
    with pytest.raises(AuthorizationError):store.load_session(demo,real_session['session_id'])
    with pytest.raises(AuthorizationError):store.load_package(demo,p['package_id'])
    assert store.list_sessions(demo)==[]
    demo_session=store.create_session(demo,'engineering','demo-jd','demo-source',{})
    assert len(store.list_sessions(demo))==1
    assert store.list_sessions(real)[0]['session_id']==real_session['session_id']
    store.ingest_event(real,'event-one','engineering','new_jd','real-jd',{})
    assert store.list_events(demo)==[]
    with pytest.raises(AuthorizationError):store.mark_event_read(demo,'event-one')
    store.ingest_event(demo,'event-one','engineering','new_jd','demo-jd',{})
    store.mark_event_read(demo,'event-one')
    assert len(store.list_events(real,unread_only=True))==1
    assert store.list_events(demo,unread_only=True)==[]


def test_gateway_receipt_cannot_override_trusted_adapter_identity(store):
    p=package(store,session(store))
    store.confirm_queue(principal(),p['package_id'],'action-one',gateway_configured=True)
    with pytest.raises(ValueError,match='override'):
        store.record_gateway_receipt(principal(),p['package_id'],'action-one','sent',
            {'gateway_id':'untrusted-gateway'},gateway_id='configured-fake')
    assert store.load_package(principal(),p['package_id'])['status']=='queued'


def test_opaque_identifiers_and_digests_survive_redaction_exactly(store):
    digest='1234567890'*6+'1234'  # Valid hexadecimal digest that resembles PII to prose redaction.
    identity='12345678-1234-4321-8123-123456789012'
    state={'report_sha256':digest,'snapshot_id':digest[:24],'session_id':identity,
        'selected_candidates':[identity,digest], 'note':'Contact invented@example.org'}
    s=store.create_session(principal(),'engineering','jd-one','source-v1',state)
    assert s['state']['report_sha256']==digest
    assert s['state']['snapshot_id']==digest[:24]
    assert s['state']['session_id']==identity
    assert s['state']['selected_candidates']==[identity,digest]
    assert 'invented@example.org' not in s['state']['note']
    m=manifest();m['report_digest']=digest;m['attachment_digests']={c:digest for c in m['candidate_ids']}
    p=store.prepare_package(principal(),s['session_id'],0,'pkg','directory',m)
    assert p['manifest']['report_digest']==digest
    assert set(p['manifest']['attachment_digests'].values())=={digest}
    m=manifest();identifiers=[str(i)*64 for i in range(10)]
    m.update(candidate_ids=identifiers[:8],comparison_ids=identifiers,
        attachment_digests={i:digest for i in identifiers[:8]})
    p=store.prepare_package(principal(),s['session_id'],0,'numeric-package','directory',m)
    assert list(p['manifest']['attachment_digests'])==identifiers[:8]


def test_person_names_cannot_bypass_redaction_as_identity_fields(store):
    for state in ({'candidate_id':'Jane Doe'}, {'candidate_ids':['Jane Doe']},
                  {'report_sha256':'invented@example.org'}):
        with pytest.raises(ValueError,match='identity'):
            store.create_session(principal(),'engineering','j','source',state)


def test_reprepare_unsigned_draft_rebinds_current_review_revision(store):
    s=session(store);p=package(store,s)
    changed=store.save_session(principal(),s['session_id'],0,{'menu':'report reviewed again'})
    again=package(store,changed)
    assert again['package_id']==p['package_id']
    assert again['session_revision']==changed['revision']
    queued=store.confirm_queue(principal(),again['package_id'],'explicit-new-confirmation',gateway_configured=True)
    assert queued['status']=='queued'
    newest=store.save_session(principal(),s['session_id'],changed['revision'],{'menu':'after queue'})
    preserved=package(store,newest)
    assert preserved['session_revision']==changed['revision']
    assert preserved['request_id']=='explicit-new-confirmation'


def test_v2_requires_exact_ten_attachments_and_rejects_unknown_versions(store):
    s=session(store);m=manifest()
    m.update(schema_version=2,candidate_ids=m['comparison_ids'])
    m['attachment_digests']={i:'a'*64 for i in m['candidate_ids']}
    p=store.prepare_package(principal(),s['session_id'],0,'v2-package','directory',m)
    assert p['manifest']['schema_version']==2
    for version in (0,3,True,'2'):
        invalid={**m,'schema_version':version}
        with pytest.raises(ValueError,match='version'):
            store.prepare_package(principal(),s['session_id'],0,'invalid','directory',invalid)


def test_mail_event_visibility_is_owner_scoped_and_unread_is_explicit(store):
    store.ingest_event(principal(),'email-one','engineering','email_captured','jd-one',{'summary':'Captured locally'})
    assert len(store.list_events(principal(),unread_only=True))==1
    assert store.list_events(principal('hr-two'))==[]
    with pytest.raises(AuthorizationError):store.mark_event_read(principal('hr-two'),'email-one')
    store.mark_event_read(principal(),'email-one')
    assert store.list_events(principal(),unread_only=True)==[]
    store.mark_event_unread(principal(),'email-one')
    assert len(store.list_events(principal(),unread_only=True))==1


def test_list_captures_filters_owner_workspace_realm_status_and_optional_session(store):
    def captured(owner,name,status='captured'):
        s=store.create_session(owner,'engineering','jd-'+name,'source',{})
        m=manifest();m.update(schema_version=2,candidate_ids=m['comparison_ids'])
        m['attachment_digests']={i:'a'*64 for i in m['candidate_ids']}
        p=store.prepare_package(owner,s['session_id'],0,'digest-'+name,'tech',m)
        if status!='draft':
            store.begin_capture(owner,p['package_id'],'request-'+name,'a'*64)
            store.record_capture(owner,p['package_id'],'request-'+name,status,{'message_sha256':'a'*64})
        return s,p
    first,one=captured(principal(),'one')
    second,two=captured(principal(),'two')
    captured(principal(),'draft','draft');captured(principal(),'failed','failed')
    foreign,foreign_package=captured(principal('another-hr'),'foreign')
    demo=replace(principal(),auth_mode='demo')
    _,demo_package=captured(demo,'demo')
    assert {p['package_id'] for p in store.list_captures(principal())}=={one['package_id'],two['package_id']}
    assert [p['package_id'] for p in store.list_captures(principal(),first['session_id'])]==[one['package_id']]
    assert [p['package_id'] for p in store.list_captures(principal('another-hr'))]==[foreign_package['package_id']]
    assert [p['package_id'] for p in store.list_captures(demo)]==[demo_package['package_id']]
    assert store.list_captures(principal(departments=('sales',)))==[]
    for forbidden,session_id in ((principal(),foreign['session_id']),
            (principal(departments=('sales',)),first['session_id']), (principal(),'missing')):
        with pytest.raises(AuthorizationError):store.list_captures(forbidden,session_id)
    with pytest.raises(AuthorizationError):store.list_captures(replace(principal(),expires_at=time.time()-1))
    store.save_session(principal(),first['session_id'],0,{},source_fingerprint='new-source')
    reopened=PortalStore(store.path)
    assert [p['package_id'] for p in reopened.list_captures(principal(),first['session_id'])]==[one['package_id']]
