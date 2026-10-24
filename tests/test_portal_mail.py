"""Synthetic end-to-end local MIME, authorization and recovery invariants."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
import sqlite3
import stat
import time
import zipfile

import pytest

from candidate_screener.app.portal_controller import PortalController, StaleReview
from candidate_screener.app.portal_data import synthetic_demo
from candidate_screener.app.portal_mail import DemoMailer, department_directory
from candidate_screener.app.portal_store import AuthorizationError, PortalStore, Principal, RevisionConflict


@pytest.fixture
def review(tmp_path):
    principal=Principal('reviewer',('demo',),time.time()+3600,'demo')
    controller=PortalController(synthetic_demo(),PortalStore(tmp_path/'private/workspace.sqlite'),principal)
    session=controller.open_job(controller.jobs()[0]['jd_id'])
    return controller,session


def prepared(review,recipient='tech'):
    controller,session=review
    package=controller.prepare_handoff(session['session_id'],recipient)
    mailer=DemoMailer(controller)
    preview=mailer.preview(session['session_id'],package['package_id'])
    return controller,session,package,mailer,preview


def capture(mailer,session,package,preview,request='click-one'):
    return mailer.capture(session['session_id'],package['package_id'],request,preview_id=preview['preview_id'])


def test_mime_matches_preview_and_contains_all_ten_redacted_source_derivatives(review):
    controller,session,package,mailer,preview=prepared(review)
    assert preview['recipient']=='tech@example.invalid'
    assert len(preview['candidate_ids'])==10 and package['manifest']['schema_version']==2
    assert not (controller.store.path.parent/'mail-captures').exists()  # Preview has no capture side effect.
    result=capture(mailer,session,package,preview)
    assert result['status']=='captured' and result['receipt']['delivered'] is False
    raw=mailer.download(package['package_id'])
    message=BytesParser(policy=policy.default).parsebytes(raw)
    assert not message.defects and str(message['Subject'])==preview['subject']
    assert message['To'].addresses[0].addr_spec==preview['recipient']
    assert message.get_body(preferencelist=('plain',)).get_content().replace('\r\n','\n')==preview['body']
    attachments=list(message.iter_attachments())
    assert len(attachments)==1 and attachments[0].get_filename()==preview['package_filename']
    content=attachments[0].get_payload(decode=True)
    assert hashlib.sha256(content).hexdigest()==package['package_digest']
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['schema_version']==2 and [a['rank'] for a in manifest['attachments']]==list(range(1,11))
        assert len([n for n in archive.namelist() if n.startswith('cvs/')])==10
        assert len([n for n in archive.namelist() if n.startswith('profiles/')])==10
        for name,digest in manifest['artifact_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    path=mailer._path(package['package_id'])
    assert stat.S_IMODE(path.stat().st_mode)==0o600
    assert stat.S_IMODE(path.parent.stat().st_mode)==0o700
    assert result['receipt']['message_sha256']==hashlib.sha256(raw).hexdigest()


def test_double_click_reload_and_concurrent_confirmations_capture_once(review):
    controller,session,package,mailer,preview=prepared(review)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda request:capture(mailer,session,package,preview,request),('click-one','click-two')))
    assert {result['status'] for result in results}=={'captured'}
    assert len({result['request_id'] for result in results})==1
    original=mailer.download(package['package_id'])
    reopened=PortalController(controller.dataset,PortalStore(controller.store.path),controller.principal)
    same=reopened.prepare_handoff(session['session_id'],'tech')
    assert same['package_id']==package['package_id']
    result=capture(DemoMailer(reopened),session,same,preview,'click-three')
    assert result['request_id']==results[0]['request_id']
    assert DemoMailer(reopened).download(package['package_id'])==original
    assert len(list((controller.store.path.parent/'mail-captures').glob('*.eml')))==1
    assert [e['kind'] for e in controller.events()]==['email_captured']


def test_write_failure_is_retryable_owner_alert_and_retry_keeps_identity(review,monkeypatch):
    controller,session,package,mailer,preview=prepared(review)
    def fail(*args):raise OSError('private path details must not leak')
    with monkeypatch.context() as patch:
        patch.setattr('candidate_screener.app.portal_mail._write_atomic',fail)
        failed=capture(mailer,session,package,preview)
        assert failed['status']=='failed' and failed['receipt']['retryable']
        assert 'private path' not in json.dumps(failed['receipt'])
        again=capture(mailer,session,package,preview,'second-click')
        assert again['request_id']==failed['request_id']
    assert not list(controller.store.path.parent.glob('mail-captures/*.eml'))
    assert [e['kind'] for e in controller.events()]==['email_failed']
    success=capture(mailer,session,package,preview,'retry-click')
    assert success['status']=='captured' and success['request_id']==failed['request_id']
    assert {e['kind'] for e in controller.events()}=={'email_failed','email_captured'}


def test_interruption_after_artifact_write_recovers_same_message_and_request(review,monkeypatch):
    controller,session,package,mailer,preview=prepared(review)
    with monkeypatch.context() as patch:
        def interrupted(*args):raise RuntimeError('interrupted before database commit')
        patch.setattr(controller.store,'record_capture',interrupted)
        with pytest.raises(RuntimeError):capture(mailer,session,package,preview)
    original=mailer._path(package['package_id']).read_bytes()
    assert controller.store.load_package(controller.principal,package['package_id'])['status']=='queued'
    result=capture(mailer,session,package,preview,'recovery-click')
    assert result['status']=='captured' and result['request_id']=='click-one'
    assert mailer.download(package['package_id'])==original
    assert len(controller.events())==1


def test_owner_workspace_expiry_and_revocation_guard_capture_and_download(review):
    controller,session,package,mailer,preview=prepared(review)
    capture(mailer,session,package,preview)
    other=PortalController(controller.dataset,controller.store,replace(controller.principal,subject='another-reviewer'))
    for operation in (lambda:DemoMailer(other).download(package['package_id']),
                      lambda:DemoMailer(other).preview(session['session_id'],package['package_id']),
                      lambda:capture(DemoMailer(other),session,package,preview)):
        with pytest.raises(AuthorizationError):operation()
    assert other.events()==[]
    event=controller.events()[0]
    with pytest.raises(AuthorizationError):controller.store.mark_event_unread(other.principal,event['event_id'])
    expired=replace(controller.principal,expires_at=time.time()-1)
    with pytest.raises(AuthorizationError):controller.store.load_package(expired,package['package_id'])
    allowed={'value':True}
    real=PortalController(replace(controller.dataset,is_demo=False),controller.store,
        Principal('real',('benchmark',),time.time()+3600,'local_token'),auth_guard=lambda:allowed['value'])
    real_session=real.open_job(real.jobs()[0]['jd_id'])
    real_package=real.prepare_handoff(real_session['session_id'],'business')
    real_mailer=DemoMailer(real)
    real_preview=real_mailer.preview(real_session['session_id'],real_package['package_id'])
    capture(real_mailer,real_session,real_package,real_preview,'real-click')
    allowed['value']=False
    for operation in (lambda:real_mailer.download(real_package['package_id']),
                      lambda:capture(real_mailer,real_session,real_package,real_preview)):
        with pytest.raises(PermissionError,match='revoked'):operation()


def test_changed_recipient_preview_snapshot_and_session_cannot_be_confirmed(review):
    controller,session,package,mailer,preview=prepared(review)
    directory=department_directory(session['jd_id'],'demo')
    directory[session['jd_id']]['tech']['label']='Changed Tech Department'
    with pytest.raises(RevisionConflict,match='Recipient configuration'):
        DemoMailer(controller,directory).capture(session['session_id'],package['package_id'],'click',preview_id=preview['preview_id'])
    with pytest.raises(RevisionConflict,match='preview changed'):
        mailer.capture(session['session_id'],package['package_id'],'click',preview_id='a'*64)
    controller.freshness_guard=lambda:False
    with pytest.raises(StaleReview):capture(mailer,session,package,preview)
    controller.freshness_guard=None
    controller.store.save_session(controller.principal,session['session_id'],session['revision'],{'menu':'changed'})
    with pytest.raises(RevisionConflict,match='stale'):capture(mailer,session,package,preview)
    assert not (controller.store.path.parent/'mail-captures').exists()


def test_changed_directory_can_be_prepared_again_but_old_preview_cannot_authorize_it(review):
    controller,session,package,mailer,preview=prepared(review)
    directory=department_directory(session['jd_id'],'demo')
    directory[session['jd_id']]['tech']['label']='Tech hiring team'
    changed=controller.prepare_handoff(session['session_id'],'tech',directory)
    assert changed['package_id']!=package['package_id']
    assert changed['package_digest']!=package['package_digest']
    current=DemoMailer(controller,directory)
    with pytest.raises(RevisionConflict,match='preview changed'):
        current.capture(session['session_id'],changed['package_id'],'changed-click',preview_id=preview['preview_id'])
    new_preview=current.preview(session['session_id'],changed['package_id'])
    assert capture(current,session,changed,new_preview)['status']=='captured'


def test_request_identity_cannot_be_reused_for_another_package(review):
    controller,session,package,mailer,preview=prepared(review)
    capture(mailer,session,package,preview)
    second=controller.prepare_handoff(session['session_id'],'business')
    second_preview=mailer.preview(session['session_id'],second['package_id'])
    with pytest.raises(ValueError,match='another package'):
        capture(mailer,session,second,second_preview)
    assert controller.store.load_package(controller.principal,second['package_id'])['status']=='draft'


def test_job_title_header_injection_rejected_before_capture(review):
    controller,session=review
    snapshots=json.loads(controller.dataset._snapshots_json)
    snapshots[session['jd_id']]['jd_title']='Role\r\nBcc: steal@example.invalid'
    controller.dataset=replace(controller.dataset,_snapshots_json=json.dumps(snapshots))
    package=controller.prepare_handoff(session['session_id'],'tech')
    with pytest.raises(ValueError,match='header controls'):
        DemoMailer(controller).preview(session['session_id'],package['package_id'])


def test_revocation_during_failed_artifact_write_cannot_record_a_result(review,monkeypatch):
    original,_=review
    allowed={'value':True}
    controller=PortalController(replace(original.dataset,is_demo=False),original.store,
        Principal('real',('benchmark',),time.time()+3600,'local_token'),auth_guard=lambda:allowed['value'])
    session=controller.open_job(controller.jobs()[0]['jd_id'])
    package=controller.prepare_handoff(session['session_id'],'tech')
    mailer=DemoMailer(controller);preview=mailer.preview(session['session_id'],package['package_id'])
    def revoke(*args):
        allowed['value']=False
        raise OSError('write failed')
    monkeypatch.setattr('candidate_screener.app.portal_mail._write_atomic',revoke)
    with pytest.raises(PermissionError,match='revoked'):capture(mailer,session,package,preview)
    assert controller.store.load_package(controller.principal,package['package_id'])['status']=='queued'
    assert controller.store.list_events(controller.principal)==[]


@pytest.mark.parametrize('field,value',[
    ('address','tech@example.invalid\r\nBcc: steal@example.invalid'),
    ('label','Tech\nBcc: steal@example.invalid'),('address','real@example.com'),
])
def test_header_injection_and_real_recipients_rejected(review,field,value):
    controller,session=review
    directory=department_directory(session['jd_id'],'demo')
    directory[session['jd_id']]['tech'][field]=value
    package=controller.prepare_handoff(session['session_id'],'tech',directory)
    with pytest.raises(ValueError,match='header controls|example.invalid'):
        DemoMailer(controller,directory).preview(session['session_id'],package['package_id'])
    assert not (controller.store.path.parent/'mail-captures').exists()


def test_authorization_workspace_is_separate_from_human_routing_and_cross_session_is_denied(review):
    controller,session=review
    directory=department_directory(session['jd_id'],'Tech Department')
    with pytest.raises(PermissionError):controller.prepare_handoff(session['session_id'],'tech',directory)
    controller,session,package,mailer,preview=prepared(review,'business')
    assert preview['recipient']=='business@example.invalid' and package['department']=='demo'
    other=controller.open_job(controller.jobs()[0]['jd_id'])
    with pytest.raises(PermissionError,match='different review'):
        mailer.preview(other['session_id'],package['package_id'])
    result=controller.confirm_handoff(session['session_id'],package['package_id'],'confirmed',preview_id=preview['preview_id'])
    assert result['status']=='captured'
    with pytest.raises(ValueError,match='gateway delivery'):
        controller.store.record_gateway_receipt(controller.principal,package['package_id'],'confirmed','sent',{},gateway_id='demo')


def test_capture_read_survives_source_change_and_detects_artifact_tampering(review):
    controller,session,package,mailer,preview=prepared(review)
    capture(mailer,session,package,preview)
    raw=mailer.download(package['package_id'])
    controller.freshness_guard=lambda:False
    assert mailer.download(package['package_id'])==raw
    with pytest.raises(StaleReview):capture(mailer,session,package,preview)
    mailer._path(package['package_id']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='integrity'):mailer.download(package['package_id'])


def test_historical_capture_list_and_download_survive_review_revision_changes(review):
    controller,session,package,mailer,preview=prepared(review)
    result=capture(mailer,session,package,preview)
    raw=mailer.download(package['package_id'])
    controller.store.save_session(controller.principal,session['session_id'],session['revision'],{'menu':'ordinary review change'})
    duplicate=controller.prepare_handoff(session['session_id'],'tech')
    assert duplicate['package_id']==package['package_id'] and duplicate['request_id']==result['request_id']
    with pytest.raises(RevisionConflict,match='stale'):mailer.preview(session['session_id'],package['package_id'])
    controller.freshness_guard=lambda:False
    listed=controller.store.list_captures(controller.principal,session_id=session['session_id'])
    assert [item['package_id'] for item in listed]==[package['package_id']]
    assert mailer.download(listed[0]['package_id'])==raw


def test_live_mail_keeps_backend_calibrated_score_meaning_and_csv_disclaimer(review):
    controller,session=review
    snapshots=json.loads(controller.dataset._snapshots_json)
    payload=json.loads(snapshots[session['jd_id']]['_payload_json'])
    semantics='Calibrated probability of Fit supplied by the backend'
    payload['source_flags'].update(source_type='operational_live',score_semantics=semantics)
    for row in payload['candidates']:row['score_semantics']=semantics
    snapshots[session['jd_id']]['_payload_json']=json.dumps(payload)
    controller.dataset=replace(controller.dataset,_snapshots_json=json.dumps(snapshots))
    controller,session,package,mailer,preview=prepared((controller,session))
    assert semantics in preview['body'] and 'not Fit probabilities' not in preview['body']
    assert 'score meaning is supplied by the backend' in preview['body']
    capture(mailer,session,package,preview)
    message=BytesParser(policy=policy.default).parsebytes(mailer.download(package['package_id']))
    assert semantics in message.get_body(preferencelist=('plain',)).get_content()
    _,_,_,_,csv_preview=prepared((PortalController(synthetic_demo(),controller.store,controller.principal),session),'business')
    assert 'not Fit probabilities' in csv_preview['body']


def test_legacy_schema_migration_preserves_event_reads_and_packages(tmp_path):
    path=tmp_path/'legacy.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE events(event_id TEXT PRIMARY KEY,department TEXT NOT NULL,kind TEXT NOT NULL,
                jd_id TEXT NOT NULL,body_json TEXT NOT NULL,created_at REAL NOT NULL);
            CREATE TABLE event_reads(event_id TEXT NOT NULL,recipient TEXT NOT NULL,read_at REAL NOT NULL,
                PRIMARY KEY(event_id,recipient));
            CREATE TABLE outbox(package_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,owner TEXT NOT NULL,
                department TEXT NOT NULL,session_revision INTEGER NOT NULL,source_fingerprint TEXT NOT NULL,
                package_digest TEXT NOT NULL,recipient_id TEXT NOT NULL,manifest_json TEXT NOT NULL,
                status TEXT NOT NULL,request_id TEXT UNIQUE,receipt_json TEXT,updated_at REAL NOT NULL,
                UNIQUE(session_id,package_digest,recipient_id));
        ''')
        db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',('local_token:legacy','benchmark','new_jd','jd','{}',1))
        db.execute('INSERT INTO event_reads VALUES(?,?,?)',('local_token:legacy','local_token:reviewer',2))
        legacy_manifest={'candidate_ids':[f'Applicant-{i}' for i in range(8)],
                         'comparison_ids':[f'Applicant-{i}' for i in range(10)],
                         'attachment_digests':{f'Applicant-{i}':'a'*64 for i in range(8)},'report_digest':'b'*64}
        encoded=json.dumps(legacy_manifest)
        db.execute('INSERT INTO outbox VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
            ('legacy-package','legacy-session','local_token:reviewer','benchmark',0,'source','old-digest','old-recipient',encoded,'sent','old-request','{}',1))
    store=PortalStore(path)
    principal=Principal('reviewer',('benchmark',),time.time()+3600,'local_token')
    with store._db() as db:
        db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?)',('legacy-session','local_token:reviewer','benchmark','jd',0,'source','{}',1))
        assert db.execute('SELECT manifest_json FROM outbox').fetchone()[0]==encoded
    historical=store.load_package(principal,'legacy-package')
    assert historical['status']=='sent' and historical['manifest']==legacy_manifest
    assert store.list_events(principal)[0]['read_at']==2
    assert store.list_events(principal,unread_only=True)==[]
    store.mark_event_unread(principal,'legacy')
    assert len(store.list_events(principal,unread_only=True))==1
    PortalStore(path)  # Reopening migrations must be harmless.
    assert store.load_package(principal,'legacy-package')['package_digest']=='old-digest'
