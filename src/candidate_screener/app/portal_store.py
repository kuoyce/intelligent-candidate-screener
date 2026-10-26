"""Local owner-authorized HR sessions, explicit event inbox and delivery outbox.

This store accepts principals produced by a trusted authentication adapter; it is
not an identity provider. Tokens and unredacted document text are never persisted.
No method contacts an email service or manufactures application events.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid

from candidate_screener.annotation.redact import redact_text
from candidate_screener.config import PROCESSED

DEFAULT_PATH = PROCESSED / 'hr-portal/workspace.sqlite'


@dataclass(frozen=True)
class Principal:
    subject: str
    departments: tuple[str, ...]
    expires_at: float
    auth_mode: str


class AuthorizationError(ValueError):
    pass


class RevisionConflict(ValueError):
    pass


def _bounded(value, name, limit=200):
    if not isinstance(value,str) or not value or len(value)>limit or any(ord(c)<32 for c in value):
        raise ValueError(f'Invalid {name}')
    return value


def _subject(principal):
    """Separate authentication realms even if a database is accidentally shared."""
    return f'{principal.auth_mode}:{principal.subject}'


def _event_id(principal,event_id):
    return f'{principal.auth_mode}:{event_id}'


_DIGEST_FIELDS=frozenset({'report_sha256','report_digest','snapshot_sha256','source_digest',
                         'source_sha256','package_digest','document_digest','directory_digest',
                         'preview_digest','message_sha256','source_scope_sha256','scope_digest','prompt_sha256'})
_ID_FIELDS=frozenset({'snapshot_id','session_id','candidate_id','cv_id','jd_id','run_id','package_id','report_id'})
_ID_LIST_FIELDS=frozenset({'candidate_ids','comparison_ids','selected_candidates','top8_ids','shortlist_ids'})


def _opaque_identity(value):
    """Only digest/UUID/source-generated identifiers bypass prose redaction."""
    if not isinstance(value,str) or not re.fullmatch(
            r'(?:[a-fA-F0-9]{16,128}|[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}'
            r'|(?:cv|jd|candidate|a2-cv|a2-jd)[-_:][a-fA-F0-9]{16,128}'
            r'|Applicant-\d{1,6}|synthetic-(?:cv|jd)-\d{1,6}(?:-\d{1,6})?'
            r'|(?:demo-|phase2-x2:)[a-fA-F0-9]{16,128})',value):
        raise ValueError('Invalid opaque source identity')
    return value


def _digest_identity(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-fA-F0-9]{64}',value):
        raise ValueError('Invalid SHA-256 identity')
    return value


def _payload(value):
    """Redact prose while preserving narrowly validated typed source identities."""
    if not isinstance(value,dict):raise ValueError('State must be a JSON object')
    def clean(item,depth=0,field=None):
        if depth>12:raise ValueError('State nesting exceeds limit')
        if isinstance(item,dict):
            if field=='attachment_digests':
                return {_opaque_identity(k):_digest_identity(v) for k,v in item.items()}
            result={}
            for key,v in item.items():
                if not isinstance(key,str) or len(key)>100:raise ValueError('Invalid state field')
                if re.search(r'(^|_)(token|password|secret|credential|authorization|cookie)(_|$)',key,re.I):
                    raise ValueError('Credentials cannot be persisted')
                result[key]=clean(v,depth+1,key)
            return result
        if isinstance(item,list):
            if len(item)>500:raise ValueError('State list exceeds limit')
            if field in _ID_LIST_FIELDS:return [_opaque_identity(v) for v in item]
            return [clean(v,depth+1) for v in item]
        if isinstance(item,str):
            if len(item)>20_000:raise ValueError('State string exceeds limit')
            if re.search(r'\bBearer\s+\S+',item,re.I):raise ValueError('Credentials cannot be persisted')
            # QA references are typed hashes, even when a digest happens to be
            # entirely numeric. Prose redaction would sever citation links.
            if field=='evidence_id' and re.fullmatch(r'qa:[a-f0-9]{24}',item):return item
            if field=='owner':
                try:return _opaque_identity(item)
                except ValueError:pass
            if field in _DIGEST_FIELDS:return _digest_identity(item)
            if field in _ID_FIELDS:return _opaque_identity(item)
            return redact_text(item)
        if item is None or isinstance(item,(bool,int)):return item
        if isinstance(item,float) and math.isfinite(item):return item
        raise ValueError('State contains unsupported values')
    text=json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False)
    if len(text.encode())>100_000:raise ValueError('State exceeds storage limit')
    return text


class PortalStore:
    def __init__(self,path:Path=DEFAULT_PATH):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        # Containing directory and all SQLite journal files inherit private mode.
        self.path.parent.chmod(0o700)
        descriptor=os.open(self.path,os.O_CREAT|os.O_RDWR,0o600)
        os.close(descriptor);self.path.chmod(0o600)
        with self._db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions(
                session_id TEXT PRIMARY KEY,owner TEXT NOT NULL,department TEXT NOT NULL,
                jd_id TEXT NOT NULL,revision INTEGER NOT NULL,source_fingerprint TEXT NOT NULL,
                state_json TEXT NOT NULL,updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
                event_id TEXT PRIMARY KEY,department TEXT NOT NULL,kind TEXT NOT NULL,
                jd_id TEXT NOT NULL,body_json TEXT NOT NULL,created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS event_reads(
                event_id TEXT NOT NULL,recipient TEXT NOT NULL,read_at REAL NOT NULL,
                PRIMARY KEY(event_id,recipient),FOREIGN KEY(event_id) REFERENCES events(event_id));
            CREATE TABLE IF NOT EXISTS outbox(
                package_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,owner TEXT NOT NULL,
                department TEXT NOT NULL,session_revision INTEGER NOT NULL,
                source_fingerprint TEXT NOT NULL,package_digest TEXT NOT NULL,recipient_id TEXT NOT NULL,
                manifest_json TEXT NOT NULL,status TEXT NOT NULL,request_id TEXT UNIQUE,
                receipt_json TEXT,updated_at REAL NOT NULL,
                UNIQUE(session_id,package_digest,recipient_id),FOREIGN KEY(session_id) REFERENCES sessions(session_id));
            ''')
            # Additive migration: old packages, event IDs and read rows are kept.
            if 'owner' not in {r['name'] for r in db.execute('PRAGMA table_info(events)')}:
                db.execute('ALTER TABLE events ADD COLUMN owner TEXT')
            if 'created_at' not in {r['name'] for r in db.execute('PRAGMA table_info(outbox)')}:
                db.execute('ALTER TABLE outbox ADD COLUMN created_at REAL')
                db.execute('UPDATE outbox SET created_at=updated_at')
            db.execute('''CREATE TABLE IF NOT EXISTS capture_requests(
                package_id TEXT PRIMARY KEY,preview_digest TEXT NOT NULL,
                schema_version INTEGER NOT NULL,FOREIGN KEY(package_id) REFERENCES outbox(package_id))''')
            db.execute('CREATE TABLE IF NOT EXISTS portal_store_migrations(version INTEGER PRIMARY KEY)')
            db.execute('INSERT OR IGNORE INTO portal_store_migrations VALUES(2)')

    @contextmanager
    def _db(self):
        connection=sqlite3.connect(self.path,timeout=10)
        connection.row_factory=sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            connection.execute('BEGIN IMMEDIATE')
            yield connection
            connection.commit()
        except Exception:
            connection.rollback();raise
        finally:
            connection.close()

    @staticmethod
    def _authorize(principal,department=None):
        if (not isinstance(principal,Principal) or not principal.subject
                or principal.auth_mode not in {'demo','local_token'}
                or not isinstance(principal.departments,tuple) or not principal.departments
                or any(not isinstance(d,str) or not d for d in principal.departments)
                or isinstance(principal.expires_at,bool) or not isinstance(principal.expires_at,(int,float))
                or not math.isfinite(principal.expires_at) or principal.expires_at<=time.time()):
            raise AuthorizationError('Verified unexpired HR identity required')
        if department is not None and department not in principal.departments:
            raise AuthorizationError('Department access denied')

    @classmethod
    def _owned(cls,db,principal,session_id):
        cls._authorize(principal)
        row=db.execute('SELECT * FROM sessions WHERE session_id=?',(session_id,)).fetchone()
        if row is None or row['owner']!=_subject(principal):
            raise AuthorizationError('Session unavailable or access denied')
        cls._authorize(principal,row['department'])
        return row

    @staticmethod
    def _session(row):
        result=dict(row);result['state']=json.loads(result.pop('state_json'));return result

    def create_session(self,principal,department,jd_id,source_fingerprint,state):
        self._authorize(principal,department)
        _bounded(jd_id,'JD ID');_bounded(source_fingerprint,'source fingerprint')
        state_json=_payload(state);identity=str(uuid.uuid4())
        with self._db() as db:
            db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?)',
                (identity,_subject(principal),department,jd_id,0,source_fingerprint,state_json,time.time()))
            return self._session(self._owned(db,principal,identity))

    def load_session(self,principal,session_id):
        with self._db() as db:return self._session(self._owned(db,principal,session_id))

    def list_sessions(self,principal):
        self._authorize(principal)
        with self._db() as db:
            rows=db.execute('SELECT * FROM sessions WHERE owner=? ORDER BY updated_at DESC',(_subject(principal),)).fetchall()
            return [self._session(r) for r in rows if r['department'] in principal.departments]

    def save_session(self,principal,session_id,revision,state,*,source_fingerprint=None):
        if type(revision) is not int or revision<0:raise ValueError('Invalid session revision')
        state_json=_payload(state)
        with self._db() as db:
            row=self._owned(db,principal,session_id)
            if revision!=row['revision']:raise RevisionConflict('Newer session revision exists; reload before saving')
            fingerprint=source_fingerprint if source_fingerprint is not None else row['source_fingerprint']
            _bounded(fingerprint,'source fingerprint')
            db.execute('UPDATE sessions SET state_json=?,revision=revision+1,source_fingerprint=?,updated_at=? WHERE session_id=?',
                       (state_json,fingerprint,time.time(),session_id))
            return self._session(self._owned(db,principal,session_id))

    @staticmethod
    def _insert_event(db,principal,event_id,department,kind,jd_id,body,*,owner_only=False):
        body_json=_payload(body)
        stored_id=_event_id(principal,event_id)
        owner=_subject(principal) if owner_only or kind in {'email_captured','email_failed'} else None
        old=db.execute('SELECT * FROM events WHERE event_id=?',(stored_id,)).fetchone()
        if old and (old['department'],old['kind'],old['jd_id'],old['body_json'],old['owner'])!=(department,kind,jd_id,body_json,owner):
            raise ValueError('Source event ID reused with different payload')
        db.execute('INSERT OR IGNORE INTO events(event_id,department,kind,jd_id,body_json,created_at,owner) VALUES(?,?,?,?,?,?,?)',
                   (stored_id,department,kind,jd_id,body_json,time.time(),owner))
        return {'event_id':event_id,'deduplicated':old is not None}

    def ingest_event(self,principal,event_id,department,kind,jd_id,body,*,owner_only=False):
        """Trusted portal-adapter entrypoint; explicit source event IDs only."""
        self._authorize(principal,department)
        _bounded(event_id,'source event ID');_bounded(jd_id,'JD ID')
        if kind not in {'new_jd','new_applicant','ranking_ready','ranking_failed','email_captured','email_failed'}:
            raise ValueError('Unsupported portal event type')
        with self._db() as db:
            return self._insert_event(db,principal,event_id,department,kind,jd_id,body,owner_only=owner_only)

    def list_events(self,principal,*,unread_only=False):
        self._authorize(principal)
        with self._db() as db:
            rows=db.execute('SELECT e.*,r.read_at FROM events e LEFT JOIN event_reads r ON e.event_id=r.event_id AND r.recipient=? WHERE e.event_id LIKE ? ORDER BY e.created_at DESC',(_subject(principal),principal.auth_mode+':%')).fetchall()
            return [{**{k:r[k] for k in r.keys() if k!='body_json'},'event_id':r['event_id'].split(':',1)[1],'body':json.loads(r['body_json'])} for r in rows
                    if r['department'] in principal.departments and r['owner'] in (None,_subject(principal))
                    and (not unread_only or r['read_at'] is None)]

    def mark_event_read(self,principal,event_id):
        self._authorize(principal)
        stored_id=_event_id(principal,event_id)
        with self._db() as db:
            row=db.execute('SELECT * FROM events WHERE event_id=?',(stored_id,)).fetchone()
            if row is None or row['owner'] not in (None,_subject(principal)):
                raise AuthorizationError('Event unavailable or access denied')
            self._authorize(principal,row['department'])
            db.execute('INSERT OR IGNORE INTO event_reads VALUES(?,?,?)',(stored_id,_subject(principal),time.time()))
        return {'event_id':event_id,'read':True}

    def mark_event_unread(self,principal,event_id):
        self._authorize(principal)
        stored_id=_event_id(principal,event_id)
        with self._db() as db:
            row=db.execute('SELECT * FROM events WHERE event_id=?',(stored_id,)).fetchone()
            if row is None or row['owner'] not in (None,_subject(principal)):
                raise AuthorizationError('Event unavailable or access denied')
            self._authorize(principal,row['department'])
            db.execute('DELETE FROM event_reads WHERE event_id=? AND recipient=?',(stored_id,_subject(principal)))
        return {'event_id':event_id,'read':False}

    @staticmethod
    def _package(row):
        result=dict(row);result['manifest']=json.loads(result.pop('manifest_json'))
        receipt=result.pop('receipt_json');result['receipt']=json.loads(receipt) if receipt else None
        return result

    @classmethod
    def _owned_package(cls,db,principal,package_id):
        cls._authorize(principal)
        row=db.execute('SELECT * FROM outbox WHERE package_id=?',(package_id,)).fetchone()
        if row is None or row['owner']!=_subject(principal):
            raise AuthorizationError('Package unavailable or access denied')
        cls._authorize(principal,row['department'])
        cls._owned(db,principal,row['session_id'])
        return row

    def prepare_package(self,principal,session_id,revision,package_digest,recipient_id,manifest):
        _bounded(package_digest,'package digest');_bounded(recipient_id,'department recipient ID')
        if type(revision) is not int or revision<0:raise ValueError('Invalid session revision')
        manifest_json=_payload(manifest);clean=json.loads(manifest_json)
        version=clean.get('schema_version',1)
        if type(version) is not int or version not in {1,2}:raise ValueError('Unsupported handoff schema version')
        count=8 if version==1 else 10
        candidates=clean.get('candidate_ids');comparison=clean.get('comparison_ids')
        attachments=clean.get('attachment_digests')
        if (not isinstance(candidates,list) or len(candidates)!=count
                or any(not isinstance(c,str) or not c or len(c)>200 for c in candidates) or len(set(candidates))!=count
                or not isinstance(comparison,list) or len(comparison)!=10
                or any(not isinstance(c,str) or not c or len(c)>200 for c in comparison) or len(set(comparison))!=10
                or candidates!=comparison[:count] or not isinstance(attachments,dict)
                or set(attachments)!=set(candidates)
                or any(not isinstance(v,str) or not re.fullmatch('[a-f0-9]{64}',v) for v in attachments.values())
                or not isinstance(clean.get('report_digest'),str)
                or not re.fullmatch('[a-f0-9]{64}',clean['report_digest'])):
            raise ValueError(f'Complete v{version} package requires ranks1–{count} attachments and the all10 comparison')
        with self._db() as db:
            session=self._owned(db,principal,session_id)
            if session['revision']!=revision:raise RevisionConflict('Package session changed; review again')
            old=db.execute('SELECT * FROM outbox WHERE session_id=? AND package_digest=? AND recipient_id=?',
                           (session_id,package_digest,recipient_id)).fetchone()
            if old:
                if old['manifest_json']!=manifest_json:raise ValueError('Package digest reused with different contents')
                # Explicit preparation is a fresh review of this unsigned draft.
                # Already confirmed requests retain their original revision and
                # delivery identity, including unknown/failed/sent outcomes.
                if old['status']=='draft' and old['request_id'] is None:
                    db.execute('UPDATE outbox SET session_revision=?,source_fingerprint=?,updated_at=? WHERE package_id=?',
                        (revision,session['source_fingerprint'],time.time(),old['package_id']))
                return self._package(self._owned_package(db,principal,old['package_id']))
            identity=str(uuid.uuid4())
            db.execute('''INSERT INTO outbox(package_id,session_id,owner,department,session_revision,
                source_fingerprint,package_digest,recipient_id,manifest_json,status,request_id,receipt_json,updated_at,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (identity,session_id,_subject(principal),session['department'],revision,
                 session['source_fingerprint'],package_digest,recipient_id,manifest_json,'draft',None,None,time.time(),time.time()))
            return self._package(self._owned_package(db,principal,identity))

    def load_package(self,principal,package_id):
        with self._db() as db:return self._package(self._owned_package(db,principal,package_id))

    def list_captures(self,principal,session_id=None):
        """Owned historical captures; no current source/revision is required.

        A supplied session ID is authorized explicitly. Unfiltered listings omit
        sessions outside the principal's current workspaces and authentication realm.
        Returned package IDs can be passed to DemoMailer.download after a reload.
        """
        self._authorize(principal)
        with self._db() as db:
            if session_id is not None:
                self._owned(db,principal,session_id)
            placeholders=','.join('?' for _ in principal.departments)
            query=f'''SELECT p.* FROM outbox p JOIN sessions s ON p.session_id=s.session_id
                WHERE p.owner=? AND s.owner=? AND p.status='captured'
                AND p.department IN ({placeholders}) AND s.department IN ({placeholders})'''
            parameters=[_subject(principal),_subject(principal),*principal.departments,*principal.departments]
            if session_id is not None:
                query+=' AND p.session_id=?';parameters.append(session_id)
            rows=db.execute(query+' ORDER BY p.updated_at DESC,p.package_id',parameters).fetchall()
            return [self._package(row) for row in rows]

    def confirm_queue(self,principal,package_id,request_id,*,gateway_configured=False):
        """Persist one confirmed request; no network/mail side effect occurs here."""
        _bounded(request_id,'send request ID')
        with self._db() as db:
            row=self._owned_package(db,principal,package_id)
            session=self._owned(db,principal,row['session_id'])
            if (session['revision']!=row['session_revision'] or
                    session['source_fingerprint']!=row['source_fingerprint']):
                raise RevisionConflict('Package is stale; refresh and review before confirmation')
            if row['request_id']:
                # Return the existing action even after a double-click supplies
                # another ID. The same package must never become a second send.
                return {**self._package(row),'gateway_available':bool(gateway_configured)}
            if not gateway_configured:
                return {**self._package(row),'gateway_available':False,
                        'unavailable_reason':'Department mail gateway is not configured; nothing was queued or sent'}
            reused=db.execute('SELECT package_id FROM outbox WHERE request_id=?',(request_id,)).fetchone()
            if reused:raise ValueError('Send request ID belongs to another package')
            db.execute('UPDATE outbox SET status=?,request_id=?,updated_at=? WHERE package_id=?',
                       ('queued',request_id,time.time(),package_id))
            return {**self._package(self._owned_package(db,principal,package_id)),'gateway_available':True}

    def record_gateway_receipt(self,principal,package_id,request_id,status,receipt,*,gateway_id):
        """Trusted injected gateway-adapter boundary, never a browser action.

        The application supplies this only from the configured gateway's verified
        acknowledgement. This local store itself cannot prove remote delivery.
        """
        _bounded(gateway_id,'configured gateway ID')
        if status not in {'sent','failed','unknown'}:raise ValueError('Invalid gateway delivery state')
        if not isinstance(receipt,dict) or ('gateway_id' in receipt and receipt['gateway_id']!=gateway_id):
            raise ValueError('Receipt cannot override the configured gateway identity')
        receipt_json=_payload({**receipt,'gateway_id':gateway_id})
        with self._db() as db:
            row=self._owned_package(db,principal,package_id)
            if db.execute('SELECT 1 FROM capture_requests WHERE package_id=?',(package_id,)).fetchone():
                raise ValueError('Demo capture cannot become a gateway delivery')
            if row['request_id']!=request_id or not request_id or row['status']=='draft':
                raise ValueError('Receipt does not match a queued delivery request')
            if row['status']=='sent' and status!='sent':raise ValueError('Sent handoff cannot regress')
            db.execute('UPDATE outbox SET status=?,receipt_json=?,updated_at=? WHERE package_id=?',
                       (status,receipt_json,time.time(),package_id))
            return self._package(self._owned_package(db,principal,package_id))

    @classmethod
    def _current_package(cls,db,principal,package_id):
        row=cls._owned_package(db,principal,package_id)
        session=cls._owned(db,principal,row['session_id'])
        if (session['revision']!=row['session_revision'] or
                session['source_fingerprint']!=row['source_fingerprint']):
            raise RevisionConflict('Package is stale; refresh and review before confirmation')
        return row,session

    def begin_capture(self,principal,package_id,request_id,preview_digest):
        """Reserve one local capture identity and bind it to the reviewed MIME."""
        _bounded(request_id,'capture request ID');_digest_identity(preview_digest)
        with self._db() as db:
            row,_=self._current_package(db,principal,package_id)
            if json.loads(row['manifest_json']).get('schema_version',1)!=2:
                raise ValueError('Local top-ten capture requires handoff schema version 2')
            binding=db.execute('SELECT * FROM capture_requests WHERE package_id=?',(package_id,)).fetchone()
            if binding:
                if binding['preview_digest']!=preview_digest:
                    raise RevisionConflict('Email preview changed; prepare and review a new package')
                return self._package(row)
            if row['status']!='draft' or row['request_id']:
                raise ValueError('Package already belongs to another delivery transport')
            if db.execute('SELECT 1 FROM outbox WHERE request_id=?',(request_id,)).fetchone():
                raise ValueError('Capture request ID belongs to another package')
            db.execute('INSERT INTO capture_requests VALUES(?,?,1)',(package_id,preview_digest))
            db.execute('UPDATE outbox SET request_id=?,status=?,updated_at=? WHERE package_id=?',
                       (request_id,'queued',time.time(),package_id))
            return self._package(self._owned_package(db,principal,package_id))

    def record_capture(self,principal,package_id,request_id,status,receipt):
        """Commit the local result and its owner-only alert in one transaction."""
        if status not in {'captured','failed'}:raise ValueError('Invalid local capture status')
        if not isinstance(receipt,dict):raise ValueError('Invalid local capture receipt')
        with self._db() as db:
            row,session=self._current_package(db,principal,package_id)
            binding=db.execute('SELECT * FROM capture_requests WHERE package_id=?',(package_id,)).fetchone()
            if not binding or not request_id or row['request_id']!=request_id:
                raise ValueError('Receipt does not match a confirmed local capture')
            if status=='captured' and receipt.get('message_sha256')!=binding['preview_digest']:
                raise ValueError('Captured message differs from the confirmed preview')
            if row['status']=='captured':return self._package(row)
            safe={**receipt,'schema_version':1,'transport':'demo_capture','delivered':False}
            receipt_json=_payload(safe)
            db.execute('UPDATE outbox SET status=?,receipt_json=?,updated_at=? WHERE package_id=?',
                       (status,receipt_json,time.time(),package_id))
            kind='email_captured' if status=='captured' else 'email_failed'
            self._insert_event(db,principal,f'{kind}:{package_id}',row['department'],kind,session['jd_id'],
                {'package_id':package_id,'session_id':row['session_id'],
                 'summary':'Email captured in demo outbox' if status=='captured' else 'Local email capture failed; retry available'},
                owner_only=True)
            return self._package(self._owned_package(db,principal,package_id))
