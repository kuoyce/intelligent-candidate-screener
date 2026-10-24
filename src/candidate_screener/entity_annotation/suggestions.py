"""Immutable annotation-assistance sets, separate from every human review.

Raw original-source predictions remain in ignored storage. Only verified,
redaction-safe, nonoverlapping display spans can leave this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from .text import digest, map_selection, python_to_utf16

VERSION = 'entity-suggestions-v1'
LABELS = frozenset({'KNOWLEDGE', 'SKILL', 'TITLE', 'DEGREE'})


def _identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()[:24]


def publish_set(campaign: dict, records: list[dict], *, model_id: str,
                training_snapshot_id: str, suggestion_root: Path) -> str:
    """Publish already-generated predictions; never runs inference or edits gold.

    Caller must supply a verified frozen model/snapshot ID. Every record contains
    doc_id, source_sha256, state (complete/incomplete/failed), and raw entities.
    A later reader revalidates every boundary against the immutable campaign.
    """
    if not model_id or not training_snapshot_id:
        raise ValueError('Frozen model and training snapshot identities required')
    documents = {d['doc_id']: d for d in campaign['documents']}
    seen = set()
    for record in records:
        doc = documents.get(record.get('doc_id'))
        if doc is None or doc['split'] != 'train' or record['doc_id'] in seen:
            raise ValueError('Suggestions require unique campaign training documents')
        seen.add(record['doc_id'])
        _validate_record(record, doc, campaign['labels'])
    payload = {'version': VERSION, 'campaign_id': campaign['campaign_id'],
               'model_id': model_id, 'training_snapshot_id': training_snapshot_id,
               'threshold': .5, 'labels': campaign['labels'],
               'display_policy': 'descending-score-then-start-end-label; greedy nonoverlap',
               'documents': sorted(records, key=lambda r: r['doc_id'])}
    set_id = _identity(payload)
    manifest = {'suggestion_set_id': set_id, **payload}
    folder = Path(suggestion_root) / 'sets' / set_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Immutable suggestion set differs')
    if not path.exists():
        import os, tempfile
        descriptor, temporary = tempfile.mkstemp(dir=folder, prefix='.manifest-')
        try:
            with os.fdopen(descriptor, 'w') as output:
                json.dump(manifest, output, ensure_ascii=False, sort_keys=True)
                output.write('\n')
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, path)  # atomic create without replacing an immutable set
            except FileExistsError:
                if json.loads(path.read_text()) != manifest:
                    raise ValueError('Immutable suggestion set differs')
        finally:
            Path(temporary).unlink(missing_ok=True)
    pointer = Path(suggestion_root) / 'active.json'
    import os, tempfile
    descriptor, temporary = tempfile.mkstemp(dir=pointer.parent, prefix='.active-')
    try:
        with os.fdopen(descriptor, 'w') as output:
            json.dump({'suggestion_set_id': set_id}, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, pointer)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return set_id


def _validate_record(record, doc, labels):
    if record.get('source_sha256') != doc['source_sha256'] or digest(doc['source_text']) != doc['source_sha256']:
        raise ValueError('Suggestion source identity mismatch')
    if record.get('state') not in {'complete', 'incomplete', 'failed'}:
        raise ValueError('Unknown suggestion processing state')
    entities = record.get('entities')
    if not isinstance(entities, list):
        raise ValueError('Suggestion entities must be a list')
    if record['state'] == 'failed' and entities:
        raise ValueError('Failed inference cannot supply predictions')
    for entity in entities:
        a, b, score = entity.get('start'), entity.get('end'), entity.get('confidence')
        if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int)
                or not isinstance(b, int) or not 0 <= a < b <= len(doc['source_text'])
                or entity.get('surface') != doc['source_text'][a:b]
                or entity.get('label') not in set(labels) & LABELS
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not .5 <= score <= 1):
            raise ValueError('Invalid suggestion evidence')


def load_for_document(campaign: dict, doc: dict, suggestion_root: Path) -> dict:
    """Map an active verified set to safe UTF16 display spans.

    The service additionally enforces annotator assignment and untouched review
    state while holding its review lock. This helper independently refuses heldout
    documents. Missing artifacts permit ordinary blank annotation.
    """
    if doc['split'] != 'train':
        raise ValueError('Held-out documents cannot expose suggestions')
    pointer = Path(suggestion_root) / 'active.json'
    if not pointer.exists():
        return {'available': False}
    set_id = json.loads(pointer.read_text()).get('suggestion_set_id')
    if not isinstance(set_id, str) or not re.fullmatch('[a-f0-9]{24}', set_id):
        raise ValueError('Invalid suggestion set identifier')
    path = Path(suggestion_root) / 'sets' / set_id / 'manifest.json'
    if not path.exists():
        return {'available': False}
    manifest = json.loads(path.read_text())
    payload = {key: value for key, value in manifest.items() if key != 'suggestion_set_id'}
    if (manifest.get('suggestion_set_id') != set_id or _identity(payload) != set_id
            or manifest.get('campaign_id') != campaign['campaign_id']
            or manifest.get('version') != VERSION or manifest.get('labels') != campaign['labels']
            or manifest.get('threshold') != .5 or not manifest.get('model_id')
            or not manifest.get('training_snapshot_id')):
        raise ValueError('Suggestion manifest identity mismatch')
    records = [r for r in manifest['documents'] if r['doc_id'] == doc['doc_id']]
    if not records:
        return {'available': False}
    if len(records) != 1:
        raise ValueError('Duplicate suggestion document identity')
    record = records[0]
    _validate_record(record, doc, campaign['labels'])
    if record['state'] != 'complete':
        return {'available': False, 'state': record['state']}
    if digest(doc['display_text']) != doc['display_sha256']:
        raise ValueError('Suggestion display identity mismatch')
    # Boundary maps can have repeated offsets at masking boundaries. Require a
    # unique exact safe source round trip, rather than guessing an inverse.
    inverse = {}
    for display, original in enumerate(doc['boundary_map']):
        if original is not None:
            inverse.setdefault(original, []).append(display)
    mapped, excluded_masked = [], 0
    for entity in record['entities']:
        candidates = []
        for a in inverse.get(entity['start'], []):
            for b in inverse.get(entity['end'], []):
                start, end = python_to_utf16(doc['display_text'], a), python_to_utf16(doc['display_text'], b)
                try:
                    if map_selection(doc, start, end) != (entity['start'], entity['end']):
                        continue
                except ValueError:
                    continue
                if doc['display_text'][a:b] == entity['surface']:
                    candidates.append({'start': start, 'end': end, 'label': entity['label'], 'score': entity['confidence']})
        if len(candidates) != 1:
            excluded_masked += 1
        else:
            mapped.append(candidates[0])
    accepted, excluded_overlaps = [], 0
    for span in sorted(mapped, key=lambda s: (-s['score'], s['start'], s['end'], s['label'])):
        if any(span['start'] < s['end'] and s['start'] < span['end'] for s in accepted):
            excluded_overlaps += 1
        else:
            accepted.append(span)
    accepted.sort(key=lambda s: (s['start'], s['end'], s['label']))
    return {'available': True, 'suggestion_set_id': set_id, 'model_id': manifest['model_id'],
            'training_snapshot_id': manifest['training_snapshot_id'], 'state': record['state'],
            'spans': accepted, 'raw_count': len(record['entities']),
            'excluded_masked': excluded_masked, 'excluded_overlaps': excluded_overlaps,
            'offset_unit': 'utf16', 'score_note': 'Raw uncalibrated model scores; not human gold'}
