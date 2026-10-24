"""Verify a real trained artifact, generate privately, then publish still-untouched train suggestions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from candidate_screener.config import PROJECT_ROOT
from .service import AnnotationService
from .suggestions import publish_set


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_model(path: Path, campaign: dict) -> dict:
    """Check full manifest and artifact digests without importing neural packages."""
    path = Path(path)
    manifest = json.loads((path / 'finetune-manifest.json').read_text())
    payload = {key: value for key, value in manifest.items() if key != 'model_id'}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    if manifest.get('model_id') != expected:
        raise ValueError('Fine-tuned model manifest identity mismatch')
    snapshot = manifest.get('snapshot', {})
    from .character_tokenization import validate_manifest
    policy = manifest.get('tokenizer_policy')
    validate_manifest(policy)
    if (manifest.get('tokenizer_sha256') != policy['tokenizer_sha256']
            or snapshot.get('jsonl_sha256') != policy['snapshot_sha256']):
        raise ValueError('Fine-tuned tokenizer snapshot provenance mismatch')
    revisions = snapshot.get('source_revisions', [])
    if not snapshot.get('jsonl_sha256') or len(revisions) != 10:
        raise ValueError('Real ten-document human snapshot required for publication')
    if manifest.get('synthetic') or manifest.get('result', {}).get('synthetic'):
        raise ValueError('Synthetic checkpoints cannot publish suggestions')
    documents = {d['doc_id']: d for d in campaign['documents']}
    primary = campaign.get('annotators', ['annotator1'])[0]
    ids = []
    for revision in revisions:
        identity = revision.get('doc_id')
        doc = documents.get(identity)
        if doc is None or doc['split'] != 'train' or doc['source'] == 'synthetic-entity-demo':
            raise ValueError('Model training snapshot is not real campaign training data')
        if (revision.get('campaign_id') != campaign['campaign_id']
                or revision.get('source_sha256') != doc['source_sha256']
                or revision.get('annotator') != primary
                or revision.get('guideline_version') != campaign['guideline_version']
                or isinstance(revision.get('revision'), bool)
                or not isinstance(revision.get('revision'), int) or revision['revision'] < 1):
            raise ValueError('Model training snapshot campaign/source/review provenance mismatch')
        ids.append(identity)
    if len(set(ids)) != 10:
        raise ValueError('Model training snapshot must have ten unique documents')
    policy_sources = {(r['doc_id'], r['source_sha256'], r['campaign_id'], r['revision'])
                      for r in policy['training_sources']}
    revision_sources = {(r['doc_id'], r['source_sha256'], r['campaign_id'], r['revision'])
                        for r in revisions}
    if policy_sources != revision_sources:
        raise ValueError('Tokenizer training sources differ from model snapshot revisions')
    artifacts = manifest.get('artifact_sha256', {})
    if not artifacts or 'representation.pt' not in artifacts:
        raise ValueError('Model artifact inventory missing')
    for name, expected_digest in artifacts.items():
        artifact = path / name
        if not artifact.resolve().is_relative_to(path.resolve()) or _sha(artifact) != expected_digest:
            raise ValueError('Fine-tuned model artifact digest mismatch')
    return manifest


def eligible_document_ids(service, campaign, records):
    primary = campaign.get('annotators', ['annotator1'])[0]
    assigned = service._assigned(campaign, primary)
    touched = {r['doc_id'] for r in records if r['annotator'] == primary}
    return [d['doc_id'] for d in campaign['documents'] if d['split'] == 'train'
            and d['doc_id'] in assigned and d['doc_id'] not in touched]


def publish_results(service, model_path, worker_output):
    """Recheck latest human revisions under lock immediately before pointer publication."""
    with service._locked():
        campaign = service.campaign()
        manifest = verify_model(model_path, campaign)
        if (worker_output.get('campaign_id') != campaign['campaign_id']
                or worker_output.get('model_id') != manifest['model_id']
                or worker_output.get('training_snapshot_id') != manifest['snapshot']['jsonl_sha256']):
            raise ValueError('Worker output provenance mismatch')
        eligible = set(eligible_document_ids(service, campaign, service.records()))
        rows = worker_output.get('records')
        if not isinstance(rows, list):
            raise ValueError('Worker output records missing')
        seen = set()
        for row in rows:
            if row.get('doc_id') in seen:
                raise ValueError('Duplicate worker document')
            seen.add(row.get('doc_id'))
        remaining = [row for row in rows if row['doc_id'] in eligible]
        set_id = publish_set(campaign, remaining, model_id=manifest['model_id'],
                             training_snapshot_id=manifest['snapshot']['jsonl_sha256'],
                             suggestion_root=service.path.parent / 'suggestions')
        return {'suggestion_set_id': set_id, 'documents': len(remaining),
                'skipped_since_generation': len(rows) - len(remaining),
                'states': {state: sum(row['state'] == state for row in remaining)
                           for state in ('complete', 'incomplete', 'failed')}}


def run(campaign_path: Path, model_path: Path):
    service = AnnotationService(campaign_path)
    with service._locked():
        campaign = service.campaign()
        manifest = verify_model(model_path, campaign)
        identities = eligible_document_ids(service, campaign, service.records())
        request = {'campaign_path': str(service.path.resolve()), 'campaign_id': campaign['campaign_id'],
                   'model_path': str(Path(model_path).resolve()), 'model_id': manifest['model_id'],
                   'training_snapshot_id': manifest['snapshot']['jsonl_sha256'], 'document_ids': identities}
    root = service.path.parent / 'suggestions'
    root.mkdir(exist_ok=True)
    # Request/output contain original evidence and must stay under ignored data.
    with tempfile.TemporaryDirectory(prefix='.generation-', dir=root) as temporary:
        request_path, output_path = Path(temporary) / 'request.json', Path(temporary) / 'output.json'
        request_path.write_text(json.dumps(request))
        subprocess.run(['uv', 'run', '--project', str(PROJECT_ROOT / 'runtime/gliner-medium'),
                        '--no-sync', 'python', '-m', 'candidate_screener.entity_annotation.suggestion_worker',
                        '--request', str(request_path), '--output', str(output_path)], check=True,
                       cwd=PROJECT_ROOT, env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT / 'src'),
                                             'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        return publish_results(service, model_path, json.loads(output_path.read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.campaign, args.model)))


if __name__ == '__main__':
    main()
