"""Offline native suggestion worker; no pandas or annotation gold imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

PROMPTS = {
    'technology, software tool or programming language': 'KNOWLEDGE',
    'professional skill or competency': 'SKILL',
    'job title or occupation': 'TITLE',
    'academic degree or qualification': 'DEGREE',
}


class TokenizationBoundaryError(ValueError):
    """A source window would change frozen lexical cut rules on retokenization."""


def validate_lexical_windows(text: str, plan: dict) -> dict:
    """Fail explicitly if a chunk clips an original lexical surface.

    Learned cuts key on complete lexical surfaces. Retokenizing a suffix/prefix
    may lose other learned cuts, even though its returned offsets remain valid.
    The guard uses source text only and never consults human spans.
    """
    from .character_tokenization import LEXICAL
    boundaries = {point for window in plan['windows'] for point in window}
    for match in LEXICAL.finditer(text):
        if any(match.start() < point < match.end() for point in boundaries):
            raise TokenizationBoundaryError('Sentence fallback clips a lexical token; inference unavailable')
    return plan


def generate_records(campaign: dict, document_ids: list[str], model,
                     *, per_document_seconds: float = 120) -> list[dict]:
    from candidate_screener.extraction import transformer
    from candidate_screener.extraction import sentence_chunking
    saved = dict(transformer.MODEL_LABELS)
    original_planner = sentence_chunking.plan_chunks
    def safe_planner(text, *args, **kwargs):
        return validate_lexical_windows(text, original_planner(text, *args, **kwargs))
    transformer.MODEL_LABELS.clear()
    transformer.MODEL_LABELS.update(PROMPTS)
    sentence_chunking.plan_chunks = safe_planner
    try:
        return _generate_records(campaign, document_ids, model,
                                 per_document_seconds=per_document_seconds)
    finally:
        transformer.MODEL_LABELS.clear()
        transformer.MODEL_LABELS.update(saved)
        sentence_chunking.plan_chunks = original_planner


def _generate_records(campaign, document_ids, model, *, per_document_seconds):
    """Use exact tokenizer units and original offsets; never reads human spans."""
    if not math.isfinite(per_document_seconds) or per_document_seconds <= 0:
        raise ValueError('Positive finite inference budget required')
    from candidate_screener.extraction.medium_inference import extract_document
    # Worker-local mutation: imported aliases in _fits/extract_document refer to
    # this same dictionary. Actual preparation and prediction both use four labels.
    docs = {doc['doc_id']: doc for doc in campaign['documents']}
    if len(document_ids) != len(set(document_ids)):
        raise ValueError('Duplicate requested inference documents')
    primary = campaign.get('annotators', ['annotator1'])[0]
    assigned = set(campaign.get('assignments', {}).get(primary, docs))
    result = []
    for identity in document_ids:
        doc = docs.get(identity)
        if doc is None or doc['split'] != 'train' or identity not in assigned:
            raise ValueError('Suggestion inference is restricted to assigned training documents')
        if hashlib.sha256(doc['source_text'].encode()).hexdigest() != doc['source_sha256']:
            raise ValueError('Inference source digest mismatch')
        started = time.monotonic()
        record = {'doc_id': identity, 'source_sha256': doc['source_sha256'], 'entities': []}
        try:
            entities, coverage = extract_document(
                {'doc_id': identity, 'source': doc['source'], 'doc_type': doc['side'],
                 'fields': {'text': doc['source_text']}}, model, threshold=.5,
                overlap_tokens=32, deadline=started + per_document_seconds)
            complete = (set(coverage) == {'text'} and coverage['text']['complete']
                        and not coverage['text'].get('quarantined_predictions', 0))
            record.update(entities=entities, coverage=coverage,
                          state='complete' if complete else 'incomplete')
        except Exception as exc:
            # No exception message or source fragments in logs or public results.
            record.update(state='failed', error=type(exc).__name__)
        record['runtime_seconds'] = time.monotonic() - started
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    # The base-runtime orchestrator validates this immutable manifest before
    # dispatch and again before publication. Keep pandas out of this runtime.
    campaign = json.loads(Path(request['campaign_path']).read_text())
    if campaign['campaign_id'] != request['campaign_id']:
        raise ValueError('Worker campaign mismatch')
    import torch
    torch.set_num_threads(4)
    from .finetune import load_finetuned_model
    model = load_finetuned_model(Path(request['model_path']),
                                 snapshot_sha256=request['training_snapshot_id'])
    if model.finetune_manifest['model_id'] != request['model_id']:
        raise ValueError('Worker model identity mismatch')
    records = generate_records(campaign, request['document_ids'], model)
    output = {key: request[key] for key in ('campaign_id', 'model_id', 'training_snapshot_id')}
    output['records'] = records
    args.output.write_text(json.dumps(output, ensure_ascii=False) + '\n')
    print(json.dumps({'documents': len(records), 'states': {
        state: sum(r['state'] == state for r in records)
        for state in ('complete', 'incomplete', 'failed')}}))


if __name__ == '__main__':
    main()
