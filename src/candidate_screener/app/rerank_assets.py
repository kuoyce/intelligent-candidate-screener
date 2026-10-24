"""Export a load-only chatbot manifest for the already selected R1/R2 weights.

Selection is copied from the committed training-validation record. No model is fitted,
no final labels are read, and missing trained weights must be restored from an archive.
The export digest pins restored local bytes; the research record has no original
checkpoint content digest against which to authenticate that restoration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from candidate_screener.app.screening_service import checkpoint_digest
from candidate_screener.config import OUTPUT, PROCESSED
from candidate_screener.rerank.models import MAX_LENGTH, NAMES

DEFAULT_RECORD = OUTPUT / 'matching' / 'stage2-metrics.json'
DEFAULT_WEIGHTS_ROOT = PROCESSED / 'rerank'
DEFAULT_MANIFEST = PROCESSED / 'chatbot' / 'reranker.json'


class MissingRerankerCache(FileNotFoundError):
    """The exact recorded trained checkpoint is unavailable."""


def selected_checkpoint(*, record_path: str | Path = DEFAULT_RECORD,
                        weights_root: str | Path = DEFAULT_WEIGHTS_ROOT,
                        rung: str = 'R2') -> Path:
    """Resolve the recorded selected index, without reevaluating selection criteria."""
    if rung not in ('R1', 'R2'):
        raise ValueError('Only recorded cross-encoder rungs R1/R2 can be exported')
    record = json.loads(Path(record_path).read_text())
    selected = record['batch']['rungs'][rung]
    index = selected['selected']
    if type(index) is not int or not 0 <= index < len(selected['grid']):
        raise ValueError('Invalid recorded selected configuration index')
    basename = selected['grid'][index]['weights']
    if (not isinstance(basename, str) or Path(basename).name != basename
            or not basename.startswith(rung.lower() + '-')):
        raise ValueError('Invalid recorded checkpoint basename')
    return Path(weights_root).resolve() / basename


def prepare_manifest(*, record_path: str | Path = DEFAULT_RECORD,
                     weights_root: str | Path = DEFAULT_WEIGHTS_ROOT,
                     destination: str | Path = DEFAULT_MANIFEST,
                     rung: str = 'R2', expected_sha256: str | None = None) -> Path:
    """Pin existing exact selected weights; never train, download, or substitute a base."""
    checkpoint = selected_checkpoint(record_path=record_path, weights_root=weights_root, rung=rung)
    if not checkpoint.is_dir() or not (checkpoint / 'config.json').is_file():
        raise MissingRerankerCache(
            f'Selected trained {rung} checkpoint {checkpoint.name} is missing. '
            f'Restore its complete model and tokenizer directory under {Path(weights_root)}. '
            'Manifest preparation never retrains or substitutes base weights.')
    if not any((checkpoint / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin',
                  'model.safetensors.index.json', 'pytorch_model.bin.index.json')):
        raise MissingRerankerCache('Selected checkpoint has no trained model weights; restore the complete cache.')
    if not (checkpoint / 'tokenizer_config.json').is_file() or not any(
            (checkpoint / name).is_file() for name in ('tokenizer.json', 'vocab.txt', 'sentencepiece.bpe.model')):
        raise MissingRerankerCache('Selected checkpoint tokenizer is incomplete; restore the complete cache.')
    digest = checkpoint_digest(checkpoint)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Restored checkpoint differs from the supplied trusted content digest')
    record_path = Path(record_path).resolve()
    destination = Path(destination).resolve()
    relative_checkpoint = os.path.relpath(checkpoint, destination.parent)
    manifest = {
        'name': NAMES[rung], 'checkpoint': relative_checkpoint, 'tokenizer': relative_checkpoint,
        'sha256': digest, 'max_length': MAX_LENGTH['ce'], 'truncation_policy': 'longest_first',
        'text_variant': 'text_v2', 'score_semantics': 'ordering_logit', 'batch_size': 8,
        'selection_record': str(record_path),
        'selection_record_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest(),
        'selected_rung': rung, 'selected_weights': checkpoint.name,
        'checkpoint_digest_provenance': 'local restored bytes at export; original training digest unavailable',
    }
    contents = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    if destination.exists():
        if destination.read_text() != contents:
            raise ValueError('Existing frozen manifest differs; use a new destination for changed artifacts')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two sessions from replacing a pinned manifest.
    try:
        with destination.open('x') as stream:
            stream.write(contents)
    except FileExistsError:
        if destination.read_text() != contents:
            raise ValueError('Concurrent frozen manifest export differs') from None
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=DEFAULT_RECORD)
    parser.add_argument('--weights-root', type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument('--destination', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--rung', choices=('R1', 'R2'), default='R2')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args(argv)
    try:
        path = prepare_manifest(record_path=args.record, weights_root=args.weights_root,
                                destination=args.destination, rung=args.rung,
                                expected_sha256=args.expected_sha256)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Frozen reranker manifest: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
