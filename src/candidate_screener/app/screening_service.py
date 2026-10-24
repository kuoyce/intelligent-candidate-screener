"""Session-owned screening snapshots and strictly local, frozen CE inference."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from threading import RLock
import uuid
from time import perf_counter

import numpy as np
import pandas as pd

from candidate_screener import pipeline

MAX_POOL_SIZE = 500
MAX_TEXT_LENGTH = 200_000


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


@dataclass
class ScreeningResult(pipeline.Result):
    retrieval_score: float = 0.0
    retrieval_rank: int = 0
    rerank_score: float | None = None
    rerank_rank: int | None = None
    rerank_status: str = "not_reranked"
    classification_status: str = "unavailable"


@dataclass(frozen=True)
class ScreeningSnapshot:
    snapshot_id: str
    digest: str
    jd_text: str
    jd_title: str
    _results_json: str
    _provenance_json: str
    _sources_json: str = field(default="{}", repr=False)

    @property
    def candidate_texts(self) -> dict[str, str]:
        """Private session-owned originals; callers receive an independent copy."""
        return json.loads(self._sources_json)

    @property
    def results(self) -> tuple[ScreeningResult, ...]:
        return tuple(ScreeningResult(**row) for row in json.loads(self._results_json))

    @property
    def provenance(self) -> dict:
        return json.loads(self._provenance_json)


def checkpoint_digest(path: Path) -> str:
    """Hash relative names and bytes; refuse symlinks to external artifacts."""
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError("Frozen checkpoint is empty")
    for file in files:
        if file.is_symlink() or not file.resolve().is_relative_to(path.resolve()):
            raise ValueError("Checkpoint cannot contain linked artifacts")
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b'\0')
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        digest.update(b'\0')
    return digest.hexdigest()


class FrozenCrossEncoder:
    """No fit method; a shared lock protects model/tokenizer inference."""
    def __init__(self, model, manifest: dict):
        self._model = model
        self._manifest_json = _json(manifest)
        self._lock = RLock()
        self.name = manifest['name']

    @property
    def provenance(self):
        return json.loads(self._manifest_json)

    def score(self, pairs: pd.DataFrame) -> pd.Series:
        import torch
        manifest = self.provenance
        batch_size = manifest.get('batch_size', 8)
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("Frozen CE batch size must be between 1 and 32")
        scores = []
        with self._lock:
            for start in range(0, len(pairs), batch_size):
                batch = pairs.iloc[start:start + batch_size]
                with torch.inference_mode():
                    encoded = self._model.tokenizer(
                        batch.jd_text.tolist(), batch.cv_text.tolist(), padding=True,
                        truncation=manifest['truncation_policy'], max_length=manifest['max_length'],
                        return_tensors='pt')
                    encoded = {k: v.to(self._model.device) for k, v in encoded.items()}
                    logits = self._model(**encoded).logits
                    if logits.ndim != 2 or logits.shape != (len(batch), 1):
                        raise ValueError("Frozen CE requires one scalar logit per pair")
                    scores.extend(logits[:, 0].detach().cpu().numpy().tolist())

        return pd.Series(scores, index=pairs.cv_id.astype(str).tolist())


def validate_reranker_manifest(manifest_path: str | Path) -> tuple[dict, Path]:
    """Verify the local manifest and pinned artifacts without importing an ML runtime."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError('Frozen manifest must be an object')
    for key in ('name', 'checkpoint', 'sha256', 'tokenizer', 'max_length',
                'truncation_policy', 'selection_record', 'text_variant', 'score_semantics'):
        if key not in manifest or not manifest[key]:
            raise ValueError(f"Frozen manifest requires {key}")
    for key in ('name', 'checkpoint', 'sha256', 'tokenizer', 'truncation_policy',
                'selection_record', 'text_variant', 'score_semantics'):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f'Frozen manifest requires string {key}')
    if type(manifest.get('batch_size', 8)) is not int or not 1 <= manifest.get('batch_size', 8) <= 32:
        raise ValueError('Frozen CE batch size must be between 1 and 32')
    if manifest['text_variant'] != 'text_v2':
        raise ValueError("Frozen manifest must use text_v2")
    if type(manifest['max_length']) is not int or not 1 <= manifest['max_length'] <= 8192:
        raise ValueError("Invalid frozen maximum length")
    if manifest['truncation_policy'] not in ('longest_first', 'only_second'):
        raise ValueError("Explicit paired-text truncation required")
    checkpoint = (path.parent / manifest['checkpoint']).resolve()
    tokenizer_path = (path.parent / manifest['tokenizer']).resolve()
    if tokenizer_path != checkpoint:
        raise ValueError("Tokenizer must be included in the hashed checkpoint directory")
    if not checkpoint.is_dir() or checkpoint_digest(checkpoint) != manifest['sha256']:
        raise ValueError("Frozen checkpoint content digest mismatch or missing artifacts")
    if manifest['score_semantics'] not in ('ordering_logit', 'binary_logit'):
        raise ValueError('Explicit scalar CE score semantics required')
    threshold = manifest.get('binary_threshold')
    if threshold is not None:
        if type(threshold) not in (float, int) or not np.isfinite(threshold):
            raise ValueError("Binary threshold must be finite")
        if manifest['score_semantics'] != 'binary_logit':
            raise ValueError('Ordering scores cannot supply binary classification')
        if not manifest.get('threshold_validation_record'):
            raise ValueError("Binary threshold requires validation lineage")
    return manifest, checkpoint


def load_reranker(manifest_path: str | Path) -> FrozenCrossEncoder:
    """Load verified local files only; missing artifacts never trigger downloads."""
    manifest, checkpoint = validate_reranker_manifest(manifest_path)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if model.config.num_labels != 1:
        raise ValueError('Frozen CE requires one scalar logit per pair')
    model.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model.eval()
    return FrozenCrossEncoder(model, manifest)


class IllustrationTfidf:
    """ID-aligned adapter around the existing submission-only illustration."""
    from candidate_screener.matching.demo import IllustrationTfidf as _Implementation
    name = _Implementation.name

    def score(self, pairs):
        ranker = self._Implementation()
        ranker.fit(pairs)
        values = ranker.score(pairs).to_numpy()
        return pd.Series(values, index=pairs.cv_id.tolist())


def _scores(ranker, frame):
    values = ranker.score(frame)
    ids = frame.cv_id.astype(str).tolist()
    if not isinstance(values, pd.Series) or values.index.tolist() != ids:
        raise ValueError("Scoring output must align explicitly with candidate IDs")
    scores = values.to_numpy(dtype=float)
    if len(scores) != len(ids) or not np.isfinite(scores).all():
        raise ValueError("One finite score per candidate required")
    return dict(zip(ids, map(float, scores), strict=True))


class _Precomputed:
    def __init__(self, scores, name):
        self.scores, self.name = scores, name

    def score(self, pairs):
        return pd.Series([self.scores[str(i)] for i in pairs.cv_id])


def screen(jd_text: str, cvs: pd.DataFrame, *, jd_title: str = '', reranker=None,
           shortlist_size: int = 50, matcher=None, entity_extractor=None) -> ScreeningSnapshot:
    started = perf_counter()
    if not isinstance(jd_text, str) or not jd_text.strip() or len(jd_text) > MAX_TEXT_LENGTH:
        raise ValueError("Readable bounded JD required")
    if not isinstance(jd_title, str) or len(jd_title) > 1000:
        raise ValueError("Bounded JD title required")
    if not {'cv_id', 'cv_text'}.issubset(cvs.columns) or not 1 <= len(cvs) <= MAX_POOL_SIZE:
        raise ValueError("Candidate pool requires IDs and text and must be bounded")
    if type(shortlist_size) is not int or not 1 <= shortlist_size <= MAX_POOL_SIZE:
        raise ValueError("Invalid shortlist size")
    allowed = [key for key in ('cv_id', 'cv_text', 'source_kind', 'translation_provenance') if key in cvs]
    pool = cvs.loc[:, allowed].copy(deep=True).reset_index(drop=True)
    if pool.cv_id.isna().any() or pool.cv_id.astype(str).duplicated().any():
        raise ValueError("Unique candidate IDs required")
    pool['cv_id'] = pool.cv_id.astype(str)
    if not pool.cv_id.map(lambda value: bool(value.strip()) and len(value) <= 100).all():
        raise ValueError('Nonempty bounded candidate IDs required')
    if not pool.cv_text.map(lambda t: isinstance(t, str) and bool(t.strip()) and len(t) <= MAX_TEXT_LENGTH).all():
        raise ValueError("Readable bounded candidate text required")
    frame = pipeline._frame(jd_text, pool, None, jd_title)
    retrieval = IllustrationTfidf()
    retrieval_scores = _scores(retrieval, frame)
    retrieved = sorted(retrieval_scores, key=lambda i: (-retrieval_scores[i], i))
    shortlist = retrieved[:shortlist_size]
    rerank_scores = {}
    provenance = {'retrieval': {'name': retrieval.name, 'fitted_on': 'submitted pool; unsupervised'},
                  'reranker': None, 'shortlist_size': shortlist_size,
                  'classification': 'unavailable', 'text_variant': 'text_v2'}
    if reranker is not None:
        provenance['reranker'] = reranker.provenance
        short_frame = frame.set_index('cv_id', drop=False).loc[shortlist].reset_index(drop=True)
        rerank_scores = _scores(reranker, short_frame)
        shortlist = sorted(shortlist, key=lambda i: (-rerank_scores[i], i))
    order = shortlist + retrieved[shortlist_size:]
    # Synthetic positions only drive evidence iteration; never expose them as model scores.
    positions = {cv_id: float(len(order) - i) for i, cv_id in enumerate(order)}
    evidence = pipeline.screen(jd_text, pool, ranker=_Precomputed(positions, retrieval.name),
                               matcher=matcher, entity_extractor=entity_extractor, jd_title=jd_title)
    results = []
    threshold = (provenance['reranker'] or {}).get('binary_threshold')
    if threshold is not None and (provenance['reranker'] or {}).get('score_semantics') != 'binary_logit':
        raise ValueError('Ordering scores cannot supply binary classification')
    if threshold is not None and not (provenance['reranker'] or {}).get('threshold_validation_record'):
        raise ValueError("Binary threshold requires validation lineage")
    if threshold is not None and (type(threshold) not in (float, int) or not np.isfinite(threshold)):
        raise ValueError("Binary threshold must be finite")
    for row in evidence:
        data = row.as_dict()
        cv_id = row.cv_id
        ce = rerank_scores.get(cv_id)
        data.update(score=ce if ce is not None else retrieval_scores[cv_id], confidence=(len(order) - row.rank + 1) / len(order),
                    retrieval_score=retrieval_scores[cv_id], retrieval_rank=retrieved.index(cv_id) + 1,
                    rerank_score=ce, rerank_rank=shortlist.index(cv_id) + 1 if ce is not None else None,
                    rerank_status='reranked' if ce is not None else 'not_reranked',
                    ranker_name=reranker.name if ce is not None else retrieval.name,
                    classification_status='available' if ce is not None and threshold is not None else 'unavailable',
                    fit_call=('Fit' if ce >= threshold else 'No Fit') if ce is not None and threshold is not None else None)
        results.append(data)
    provenance['confidence_semantics'] = 'display-order percentile; not probability'
    provenance['classification'] = 'available_for_reranked_shortlist' if threshold is not None else 'unavailable'
    inputs = {'jd_text': jd_text, 'jd_title': jd_title, 'candidates': pool.to_dict('records'),
              'configuration': provenance, 'results': results}
    digest = hashlib.sha256(_json(inputs).encode()).hexdigest()
    provenance['completion_status'] = 'complete'
    provenance['timings'] = {'total_seconds': perf_counter() - started}
    return ScreeningSnapshot("snapshot_" + uuid.uuid4().hex, digest, jd_text, jd_title,
                             _json(results), _json(provenance),
                             _json(dict(zip(pool.cv_id, pool.cv_text, strict=True))))
