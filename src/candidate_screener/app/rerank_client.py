"""Loopback-only frozen CE proxy for a separately provisioned local ML runtime."""
from __future__ import annotations

import ipaddress
import json
import math
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests

from candidate_screener.app.screening_service import (
    MAX_POOL_SIZE, MAX_TEXT_LENGTH, validate_reranker_manifest,
)

MAX_RESPONSE_BYTES = 64 * 1024
MAX_REQUEST_BYTES = 4 * 1024 * 1024
TIMEOUT_SECONDS = 60


class RerankerUnavailable(RuntimeError):
    """Safe operational message without provider errors or document content."""


def _endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('A loopback HTTP(S) reranker service URL is required')
    try:
        loopback = parsed.hostname == 'localhost' or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = False
    if not loopback or parsed.path.rstrip('/') not in ('', '/v1'):
        raise ValueError('Only the local loopback reranker service is supported')
    return urlunsplit((parsed.scheme, parsed.netloc, '/v1/rerank', '', ''))


class ServiceReranker:
    """No document cache or fit method; each request owns its HTTP session."""
    def __init__(self, manifest: dict, base_url: str):
        self._endpoint = _endpoint(base_url)
        self._manifest_json = json.dumps(manifest, sort_keys=True, allow_nan=False)
        self.name = manifest['name']

    @property
    def provenance(self) -> dict:
        manifest = json.loads(self._manifest_json)
        manifest['inference_transport'] = 'loopback local service'
        return manifest

    def score(self, pairs: pd.DataFrame) -> pd.Series:
        if not {'cv_id', 'jd_text', 'cv_text'}.issubset(pairs.columns) or not 1 <= len(pairs) <= MAX_POOL_SIZE:
            raise ValueError('A bounded candidate scoring frame is required')
        ids = pairs.cv_id.astype(str).tolist()
        if pairs.cv_id.isna().any() or len(set(ids)) != len(ids):
            raise ValueError('Unique candidate IDs are required')
        for column in ('jd_text', 'cv_text'):
            if not pairs[column].map(lambda t: isinstance(t, str) and bool(t.strip())
                                     and len(t) <= MAX_TEXT_LENGTH).all():
                raise ValueError('Readable bounded scoring text is required')
        expected = json.loads(self._manifest_json)['sha256']
        deadline = monotonic() + TIMEOUT_SECONDS
        all_scores = []
        try:
            with requests.Session() as session:
                session.trust_env = False
                start = 0
                while start < len(pairs):
                    count = min(8, len(pairs) - start)
                    while count > 0:
                        batch = pairs.iloc[start:start + count]
                        body = {'checkpoint_sha256': expected, 'jd_text': batch.jd_text.tolist(),
                                'cv_text': batch.cv_text.tolist()}
                        encoded = json.dumps(body, ensure_ascii=False).encode('utf-8')
                        if len(encoded) <= MAX_REQUEST_BYTES:
                            break
                        count //= 2
                    if count == 0:
                        raise ValueError('Scoring pair exceeds the local service request limit')
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise RerankerUnavailable('The local reranker service exceeded its time budget.')
                    with session.post(self._endpoint, data=encoded, headers={'Content-Type': 'application/json'},
                                      timeout=(min(5, remaining), remaining),
                                      allow_redirects=False, stream=True) as response:
                        if response.status_code != 200:
                            raise RerankerUnavailable('The local reranker service is unavailable.')
                        content = bytearray()
                        for chunk in response.iter_content(8192):
                            if monotonic() > deadline:
                                raise RerankerUnavailable('The local reranker service exceeded its time budget.')
                            content.extend(chunk)
                            if len(content) > MAX_RESPONSE_BYTES:
                                raise RerankerUnavailable('The local reranker service returned an oversized response.')
                        payload = json.loads(content)
                        if not isinstance(payload, dict) or payload.get('checkpoint_sha256') != expected:
                            raise RerankerUnavailable('The local reranker checkpoint differs from its pinned manifest.')
                        scores = payload.get('scores')
                        if (not isinstance(scores, list) or len(scores) != count
                                or any(type(score) not in (int, float) or not math.isfinite(score) for score in scores)):
                            raise RerankerUnavailable('The local reranker returned invalid scores.')
                        all_scores.extend(scores)
                    start += count
            return pd.Series(all_scores, index=ids, dtype=float)

        except (requests.RequestException, ValueError, TypeError, OverflowError):
            raise RerankerUnavailable('The local reranker service is unavailable or returned invalid data.') from None


def load_service_reranker(manifest_path: str | Path, base_url: str) -> ServiceReranker:
    """Verify restored local weights; service must attest the same digest per response."""
    _endpoint(base_url)
    manifest, _ = validate_reranker_manifest(manifest_path)
    return ServiceReranker(manifest, base_url)
