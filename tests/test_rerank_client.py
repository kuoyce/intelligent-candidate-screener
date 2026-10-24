"""Local frozen CE transport integrity without importing an ML runtime."""
import json

import pandas as pd
import pytest

from candidate_screener.app import rerank_client as client
from candidate_screener.app.screening_service import checkpoint_digest


def manifest(tmp_path):
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'model.safetensors').write_bytes(b'synthetic checkpoint')
    path = tmp_path / 'manifest.json'
    values = {'name': 'synthetic frozen R2', 'checkpoint': 'weights', 'tokenizer': 'weights',
              'sha256': checkpoint_digest(weights), 'max_length': 384,
              'truncation_policy': 'longest_first', 'text_variant': 'text_v2',
              'selection_record': 'synthetic training validation', 'score_semantics': 'ordering_logit'}
    path.write_text(json.dumps(values))
    return path, values


def fake_http(monkeypatch, payload, status=200):
    captured = {}

    class Response:
        status_code = status

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            yield payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, endpoint, **kwargs):
            captured.update(endpoint=endpoint, trust_env=self.trust_env, **kwargs)
            return Response()

    monkeypatch.setattr(client.requests, 'Session', Session)
    return captured


def pairs():
    return pd.DataFrame({'cv_id': ['z', 'a'], 'jd_text': ['python required'] * 2,
                         'cv_text': ['python experience', 'gardening experience']})


def test_loopback_scores_preserve_precision_ids_and_pinned_provenance(tmp_path, monkeypatch):
    path, values = manifest(tmp_path)
    captured = fake_http(monkeypatch, {'checkpoint_sha256': values['sha256'], 'scores': [0.50000009, -1.2]})
    reranker = client.load_service_reranker(path, 'http://127.0.0.1:8766/v1')
    result = reranker.score(pairs())
    assert result.index.tolist() == ['z', 'a']
    assert result.tolist() == [0.50000009, -1.2]
    assert captured['endpoint'] == 'http://127.0.0.1:8766/v1/rerank'
    assert captured['trust_env'] is False and captured['allow_redirects'] is False
    assert json.loads(captured['data']) == {'checkpoint_sha256': values['sha256'],
                               'jd_text': pairs().jd_text.tolist(), 'cv_text': pairs().cv_text.tolist()}
    assert reranker.provenance['score_semantics'] == 'ordering_logit'
    copied = reranker.provenance
    copied['sha256'] = 'changed'
    assert reranker.provenance['sha256'] == values['sha256']
    assert not hasattr(reranker, 'fit')


@pytest.mark.parametrize('url', ['https://example.org/v1', 'http://192.168.1.2',
                                 'http://user:password@localhost', 'http://localhost/v1?token=x',
                                 'http://localhost/arbitrary'])
def test_rejects_remote_or_credential_urls(tmp_path, url):
    path, _ = manifest(tmp_path)
    with pytest.raises(ValueError, match='loopback|local'):
        client.load_service_reranker(path, url)


@pytest.mark.parametrize('scores', [[1], [float('nan'), 1], [True, 1], ['1', 2]])
def test_invalid_service_scores_fail_closed(tmp_path, monkeypatch, scores):
    path, values = manifest(tmp_path)
    fake_http(monkeypatch, {'checkpoint_sha256': values['sha256'], 'scores': scores})
    with pytest.raises(client.RerankerUnavailable, match='invalid scores'):
        client.load_service_reranker(path, 'http://localhost').score(pairs())


def test_checkpoint_mismatch_and_oversized_response_fail_closed(tmp_path, monkeypatch):
    path, _ = manifest(tmp_path)
    fake_http(monkeypatch, {'checkpoint_sha256': 'wrong', 'scores': [1, 2]})
    with pytest.raises(client.RerankerUnavailable, match='checkpoint differs'):
        client.load_service_reranker(path, 'http://localhost').score(pairs())
    fake_http(monkeypatch, b'x' * (client.MAX_RESPONSE_BYTES + 1))
    with pytest.raises(client.RerankerUnavailable, match='oversized'):
        client.load_service_reranker(path, 'http://localhost').score(pairs())


def test_tampered_local_checkpoint_rejected_before_request(tmp_path, monkeypatch):
    path, _ = manifest(tmp_path)
    (tmp_path / 'weights' / 'model.safetensors').write_bytes(b'changed')
    fake_http(monkeypatch, {})
    with pytest.raises(ValueError, match='digest mismatch'):
        client.load_service_reranker(path, 'http://localhost')


def test_batches_at_most_eight_and_preserves_pool_order(tmp_path, monkeypatch):
    path, values = manifest(tmp_path)
    observed = []

    class Response:
        status_code = 200

        def __init__(self, count):
            self.count = count

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, size):
            yield json.dumps({'checkpoint_sha256': values['sha256'],
                              'scores': list(range(len(observed) * 10, len(observed) * 10 + self.count))}).encode()

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, endpoint, **kwargs):
            body = json.loads(kwargs['data'])
            count = len(body['cv_text'])
            observed.append((count, len(kwargs['data'])))
            return Response(count)

    monkeypatch.setattr(client.requests, 'Session', Session)
    frame = pd.DataFrame({'cv_id': [str(i) for i in range(17)], 'jd_text': ['JD'] * 17,
                          'cv_text': ['CV'] * 17})
    reranker = client.load_service_reranker(path, 'http://localhost')
    result = reranker.score(frame)
    assert result.index.tolist() == frame.cv_id.tolist()
    assert [count for count, _ in observed] == [8, 8, 1]
    assert result.tolist() == list(range(10, 18)) + list(range(20, 28)) + [30]
    observed.clear()
    frame.loc[:, 'jd_text'] = '\u4e00' * 200_000
    frame.loc[:, 'cv_text'] = '\u4e00' * 200_000
    result = reranker.score(frame.iloc[:8])
    assert len(result) == 8
    assert all(count <= 8 and size <= client.MAX_REQUEST_BYTES for count, size in observed)
    assert len(observed) > 1
