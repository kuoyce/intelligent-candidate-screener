"""Isolated runtime CE scoring uses inline tensor doubles, not optional Torch."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from candidate_screener.llm.local_reranker import LocalRerankRuntime


class Vector:
    def __init__(self, values):
        self.values = values
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.values


class Logits:
    ndim = 2
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), 1)
    def __getitem__(self, key):
        assert key == (slice(None), 0)
        return Vector(self.values)


def runtime(values=None):
    batches = []
    class Model:
        def __call__(self, **inputs):
            return SimpleNamespace(logits=Logits(values if values is not None else [len(t) for t in inputs['texts']]))
    def tokenizer(left, right, **kwargs):
        assert kwargs['truncation'] == 'longest_first' and kwargs['max_length'] == 384
        batches.append(len(right))
        return {'texts': right}
    return LocalRerankRuntime(tokenizer, Model(), SimpleNamespace(inference_mode=nullcontext),
                             {'name': 'synthetic CE', 'sha256': 'a' * 64, 'max_length': 384,
                              'truncation_policy': 'longest_first', 'batch_size': 2, 'score_semantics': 'ordering_logit'}), batches


def body(**overrides):
    return {'checkpoint_sha256': 'a' * 64, 'jd_text': ['Python'] * 3,
            'cv_text': ['a', 'bbb', 'cc'], **overrides}


def test_ce_logits_preserve_input_order_and_batch_bounds():
    model, batches = runtime()
    result = model.score(body())
    assert result['scores'] == [1., 3., 2.]
    assert result['checkpoint_sha256'] == 'a' * 64
    assert batches == [2, 1]


@pytest.mark.parametrize('payload', [body(checkpoint_sha256='b' * 64), body(cv_text=['one']),
                                  body(jd_text=['']), body(cv_text=[1, 2, 3]), body(extra='instruction')])
def test_invalid_ce_requests_cannot_produce_scores(payload):
    model, batches = runtime()
    with pytest.raises(ValueError):
        model.score(payload)
    assert not batches


def test_nonfinite_ce_logits_cannot_be_returned():
    model, _ = runtime([float('nan')])
    with pytest.raises(ValueError, match='invalid scores'):
        model.score(body(jd_text=['Python'], cv_text=['Python']))
