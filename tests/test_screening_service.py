"""Frozen inference boundaries and score semantics on synthetic documents."""
import json

import pandas as pd
import pytest

from candidate_screener.app import screening_service as service
from candidate_screener.extraction.rules import SkillMatcher


class FakeCE:
    name = 'synthetic frozen CE'
    provenance = {'name': name, 'score_semantics': 'binary_logit', 'binary_threshold': 0.5000001,
                  'threshold_validation_record': 'synthetic train validation'}

    def score(self, frame):
        return pd.Series([0.50000009 if i == 'a' else 0.8 for i in frame.cv_id],
                         index=frame.cv_id.tolist())


def run(**kwargs):
    pool = pd.DataFrame({'cv_id': ['a', 'b', 'c'],
                         'cv_text': ['Python developer', 'Python engineer', 'gardening plants']})
    return service.screen('Python required', pool, matcher=SkillMatcher(pd.DataFrame({'alias': ['Python'], 'canonical': ['python']})), **kwargs)


def test_shortlist_tail_and_full_precision_threshold():
    snapshot = run(reranker=FakeCE(), shortlist_size=2)
    rows = snapshot.results
    assert [r.cv_id for r in rows] == ['b', 'a', 'c']
    assert rows[1].fit_call == 'No Fit'  # rounds to .5; must compare full precision
    assert rows[1].rerank_score == 0.50000009
    assert rows[2].rerank_score is None and rows[2].rerank_rank is None
    assert rows[2].fit_call is None and rows[2].rerank_status == 'not_reranked'
    assert rows[0].retrieval_rank == 2


def test_snapshot_copies_do_not_mutate_grounding():
    snapshot = run()
    rows = snapshot.results
    rows[0].evidence.clear()
    rows[0].rank = 999
    provenance = snapshot.provenance
    provenance.clear()
    assert snapshot.results[0].rank == 1
    assert snapshot.provenance['classification'] == 'unavailable'
    assert snapshot.digest == run().digest
    assert snapshot.snapshot_id != run().snapshot_id


@pytest.mark.parametrize('bad', ['alignment', 'nan'])
def test_rejects_misaligned_or_nonfinite_scores(bad):
    class Bad(FakeCE):
        def score(self, frame):
            return pd.Series([float('nan')] * len(frame) if bad == 'nan' else [1.] * len(frame),
                             index=frame.cv_id.tolist()[::-1] if bad == 'alignment' else frame.cv_id.tolist())
    with pytest.raises(ValueError):
        run(reranker=Bad(), shortlist_size=2)


def test_threshold_requires_validation_lineage():
    class Bad(FakeCE):
        provenance = {'binary_threshold': 0.5, 'score_semantics': 'binary_logit'}
    with pytest.raises(ValueError, match='validation lineage'):
        run(reranker=Bad())


def test_missing_checkpoint_fails_before_model_loading(tmp_path):
    manifest = {'name': 'test', 'checkpoint': 'missing', 'tokenizer': 'missing',
                'sha256': '0' * 64, 'max_length': 512, 'truncation_policy': 'longest_first',
                'score_semantics': 'ordering_logit', 'selection_record': 'train-selected', 'text_variant': 'text_v2'}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='digest mismatch'):
        service.load_reranker(path)


def test_duplicate_ids_and_pool_bound():
    pool = pd.DataFrame({'cv_id': ['a', 'a'], 'cv_text': ['python', 'python']})
    with pytest.raises(ValueError, match='Unique'):
        service.screen('python', pool)
    with pytest.raises(ValueError, match='shortlist'):
        run(shortlist_size=0)


def test_completion_and_percentile_are_explicit():
    snapshot = run()
    assert snapshot.provenance['completion_status'] == 'complete'
    assert snapshot.provenance['timings']['total_seconds'] >= 0
    assert 'not probability' in snapshot.provenance['confidence_semantics']
    assert [r.confidence for r in snapshot.results] == [1., 2 / 3, 1 / 3]


@pytest.mark.parametrize('threshold', [True, '0.5'])
def test_threshold_rejects_non_numeric_types(threshold):
    class Bad(FakeCE):
        provenance = {**FakeCE.provenance, 'binary_threshold': threshold}
    with pytest.raises(ValueError, match='finite'):
        run(reranker=Bad())


def test_manifest_requires_object(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text('[]')
    with pytest.raises(ValueError, match='object'):
        service.load_reranker(path)


def test_ce_batches_and_preserves_input_ids():
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    sizes = []

    class Model:
        device = 'cpu'

        def tokenizer(self, left, right, **kwargs):
            sizes.append(len(left))
            return {'input_ids': torch.zeros((len(left), 1))}

        def __call__(self, **encoded):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(logits=torch.ones((len(encoded['input_ids']), 1)))

    model = service.FrozenCrossEncoder(Model(), {'name': 'fake', 'batch_size': 2,
                                      'truncation_policy': 'longest_first', 'max_length': 512})
    frame = pd.DataFrame({'cv_id': ['c', 'a', 'b', 'd', 'e'], 'jd_text': ['x'] * 5,
                          'cv_text': ['y'] * 5})
    assert model.score(frame).index.tolist() == frame.cv_id.tolist()
    assert sizes == [2, 2, 1]


def test_cv_metadata_is_ignored_even_when_missing_or_changed():
    pool = pd.DataFrame({'cv_id': ['a'], 'cv_text': ['Python'],
                         'Position': [float('nan')], 'Primary Keyword': ['oracle']})
    matcher = SkillMatcher(pd.DataFrame({'canonical': [], 'alias': []}))
    first = service.screen('Python', pool, matcher=matcher)
    pool['Position'] = 'Changed oracle'
    second = service.screen('Python', pool, matcher=matcher)
    assert first.digest == second.digest
    assert first.results[0].score == second.results[0].score


def test_valid_manifest_loader_is_local_only_and_never_fits(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{"num_labels":1}')
    manifest = {'name': 'synthetic frozen artifact', 'checkpoint': 'checkpoint', 'tokenizer': 'checkpoint',
        'sha256': service.checkpoint_digest(checkpoint), 'text_variant': 'text_v2',
        'score_semantics': 'ordering_logit', 'selection_record': 'synthetic dev selection',
        'max_length': 384, 'truncation_policy': 'longest_first'}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    calls = []
    class Model:
        config = SimpleNamespace(num_labels=1)
        def eval(self):
            calls.append('eval')
    def model_load(path, **kwargs):
        assert kwargs == {'local_files_only': True, 'trust_remote_code': False}
        calls.append('model')
        return Model()
    def tokenizer_load(path, **kwargs):
        assert kwargs == {'local_files_only': True, 'trust_remote_code': False}
        calls.append('tokenizer')
        return object()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=model_load),
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load)))
    loaded = service.load_reranker(path)
    assert not hasattr(loaded, 'fit')
    assert calls == ['model', 'tokenizer', 'eval']
    (checkpoint / 'config.json').write_text('{"num_labels":2}')
    with pytest.raises(ValueError, match='digest mismatch'):
        service.load_reranker(path)
