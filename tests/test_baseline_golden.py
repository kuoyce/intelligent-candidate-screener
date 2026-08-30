"""End-to-end: does the baseline still reproduce the committed record? *(D21)*

The refit tests here are the only ones in the suite that need `data/` on disk, so
they skip — visibly, with a reason — when the leak-free split has not been built.
Everything else runs on inline synthetic corpora, which is what makes `uv run pytest`
meaningful on a fresh clone.

The assertions **about the committed record itself** are deliberately not skipped:
`baseline-metrics.json` is in git, so a fresh clone has it, and what it claims about
the split and the floor is checkable without refitting anything.
"""
from __future__ import annotations

import json

import pytest

from candidate_screener.baselines import run

#: Applied to the tests that refit, not to the whole module — see the docstring.
needs_data = pytest.mark.skipif(
    not (run.FIT_SPLIT / "test.parquet").exists(),
    reason=f"{run.FIT_SPLIT} not built — run `python -m candidate_screener.data.build "
           "--task fit-split --seed 0`. Only the refit tests need it.")


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return run.metrics_document(run.build(seed=0))


@pytest.fixture(scope="module")
def committed() -> dict:
    from candidate_screener.config import BASELINE_METRICS
    if not BASELINE_METRICS.exists():
        pytest.fail(f"{BASELINE_METRICS} is missing — it is committed, so this means "
                    "the record was deleted rather than that the build was skipped.")
    return json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))


@needs_data
def test_baseline_reproduces_the_committed_record(rebuilt, committed):
    failures = [message for ok, message in run.check(rebuilt, committed) if not ok]
    assert not failures, "\n".join(failures)


@needs_data
def test_check_fails_when_a_metric_is_perturbed(rebuilt, committed):
    """A check that cannot fail is not a check.

    Perturb one accuracy by 1e-6 — far below anything a human would notice in a
    report, far above `FLOAT_TOLERANCE` — and the diff must catch it.
    """
    tampered = json.loads(json.dumps(committed))
    tampered["models"]["tfidf"]["accuracy"] += 1e-6
    assert any(not ok for ok, _ in run.check(rebuilt, tampered))


@needs_data
def test_check_fails_when_a_confusion_count_moves(rebuilt, committed):
    """Confusion-matrix integers are compared exactly *(Q23)* — a single row's
    predicted class flipping is the failure mode that actually matters."""
    tampered = json.loads(json.dumps(committed))
    tampered["models"]["bm25"]["confusion"][0][0] += 1
    assert any(not ok for ok, _ in run.check(rebuilt, tampered))


@needs_data
def test_environment_drift_alone_does_not_fail_the_check(rebuilt, committed):
    """`environment` explains a failure; it is not itself an assertion.

    A `uv lock` bump must not fail the check on its own — otherwise every re-lock is
    a false alarm and the check stops being read.
    """
    tampered = json.loads(json.dumps(committed))
    tampered["environment"]["numpy"] = "0.0.0-not-a-real-version"
    assert all(ok for ok, _ in run.check(rebuilt, tampered))


def test_the_split_is_the_leak_free_one(committed):
    """The most consequential single fact about every number in the record *(Q19)*."""
    assert committed["split"]["source"] == "processed/fit"
    assert committed["split"]["train"]["pairs"] == 3990
    assert committed["split"]["test"]["pairs"] == 659


def test_both_models_beat_the_floor_on_macro_f1_and_not_on_accuracy(committed):
    """The measured position, pinned so it is read as a finding and not a bug.

    One linear threshold over three overlapping, imbalanced classes does exactly
    this. If a future change moves it, that is a result worth stopping on.
    """
    floor = committed["floor"]
    for name in run.MODELS:
        model = committed["models"][name]
        assert model["macro_f1"] > floor["macro_f1"]
        assert model["accuracy"] < floor["accuracy"]
