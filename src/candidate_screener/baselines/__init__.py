"""The Stage 1 classical baseline — TF-IDF cosine and BM25, each over one feature.

A sibling of `data/` and `evaluation/`, not a subpackage of either: building an
artefact, scoring one, and *being* a model are three different lifetimes *(Phase 3,
deviation W11)*.

Everything here except `run` is a pure function of its arguments, so the whole
package is testable without `data/` on disk — which matters, because `data/` is
git-ignored and a fresh clone has none of it.

    uv run python -m candidate_screener.baselines.run --seed 0
    uv run python -m candidate_screener.baselines.run --check
"""
