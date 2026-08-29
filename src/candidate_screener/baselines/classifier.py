"""The single-feature classifier, with its two determinism guards.

The baseline turns one similarity score into one of A1's three labels through a
`LogisticRegression` over that single feature. Two things about that fit could make
a re-run differ from the committed record, and both are closed here rather than
documented and hoped for:

1. **The solver is pinned explicitly.** `random_state` is consumed only by `sag`,
   `saga` and `liblinear`; under the default `lbfgs` it is inert. Passing it without
   pinning the solver reads like a control that exists, so both are set — the seed
   is insurance against a future solver swap making it live, not a knob in use today.
2. **Non-convergence raises.** A fit truncated at `max_iter` is a function of the
   iteration cap, and the cap is not a modelling decision anyone made. A converged
   fit is reproducible; a truncated one drifts on any perturbation, which is exactly
   the drift `run --check` would then report without explaining.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

#: `class_weight="balanced"` is load-bearing: A1 is 52% No Fit, and an unweighted
#: single-feature fit collapses onto that majority and reports the floor as a result.
CLASSIFIER_CONFIG: dict = {
    "solver": "lbfgs",
    "max_iter": 5000,
    "class_weight": "balanced",
}


def fit_single_feature(scores: Sequence[float], labels: Sequence[str],
                       seed: int = 0) -> LogisticRegression:
    """Fit one logistic regression on one score column.

    `seed` is passed as `random_state` and is **inert under lbfgs** — see the module
    docstring. Raises `RuntimeError` if the solver hits `max_iter`.
    """
    X = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(labels)
    clf = LogisticRegression(random_state=seed, **CLASSIFIER_CONFIG)
    clf.fit(X, y)
    if int(np.max(clf.n_iter_)) >= CLASSIFIER_CONFIG["max_iter"]:
        raise RuntimeError(
            f"lbfgs hit max_iter={CLASSIFIER_CONFIG['max_iter']} without converging "
            f"(n_iter_={clf.n_iter_.tolist()}). The fit is a function of the iteration "
            "cap, not of the data, and is not reproducible — raise the cap or rescale "
            "the feature rather than recording this model.")
    return clf


def predict(clf: LogisticRegression, scores: Sequence[float]) -> np.ndarray:
    """Predict labels for a score column, reshaping so callers never have to."""
    return clf.predict(np.asarray(scores, dtype=float).reshape(-1, 1))
