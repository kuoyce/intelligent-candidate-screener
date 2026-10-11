"""C1: the JD title -> one role family (plan 2026-10-07-shortlist-recall, 01 §2 C1; D54).

The recruiter types the job title (`jd_title`, the A2 JD `Position` as given, Q-R2). It is
mapped to one of P3's 60 role families in two steps:

1. **Lookup.** `title_profile.normalise_title(title)` in P3's alias table (518 normalised
   train titles, each with its modal family; fit part only).
2. **Title classifier**, where the lookup misses: hashed word 1-2-grams and character
   2-4-grams (within word boundaries) of the lower-cased title -> logistic regression,
   trained on the **fit part's** JD and CV `Position` strings with their oracle families
   (P3's split; the dev set and the final set are outside it). Distinct (title, family)
   rows are weighted by their count.

`jd_family(titles)` returns the family, its probability (the classifier's; 1.0 for a
lookup hit) and the source. The title is the only input; JD text is never read here.

Never reads `a2-final-labels.csv` or `judgements.csv` (test).
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.extraction.title_profile import normalise_title

CACHE = PROCESSED / "extraction" / "jd-title"
SEED = 0
HASH_FEATURES = 2 ** 18
LR_C = 10.0
CONFIG = {"lookup": "normalise_title -> P3 alias table",
          "classifier": {"input": "lower-cased title as given",
                         "features": "HashingVectorizer word 1-2 + char_wb 2-4, 2^18 each, "
                                     "alternate_sign=False, l2 rows",
                         "lr": {"C": LR_C, "solver": "saga", "max_iter": 50, "tol": 1e-3,
                                "random_state": SEED},
                         "train": "fit part JD + CV Position, distinct (title, family), "
                                  "weight = count"}}
MODULE = "candidate_screener.extraction.jd_title"


def _blank(v: object) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == ""


def _clean(title: object) -> str:
    return "" if _blank(title) else " ".join(str(title).lower().split())


def _vectorise(titles: Sequence[str]):
    from scipy import sparse
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize
    word = HashingVectorizer(ngram_range=(1, 2), n_features=HASH_FEATURES, alternate_sign=False,
                             norm=None, dtype=np.float32)
    char = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=HASH_FEATURES,
                             alternate_sign=False, norm=None, dtype=np.float32)
    return normalize(sparse.hstack([word.transform(titles), char.transform(titles)]).tocsr())


@dataclass
class TitleModel:
    labels: tuple[str, ...]
    aliases: dict[str, str]
    clf: object
    fingerprint: str

    def proba(self, titles: Sequence[str]) -> np.ndarray:
        """(n, len(labels)) classifier probabilities; a class it never saw gets 0."""
        out = np.zeros((len(titles), len(self.labels)))
        if not len(titles):
            return out
        p = self.clf.predict_proba(_vectorise([_clean(t) for t in titles]))
        out[:, [self.labels.index(c) for c in self.clf.classes_]] = p
        return out


def fit_title_classifier(titles: Sequence[str], families: Sequence[str]):
    """Logistic regression on distinct (title, family) rows, weighted by count. Pure."""
    from sklearn.linear_model import LogisticRegression
    df = pd.DataFrame({"t": [_clean(t) for t in titles], "y": list(families)})
    df = df[(df.t != "") & df.y.notna()]
    agg = df.groupby(["t", "y"], sort=True).size().reset_index(name="w")
    clf = LogisticRegression(C=LR_C, solver="saga", max_iter=50, tol=1e-3, random_state=SEED)
    clf.fit(_vectorise(agg.t.tolist()), agg.y.to_numpy(), sample_weight=agg.w.to_numpy())
    return clf


def jd_family(titles: Sequence[object], model: TitleModel | None = None) -> pd.DataFrame:
    """Per title: `family`, `p` (its probability), `source` (`lookup` | `classifier` |
    `none` for an empty title, family None)."""
    model = model or load_title_model()
    norm = [normalise_title(t) for t in titles]
    hit = [model.aliases.get(n) if n else None for n in norm]
    miss = [i for i, (h, t) in enumerate(zip(hit, titles)) if h is None and not _blank(t)]
    proba = model.proba([titles[i] for i in miss])
    fam, p, src = list(hit), [1.0 if h else np.nan for h in hit], [
        "lookup" if h else "none" for h in hit]
    for row, i in enumerate(miss):
        j = int(np.argmax(proba[row]))
        fam[i], p[i], src[i] = model.labels[j], float(proba[row, j]), "classifier"
    return pd.DataFrame({"family": pd.Series(fam, dtype=object), "p": p, "source": src})


# --- training and cache -----------------------------------------------------------------------

def train(docs: Mapping[str, pd.DataFrame] | None = None, prep: Mapping | None = None
          ) -> TitleModel:
    """Fit on the P3 fit part (both sides' `Position` and oracle family) and cache it."""
    from candidate_screener.extraction import role_family as rf
    if docs is None or prep is None:
        docs = rf.load_train_documents()
        prep = rf.prepare(docs)
    role = rf.load_role_model()
    titles, fams = [], []
    for s in rf.SIDES:
        fit = prep["parts"][s]["fit"]
        titles += docs[s].Position.to_numpy()[fit].tolist()
        fams += prep["y"][s].to_numpy()[fit].tolist()
    clf = fit_title_classifier(titles, fams)
    model = TitleModel(tuple(role.labels), dict(role.aliases), clf, fingerprint(role.fingerprint))
    save(model)
    return model


def fingerprint(role_fp: str) -> str:
    blob = json.dumps({"config": CONFIG, "role_model": role_fp}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def save(model: TitleModel) -> None:
    if type(model).__module__ != MODULE:
        raise TypeError(f"{type(model).__module__}.TitleModel would not unpickle elsewhere")
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / f"title-model-{model.fingerprint}.pkl").open("wb") as fh:
        pickle.dump(model, fh)


_LOADED: dict[str, TitleModel] = {}


def load_title_model() -> TitleModel:
    """The cached title model for the current role model; trained on first use (~1 min)."""
    from candidate_screener.extraction import role_family as rf
    fp = fingerprint(rf.load_role_model().fingerprint)
    if fp not in _LOADED:
        path = CACHE / f"title-model-{fp}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                _LOADED[fp] = pickle.load(fh)
        else:
            import importlib
            _LOADED[fp] = importlib.import_module(MODULE).train()
    return _LOADED[fp]
