"""E2: job title -> role family, from text only (plan 2026-10-06, P3).

Four parts (02-design-spec §3, 03-work-plan P3; the record is `12-p3-role-family.md`):

1. **The label** `role_family(keyword, position)` (Q-B1, user decision 6 Oct 2026, "T-A,
   split generic"). The label is the A2 `Primary Keyword`, except the generic keywords
   `Data Science`, `Lead` and `Other`, which are split by the document's normalised
   `Position` (`title_profile.normalise_title`, T-B1) into the families with at least
   `MIN_SUPPORT` train JDs **and** train CVs under that keyword. A document of a generic
   keyword whose title is in no family gets the residual `<keyword>/other`. A split family
   whose title is a specific keyword's own title (that keyword's modal normalised title,
   and the keyword holds `MERGE_PURITY` of the train documents carrying that title under a
   non-generic keyword) is **merged** into that keyword, so `Data Science` + `data engineer`
   is `Data Engineer`, not a near-duplicate class, while `data scientist` (mostly tagged
   `Python` outside `Data Science`, but not Python's own title) stays a family. A keyword
   with fewer than `MIN_SUPPORT` train JDs or CVs (the four JD-only singletons) is folded
   into `Other` before the split. JDs and CVs therefore share one label space.
   **The label reads metadata**: it is for training labels and the oracle only, never for
   inference (D53).
2. **Title spans** `title_spans(text)`: the shared normalisation and n-gram find-in-text of
   `title_profile`, against an alias table of normalised A2-train `Position` strings, each
   mapped to its modal role family. Spans carry offsets into the original text (UI
   evidence) and feed the classifier as indicator features.
3. **The classifier**: one model per side (JD, CV), text -> role family, trained on the A2
   train region only (D19), on `text_v2` (no `Position` in CV text, D53). Two candidates are
   fixed in advance and selected on a document-disjoint, seeded validation split:
   (a) hashed word 1-2-gram TF-IDF + span indicators -> logistic regression;
   (b) frozen MiniLM chunk-mean embeddings + span indicators -> logistic regression.
   Probabilities are temperature-calibrated on a separate validation draw.
4. **Measurement** on the final-set documents (60 JDs, the 600 pairs' CVs): against the
   oracle label, and pair-level same-family agreement. Measurement only (D46); this module
   never reads `a2-final-labels.csv` or `judgements.csv`.

    uv run python -m candidate_screener.extraction.role_family --train      # fit + record
    uv run python -m candidate_screener.extraction.role_family --evaluate   # record from cache
    uv run python -m candidate_screener.extraction.role_family --check      # refit + diff
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import math
import pickle
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT, PROCESSED
from candidate_screener.extraction.title_profile import (_PRE, _TOKEN, MAX_NGRAM,
                                                         canonical_tokens, normalise_title)

RECORD = OUTPUT / "extraction" / "role-family-metrics.json"
CACHE = PROCESSED / "extraction" / "role-family"

# --- 1. the label ----------------------------------------------------------------------

GENERIC_KEYWORDS = ("Data Science", "Lead", "Other")
#: Where an under-supported keyword goes before the split.
FOLD_INTO = "Other"
#: *m*: a title is a family under a generic keyword with at least this many train JDs and
#: train CVs carrying (keyword, title); the same bar admits a keyword as a class. P2's
#: headline m (11-p2-data-checks §1, §3.2).
MIN_SUPPORT = 20
#: A split family is merged into a specific keyword when at least this share of the train
#: documents with that normalised title and a non-generic keyword carry that keyword.
MERGE_PURITY = 0.5
RESIDUAL = "other"
SEP = "/"

#: The label space, derived from the A2 train region by `derive_label_space` and committed
#: here so the label function needs no data. `--check` re-derives and diffs it.
KEYWORD_CLASSES: tuple[str, ...] = (
    ".NET", "Android", "Artist", "Business Analyst", "C++", "Data Analyst", "Data Engineer",
    "Design", "DevOps", "Flutter", "Golang", "HR", "Java", "JavaScript", "Lead Generation",
    "Marketing", "Node.js", "PHP", "Product Manager", "Project Manager", "Python", "QA",
    "QA Automation", "Recruiter", "Ruby", "Rust", "SEO", "SQL", "Sales", "Salesforce", "Scala",
    "Scrum Master", "Security", "Support", "Sysadmin", "Technical Writing", "Unity", "iOS",
)
SPLIT_FAMILIES: Mapping[str, Mapping[str, str]] = {
    "Data Science": {
        "computer vision engineer": "Data Science/computer vision engineer",
        "data analyst": "Data Analyst",                     # merged: Data Analyst's own title
        "data engineer": "Data Engineer",                   # merged
        "data scientist": "Data Science/data scientist",
        "ml engineer": "Data Science/ml engineer",
    },
    "Lead": {
        "architect": "Lead/architect",
        "cto": "Lead/cto",
        "engineering manager": "Lead/engineering manager",
        "software architect": "Lead/software architect",
        "solution architect": "Lead/solution architect",
    },
    "Other": {
        "account manager": "Other/account manager",
        "content manager": "Other/content manager",
        "content writer": "Other/content writer",
        "copywriter": "Other/copywriter",
        "data analyst": "Data Analyst",                     # merged
        "finance manager": "Other/finance manager",
        "financial analyst": "Other/financial analyst",
        "financial manager": "Other/financial manager",
        "frontend developer": "JavaScript",                 # merged
        "lawyer": "Other/lawyer",
        "legal counsel": "Other/legal counsel",
        "operations manager": "Other/operations manager",
        "software developer": "Other/software developer",
        "technical writer": "Technical Writing",            # merged
    },
}


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == ""


def role_family(keyword: object, position: object, *,
                keyword_classes: Iterable[str] | None = None,
                split_families: Mapping[str, Mapping[str, str]] | None = None) -> str | None:
    """The role-family **label** of an A2 document (training labels and the oracle only).

    None when the keyword is missing (30 train CVs): such a document has no label.
    """
    if _missing(keyword):
        return None
    classes = set(KEYWORD_CLASSES if keyword_classes is None else keyword_classes)
    split = SPLIT_FAMILIES if split_families is None else split_families
    kw = str(keyword)
    if kw not in classes and kw not in GENERIC_KEYWORDS:
        kw = FOLD_INTO
    if kw not in GENERIC_KEYWORDS:
        return kw
    return split.get(kw, {}).get(normalise_title(position), f"{kw}{SEP}{RESIDUAL}")


def label_space(keyword_classes: Iterable[str] | None = None,
                split_families: Mapping[str, Mapping[str, str]] | None = None) -> tuple[str, ...]:
    """Every label `role_family` can return, sorted: the one space both sides share."""
    classes = set(KEYWORD_CLASSES if keyword_classes is None else keyword_classes)
    split = SPLIT_FAMILIES if split_families is None else split_families
    out = classes | {f"{g}{SEP}{RESIDUAL}" for g in GENERIC_KEYWORDS}
    out |= {lab for fams in split.values() for lab in fams.values()}
    return tuple(sorted(out))


def derive_label_space(jd: pd.DataFrame, cv: pd.DataFrame, m: int = MIN_SUPPORT,
                       purity: float = MERGE_PURITY) -> dict:
    """Keyword classes and split families from train-region `Primary Keyword` + `Position`.

    Returns `{"keyword_classes": [...], "split_families": {generic: {title: label}},
    "merges": {...}, "folded": [...]}`. Pure: the caller filters to the train region.
    """
    jk, ck = jd["Primary Keyword"], cv["Primary Keyword"]
    jn, cn = jd.Position.map(normalise_title), cv.Position.map(normalise_title)
    jvc, cvc = jk.value_counts(), ck.value_counts()
    keywords = sorted(set(jvc.index) | set(cvc.index))
    classes = sorted(k for k in keywords if k not in GENERIC_KEYWORDS
                     and jvc.get(k, 0) >= m and cvc.get(k, 0) >= m)
    folded = sorted(k for k in keywords if k not in GENERIC_KEYWORDS and k not in classes)

    def fold(s: pd.Series) -> pd.Series:
        return s.where(~s.isin(folded), FOLD_INTO)
    jk, ck = fold(jk), fold(ck)
    # Owner of each title among specific (non-generic) keywords, both sides pooled.
    spec = pd.DataFrame({"kw": pd.concat([jk, ck]), "t": pd.concat([jn, cn])})
    spec = spec[spec.kw.isin(classes) & (spec.t != "")]
    owner = spec.groupby("t").kw.agg(lambda s: s.value_counts().index[0])
    share = spec.groupby("t").kw.agg(lambda s: s.value_counts().iloc[0] / len(s))
    size = spec.groupby("t").size()
    # Each specific keyword's own (modal) title: only that title may be merged into it.
    head = spec.groupby("kw").t.agg(lambda s: s.value_counts().index[0])

    split: dict[str, dict[str, str]] = {}
    merges: dict[str, dict] = {}
    for g in GENERIC_KEYWORDS:
        a = jn[(jk == g) & (jn != "")].value_counts()
        b = cn[(ck == g) & (cn != "")].value_counts()
        fams = sorted(set(a[a >= m].index) & set(b[b >= m].index))
        split[g] = {}
        for t in fams:
            if (t in owner.index and size[t] >= m and share[t] >= purity
                    and head.get(owner[t]) == t):
                split[g][t] = str(owner[t])
                merges[f"{g}{SEP}{t}"] = {"into": str(owner[t]), "share": round(float(share[t]), 4),
                                          "n_specific": int(size[t])}
            else:
                split[g][t] = f"{g}{SEP}{t}"
    return {"keyword_classes": classes, "split_families": split, "merges": merges,
            "folded": folded}


def labels_for(df: pd.DataFrame, space: Mapping | None = None) -> pd.Series:
    """`role_family` over a frame with `Primary Keyword` and `Position`."""
    kw = dict(keyword_classes=space["keyword_classes"], split_families=space["split_families"]
              ) if space else {}
    return pd.Series([role_family(k, p, **kw) for k, p in zip(df["Primary Keyword"], df.Position)],
                     index=df.index, dtype=object)


# --- text (text_v2, D53) -----------------------------------------------------------------

#: `annotation/sample.py` CV_FIELDS minus `Position` (the annotation module is frozen,
#: D42). P1's `matching.corpus.CV_FIELDS_BY_VARIANT["v2"]` is the same list; unify later.
CV_FIELDS_V2 = ("CV", "Highlights", "Moreinfo", "Looking For")
JD_FIELD = "Long Description"


def cv_text_v2(cv: pd.DataFrame) -> pd.Series:
    """CV text without `Position`, joined exactly as `annotation.sample.cv_text` joins."""
    parts = [cv[f].fillna("").astype(str).str.strip() for f in CV_FIELDS_V2]
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.str.cat(p, sep="\n\n").str.strip()
    return joined.str.replace(r"\n{3,}", "\n\n", regex=True)


def jd_text(jd: pd.DataFrame) -> pd.Series:
    return jd[JD_FIELD].fillna("").astype(str).str.strip()


# --- 2. title spans ----------------------------------------------------------------------

@dataclass(frozen=True)
class TitleSpan:
    start: int          # offsets into the original text
    end: int
    text: str           # the original surface string
    title: str          # the normalised alias that matched
    family: str         # its role family


def _lowered_with_map(text: str) -> tuple[str, list[int]]:
    """`title_profile`'s pre-substituted lower-case text, with each character's origin."""
    chars: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text):
        low = ch.lower()
        chars.extend(low)
        origin.extend([i] * len(low))
    s = "".join(chars)
    for pat, rep in _PRE:
        out, omap, last = [], [], 0
        for mt in pat.finditer(s):
            out.append(s[last:mt.start()])
            omap.extend(origin[last:mt.start()])
            out.append(rep)
            omap.extend([origin[mt.start()]] * len(rep))
            last = mt.end()
        out.append(s[last:])
        omap.extend(origin[last:])
        s, origin = "".join(out), omap
    return s, origin


def title_spans(text: object, aliases: Mapping[str, str] | None = None,
                max_ngram: int = MAX_NGRAM) -> list[TitleSpan]:
    """Alias titles found in `text`, leftmost-longest and non-overlapping, with offsets.

    `aliases` maps a normalised title to its role family; by default the cached model's
    table (`load_role_model().aliases`). Matching is on `title_profile`'s token rule, so a
    span is token-bounded (`java` never matches inside `javascript`).
    """
    if aliases is None:
        aliases = load_role_model().aliases
    if _missing(text):
        return []
    text = str(text)
    memo = _span_memo(aliases)
    key = (text, max_ngram)
    if key not in memo:
        if len(memo) >= SPAN_MEMO_MAX:
            memo.clear()
        memo[key] = tuple(_title_spans(text, aliases, max_ngram))
    return list(memo[key])


#: Per-text memo of `title_spans` (plan 2026-10-08, W5), one table per alias-table object
#: (the object is held and compared by identity, so a new model never reads another
#: model's spans). Spans are discrete (offsets, strings) and `TitleSpan` is frozen; callers
#: get a fresh list. Bounded: a table is cleared when it reaches `SPAN_MEMO_MAX` texts.
SPAN_MEMO_MAX = 200_000
_SPANS: dict[int, tuple[Mapping[str, str], dict[tuple[str, int], tuple[TitleSpan, ...]]]] = {}


def _span_memo(aliases: Mapping[str, str]) -> dict[tuple[str, int], tuple[TitleSpan, ...]]:
    hit = _SPANS.get(id(aliases))
    if hit is None or hit[0] is not aliases:
        _SPANS[id(aliases)] = hit = (aliases, {})
    return hit[1]


def _title_spans(text: str, aliases: Mapping[str, str], max_ngram: int) -> list[TitleSpan]:
    """`title_spans` without the memo: `text` is a string, `aliases` resolved."""
    toks = canonical_tokens(text)
    if not toks:
        return []
    longest = _first_token_index(aliases)
    starts = [o for _, o in toks]
    words = [t for t, _ in toks]
    found: list[tuple[int, int, str]] = []                 # (first token, n, alias)
    i = 0
    while i < len(words):
        top = min(longest.get(words[i], 0), max_ngram, len(words) - i)
        hit = next(((n, g) for n in range(top, 0, -1)
                    if (g := " ".join(words[i:i + n])) in aliases), None)
        if hit is None:
            i += 1
            continue
        found.append((i, *hit))
        i += hit[0]
    if not found:
        return []
    s, origin = _lowered_with_map(text)
    raw_starts, raw_ends = [], []
    for mt in _TOKEN.finditer(s):
        raw_starts.append(mt.start())
        raw_ends.append(mt.end())

    def end_of(j: int) -> int:
        """End (in `s`) of canonical token j: the last raw token before token j+1."""
        nxt = starts[j + 1] if j + 1 < len(starts) else len(s) + 1
        return raw_ends[bisect.bisect_left(raw_starts, nxt) - 1]

    out: list[TitleSpan] = []
    for i, n, g in found:
        a, b = origin[starts[i]], origin[end_of(i + n - 1) - 1] + 1
        out.append(TitleSpan(a, b, text[a:b], g, aliases[g]))
    return out


_INDEX: dict[int, tuple[Mapping[str, str], dict[str, int]]] = {}


def _first_token_index(aliases: Mapping[str, str]) -> dict[str, int]:
    """First token -> longest alias length starting with it (cached per table object)."""
    hit = _INDEX.get(id(aliases))
    if hit is None or hit[0] is not aliases:
        index: dict[str, int] = {}
        for a in aliases:
            w = a.split()
            if w:
                index[w[0]] = max(index.get(w[0], 0), len(w))
        _INDEX[id(aliases)] = hit = (aliases, index)
    return hit[1]


def build_alias_table(titles: pd.Series, labels: pd.Series, m: int = MIN_SUPPORT,
                      purity: float = 0.5, single_token_purity: float = 0.8) -> dict[str, str]:
    """Normalised train `Position` -> modal role family.

    Kept when the title has at least `m` documents and its modal family holds `purity` of
    them; a one-token title (`cto`, `copywriter`, but also `developer`) must reach
    `single_token_purity`, since a single word is weaker evidence of a title.
    """
    df = pd.DataFrame({"t": titles.to_numpy(), "y": labels.to_numpy()})
    df = df[(df.t != "") & df.y.notna()]
    out: dict[str, str] = {}
    for t, grp in df.groupby("t", sort=True):
        if len(grp) < m:
            continue
        vc = grp.y.value_counts()
        share = vc.iloc[0] / len(grp)
        bar = single_token_purity if len(t.split()) == 1 else purity
        if share >= bar:
            out[str(t)] = str(sorted(vc[vc == vc.iloc[0]].index)[0])
    return out


def span_features(texts: Sequence[str], aliases: Mapping[str, str],
                  labels: Sequence[str]):
    """Sparse indicators per document: [family k named anywhere | first span is family k]."""
    from scipy import sparse
    index = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    rows, cols = [], []
    for r, t in enumerate(texts):
        spans = title_spans(t, aliases)
        seen = {index[s.family] for s in spans if s.family in index}
        for c in seen:
            rows.append(r)
            cols.append(c)
        if spans and spans[0].family in index:
            rows.append(r)
            cols.append(k + index[spans[0].family])
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), 2 * k))


# --- 3. classifier -----------------------------------------------------------------------

SIDES = ("jd", "cv")
CANDIDATES = ("tfidf", "minilm")
SEED = 0
VAL_SHARE = 0.10
#: Documents per side in the validation draw both candidates are selected on, and in the
#: separate draw the temperature is fitted on.
N_SELECT = 2_500
N_CALIB = 1_000
#: MiniLM is ~20 chunks/s on this CPU, so (b) trains on a seeded subsample of the fit
#: part, with a per-class floor so no class is absent.
MINILM_TRAIN_N = 6_000
MINILM_CLASS_FLOOR = 15
HASH_FEATURES = 2 ** 18
LR_C = 10.0
LR_MAX_ITER = 30
TOP_K = 3
ECE_BINS = 15

CONFIG = {
    "label": {"generic": list(GENERIC_KEYWORDS), "fold_into": FOLD_INTO, "m": MIN_SUPPORT,
              "merge_purity": MERGE_PURITY, "residual": RESIDUAL,
              "normaliser": "extraction.title_profile.normalise_title"},
    "text": {"jd": [JD_FIELD], "cv": list(CV_FIELDS_V2), "variant": "v2 (D53)"},
    "aliases": {"source": "normalised A2-train Position (fit part)", "m": MIN_SUPPORT,
                "purity": 0.5, "single_token_purity": 0.8, "max_ngram": MAX_NGRAM},
    "split": {"seed": SEED, "val_share": VAL_SHARE, "group": "sha256 of whitespace-normalised text",
              "n_select": N_SELECT, "n_calib": N_CALIB},
    "tfidf": {"vectoriser": "HashingVectorizer word 1-2 grams, alternate_sign=False",
              "n_features": HASH_FEATURES, "idf": "TfidfTransformer sublinear_tf, fit part",
              "span_features": True, "lr": {"C": LR_C, "solver": "saga", "max_iter": LR_MAX_ITER,
                                            "random_state": SEED}},
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2", "pooling": "sbert.Encoder chunk mean (800 chars)",
               "train_n": MINILM_TRAIN_N, "class_floor": MINILM_CLASS_FLOOR, "span_features": True,
               "lr": {"C": LR_C, "solver": "lbfgs", "max_iter": 1000}},
    "calibration": "temperature on the calibration draw, NLL",
    "selection": "higher mean of the JD and CV validation macro-F1",
}


def _fingerprint(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _text_group(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest()


def split_by_document(texts: Sequence[str], seed: int = SEED, val_share: float = VAL_SHARE
                      ) -> np.ndarray:
    """True for validation rows. Identical texts (after whitespace and case folding) share
    a side, so a re-posted JD cannot sit on both."""
    groups = np.array([_text_group(t) for t in texts])
    uniq = np.unique(groups)                                     # sorted: seed means one thing
    perm = np.random.default_rng(seed).permutation(len(uniq))
    val = set(uniq[perm[: int(round(val_share * len(uniq)))]])
    return np.array([g in val for g in groups])


def seeded_draw(idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    idx = np.sort(idx)
    return np.sort(np.random.default_rng(seed).permutation(idx)[:n])


def floor_sample(y: np.ndarray, idx: np.ndarray, n: int, floor: int, seed: int) -> np.ndarray:
    """`n` seeded rows of `idx`, topped up so each class has min(floor, its size) rows."""
    rng = np.random.default_rng(seed)
    idx = np.sort(idx)
    take = set(rng.permutation(idx)[:n].tolist())
    for c in sorted(set(y[idx])):
        members = idx[y[idx] == c]
        have = sum(1 for i in members if i in take)
        if have < min(floor, len(members)):
            rest = np.array([i for i in members if i not in take])
            take |= set(rng.permutation(rest)[: min(floor, len(members)) - have].tolist())
    return np.array(sorted(take))


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """The temperature minimising NLL (argmax, so accuracy, is unchanged)."""
    from scipy.optimize import minimize_scalar
    rows = np.arange(len(y))

    def nll(log_t: float) -> float:
        p = softmax(logits, math.exp(log_t))
        return float(-np.log(np.clip(p[rows, y], 1e-12, None)).mean())
    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded", options={"xatol": 1e-4})
    return round(float(math.exp(res.x)), 4)


@dataclass
class SideModel:
    """One side's fitted classifier over the full label space."""
    kind: str
    labels: tuple[str, ...]
    clf: object
    idf: object = None
    temperature: float = 1.0

    def _features(self, texts: Sequence[str], aliases: Mapping[str, str], encoder=None):
        from scipy import sparse
        spans = span_features(texts, aliases, self.labels)
        if self.kind == "tfidf":
            return sparse.hstack([self.idf.transform(_hasher().transform(texts)), spans]).tocsr()
        emb = encoder.encode(texts)
        return np.hstack([emb, spans.toarray()])

    def logits(self, texts: Sequence[str], aliases: Mapping[str, str], encoder=None) -> np.ndarray:
        """Decision values over `self.labels` (a class unseen in training gets -inf)."""
        raw = self.clf.decision_function(self._features(texts, aliases, encoder))
        if raw.ndim == 1:                    # two classes: sklearn returns one logit
            raw = np.column_stack([np.zeros_like(raw), raw])
        out = np.full((len(texts), len(self.labels)), -np.inf)
        cols = [self.labels.index(c) for c in self.clf.classes_]
        out[:, cols] = raw
        return out

    def proba(self, texts: Sequence[str], aliases: Mapping[str, str], encoder=None) -> np.ndarray:
        return softmax(self.logits(texts, aliases, encoder), self.temperature)


@dataclass
class RoleModel:
    """Both sides, the alias table and the label space: what `load_role_model` returns."""
    kind: str
    labels: tuple[str, ...]
    aliases: dict[str, str]
    sides: dict[str, SideModel]
    fingerprint: str
    _encoder: object = field(default=None, repr=False)

    def encoder(self):
        if self.kind != "minilm":
            return None
        if self._encoder is None:
            from candidate_screener.matching.neural import MODEL_ID
            from candidate_screener.matching.sbert import Encoder
            self._encoder = Encoder(MODEL_ID, "minilm-role-family")
        return self._encoder


def _hasher():
    from sklearn.feature_extraction.text import HashingVectorizer
    return HashingVectorizer(ngram_range=(1, 2), n_features=HASH_FEATURES,
                             alternate_sign=False, norm=None, dtype=np.float32)


def fit_tfidf(texts: Sequence[str], y: Sequence[str], aliases: Mapping[str, str],
              labels: tuple[str, ...]) -> SideModel:
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LogisticRegression
    counts = _hasher().transform(texts)
    idf = TfidfTransformer(sublinear_tf=True).fit(counts)
    x = sparse.hstack([idf.transform(counts), span_features(texts, aliases, labels)]).tocsr()
    clf = LogisticRegression(C=LR_C, solver="saga", max_iter=LR_MAX_ITER, tol=1e-3,
                             random_state=SEED).fit(x, np.asarray(y))
    return SideModel("tfidf", labels, clf, idf)


def fit_minilm(texts: Sequence[str], y: Sequence[str], aliases: Mapping[str, str],
               labels: tuple[str, ...], encoder) -> SideModel:
    from sklearn.linear_model import LogisticRegression
    x = np.hstack([encoder.encode(texts), span_features(texts, aliases, labels).toarray()])
    clf = LogisticRegression(C=LR_C, solver="lbfgs", max_iter=1000).fit(x, np.asarray(y))
    return SideModel("minilm", labels, clf)


# --- metrics -----------------------------------------------------------------------------

def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over the classes present in `y` (integer codes)."""
    tp = np.bincount(y[y == pred], minlength=n_classes).astype(float)
    pos = np.bincount(y, minlength=n_classes).astype(float)
    called = np.bincount(pred, minlength=n_classes).astype(float)
    present = pos > 0
    den = pos + called
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    return float(f1[present].mean()) if present.any() else float("nan")


def topk_hits(proba: np.ndarray, y: np.ndarray, k: int = TOP_K) -> np.ndarray:
    top = np.argsort(-proba, axis=1, kind="stable")[:, :k]
    return (top == y[:, None]).any(axis=1)


def ece(proba: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    conf = proba.max(axis=1)
    hit = proba.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def classification_block(proba: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                         n_boot: int = 1_000, seed: int = SEED) -> dict:
    """Accuracy, macro-F1, top-3, with 95% percentile CIs.

    The bootstrap resamples `groups` (JDs, for final-set CVs nested in JDs) or, with no
    groups, documents.
    """
    pred = proba.argmax(axis=1)
    hit3 = topk_hits(proba, y)
    k = proba.shape[1]
    stats = {"accuracy": lambda r: float((pred[r] == y[r]).mean()),
             "macro_f1": lambda r: macro_f1(y[r], pred[r], k),
             "top3_accuracy": lambda r: float(hit3[r].mean())}
    blocks = ([np.flatnonzero(groups == g) for g in np.unique(groups)] if groups is not None
              else [np.array([i]) for i in range(len(y))])
    rng = np.random.default_rng(seed)
    draws = {s: [] for s in stats}
    for _ in range(n_boot):
        rows = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        for s, f in stats.items():
            draws[s].append(f(rows))
    allrows = np.arange(len(y))
    out = {"n": int(len(y)), "n_classes_present": int(len(np.unique(y)))}
    if groups is not None:
        out["n_groups"] = int(len(blocks))
    for s, f in stats.items():
        lo, hi = np.percentile(draws[s], [2.5, 97.5])
        out[s] = {"value": round(f(allrows), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


# --- data --------------------------------------------------------------------------------

def load_train_documents() -> dict[str, pd.DataFrame]:
    """A2 train-region JDs and CVs (D19) with `text_v2`, label inputs and nothing else."""
    from candidate_screener.data.a2_finetune import load_train_region
    from candidate_screener.data.profile import load_split
    jt, ct = load_train_region()
    jd = load_split("djinni-jd", "train")
    cv = load_split("djinni-cv", "train")
    jd = jd[jd["id"].isin(jt) & (jd["Long Description_lang"] == "en")].copy()
    cv = cv[cv["id"].isin(ct) & (cv["CV_lang"] == "en")].copy()
    jd["text"] = jd_text(jd)
    cv["text"] = cv_text_v2(cv)
    keep = ["id", "text", "Primary Keyword", "Position"]
    jd = jd[jd.text.str.len() > 0][keep].sort_values("id").reset_index(drop=True)
    cv = cv[cv.text.str.len() > 0][keep].sort_values("id").reset_index(drop=True)
    return {"jd": jd, "cv": cv}


def load_final_documents() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The 600 pairs (ids and stratum only, no labels) and their JDs and CVs, `text_v2`."""
    from candidate_screener.annotation.sample import load_english
    from candidate_screener.evaluation.final_set import PAIRS_CSV
    pairs = pd.read_csv(PAIRS_CSV, usecols=["pair_id", "stratum", "jd_id", "cv_id"])
    jd, cv = load_english()
    jd = jd[jd["id"].isin(pairs.jd_id)].copy()
    cv = cv[cv["id"].isin(pairs.cv_id)].copy()
    jd["text"], cv["text"] = jd_text(jd), cv_text_v2(cv)
    keep = ["id", "text", "Primary Keyword", "Position"]
    return pairs, {"jd": jd[keep].set_index("id"), "cv": cv[keep].set_index("id")}


# --- training ----------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(docs: dict[str, pd.DataFrame]) -> dict:
    """Labels, the document split and the draws; the label space is re-derived here."""
    fit_mask = {s: ~split_by_document(list(docs[s].text), SEED + i) for i, s in enumerate(SIDES)}
    # The label space and the alias table see the fit part only.
    space = derive_label_space(docs["jd"][fit_mask["jd"]], docs["cv"][fit_mask["cv"]])
    labels = label_space(space["keyword_classes"], space["split_families"])
    y = {s: labels_for(docs[s], space) for s in SIDES}
    titles = {s: docs[s].Position.map(normalise_title) for s in SIDES}
    aliases = build_alias_table(
        pd.concat([titles[s][fit_mask[s]] for s in SIDES], ignore_index=True),
        pd.concat([y[s][fit_mask[s]] for s in SIDES], ignore_index=True))
    parts = {}
    for i, s in enumerate(SIDES):
        has = y[s].notna().to_numpy()
        fit_idx = np.flatnonzero(fit_mask[s] & has)
        val_idx = np.flatnonzero(~fit_mask[s] & has)
        select = seeded_draw(val_idx, N_SELECT, SEED + 10 + i)
        calib = seeded_draw(np.setdiff1d(val_idx, select), N_CALIB, SEED + 20 + i)
        codes = pd.Categorical(y[s], categories=labels).codes
        mini = floor_sample(codes, fit_idx, MINILM_TRAIN_N, MINILM_CLASS_FLOOR, SEED + 30 + i)
        parts[s] = {"fit": fit_idx, "val": val_idx, "select": select, "calib": calib,
                    "minilm_train": mini, "unlabelled": int((~has).sum())}
    return {"space": space, "labels": labels, "y": y, "aliases": aliases, "parts": parts,
            "titles": titles, "fit_mask": fit_mask}


def train_all(docs: dict[str, pd.DataFrame], prep: dict) -> dict:
    """Both candidates, both sides; validation figures; the selected model, calibrated."""
    from candidate_screener.matching.neural import MODEL_ID
    from candidate_screener.matching.sbert import Encoder
    labels, aliases, y = prep["labels"], prep["aliases"], prep["y"]
    lab_index = {lab: i for i, lab in enumerate(labels)}
    encoder = Encoder(MODEL_ID, "minilm-role-family")
    models: dict[str, dict[str, SideModel]] = {c: {} for c in CANDIDATES}
    validation: dict = {c: {} for c in CANDIDATES}
    timing: dict = {}
    for s in SIDES:
        p, texts, ys = prep["parts"][s], docs[s].text.to_numpy(), y[s].to_numpy()
        sel = p["select"]
        y_sel = np.array([lab_index[v] for v in ys[sel]])
        t0 = time.time()
        _log(f"{s}: tfidf on {len(p['fit'])} documents")
        models["tfidf"][s] = fit_tfidf(list(texts[p["fit"]]), ys[p["fit"]], aliases, labels)
        timing[f"tfidf_{s}_fit_s"] = round(time.time() - t0)
        t0 = time.time()
        _log(f"{s}: minilm on {len(p['minilm_train'])} documents (embedding)")
        for chunk in np.array_split(p["minilm_train"], max(1, len(p["minilm_train"]) // 1000)):
            encoder.encode(list(texts[chunk]))
        encoder.encode(list(texts[sel]))
        models["minilm"][s] = fit_minilm(list(texts[p["minilm_train"]]), ys[p["minilm_train"]],
                                         aliases, labels, encoder)
        timing[f"minilm_{s}_fit_s"] = round(time.time() - t0)
        for c in CANDIDATES:
            enc = encoder if c == "minilm" else None
            logit = models[c][s].logits(list(texts[sel]), aliases, enc)
            validation[c][s] = {**classification_block(softmax(logit), y_sel),
                                "n_train": int(len(p["fit"] if c == "tfidf" else p["minilm_train"]))}
    score = {c: float(np.mean([validation[c][s]["macro_f1"]["value"] for s in SIDES]))
             for c in CANDIDATES}
    chosen = max(CANDIDATES, key=lambda c: (score[c], c == "tfidf"))
    calibration = {}
    for s in SIDES:
        p, texts, ys = prep["parts"][s], docs[s].text.to_numpy(), y[s].to_numpy()
        enc = encoder if chosen == "minilm" else None
        logit = models[chosen][s].logits(list(texts[p["calib"]]), aliases, enc)
        yc = np.array([lab_index[v] for v in ys[p["calib"]]])
        t = fit_temperature(np.where(np.isfinite(logit), logit, -1e9), yc)
        models[chosen][s].temperature = t
        sel_logit = models[chosen][s].logits(list(texts[p["select"]]), aliases, enc)
        ysel = np.array([lab_index[v] for v in ys[p["select"]]])
        calibration[s] = {"temperature": t, "n_calib": int(len(yc)),
                          "select_ece_before": round(ece(softmax(sel_logit), ysel), 4),
                          "select_ece_after": round(ece(softmax(sel_logit, t), ysel), 4)}
    return {"models": models, "validation": validation, "selection_score": score,
            "chosen": chosen, "calibration": calibration, "timing": timing}


def config_fingerprint(prep: dict) -> str:
    from candidate_screener.data.a2_finetune import PARTITION_MANIFEST
    return _fingerprint({"config": CONFIG, "labels": prep["labels"],
                         "partition": hashlib.sha256(PARTITION_MANIFEST.read_bytes()).hexdigest()})


#: The importable module the pickled classes must belong to.
MODULE = "candidate_screener.extraction.role_family"


def save_model(model: RoleModel) -> None:
    """Pickle the model. Refuses classes defined in `__main__` (a `python -m` run), which
    no other process could unpickle; `main` re-imports the module to avoid that."""
    for side in model.sides.values():
        if type(side).__module__ != MODULE:
            raise TypeError(f"{type(side).__module__}.SideModel would not unpickle "
                            f"outside this run; train through {MODULE}.main")
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"role-model-{model.fingerprint}.pkl"
    with path.open("wb") as fh:
        pickle.dump({"kind": model.kind, "labels": model.labels, "aliases": model.aliases,
                     "sides": model.sides, "fingerprint": model.fingerprint}, fh)
    (CACHE / "LATEST").write_text(model.fingerprint + "\n", encoding="utf-8")


_LOADED: dict[str, RoleModel] = {}


def load_role_model(fingerprint: str | None = None) -> RoleModel:
    """The cached, calibrated role-family model (run `--train` first)."""
    if fingerprint is None:
        latest = CACHE / "LATEST"
        if not latest.exists():
            raise FileNotFoundError(f"no role-family model under {CACHE}: run "
                                    "`uv run python -m candidate_screener.extraction.role_family --train`")
        fingerprint = latest.read_text(encoding="utf-8").strip()
    if fingerprint not in _LOADED:
        with (CACHE / f"role-model-{fingerprint}.pkl").open("rb") as fh:
            d = pickle.load(fh)
        _LOADED[fingerprint] = RoleModel(d["kind"], tuple(d["labels"]), d["aliases"],
                                         d["sides"], d["fingerprint"])
    return _LOADED[fingerprint]


# --- 5. public API (P7) ------------------------------------------------------------------

def predict_proba(texts: Sequence[str], side: str, model: RoleModel | None = None) -> np.ndarray:
    """Calibrated role-family probabilities, columns in `model.labels` order.

    Text only: no metadata field is read. Duplicate texts are scored once.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    model = model or load_role_model()
    texts = ["" if _missing(t) else str(t) for t in texts]
    uniq = sorted(set(texts))
    if not uniq:
        return np.zeros((0, len(model.labels)))
    p = _proba_memo(model, side, uniq)
    row = {t: i for i, t in enumerate(uniq)}
    return p[[row[t] for t in texts]]


#: Call-level memo of the classifier's probabilities (plan 2026-10-08, W5): the X rungs
#: call `role_match` on the same frame once per rung. A hit needs the same model object
#: (held and compared by identity), the same side and the same sorted unique texts, so
#: only an identical batch is replayed and no row is ever computed in another batch.
PROBA_MEMO_MAX = 64
_PROBA: dict[tuple[int, str, str], tuple[RoleModel, np.ndarray]] = {}


def _proba_memo(model: RoleModel, side: str, uniq: list[str]) -> np.ndarray:
    h = hashlib.sha256()
    for t in uniq:
        h.update(hashlib.sha256(t.encode("utf-8", "surrogatepass")).digest())
    key = (id(model), side, h.hexdigest())
    hit = _PROBA.get(key)
    if hit is None or hit[0] is not model:
        if len(_PROBA) >= PROBA_MEMO_MAX:
            _PROBA.clear()
        hit = _PROBA[key] = (model, model.sides[side].proba(uniq, model.aliases,
                                                             model.encoder()))
    return hit[1]


def soft_agreement(p_jd: np.ndarray, p_cv: np.ndarray) -> np.ndarray:
    """`sum_k p_jd(k) * p_cv(k)` per row: the probability two independent draws agree."""
    return np.einsum("ij,ij->i", p_jd, p_cv)


def role_match(jd_texts: Sequence[str], cv_texts: Sequence[str],
               model: RoleModel | None = None) -> np.ndarray:
    """Soft role agreement of aligned (JD, CV) pairs, in [0, 1] (02 §3.4)."""
    if len(jd_texts) != len(cv_texts):
        raise ValueError("jd_texts and cv_texts must be aligned pairs")
    model = model or load_role_model()
    return soft_agreement(predict_proba(jd_texts, "jd", model), predict_proba(cv_texts, "cv", model))


# --- 4. final-set measurement --------------------------------------------------------------

def _auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = np.empty(len(allv))
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    ranks[order] = r
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pair_block(frame: pd.DataFrame, n_boot: int = 2_000, seed: int = SEED) -> dict:
    """Same-family agreement over pairs, CI by resampling JDs (never pairs)."""
    from candidate_screener.evaluation.resample import pair_stat_ci
    from candidate_screener.evaluation.resample import binary_kappa
    stats = {
        "oracle_same_share": (lambda o: float(o.mean()), ["oracle_same"]),
        "predicted_same_share": (lambda p: float(p.mean()), ["pred_same"]),
        "agreement": (lambda o, p: float((o == p).mean()), ["oracle_same", "pred_same"]),
        "kappa": (binary_kappa, ["oracle_same", "pred_same"]),
        "soft_auc_vs_oracle_same": (_auc, ["soft", "oracle_same"]),
        "soft_mean_oracle_same": (lambda s, o: float(s[o == 1].mean()) if (o == 1).any() else float("nan"),
                                  ["soft", "oracle_same"]),
        "soft_mean_oracle_differ": (lambda s, o: float(s[o == 0].mean()) if (o == 0).any() else float("nan"),
                                    ["soft", "oracle_same"]),
    }
    out = {"n_pairs": int(len(frame)), "n_jds": int(frame.jd_id.nunique())}
    for name, (f, cols) in stats.items():
        v = f(*(frame[c].to_numpy() for c in cols))
        lo, hi = pair_stat_ci(frame, f, cols, n_boot=n_boot, seed=seed)
        out[name] = {"value": None if math.isnan(v) else round(v, 4),
                     "ci95": [None if math.isnan(lo) else round(lo, 4),
                              None if math.isnan(hi) else round(hi, 4)]}
    return out


def evaluate_final(model: RoleModel, space: Mapping) -> dict:
    """Final-set documents vs the oracle label; pair-level same-family agreement."""
    pairs, docs = load_final_documents()
    idx = {lab: i for i, lab in enumerate(model.labels)}
    probs, oracle = {}, {}
    for s in SIDES:
        d = docs[s]
        probs[s] = dict(zip(d.index, predict_proba(list(d.text), s, model)))
        oracle[s] = dict(zip(d.index, labels_for(d, space)))
    out: dict = {"text_variant": "v2 (D53)", "documents": {}, "pairs": {}}
    pairs = pairs.assign(
        jd_oracle=pairs.jd_id.map(oracle["jd"]), cv_oracle=pairs.cv_id.map(oracle["cv"]))
    from candidate_screener.evaluation.final_set import SCOPES
    for scope in SCOPES:
        sub = pairs if scope == "total" else pairs[pairs.stratum == scope]
        jds = sub.drop_duplicates("jd_id")
        block = {}
        for s, frame, grp in (("jd", jds, None), ("cv", sub, sub.jd_id.to_numpy())):
            ids = frame[f"{s}_id"].to_numpy()
            ok = np.array([oracle[s][i] is not None for i in ids])
            p = np.stack([probs[s][i] for i in ids[ok]])
            yv = np.array([idx[oracle[s][i]] for i in ids[ok]])
            block[s] = {**classification_block(p, yv, None if grp is None else grp[ok]),
                        "unique_documents": int(len(set(ids[ok]))), "no_label": int((~ok).sum())}
        out["documents"][scope] = block
        f = sub[sub.jd_oracle.notna() & sub.cv_oracle.notna()]
        pj = np.stack([probs["jd"][i] for i in f.jd_id])
        pc = np.stack([probs["cv"][i] for i in f.cv_id])
        frame = pd.DataFrame({
            "jd_id": f.jd_id.to_numpy(),
            "oracle_same": (f.jd_oracle.to_numpy() == f.cv_oracle.to_numpy()).astype(int),
            "pred_same": (pj.argmax(1) == pc.argmax(1)).astype(int),
            "soft": soft_agreement(pj, pc)})
        out["pairs"][scope] = pair_block(frame)
    labels_seen = collections.Counter(
        [oracle["jd"][i] for i in pairs.jd_id.unique()])
    out["oracle_jd_classes"] = len(labels_seen)
    out["oracle_cv_classes"] = len({oracle["cv"][i] for i in pairs.cv_id.unique()})
    out["note"] = ("Measurement only (D46): no figure here chose anything. Oracle = "
                   "role_family(Primary Keyword, Position) of the final-set document. CIs "
                   "resample JDs; CV figures are over the 600 pair rows (CVs nested in JDs).")
    return out


# --- record ------------------------------------------------------------------------------

def class_profile(prep: dict) -> dict:
    out = {}
    for s in SIDES:
        y = prep["y"][s]
        fit = y[prep["fit_mask"][s]].dropna()
        vc = y.dropna().value_counts()
        out[s] = {"documents": int(len(y)), "labelled": int(y.notna().sum()),
                  "classes_present": int(len(vc)), "min": int(vc.min()),
                  "median": float(vc.median()), "max": int(vc.max()),
                  "fit_part": int(len(fit)),
                  "sizes": {str(k): int(v) for k, v in vc.sort_index().items()}}
    return out


def alias_profile(prep: dict) -> dict:
    """How much of the train `Position` the alias table recognises (exact, normalised)."""
    out = {"entries": len(prep["aliases"]),
           "multi_token": sum(1 for a in prep["aliases"] if len(a.split()) > 1),
           "digest": _fingerprint(sorted(prep["aliases"].items()))}
    for s in SIDES:
        t, val = prep["titles"][s], ~prep["fit_mask"][s]
        nonempty = t[val & (t != "")]
        hit = nonempty.isin(set(prep["aliases"]))
        y = prep["y"][s][nonempty.index]
        right = nonempty[hit].map(prep["aliases"]) == y[hit]
        out[f"{s}_validation_part"] = {
            "position_nonempty": int(len(nonempty)),
            "covered": round(float(hit.mean()), 4),
            "alias_family_equals_label": round(float(right.mean()), 4) if hit.any() else None}
    return out


def span_rate(docs: dict[str, pd.DataFrame], prep: dict) -> dict:
    """On the selection draw: share of texts with any span; first span's family = label."""
    out = {}
    for s in SIDES:
        sel = prep["parts"][s]["select"]
        texts, ys = docs[s].text.to_numpy()[sel], prep["y"][s].to_numpy()[sel]
        spans = [title_spans(t, prep["aliases"]) for t in texts]
        anyspan = np.array([bool(sp) for sp in spans])
        first_ok = np.array([bool(sp) and sp[0].family == y for sp, y in zip(spans, ys)])
        named = np.array([any(x.family == y for x in sp) for sp, y in zip(spans, ys)])
        out[s] = {"n": int(len(texts)), "any_span": round(float(anyspan.mean()), 4),
                  "label_family_named": round(float(named.mean()), 4),
                  "first_span_is_label": round(float(first_ok.mean()), 4),
                  "first_span_is_label_given_any": round(float(first_ok[anyspan].mean()), 4)
                  if anyspan.any() else None}
    return out


def committed_space_matches(space: Mapping) -> bool:
    return (tuple(space["keyword_classes"]) == tuple(KEYWORD_CLASSES)
            and {g: dict(v) for g, v in space["split_families"].items()}
            == {g: dict(v) for g, v in SPLIT_FAMILIES.items()})


def build(train: bool = True) -> dict:
    t0 = time.time()
    docs = load_train_documents()
    _log(f"train region: {len(docs['jd'])} JDs, {len(docs['cv'])} CVs")
    prep = prepare(docs)
    fp = config_fingerprint(prep)
    if train:
        res = train_all(docs, prep)
        model = RoleModel(res["chosen"], prep["labels"], prep["aliases"],
                          res["models"][res["chosen"]], fp)
        save_model(model)
        (CACHE / f"training-{fp}.json").write_text(json.dumps(
            {k: res[k] for k in ("validation", "selection_score", "chosen", "calibration",
                                 "timing")}, indent=2) + "\n", encoding="utf-8")
    else:
        model = load_role_model(fp)
    trained = json.loads((CACHE / f"training-{fp}.json").read_text(encoding="utf-8"))
    space = prep["space"]
    record = {
        "plan": "plan/2026-10-06-extraction-and-rerank (P3, E2)",
        "region": "A2 train region only for fitting (D19); final-set documents for measurement only (D46)",
        "fingerprint": fp,
        "config": CONFIG,
        "label_space": {
            "labels": list(prep["labels"]), "n_labels": len(prep["labels"]),
            "keyword_classes": space["keyword_classes"], "folded_into_other": space["folded"],
            "split_families": space["split_families"], "merges": space["merges"],
            "committed_constants_match": committed_space_matches(space)},
        "class_profile_train_region": class_profile(prep),
        "alias_table": alias_profile(prep),
        "spans_on_validation": span_rate(docs, prep),
        "split": {s: {k: (int(len(v)) if hasattr(v, "__len__") else v)
                      for k, v in prep["parts"][s].items()} for s in SIDES},
        "validation": trained["validation"],
        "selection": {"rule": CONFIG["selection"], "score": {k: round(v, 4) for k, v in
                                                             trained["selection_score"].items()},
                      "chosen": trained["chosen"]},
        "calibration": trained["calibration"],
        "final_set": evaluate_final(model, space),
    }
    _log(f"done in {time.time() - t0:.0f}s")
    return record


def render(record: dict) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--train", action="store_true", help="fit both candidates, write the record")
    g.add_argument("--evaluate", action="store_true", help="record from the cached model")
    g.add_argument("--check", action="store_true", help="refit and diff the committed record")
    g.add_argument("--derive", action="store_true", help="print the derived label space")
    a = ap.parse_args(argv)
    if a.derive:
        docs = load_train_documents()
        prep = prepare(docs)
        print(json.dumps({k: prep["space"][k] for k in ("keyword_classes", "split_families")},
                         indent=1, ensure_ascii=False))
        print("committed constants match:", committed_space_matches(prep["space"]))
        return 0
    fresh = render(build(train=not a.evaluate))
    if a.check:
        ok = RECORD.exists() and RECORD.read_text(encoding="utf-8") == fresh
        print(f"role-family record {'OK' if ok else 'MISMATCH'} ({RECORD.name})")
        return 0 if ok else 1
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(fresh, encoding="utf-8")
    print(f"wrote {RECORD}")
    return 0


if __name__ == "__main__":
    # Run the imported module, not `__main__`, so pickled classes resolve everywhere.
    import importlib
    raise SystemExit(importlib.import_module(MODULE).main())
