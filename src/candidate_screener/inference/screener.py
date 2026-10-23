"""The reported system behind one object an app loads once (plan 2026-10-23-inference-api).

    from candidate_screener import inference
    s = inference.Screener.load()                       # st.cache_resource in a Streamlit app
    res = s.rank(jd_text, cvs, jd_title="QA Engineer")  # D63: L1 on F12 over the pool
    res = s.rank(jd_text, cvs, rerank=True)             # D64: L1's top 50 reordered with C2
    for r in res.top(10): ...
    s.explain(res, r.cv_id)                             # for the recruiter

Systems: **D63** (default) = the frozen L1 (`output/rerank/l1-chat3-model.json`) on F12;
**D64** = `rerank=True`, D63's top 50 reordered by min-max L1 + 0.25 x min-max C2 (k, w from
`c2-validation-select.json`; C2 costs ~5-7 s per pair on CPU); **X2** = the baseline,
`2 role_match + experience_compatible + 0.999 tfidf_scaled`. Nothing is fitted here: every
part is loaded (role model, vocabulary, the saved concept frequencies and TF-IDF space, SBERT
weak, the frozen L1), so a new JD or CV is inference only.

Inputs (D53, D54): JD text, an optional JD title, and CVs as `cv_id` + `cv_text` only. Labels
never enter (`labels()` is separate, for checking a demo JD, D46).

**`tfidf_scaled` depends on the pool**: it is the TF-IDF cosine min-max scaled over the CVs
scored together. `scaling="background"` (option B, **the default**, owner 23 Oct 2026) pads the
CVs passed in to 500 with a fixed sample of A2 train-region CVs, as the system was trained and
tested, and returns only the CVs passed in; `scaling="pool"` (option A) scales over the CVs passed
in. On small pools A reorders noticeably (`output/inference/pool-scaling.json`: at 10 CVs the top
candidate differs from B's about 1 time in 7); on a 500-CV pool the two coincide.

Pair ids: with `jd_id` given they are `f"{jd_id}__{cv_id}"`, as the evaluation builds them, so a
final-set JD reuses the C2 scores already cached (`a2_pool` callers pass it); without it (uploads)
they derive from text digests, so the C2 cache never confuses two documents.

"present" in a CV date range is `reference_date`: today by default (uploads), A2's 2023-09-01
(`experience.REFERENCE_DATE`) for A2 documents (`a2_pool` returns it).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import threading
from functools import lru_cache, wraps

import numpy as np
import pandas as pd

from candidate_screener.config import OUTPUT
from candidate_screener.inference.schema import CVProfile, JDProfile, Ranked, Screening

SYSTEMS = ("D63", "X2")
SCALINGS = ("pool", "background")
COLUMNS = ("cv_id", "cv_text")
BACKGROUND_SIZE, BACKGROUND_SEED = 500, 0
C2_RECORD = OUTPUT / "rerank" / "c2-validation-select.json"
C2_CHUNK = 5           # C2 cache file per 5 pairs, so `progress` moves every ~30 s
DECIMALS = 6


def _digest(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _cvs_frame(cvs) -> pd.DataFrame:
    """`cv_id`, `cv_text` only (D53): unique non-empty ids, non-empty texts."""
    frame = cvs.copy() if isinstance(cvs, pd.DataFrame) else pd.DataFrame(list(cvs))
    extra, missing = set(frame.columns) - set(COLUMNS), set(COLUMNS) - set(frame.columns)
    if extra:
        raise ValueError(f"CVs carry only cv_id and cv_text (D53: no metadata at inference); got {sorted(extra)}")
    if missing or frame.empty:
        raise ValueError("at least one CV with cv_id and cv_text is required")
    frame = frame[list(COLUMNS)].reset_index(drop=True)
    if frame.cv_id.isna().any() or frame.cv_id.astype(str).str.strip().eq("").any():
        raise ValueError("every CV needs a non-empty cv_id")
    frame["cv_id"] = frame.cv_id.astype(str)
    if frame.cv_id.duplicated().any():
        raise ValueError("cv_id must be unique")
    if not frame.cv_text.map(lambda t: isinstance(t, str) and bool(t.strip())).all():
        raise ValueError("every CV needs non-empty text")
    return frame


def _years(lo, hi) -> str:
    def n(v):
        return f"{v:g} year" + ("" if v == 1 else "s")
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g} years"
    return f"at least {n(lo)}" if lo is not None else "no number of years"


def _serialised(method):
    """One call at a time per `Screener`: Streamlit shares one instance across sessions, and the
    feature path sets the CV reference date on shared state for the length of a call
    (`learned.Features`), as the per-text caches are shared too."""
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


def _list(items, n: int = 5) -> str:
    items = list(items)
    return ", ".join(items[:n]) + (f" and {len(items) - n} more" if len(items) > n else "")


class Screener:
    """Holds every loaded part; `rank`, `features`, `extract_*`, `explain`. See the module doc.

    Components are injected so the logic is testable without `data/`:
    `featurize(pairs, reference) -> F12 frame by pair_id`; `l1` an `L1Model`;
    `extract_jds(texts, titles)` and `extract_cvs(texts, reference)` -> profiles;
    `c2(pairs, progress) -> scores by pair_id`; `background()` -> frame of `id`, `cv_text`."""

    def __init__(self, featurize, l1, extract_jds, extract_cvs, c2=None, background=None,
                 c2_k: int = 50, c2_w: float = 0.25) -> None:
        self.featurize, self.l1 = featurize, l1
        self._extract_jds, self._extract_cvs = extract_jds, extract_cvs
        self.c2, self._background, self.c2_k, self.c2_w = c2, background, c2_k, c2_w
        self._jd_cache: dict = {}
        self._cv_cache: dict = {}
        self._lock = threading.RLock()

    @classmethod
    def load(cls, parts_fingerprint: str | None = None, c2_cached_only: bool = False) -> "Screener":
        """The real parts, loaded, never refitted (needs `--cache-parts` and `--freeze` once).
        `c2_cached_only` makes D64 refuse to score a C2 pair that is not cached (`--check`)."""
        from candidate_screener.extraction import role_family, rules
        from candidate_screener.inference import parts, profiles
        from candidate_screener.inference.model import L1Model
        from candidate_screener.matching import learned, lexical, sbert
        from candidate_screener.matching.entity_weighted import ConceptExtractor, EntityWeightedX2
        from candidate_screener.matching.extracted import ExtractedRule
        role, matcher = role_family.load_role_model(), rules.default_matcher()
        extractor = ConceptExtractor()
        p = parts.load_parts(parts_fingerprint, extractor=extractor)
        rule = ExtractedRule(role_model=role)
        rule.fit(None)                                   # records provenance; loads nothing more
        text = lexical.Tfidf("v2")
        text.vectorizer = p.vectorizer                   # the saved space; `fit` would refit it
        # `a1_weights.build_scorer`'s construction, from saved parts and not fitted.
        scorer = EntityWeightedX2(0.5, False, True, extractor=extractor, frequencies=p.frequencies,
                                  rule=rule, text_system=text)
        weak = sbert.SbertWeak("v2")
        if not (sbert.WEIGHTS / weak.cache_key("v2") / "config.json").exists():
            raise FileNotFoundError("SBERT weak weights are missing; inference never trains them")
        weak.fit(None)
        features = learned.Features()
        features.scorer, features.sbert = scorer, weak    # one definition of the F12 maths

        def featurize(pairs, reference):
            return features(pairs, learned.F12, reference=reference)

        def c2(pairs, progress=None):
            from candidate_screener.evaluation.rerank_embed import CASCADE_SYSTEM
            from candidate_screener.rerank import cross
            return cross.score_pairs(pairs, CASCADE_SYSTEM, chunk=C2_CHUNK, progress=progress,
                                     cached_only=c2_cached_only)

        win = json.loads(C2_RECORD.read_text(encoding="utf-8"))["winner"]
        return cls(featurize, L1Model.load(),
                   lambda texts, titles: profiles.extract_jds(texts, titles, role, matcher),
                   lambda texts, reference: profiles.extract_cvs(texts, role, matcher, reference),
                   c2=c2, background=a2_background, c2_k=int(win["k"]), c2_w=float(win["w"]))

    # --- documents -----------------------------------------------------------------------

    @_serialised
    def extract_jd(self, jd_text: str, jd_title: str | None = None) -> JDProfile:
        key = (_digest(jd_text), jd_title)
        if key not in self._jd_cache:
            self._jd_cache[key] = self._extract_jds([jd_text], [jd_title])[0]
        return self._jd_cache[key]

    def _cv_profiles(self, texts, reference: dt.date) -> list[CVProfile]:
        keys = [(_digest(t), reference) for t in texts]
        todo = {k: t for k, t in zip(keys, texts) if k not in self._cv_cache}
        if todo:
            for k, prof in zip(todo, self._extract_cvs(list(todo.values()), reference)):
                self._cv_cache[k] = prof
        return [self._cv_cache[k] for k in keys]

    @_serialised
    def extract_cv(self, cv_text: str, cv_id: str | None = None,
                   reference_date: dt.date | None = None) -> CVProfile:
        prof = self._cv_profiles([cv_text], reference_date or dt.date.today())[0]
        return dataclasses.replace(prof, cv_id=cv_id)

    # --- features and ranking --------------------------------------------------------------

    def _pairs(self, jd_text, jd_title, cvs: pd.DataFrame, scaling: str,
               jd_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(pairs of the CVs passed in, one per distinct text; every pair scored together).
        Pair ids are `jd_id__cv_id` when `jd_id` is given (a duplicate text keeps its first
        `cv_id`), else built from text digests."""
        if scaling not in SCALINGS:
            raise ValueError(f"scaling must be one of {SCALINGS}")
        if not isinstance(jd_text, str) or not jd_text.strip():
            raise ValueError("a non-empty JD text is required")
        uniq = cvs.drop_duplicates("cv_text")
        if jd_id is None:
            jd_id, keys = "jd-" + _digest(jd_text), "cv-" + uniq.cv_text.map(_digest)
        else:
            jd_id, keys = str(jd_id), uniq.cv_id
        own = pd.DataFrame({"pair_id": (jd_id + "__" + keys).to_numpy(), "jd_id": jd_id,
                            "cv_id": keys.to_numpy(), "jd_text": jd_text, "jd_title": jd_title,
                            "cv_text": uniq.cv_text.to_numpy()})
        scored = own
        if scaling == "background":
            bg = self.background_cvs(len(own))
            if len(bg):
                ids = "bg-" + bg["id"].astype(str)
                scored = pd.concat([own, pd.DataFrame({
                    "pair_id": (jd_id + "__" + ids).to_numpy(), "jd_id": jd_id, "cv_id": ids.to_numpy(),
                    "jd_text": jd_text, "jd_title": jd_title, "cv_text": bg.cv_text.to_numpy()})],
                    ignore_index=True)
                if scored.pair_id.duplicated().any():
                    raise ValueError("a cv_id collides with a background id ('bg-…')")
        return own, scored

    def background_cvs(self, n_own: int) -> pd.DataFrame:
        """The first `500 - n_own` CVs of the fixed background sample (`id`, `cv_text`)."""
        if self._background is None:
            raise ValueError("no background sample configured")
        bg = self._background()
        return bg.iloc[:max(0, BACKGROUND_SIZE - n_own)]

    def _features(self, jd_text, jd_title, cvs, scaling, reference, jd_id=None):
        own, scored = self._pairs(jd_text, jd_title, cvs, scaling, jd_id)
        X = self.featurize(scored, reference)
        X = X.loc[own.pair_id.to_numpy()]
        if len(X) != len(own) or X.isna().any().any():
            raise ValueError("a CV has no complete feature row")
        return own, X, len(scored)

    @_serialised
    def features(self, jd_text: str, cvs, jd_title: str | None = None, scaling: str = "background",
                 reference_date: dt.date | None = None, jd_id: str | None = None) -> pd.DataFrame:
        """F12 per CV passed in, indexed by `cv_id`."""
        cvs = _cvs_frame(cvs)
        own, X, _ = self._features(jd_text, jd_title, cvs, scaling, reference_date or dt.date.today(), jd_id)
        key = dict(zip(own.cv_text, own.pair_id))
        return X.loc[cvs.cv_text.map(key).to_numpy()].set_axis(cvs.cv_id.to_numpy())

    @_serialised
    def rank(self, jd_text: str, cvs, jd_title: str | None = None, top_k: int = 10,
             system: str = "D63", rerank: bool = False, scaling: str = "background",
             reference_date: dt.date | None = None, progress=None,
             jd_id: str | None = None) -> Screening:
        """Every CV passed in, best first. `system` "D63" (L1) or "X2"; `rerank=True` is D64
        (C2 on L1's top 50, ~2.5-7 s per uncached pair; `progress(done, total)` reports it). Ties
        by cv_id. `scaling`: "background" (default) or "pool"; `jd_id`: only to reuse cached C2
        scores of a known JD; see the module doc."""
        if system not in SYSTEMS:
            raise ValueError(f"system must be one of {SYSTEMS}")
        if rerank and system != "D63":
            raise ValueError("rerank (D64) reorders D63's top 50; use system='D63'")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        cvs = _cvs_frame(cvs)
        reference = reference_date or dt.date.today()
        own, X, pool_size = self._features(jd_text, jd_title, cvs, scaling, reference, jd_id)
        ids = own.pair_id.to_numpy()
        l1 = pd.Series(self.l1.predict(X), index=ids)
        x2 = pd.Series(2 * X.role_match.to_numpy() + X.experience_compatible.to_numpy()
                       + 0.999 * X.tfidf_scaled.to_numpy(), index=ids)
        c2 = pd.Series(dtype=float)
        if rerank:
            from candidate_screener.evaluation.c2_validation import fused_order
            from candidate_screener.rerank.chat_targets import top_k_mask
            if self.c2 is None:
                raise ValueError("no C2 scorer configured")
            k = min(self.c2_k, len(own))
            top = own[top_k_mask(ids, l1.loc[ids].to_numpy(), k)]
            c2 = self.c2(top[["pair_id", "jd_text", "cv_text"]], progress)
            score = fused_order(own, l1, c2, k, self.c2_w)
        else:
            score = l1 if system == "D63" else x2
        name = "D64" if rerank else system
        key = cvs.cv_text.map(dict(zip(own.cv_text, own.pair_id)))
        rows = pd.DataFrame({"cv_id": cvs.cv_id, "pair_id": key, "score": score.loc[key].to_numpy()})
        rows = rows.sort_values(["score", "cv_id"], ascending=[False, True], kind="mergesort")
        if not np.isfinite(rows.score.to_numpy(dtype=float)).all():
            raise ValueError("a CV got a non-finite score")
        feats = X.round(DECIMALS)
        ranked = tuple(Ranked(
            cv_id=r.cv_id, rank=i, score=round(float(r.score), DECIMALS),
            l1_score=round(float(l1[r.pair_id]), DECIMALS), x2_score=round(float(x2[r.pair_id]), DECIMALS),
            c2_score=round(float(c2[r.pair_id]), DECIMALS) if r.pair_id in c2.index else None,
            features={c: float(v) for c, v in feats.loc[r.pair_id].items()}, in_top_k=i <= top_k)
            for i, r in enumerate(rows.itertuples(index=False), start=1))
        profiles = self._cv_profiles(cvs.cv_text.tolist(), reference)
        return Screening(
            jd=dataclasses.replace(self.extract_jd(jd_text, jd_title), jd_id=jd_id),
            cvs={i: dataclasses.replace(p, cv_id=i) for i, p in zip(cvs.cv_id, profiles)},
            ranked=ranked, system=name, scaling=scaling, pool_size=pool_size, top_k=top_k,
            reference_date=reference)

    # --- explanation for the recruiter -------------------------------------------------------

    @_serialised
    def explain(self, screening: Screening, cv_id: str) -> dict:
        """Why this CV sits where it does, in the recruiter's terms first: role, skills, years,
        then `reasons` (plain sentences); the L1 feature shares come last, for debugging. No
        CV text is quoted."""
        r, cv, jd = screening.get(cv_id), screening.cvs[cv_id], screening.jd
        held = set(cv.skills)
        matched = [s for s in jd.skills if s in held]
        missing = [s for s in jd.skills if s not in held]
        X = pd.DataFrame([r.features])
        shares = self.l1.contributions(X).iloc[0]
        compatible = bool(r.features.get("experience_compatible", 0) >= 0.5)
        role_match = r.features.get("role_match")
        reasons = []
        if jd.role == cv.role:
            reasons.append(f"Same role family as the job: {jd.role} (role match {role_match:.2f}).")
        else:
            reasons.append(f"The job reads as {jd.role}; the CV reads as {cv.role} "
                           f"(role match {role_match:.2f}).")
        if jd.skills:
            reasons.append(f"Has {len(matched)} of the {len(jd.skills)} skills the job names"
                           + (f": {_list(matched)}." if matched else "."))
            if missing:
                reasons.append(f"Not found in the CV: {_list(missing)}.")
        else:
            reasons.append("The job names no skills the vocabulary knows.")
        asked = _years(jd.min_years, jd.max_years)
        if jd.min_years is None:
            reasons.append("The job states no years of experience.")
        elif cv.years is None:
            reasons.append(f"The job asks for {asked}; the CV states no years, so experience was "
                           "not held against it.")
        else:
            reasons.append(f"The job asks for {asked}; the CV shows about {cv.years:g} years "
                           f"({'compatible' if compatible else 'outside the range'}).")
        if r.c2_score is not None:
            reasons.append("Its place within the top 50 was refined by the reranker (C2).")
        reasons.append(f"Ranked {r.rank} of {len(screening.ranked)} ({screening.system}).")
        return {
            "cv_id": cv_id, "rank": r.rank, "of": len(screening.ranked), "system": screening.system,
            "score": r.score, "reasons": reasons,
            "jd_role": jd.role, "cv_role": cv.role, "jd_top_roles": [list(t) for t in jd.top_roles],
            "cv_top_roles": [list(t) for t in cv.top_roles], "role_match": role_match,
            "matched_skills": matched, "missing_skills": missing,
            "years_required": {"min": jd.min_years, "max": jd.max_years},
            "years_stated": cv.years, "years_source": cv.years_source,
            "experience_compatible": compatible,
            "feature_shares": [{"feature": c, "value": float(X.iloc[0][c]), "share": float(shares[c])}
                               for c in sorted(shares.index, key=lambda c: -abs(shares[c]))],
            "l1_intercept": float(self.l1.intercept),
            "note": "feature_shares are L1's: intercept + the shares = l1_score",
        }

    # --- A2 demo (needs data/) ------------------------------------------------------------------

    @staticmethod
    def a2_pool(jd_id: str) -> tuple[str, str | None, pd.DataFrame, dt.date]:
        """(JD text, JD title, its 500-CV pool as `cv_id`/`cv_text`, reference date) for one of
        the 60 final-set JDs, ready for `rank` (pass `jd_id` too, so cached C2 scores are reused).
        A2 documents, so the date is 2023-09-01."""
        from candidate_screener.extraction.experience import REFERENCE_DATE
        pools, titles = _a2_pools()
        g = pools[pools.jd_id == jd_id]
        if g.empty:
            raise KeyError(f"{jd_id} is not a final-set JD")
        return (g.jd_text.iloc[0], titles.get(jd_id),
                g[list(COLUMNS)].reset_index(drop=True), REFERENCE_DATE)

    @staticmethod
    def a2_jds() -> pd.DataFrame:
        """The 60 final-set JDs: `jd_id`, `stratum`, `jd_title`."""
        from candidate_screener.matching import corpus
        p = corpus.a2_pairs(text_variant="v2").drop_duplicates("jd_id")
        return p[["jd_id", "stratum", "jd_position"]].rename(columns={"jd_position": "jd_title"}).reset_index(drop=True)

    @staticmethod
    def labels(jd_id: str) -> pd.Series:
        """Final labels (Fit / No Fit) of a final-set JD's 10 labelled CVs, by `cv_id`. For checking
        a demo, never an input (D46)."""
        from candidate_screener.evaluation import final_set
        f = final_set.load()
        return f[f.jd_id == jd_id].set_index("cv_id").final_label


@lru_cache(maxsize=1)
def _a2_pools() -> tuple[pd.DataFrame, dict]:
    from candidate_screener.evaluation import retrieval_pilot
    from candidate_screener.matching import corpus
    pools = retrieval_pilot.build_pools(0, "v2")[["jd_id", "cv_id", "jd_text", "cv_text"]]
    p = corpus.a2_pairs(text_variant="v2").drop_duplicates("jd_id")
    return pools, dict(zip(p.jd_id, p.jd_position))


@lru_cache(maxsize=1)
def a2_background() -> pd.DataFrame:
    """A fixed, seeded order of A2 train-region CVs (`text_v2`), final-set CVs excluded; option B
    takes its first `500 - n` rows."""
    from candidate_screener.matching import corpus
    _, cv = corpus.a2_train_documents(text_variant="v2")
    final = set(corpus.a2_pairs(text_variant="v2").cv_id)
    cv = cv[~cv["id"].isin(final) & cv.cv_text.map(lambda t: isinstance(t, str) and bool(t.strip()))]
    cv = cv.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    order = np.random.default_rng(BACKGROUND_SEED).permutation(len(cv))[:BACKGROUND_SIZE]
    return cv.iloc[order][["id", "cv_text"]].reset_index(drop=True)
