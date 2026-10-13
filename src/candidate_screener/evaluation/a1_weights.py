"""Learn the four score weights from A1 labels, score them on A2 (exploratory, user exception).

The hand-set X2 score is `2·role_match + experience_compatible + 0.999·tfidf_scaled` (alpha 0).
This asks whether weights *learned* from A1 beat it on the A2 final set. Three arms, one
four-feature logistic regression each (Fit = Good Fit or Potential Fit, D47):

- `A1 original, all`      the 3,990 A1 train pairs, A1's own labels;
- `A1 original, matched`  the 639 pairs the LLM also labelled, A1's own labels (size control);
- `A1 LLM, matched`       the same 639 pairs, LLM labels (claude-sonnet-5, run 2).

**Exception.** AGENTS.md (D43, D51) lets only a reranker train on LLM output, and only on the
D50 batch. The user authorised this one test on 12 Oct 2026. It reads existing label files and
starts no model call. The result is `exploratory` (rule 8) and nothing here is adopted.
The features come from the A1 pairs' text only; nothing reads the A2 final labels except the
scorer (D46).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from candidate_screener.baselines.classifier import CLASSIFIER_CONFIG
from candidate_screener.evaluation import final_set

FEATURES = ("role_match", "experience_compatible", "tfidf_scaled", "coverage")
#: Extra text-only terms (part 2). Concepts: the extractor's affirmative concepts; skills: the
#: vocabulary matcher without the context filter; experience: the extracted years (P4).
CONCEPT_FEATURES = ("coverage_unweighted", "log_matched", "log_jd_concepts", "log_cv_concepts",
                    "skill_overlap")
EXPERIENCE_FEATURES = ("exp_gap", "exp_known")
FEATURE_SETS = {"base (4)": FEATURES,
                "+ concepts / skills": FEATURES + CONCEPT_FEATURES,
                "+ experience": FEATURES + EXPERIENCE_FEATURES,
                "all": FEATURES + CONCEPT_FEATURES + EXPERIENCE_FEATURES}
HAND_WEIGHTS = np.array([2.0, 1.0, 0.999, 0.0])           # X2 at alpha = 0
LLM_FILE = final_set.MANIFESTS / "llm-recheck-full-a1.csv"
LLM_RUN = 2


def build_scorer(seed: int = 0, extractor=None):
    """An `EntityWeightedX2` with explicit parts, used only for `.components()`.

    IDF population: the 20k-CV train sample minus every dev-pool CV (as the pilot does).
    `extractor` swaps the concept source (default: the seed rules)."""
    from candidate_screener.evaluation import dev_pools
    from candidate_screener.extraction import role_family
    from candidate_screener.matching import corpus, lexical
    from candidate_screener.matching.entity_weighted import (
        ConceptExtractor, DocumentFrequency, EntityWeightedX2)
    from candidate_screener.matching.extracted import ExtractedRule
    _, cv = corpus.a2_train_documents(text_variant="v2")
    dev = dev_pools.build()
    docs = (cv[~cv.id.isin(dev["cv"].id)].rename(columns={"cv_text": "text"})[["id", "text"]]
            .assign(region="train", doc_type="cv"))
    extractor = extractor or ConceptExtractor()
    freq = DocumentFrequency(extractor).fit(docs)
    rule = ExtractedRule(role_model=role_family.load_role_model())
    text = lexical.Tfidf("v2")
    model = EntityWeightedX2(0.5, False, True, extractor=extractor, frequencies=freq,
                             rule=rule, text_system=text)
    model.fit(None)
    return model


def features(model, pairs: pd.DataFrame) -> pd.DataFrame:
    """The four alpha-free terms for `pairs` (text columns and `jd_id` only)."""
    x = pairs[["pair_id", "jd_id", "cv_id", "jd_text", "cv_text"]].reset_index(drop=True)
    return model.components(x)[list(FEATURES)].set_index(x.pair_id.to_numpy())


def llm_labels() -> pd.Series:
    """`pair_id -> 1 (Fit) / 0 (No Fit)` from the one LLM series on all of A1's recheck pairs."""
    d = pd.read_csv(LLM_FILE)
    d = d[(d.run == LLM_RUN) & d.parsed_ok]
    if d.pair_id.duplicated().any():
        raise ValueError("LLM labels pooled across runs or draws")
    return d.set_index("pair_id").label.isin(final_set.FIT_CLASSES).astype(int)


def extra_features(model, pairs: pd.DataFrame) -> pd.DataFrame:
    """Concept, skill and experience terms for `pairs`, from text alone (D53)."""
    from candidate_screener.extraction import experience
    from candidate_screener.extraction.rules import default_matcher
    matcher, ex, freq = default_matcher(), model.extractor, model.frequencies
    rows = []
    for jd, cv in zip(pairs.jd_text, pairs.cv_text):
        j, c = ex.concepts(jd, "jd"), ex.concepts(cv, "cv")
        sj, sc = matcher.canonicals(jd), matcher.canonicals(cv)
        jd_years = experience.jd_experience(jd).min_years
        cv_years = experience.cv_experience(cv).years
        known = jd_years is not None and cv_years is not None
        rows.append({
            "coverage_unweighted": freq.coverage(j, c, False),
            "log_matched": np.log1p(len(j & c)),
            "log_jd_concepts": np.log1p(len(j)), "log_cv_concepts": np.log1p(len(c)),
            "skill_overlap": len(sj & sc) / len(sj) if sj else 0.0,
            "exp_gap": float(np.clip(cv_years - jd_years, -10, 10)) / 10 if known else 0.0,
            "exp_known": float(known)})
    return pd.DataFrame(rows, index=pairs.pair_id.to_numpy())


def fit_weights(X: pd.DataFrame, y: pd.Series, columns=FEATURES) -> pd.Series:
    """Logistic-regression coefficients by feature name (features are on comparable scales)."""
    clf = LogisticRegression(**CLASSIFIER_CONFIG)
    clf.fit(X[list(columns)].to_numpy(), y.loc[X.index].to_numpy())
    if int(np.max(clf.n_iter_)) >= CLASSIFIER_CONFIG["max_iter"]:
        raise RuntimeError("lbfgs hit max_iter: the fit is a function of the cap, not of the data")
    return pd.Series(clf.coef_[0], index=list(columns))


def linear_score(X: pd.DataFrame, weights) -> pd.Series:
    """`weights` is a name-indexed Series, or an array over `FEATURES`."""
    w = weights if isinstance(weights, pd.Series) else pd.Series(weights, index=FEATURES)
    return pd.Series(X[list(w.index)].to_numpy() @ w.to_numpy(), index=X.index)


# --- learning curve: does a learner keep improving as LLM-labelled A1 JDs are added? ---
ALL_FEATURES = FEATURE_SETS["all"]
LEARNERS = {"logistic base (4)": ("logit", FEATURES),
            "logistic all (11)": ("logit", ALL_FEATURES),
            "boosting all (11)": ("gbm", ALL_FEATURES)}


def fit_learner(kind: str, columns, X: pd.DataFrame, y: pd.Series):
    """A fitted `score(X) -> Series` for `kind` in {'logit', 'gbm'}; classes are balanced in both."""
    cols, yy = list(columns), y.loc[X.index].to_numpy()
    if kind == "logit":
        w = fit_weights(X, y, cols)
        return lambda Z: linear_score(Z, w)
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=0)
    gbm.fit(X[cols].to_numpy(), yy, sample_weight=compute_sample_weight("balanced", yy))
    return lambda Z: pd.Series(gbm.decision_function(Z[cols].to_numpy()), index=Z.index)


def mean_jd_auc(scores: pd.Series, y: pd.Series, jd: pd.Series) -> tuple[float, int]:
    """Mean per-JD AUC over JDs that have both classes, and how many JDs that was."""
    d = pd.DataFrame({"s": scores, "y": y.loc[scores.index], "j": jd.loc[scores.index]})
    a = [roc_auc_score(g.y, g.s) for _, g in d.groupby("j") if g.y.nunique() == 2]
    return (float(np.mean(a)) if a else float("nan")), len(a)


# --- the same curve on the D50 batch: LLM labels drawn from A2's own train region ---
def d50_pairs() -> tuple[pd.DataFrame, pd.Series]:
    """The 1,000 D50 pairs with `text_v2` CVs, and `pair_id -> 1 (Fit) / 0 (No Fit)` from run 11.

    A2 train region only: `load_train_documents` removes the eval region and the final set."""
    from candidate_screener.matching import corpus
    from candidate_screener.rerank import scoring_batch as sb
    s = pd.read_csv(sb.SCORES_CSV)
    s = s[(s.run == sb.RUN) & s.parsed_ok]
    if s.pair_id.duplicated().any():
        raise ValueError("D50 scores pooled across runs or draws")
    jd, cv = sb.load_train_documents()
    jd, cv = jd[jd.id.isin(s.jd_id)].copy(), cv[cv.id.isin(s.cv_id)].copy()   # keep memory low
    cv["cv_text"] = corpus.build_cv_text(cv, "v2")
    jd, cv = jd.set_index("id"), cv.set_index("id")
    pairs = s[["pair_id", "jd_id", "cv_id"]].assign(
        jd_text=s.jd_id.map(jd.jd_text), cv_text=s.cv_id.map(cv.cv_text))
    if pairs[["jd_text", "cv_text"]].isna().any().any():
        raise ValueError("a D50 pair has no text")
    return pairs.set_index("pair_id", drop=False), s.set_index("pair_id").label.isin(
        final_set.FIT_CLASSES).astype(int)


def learning_curve(X: pd.DataFrame, y: pd.Series, jd: pd.Series, X_a2: pd.DataFrame,
                   a2_y: pd.Series, a2_jd: pd.Series, sizes=(10, 25, 50, 70), reps: int = 20,
                   n_test_jds: int = 30, seed: int = 0) -> pd.DataFrame:
    """Train each learner on random JD subsets of `X` and score A2 and the held-out JDs.

    Every repetition holds out `n_test_jds` JDs of `X`; training JDs come from the rest."""
    rng = np.random.default_rng(seed)
    jds = np.array(sorted(jd.loc[X.index].unique()))
    rows = []
    for rep in range(reps):
        test = rng.choice(jds, n_test_jds, replace=False)
        pool = np.setdiff1d(jds, test)
        te = X.index[jd.loc[X.index].isin(test)]
        hand = mean_jd_auc(linear_score(X.loc[te], HAND_WEIGHTS), y, jd)[0]
        for n in sizes:
            tr = X.index[jd.loc[X.index].isin(rng.choice(pool, n, replace=False))]
            if y.loc[tr].nunique() < 2:
                continue
            for name, (kind, cols) in LEARNERS.items():
                f = fit_learner(kind, cols, X.loc[tr], y)
                rows.append({"learner": name, "train JDs": n, "rep": rep,
                             "A2 AUC": mean_jd_auc(f(X_a2), a2_y, a2_jd)[0],
                             "held-out AUC": mean_jd_auc(f(X.loc[te]), y, jd)[0], "hand held-out": hand})
    return pd.DataFrame(rows)


# --- error analysis ---
def fit_gap(X: pd.DataFrame, y: pd.Series, jd: pd.Series, n_train: int = 70, reps: int = 20,
            seed: int = 0) -> pd.DataFrame:
    """Per-JD AUC of each learner on its own training JDs and on held-out JDs (`n_train` / rest)."""
    rng = np.random.default_rng(seed)
    jds = np.array(sorted(jd.loc[X.index].unique()))
    rows = []
    for rep in range(reps):
        train = rng.choice(jds, n_train, replace=False)
        tr = X.index[jd.loc[X.index].isin(train)]
        te = X.index.difference(tr)
        scorers = {name: fit_learner(kind, cols, X.loc[tr], y) for name, (kind, cols) in LEARNERS.items()}
        scorers["hand-set X2"] = lambda Z: linear_score(Z, HAND_WEIGHTS)
        for name, f in scorers.items():
            rows.append({"scorer": name, "rep": rep, "train AUC": mean_jd_auc(f(X.loc[tr]), y, jd)[0],
                         "held-out AUC": mean_jd_auc(f(X.loc[te]), y, jd)[0]})
    return pd.DataFrame(rows)


def pair_inversions(scores: pd.Series, y: pd.Series, jd: pd.Series) -> pd.DataFrame:
    """Every (Fit, No Fit) pair within a JD: `inv` is 1 when the No Fit scores higher, 0.5 on a tie.

    The mean of `inv` is 1 - the per-pair AUC; grouping by `fit_id` or `nofit_id` gives each
    document's miss rate (a Fit ranked under No Fit) or false-alarm rate."""
    d = pd.DataFrame({"s": scores, "y": y.loc[scores.index], "j": jd.loc[scores.index]})
    out = []
    for j, g in d.groupby("j"):
        pos, neg = g[g.y == 1], g[g.y == 0]
        if pos.empty or neg.empty:
            continue
        diff = pos.s.to_numpy()[:, None] - neg.s.to_numpy()[None, :]
        inv = (diff < 0) + 0.5 * (diff == 0)
        fi, ni = np.meshgrid(pos.index, neg.index, indexing="ij")
        out.append(pd.DataFrame({"jd_id": j, "fit_id": fi.ravel(), "nofit_id": ni.ravel(), "inv": inv.ravel()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["jd_id", "fit_id", "nofit_id", "inv"])


def llm_notes(frame: pd.DataFrame) -> pd.DataFrame:
    """The final set's LLM rationale per pair: the same series (agent, run) that sits in the set.

    Diagnostic reading only. The LLM is a secondary annotator (D43), and its note says why *it*
    chose its label, which can differ from the human's."""
    d = pd.concat([pd.read_csv(final_set.MANIFESTS / f"llm-a2-{s}.csv") for s in ("generic", "targeted")])
    f = frame.reset_index()[["pair_id", "llm_run", "llm_agent_sha256"]]
    m = f.merge(d, left_on=["pair_id", "llm_run", "llm_agent_sha256"], right_on=["pair_id", "run", "agent_sha256"], how="left")
    if len(m) != len(f) or m.notes.isna().any():
        raise ValueError("final-set LLM rows and the llm-a2 files disagree")
    return m.set_index("pair_id")[["label", "notes", "score"]].rename(columns={"label": "llm_label", "score": "llm_score"})
