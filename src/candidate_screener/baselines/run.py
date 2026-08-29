"""Build the Stage 1 baseline, and assert it still reproduces *(D19–D21)*.

    uv run python -m candidate_screener.baselines.run --seed 0     # rebuild + write
    uv run python -m candidate_screener.baselines.run --check      # rebuild + diff

`--check` refits everything in memory, diffs it field by field against the committed
`output/baselines/baseline-metrics.json`, prints a `[PASS]`/`[FAIL]` per group and
exits 1 on any mismatch. It writes nothing, so it is safe to run anywhere. This is
the direct analogue of the `determinism:` check in `verify_derived.check_fit_split`,
which rebuilds the split manifest in memory and diffs it against the committed one.

Why a committed record at all: before this phase, the baseline existed only as
notebook cells and wrote its outputs into git-ignored `data/`. A run that quietly
differed had nothing to differ *from*, and three separate bugs — two vectorisers, a
`.split()` tokeniser, and a BM25 corpus lookup — each shipped for months producing a
plausible number. The record is what makes the fourth one loud.

**It deliberately does not register in `data.verify --derived`.** That registry is
the acceptance test on derived *data* artefacts, and `data/` stays about data
*(Phase 3, deviation W11)*.

Precondition: `data/processed/fit/` built — `build --task fit-split`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from candidate_screener.baselines import bm25, classifier, skills, tfidf
from candidate_screener.baselines.lexical import TOKEN_PATTERN
from candidate_screener.config import BASELINE_CACHE, BASELINE_METRICS, PROCESSED
from candidate_screener.evaluation import classification

FIT_SPLIT = PROCESSED / "fit"

#: The two models, in the order they are reported everywhere.
MODELS = ("tfidf", "bm25")

#: Written *into* the record, so the warning cannot be separated from the numbers by
#: a copy-paste. Mirrors `fit_split.NOT_COMPARABLE`.
NOT_COMPARABLE = (
    "Measured on the leak-free data/processed/fit/ split (task 3.2). NOT comparable "
    "to any number published against the shipped cnamuangtoun train/test partition, "
    "which leaks 99.8% of its test resumes into train.")

#: Comparison tolerance for `--check`, settled by measurement in task 4.3 rather
#: than by preference *(Q23)*. Integers — vocabulary size, confusion-matrix counts,
#: n, support — are compared **exactly**: those are the numbers that move when a
#: score sitting on a decision boundary flips a row's predicted class, which is the
#: real risk, not the last bits of a float. Floats are compared at 1e-12, which is
#: far tighter than any metric difference that would matter and far looser than a
#: BLAS-thread-count reordering of a sum.
FLOAT_TOLERANCE = 1e-12

#: Recorded so that a golden-check failure after a `uv lock` bump is diagnosable in
#: one diff rather than bisected. Unlike a timestamp this changes only when someone
#: deliberately re-locks, so it does not defeat the check — it explains it *(W6)*.
TRACKED_PACKAGES = ("scikit-learn", "numpy", "scipy", "pandas", "rank-bm25")


# --- loading ---------------------------------------------------------------

def load_split(name: str) -> pd.DataFrame:
    """Load one split of the **leak-free** A1 fit data *(Q19, task 3.2)*.

    Not the shipped `cnamuangtoun` partition, which leaks 99.8% of its test resumes
    into train and is not reportable. `resume_id`/`jd_id` are already present in the
    processed parquet, so no ID minting step is needed here.
    """
    path = FIT_SPLIT / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run python -m candidate_screener.data.build "
            "--task fit-split --seed 0` first")
    return pd.read_parquet(path)


def augment(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the skill-boosted representation both scorers consume."""
    return df.assign(resume_repr=skills.augment(df.resume_text),
                     jd_repr=skills.augment(df.job_description_text))


# --- the build -------------------------------------------------------------

def build(seed: int = 0) -> dict:
    """Fit both models on train, score both splits, and return everything measured.

    The vectoriser and the BM25 statistics are fitted on **train only**. Fitting
    either over the test documents would leak their document frequencies into the
    representation the model is then scored on — a subtler cousin of the split
    leakage Phase 3 removed.
    """
    train, test = augment(load_split("train")), augment(load_split("test"))

    vectorizer = tfidf.fit_shared_vectorizer(train.resume_repr, train.jd_repr)
    bm25_model = bm25.fit_bm25(train.resume_repr.drop_duplicates().fillna(""))

    scores = {
        "tfidf": {name: tfidf.pair_cosine_scores(vectorizer, df.resume_repr, df.jd_repr)
                  for name, df in (("train", train), ("test", test))},
        "bm25": {name: bm25.pair_bm25_scores(bm25_model, df.resume_repr, df.jd_repr)
                 for name, df in (("train", train), ("test", test))},
    }

    classifiers, reports = {}, {}
    for model in MODELS:
        clf = classifier.fit_single_feature(scores[model]["train"], train.label, seed)
        classifiers[model] = clf
        reports[model] = classification.report(
            model, test.label, classifier.predict(clf, scores[model]["test"]))

    return {"seed": seed, "train": train, "test": test, "vectorizer": vectorizer,
            "bm25_model": bm25_model, "scores": scores, "classifiers": classifiers,
            "reports": reports}


# --- the record ------------------------------------------------------------

def metrics_document(built: dict) -> dict:
    """The committed golden record. **No timestamp** — with one, every re-run diffs
    and the reproducibility check tests nothing *(Phase 3, W6)*."""
    train, test = built["train"], built["test"]
    vectorizer = built["vectorizer"]

    def split_stats(df: pd.DataFrame) -> dict:
        return {"pairs": int(len(df)),
                "unique_resumes": int(df.resume_id.nunique()),
                "unique_jds": int(df.jd_id.nunique()),
                "label_counts": {label: int((df.label == label).sum())
                                 for label in classification.LABELS}}

    models = {}
    for name in MODELS:
        clf, rep = built["classifiers"][name], built["reports"][name]
        # coef/intercept turn "the metrics happen to match" into "the same model was
        # fitted" — two different models can land on the same accuracy.
        models[name] = rep.to_dict() | {
            "classes": clf.classes_.tolist(),
            "coef": [float(c) for row in clf.coef_ for c in row],
            "intercept": [float(v) for v in clf.intercept_],
            "n_iter": int(max(clf.n_iter_)),
        }

    return {
        "split": {"source": "processed/fit", "note_not_comparable": NOT_COMPARABLE,
                  "train": split_stats(train), "test": split_stats(test)},
        "config": {
            "seed": built["seed"],
            "skill_patterns_sha256": skills.digest(),
            "token_pattern": TOKEN_PATTERN.pattern,
            "tfidf": {k: list(v) if isinstance(v, tuple) else v
                      for k, v in tfidf.TFIDF_CONFIG.items()},
            "classifier": classifier.CLASSIFIER_CONFIG | {"random_state": built["seed"]},
        },
        "vocabulary": {"size": int(len(vectorizer.get_feature_names_out())),
                       "sha256": tfidf.vocabulary_digest(vectorizer)},
        "floor": {"majority_class": built["reports"]["tfidf"].floor.majority_class,
                  "accuracy": built["reports"]["tfidf"].floor.accuracy,
                  "macro_f1": built["reports"]["tfidf"].floor.macro_f1},
        "models": models,
        "environment": {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
                       | {pkg: version(pkg) for pkg in TRACKED_PACKAGES},
    }


def write_cache(built: dict) -> None:
    """The pickles and the per-pair scores — a **cache, never an artefact** *(D23)*.

    sklearn pickles are version-fragile and not byte-stable across library versions,
    so treating them as the record would make every `uv lock` a false failure. They
    are git-ignored, never hashed, and no acceptance check reads them.
    """
    BASELINE_CACHE.mkdir(parents=True, exist_ok=True)
    artefacts = {"tfidf_vectorizer": built["vectorizer"], "bm25_model": built["bm25_model"],
                 "tfidf_classifier": built["classifiers"]["tfidf"],
                 "bm25_classifier": built["classifiers"]["bm25"]}
    for name, obj in artefacts.items():
        (BASELINE_CACHE / f"{name}.pkl").write_bytes(pickle.dumps(obj))

    # `scores.parquet` replaces the old `baseline_predictions.csv` and gains a
    # `split` column, so the train scores needed to refit or to inspect the decision
    # threshold are not thrown away.
    frames = []
    for name in ("train", "test"):
        df = built[name]
        frames.append(pd.DataFrame({
            "resume_id": df.resume_id.to_numpy(), "jd_id": df.jd_id.to_numpy(),
            "split": name, "label": df.label.to_numpy(),
            "tfidf_score": built["scores"]["tfidf"][name],
            "bm25_score": built["scores"]["bm25"][name]}))
    pd.concat(frames, ignore_index=True).to_parquet(
        BASELINE_CACHE / "scores.parquet", index=False)


def write_metrics(document: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


# --- the golden diff -------------------------------------------------------

def _differences(expected, actual, path: str = "") -> list[str]:
    """Recursive field-by-field diff. Ints exact, floats within `FLOAT_TOLERANCE`."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            here = f"{path}.{key}" if path else key
            if key not in expected:
                out.append(f"{here}: unexpected field {actual[key]!r}")
            elif key not in actual:
                out.append(f"{here}: missing")
            else:
                out += _differences(expected[key], actual[key], here)
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(actual)}, expected {len(expected)}"]
        return [d for i, (e, a) in enumerate(zip(expected, actual))
                for d in _differences(e, a, f"{path}[{i}]")]
    if isinstance(expected, bool) or isinstance(actual, bool):
        # `True == 1` in Python; a bool where an int was frozen is a schema change,
        # not a value that happens to match, so the types must agree too.
        same = (isinstance(expected, bool) and isinstance(actual, bool)
                and expected == actual)
        return [] if same else [f"{path}: {actual!r}, expected {expected!r}"]
    if isinstance(expected, int) and isinstance(actual, int):
        return [] if expected == actual else [f"{path}: {actual}, expected {expected}"]
    if isinstance(expected, float) or isinstance(actual, float):
        delta = abs(float(actual) - float(expected))
        return [] if delta <= FLOAT_TOLERANCE else [
            f"{path}: {actual!r}, expected {expected!r} (delta {delta:.3e} > {FLOAT_TOLERANCE:g})"]
    return [] if expected == actual else [f"{path}: {actual!r}, expected {expected!r}"]


#: Group -> the JSON key it covers. `environment` is reported but never fails: it is
#: the one non-content field, there to *explain* a failure elsewhere.
CHECK_GROUPS = ("split", "config", "vocabulary", "floor", "models")


def check(rebuilt: dict, committed: dict) -> list[tuple[bool, str]]:
    results = []
    for group in CHECK_GROUPS:
        diffs = _differences(committed.get(group), rebuilt.get(group), group)
        summary = f"{group}: reproduces the committed record" if not diffs else \
                  f"{group}: {len(diffs)} field(s) differ — " + "; ".join(diffs[:4])
        results.append((not diffs, summary))
    return results


def print_environment(rebuilt: dict, committed: dict) -> None:
    was = committed.get("environment", {})
    moved = [f"{k} {was.get(k, 'unrecorded')}->{v}"
             for k, v in rebuilt["environment"].items() if was.get(k) != v]
    print(f"  [info] environment: {'unchanged since the freeze' if not moved else ', '.join(moved)}"
          + ("" if not moved else " — a re-lock, not a code change; read any FAIL above "
                                 "against it before assuming a regression"))


# --- CLI -------------------------------------------------------------------

def print_build(built: dict, document: dict, out: Path) -> None:
    s = document["split"]
    print(f"\n=== Leak-free A1 split — train {s['train']['pairs']} pairs "
          f"({s['train']['unique_resumes']} resumes / {s['train']['unique_jds']} JDs), "
          f"test {s['test']['pairs']} pairs "
          f"({s['test']['unique_resumes']} / {s['test']['unique_jds']})")
    print(f"  shared TF-IDF vocabulary: {document['vocabulary']['size']} terms, "
          f"sha256 {document['vocabulary']['sha256'][:12]}…")
    print(f"\n=== Test-set figures, seed {built['seed']}")
    for name in MODELS:
        print("  " + str(built["reports"][name]).replace("\n", "\n  "))
    print(f"\n  metrics -> {out}")
    print(f"  cache   -> {BASELINE_CACHE}/ (git-ignored: 4 pickles + scores.parquet)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0,
                    help="classifier random_state; inert under lbfgs (see classifier.py)")
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and diff against the committed record; writes nothing")
    ap.add_argument("--out", type=Path, default=BASELINE_METRICS,
                    help="where the golden record lives")
    args = ap.parse_args(argv)

    built = build(args.seed)
    document = metrics_document(built)

    if not args.check:
        write_cache(built)
        write_metrics(document, args.out)
        print_build(built, document, args.out)
        return 0

    if not args.out.exists():
        print(f"[FAIL] {args.out} does not exist — nothing to check against. "
              "Run without --check to freeze it.")
        return 1
    committed = json.loads(args.out.read_text(encoding="utf-8"))
    print(f"\n=== baseline --check against {args.out}")
    failed = 0
    for ok, message in check(document, committed):
        print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
        failed += not ok
    print_environment(document, committed)
    print(f"\n{'the baseline reproduces' if not failed else f'{failed} CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
