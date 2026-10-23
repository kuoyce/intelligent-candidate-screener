"""The inference API against the committed records (plan 2026-10-23-inference-api, step 5).

`check()` drives the API exactly as an app would (`Screener.load`, `rank`, `features`,
`extract_*`) and compares what it returns with the records the evaluation code wrote. Writes
nothing. C2 is read from its cache only (`Screener.load(c2_cached_only=True)` refuses to score a
pair); the C2 scores themselves are reproduced cold by `evaluation.c2_validation`.

1. Final set, per JD over its 10 labelled CVs (`scaling="pool"`, A2's reference date):
   `a2-features.csv` (F12), `a2-x2-scores.csv`, `a2-jd-extraction.csv`, `a2-cv-extraction.csv`,
   each rendered as `evaluation.a2_export` renders it and compared as text.
2. L1 refitted from chat-3 as `model.freeze` fits it equals `l1-chat3-model.json` (1e-9).
3. D63 over the 60 final-set 500-CV pools (within-pool scaling over 500 is the tested condition):
   `chat3-l1-pool.json`'s "on all 500" blocks against `chat-1` and `chat-1 + chat-2` (Recall@10
   0.484413), and every JD's top 10 equals the evaluation path's (`learned.Features` + frozen L1).
4. D64 (`rerank=True`, `jd_id` passed so C2's cache is hit) and D63 against
   `c2-validation-test.json`: the winner's and L1's reference-A summaries, reference-B recall and
   the winner's deltas vs L1.

`demo(jd_id)` prints one final-set JD's top 10 and one explanation; no CV text (AGENTS).
"""
from __future__ import annotations

import io
import json
import time

import numpy as np
import pandas as pd

SEED = 0
TOL = 1e-9


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _same_csv(frame: pd.DataFrame, path) -> bool:
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    return buf.getvalue() == path.read_text(encoding="utf-8")


def _json(doc) -> str:
    return json.dumps(doc, sort_keys=True)


def final_set_tables(s) -> dict[str, bool]:
    """Item 1: the hand-off tables, rebuilt through the API."""
    from candidate_screener.evaluation import a2_export as ax
    from candidate_screener.extraction.experience import REFERENCE_DATE
    from candidate_screener.inference import profiles
    from candidate_screener.matching import corpus, learned
    pairs = corpus.a2_pairs(text_variant=ax.TEXT_VARIANT)
    feats, x2 = [], []
    for jd_id, g in pairs.groupby("jd_id", sort=True):
        cvs = g[["cv_id", "cv_text"]]
        res = s.rank(g.jd_text.iloc[0], cvs, jd_title=g.jd_position.iloc[0], system="X2",
                     scaling="pool", reference_date=REFERENCE_DATE, jd_id=jd_id)
        X = s.features(g.jd_text.iloc[0], cvs, jd_title=g.jd_position.iloc[0], scaling="pool",
                       reference_date=REFERENCE_DATE, jd_id=jd_id)
        feats.append(X.set_axis(jd_id + "__" + X.index))
        x2.append(pd.Series({jd_id + "__" + r.cv_id: r.x2_score for r in res.ranked}))
    X = pd.concat(feats).round(ax.DECIMALS)
    want = pd.read_csv(ax.FEATURES_RECORD).set_index("pair_id")
    feat_ok = (set(X.index) == set(want.index)       # 6 dp, as the table was written
               and (X.loc[want.index, list(learned.F12)].to_numpy() == want[list(learned.F12)].to_numpy()).all())
    out = ax.rank_within_jd(pairs, pd.concat(x2).loc[pairs.pair_id.to_numpy()].reset_index(drop=True))
    labels = pd.read_csv(ax.RECORD).set_index("pair_id").final_label   # carried over, not computed
    out["final_label"] = out.pair_id.map(labels)
    x2_ok = _same_csv(out[ax.COLUMNS].reset_index(drop=True), ax.RECORD)

    jds = pairs.drop_duplicates("jd_id").reset_index(drop=True)
    cvs = pairs.drop_duplicates("cv_id").reset_index(drop=True)
    jp = [s.extract_jd(t, title) for t, title in zip(jds.jd_text, jds.jd_position)]
    jd = pd.DataFrame({
        "jd_id": jds.jd_id, "stratum": jds.stratum, "jd_title": jds.jd_position,
        "role": [p.role for p in jp], "role_prob": [p.role_prob for p in jp],
        "top_roles": [profiles.roles_text(p.top_roles) for p in jp],
        "skills": [";".join(p.skills) for p in jp],
        "skills_qualified": [";".join(p.skills_qualified) for p in jp],
        "min_years": [p.min_years for p in jp], "max_years": [p.max_years for p in jp],
        "seniority": [p.seniority for p in jp]})
    cp = [s.extract_cv(t, i, REFERENCE_DATE) for t, i in zip(cvs.cv_text, cvs.cv_id)]
    cv = pd.DataFrame({
        "cv_id": cvs.cv_id, "role": [p.role for p in cp], "role_prob": [p.role_prob for p in cp],
        "top_roles": [profiles.roles_text(p.top_roles) for p in cp],
        "skills": [";".join(p.skills) for p in cp],
        "skills_qualified": [";".join(p.skills_qualified) for p in cp],
        "years": [p.years for p in cp], "years_source": [p.years_source for p in cp],
        "seniority": [p.seniority for p in cp], "seniority_source": [p.seniority_source for p in cp]})
    return {ax.FEATURES_RECORD.name + " (F12)": bool(feat_ok), ax.RECORD.name: x2_ok,
            ax.JD_EXTRACT_RECORD.name: _same_csv(jd, ax.JD_EXTRACT_RECORD),
            ax.CV_EXTRACT_RECORD.name: _same_csv(cv, ax.CV_EXTRACT_RECORD)}


def frozen_l1(features) -> dict[str, bool]:
    """Item 2: chat-3's L1 refitted (as `model.freeze` fits it, nothing written) = the record."""
    from candidate_screener.evaluation import chat3
    from candidate_screener.inference.model import RECORD, L1Model
    ridge = chat3.fit_stage2(chat3.targets(), features(chat3.pairs()), "L1", "F12")
    fit, rec = L1Model.from_ridge(ridge), L1Model.load()
    ok = fit.columns == rec.columns and all(
        np.max(np.abs(np.asarray(getattr(fit, a)) - np.asarray(getattr(rec, a)))) <= TOL
        for a in ("mu", "sd", "coef", "intercept"))
    return {RECORD.name + " (refit, 1e-9)": bool(ok)}


def pools_and_tests(s, features) -> dict[str, bool]:
    """Items 3 and 4: D63 and D64 through the API over the 60 final-set pools."""
    from candidate_screener.evaluation import c2_validation as cv2, chat3, rerank_chat as rc
    from candidate_screener.evaluation import rerank_embed as re_, retrieval_pilot
    from candidate_screener.inference.model import L1Model
    from candidate_screener.rerank import chat_targets as ct, chunked
    pools = retrieval_pilot.build_pools(SEED, re_.TEXT_VARIANT)
    l1, d63, d64, same_top = {}, {}, {}, {}
    model = L1Model.load()
    eval_l1 = pd.Series(model.predict(features(pools).loc[pools.pair_id.to_numpy()]),
                        index=pools.pair_id.to_numpy())
    for n, jd_id in enumerate(sorted(pools.jd_id.unique()), start=1):
        jd_text, title, cvs, ref_date = s.a2_pool(jd_id)
        kw = dict(jd_title=title, scaling="pool", reference_date=ref_date, jd_id=jd_id)
        X = s.features(jd_text, cvs, **kw)
        l1.update(zip(jd_id + "__" + X.index, s.l1.predict(X)))
        a = s.rank(jd_text, cvs, **kw)
        b = s.rank(jd_text, cvs, rerank=True, **kw)
        d63.update({jd_id + "__" + r.cv_id: -float(r.rank) for r in a.ranked})
        d64.update({jd_id + "__" + r.cv_id: -float(r.rank) for r in b.ranked})
        ids = pools[pools.jd_id == jd_id].pair_id.to_numpy()
        want = set(ids[ct.top_k_mask(ids, eval_l1.loc[ids].to_numpy(dtype=float), ct.K)])
        same_top[jd_id] = want == {jd_id + "__" + r.cv_id for r in a.top(ct.K)}
        if n % 10 == 0:
            _log(f"{n} JDs through the API")
    l1, d63, d64 = pd.Series(l1), pd.Series(d63), pd.Series(d64)

    rec = json.loads(chat3.L1_POOL_RECORD.read_text(encoding="utf-8"))
    chat = chunked.chat_frame()
    chat1 = chat[chat.source == ct.TRAIN_SOURCE]
    refs = {"chat-1": chat1, "chat-1 + chat-2": ct.load_latest()}
    stratum = chat1.drop_duplicates("jd_id").set_index("jd_id").stratum
    out = {}
    for rname, ref in refs.items():
        a = chat3.pool_coverage(pools, ref, l1, 10)
        got = {"recall": re_.block(a.coverage, stratum, rec["seed"]), "scored": round(float(a.scored.mean()), 4)}
        out[f"{chat3.L1_POOL_RECORD.name}: {rname}"] = _json(got) == _json(rec["results"][rname][chat3.L1_POOL])
    out[f"{chat3.L1_POOL_RECORD.name}: Recall@10 0.484413"] = (
        round(float(chat3.pool_coverage(pools, chat1, l1, 10).coverage.mean()), 6) == 0.484413)
    out["D63 top 10 = evaluation path's, every JD"] = all(same_top.values()) and len(same_top) == 60

    test = json.loads(cv2.TEST_RECORD.read_text(encoding="utf-8"))
    ref = ct.load_latest()
    st = ref.drop_duplicates("jd_id").set_index("jd_id").stratum
    per = {"winner": ct.per_jd(ref, d64), cv2.L1_ALONE: ct.per_jd(ref, d63)}
    for name, scores in (("winner", d64), (cv2.L1_ALONE, d63)):
        r = test["results"][name]
        out[f"{cv2.TEST_RECORD.name}: {name} reference A"] = (
            _json(rc.summary(per[name], st, test["seed"])) == _json(r["reference_A"]["summary"]))
        out[f"{cv2.TEST_RECORD.name}: {name} reference B"] = (
            _json(re_.block(chat3.pool_coverage(pools, chat1, scores, 10).coverage, st, test["seed"]))
            == _json(r["reference_B_recall"]))
    out[f"{cv2.TEST_RECORD.name}: winner delta vs L1"] = (
        _json(rc.deltas(per["winner"], per[cv2.L1_ALONE], st, test["seed"]))
        == _json(test["results"]["winner"]["reference_A"]["delta_vs_L1"]))
    return out


def check() -> dict[str, bool]:
    from candidate_screener.inference.screener import Screener
    from candidate_screener.matching import learned
    t0 = time.time()
    s = Screener.load(c2_cached_only=True)
    _log(f"Screener loaded ({time.time() - t0:.1f} s); item 1: the final-set tables")
    out = final_set_tables(s)
    _log("item 2: L1 refitted from chat-3 (the evaluation's feature path, refitted parts)")
    features = learned.Features()
    out.update(frozen_l1(features))
    _log("items 3-4: D63 and D64 over the 60 pools")
    out.update(pools_and_tests(s, features))
    _log(f"done in {time.time() - t0:.0f} s")
    return out


def demo(jd_id: str, rerank: bool = False) -> None:
    """One final-set JD's top 10 through the API, and the explanation of its first CV."""
    from candidate_screener.inference.screener import Screener
    s = Screener.load()
    jd_text, title, cvs, ref_date = s.a2_pool(jd_id)
    res = s.rank(jd_text, cvs, jd_title=title, reference_date=ref_date, jd_id=jd_id, rerank=rerank)
    jd = res.jd
    print(f"JD {jd_id}: {title!r}; role {jd.role} ({jd.role_prob:.2f}); {len(jd.skills)} skills; "
          f"years {jd.min_years}-{jd.max_years}; {res.system}, pool {res.pool_size}, scaling {res.scaling}")
    rows = []
    for r in res.top(10):
        cv = res.cvs[r.cv_id]
        held = set(cv.skills)
        rows.append({"rank": r.rank, "cv_id": r.cv_id[:8], "score": round(r.score, 3), "role": cv.role,
                     "matched": sum(k in held for k in jd.skills),
                     "missing": sum(k not in held for k in jd.skills), "years": cv.years})
    print(pd.DataFrame(rows).to_string(index=False))
    why = s.explain(res, res.ranked[0].cv_id)
    print("\nWhy rank 1:")
    for line in why["reasons"]:
        print(" -", line)
    print("Largest L1 feature shares:",
          ", ".join(f"{f['feature']} {f['share']:+.2f}" for f in why["feature_shares"][:5]))
