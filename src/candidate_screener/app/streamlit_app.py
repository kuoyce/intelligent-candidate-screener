"""Recruiter UI — thin Streamlit transport over `pipeline.screen` and `app.intents`.

    uv run streamlit run src/candidate_screener/app/streamlit_app.py

Everything shown passes `annotation.redact`. The chat answers come from `intents.answer`,
templates over the pipeline's results: no generative model runs here.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from candidate_screener import pipeline
from candidate_screener.annotation.redact import redact_text
from candidate_screener.app import files, intents

st.set_page_config(page_title="Candidate screening", layout="wide")


@st.cache_resource
def _matcher():
    from candidate_screener.extraction.rules import default_matcher
    return default_matcher()


@st.cache_resource
def _ranker(has_metadata: bool):
    ranker = pipeline.default_ranker(has_metadata)
    ranker.fit(None)
    return ranker


@st.cache_data
def _demo_jobs() -> pd.DataFrame:
    from candidate_screener.matching import corpus
    pairs = corpus.a2_pairs()
    jobs = pairs.drop_duplicates("jd_id")[["jd_id", "stratum", "jd_keyword", "jd_position",
                                           "jd_exp_band", "jd_text"]]
    return jobs.sort_values(["jd_keyword", "jd_id"]).reset_index(drop=True)


@st.cache_data
def _demo_candidates(jd_id: str, pool: bool) -> pd.DataFrame:
    if pool:
        from candidate_screener.evaluation.retrieval_pilot import build_pools
        frame = build_pools(0)
    else:
        from candidate_screener.matching import corpus
        frame = corpus.a2_pairs()
    frame = frame[frame.jd_id == jd_id]
    return frame[["cv_id", "cv_text", "cv_keyword", "cv_exp_band"]].reset_index(drop=True)


def _run(jd_text: str, cvs: pd.DataFrame, jd_meta: dict | None) -> None:
    has_meta = bool(jd_meta and jd_meta.get("keyword")) and "cv_keyword" in cvs
    start = time.perf_counter()
    results = pipeline.screen(jd_text, cvs, jd_meta=jd_meta, ranker=_ranker(has_meta),
                              matcher=_matcher())
    st.session_state.update(results=results, jd_text=jd_text, chat=[], cvs=cvs,
                            seconds=time.perf_counter() - start, n=len(cvs))


# --- sidebar: what to screen ---------------------------------------------------------
with st.sidebar:
    st.header("Screen candidates")
    source = st.radio("Source", ["Demo job (A2 in-domain)", "Upload files"])
    jobs = None
    if source.startswith("Demo"):
        try:
            jobs = _demo_jobs()
        except (FileNotFoundError, OSError, ValueError) as exc:
            st.warning(f"Demo data is not built on this machine ({type(exc).__name__}). "
                       "Upload files instead.")
    if jobs is not None:
        label = jobs.apply(lambda r: f"{r.jd_keyword} · {r.jd_position} ({r.stratum})", axis=1)
        pick = st.selectbox("Job", range(len(jobs)), format_func=lambda i: label.iloc[i])
        pool = st.toggle("500-candidate pool (else the 10 labelled candidates)", value=False)
        if st.button("Screen", type="primary"):
            job = jobs.iloc[pick]
            _run(job.jd_text, _demo_candidates(job.jd_id, pool),
                 {"jd_id": job.jd_id, "keyword": job.jd_keyword, "exp_band": job.jd_exp_band})
    elif source == "Upload files":
        jd_text = st.text_area("Job description", height=200)
        uploads = st.file_uploader("Resumes (.txt, .pdf, .html)", accept_multiple_files=True)
        if st.button("Screen", type="primary", disabled=not (jd_text and uploads)):
            cvs = pd.DataFrame({"cv_id": [u.name for u in uploads],
                                "cv_text": [files.to_text(u.name, u.getvalue()) for u in uploads]})
            _run(jd_text, cvs, None)

# --- main: results, evidence, chat -------------------------------------------------------
st.title("Candidate screening")
results = st.session_state.get("results")
if not results:
    st.info("Pick a demo job or upload files, then press **Screen**.")
    st.stop()

st.caption(f"{st.session_state.n} candidates screened in {st.session_state.seconds:.2f} s. "
           "Decision support only — the ranking and fit call are screening aids, not verdicts.")
table = pd.DataFrame([{"rank": r.rank, "candidate": str(r.cv_id)[:12],
                       "fit call": r.fit_call or "not assessed",
                       "coverage": None if r.coverage is None else round(100 * r.coverage),
                       "relative strength": round(100 * r.confidence),
                       "years stated": r.years_stated} for r in results])
left, right = st.columns([3, 2])
with left:
    st.subheader("Ranked shortlist")
    st.dataframe(table.head(50), hide_index=True, use_container_width=True)
    pick = st.number_input("Show evidence for rank", 1, len(results), 1)
    chosen = results[int(pick) - 1]
    st.markdown(redact_text(intents.answer(intents.Intent("explain", (str(chosen.rank),)), results)))
    with st.expander("Resume text (redacted)"):
        cvs = st.session_state.cvs
        text = cvs.loc[cvs.cv_id.astype(str) == str(chosen.cv_id), "cv_text"].iloc[0]
        st.text(redact_text(text)[:5000])
with right:
    st.subheader("Ask about the shortlist")
    st.caption(intents.HELP)
    for role, text in st.session_state.get("chat", []):
        with st.chat_message(role):
            st.markdown(text)
    if prompt := st.chat_input("e.g. why is 1 above 3"):
        reply = redact_text(intents.answer(intents.parse(prompt), results))
        st.session_state.chat += [("user", prompt), ("assistant", reply)]
        st.rerun()
