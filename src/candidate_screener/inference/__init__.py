"""The reported system as plain functions an app can call (plan 2026-10-23-inference-api).

    from candidate_screener import inference
    s = inference.Screener.load()
    res = s.rank(jd_text, cvs, jd_title=...)      # D63; rerank=True for D64; system="X2" baseline

A Streamlit page needs no more than this (uploads only; `cvs` carry `cv_id` and `cv_text`):

    import pandas as pd
    import streamlit as st
    from candidate_screener import inference
    from candidate_screener.annotation.redact import redact_text

    @st.cache_resource
    def screener():
        return inference.Screener.load()          # ~4 s, once per process

    jd = st.text_area("Job description")
    files = st.file_uploader("CVs (.txt)", accept_multiple_files=True)
    c2 = st.checkbox("Refine the top 50 with the reranker (C2, minutes)")
    if st.button("Screen", disabled=not (jd.strip() and files)):
        cvs = [{"cv_id": f.name, "cv_text": f.getvalue().decode()} for f in files]
        bar = st.progress(0.0)
        res = screener().rank(jd, cvs, rerank=c2, progress=lambda d, n: bar.progress(d / n))
        st.dataframe(pd.DataFrame([r.as_dict() for r in res.top(10)])[["rank", "cv_id", "score"]])
        for r in res.top(10):
            with st.expander(f"{r.rank}. {r.cv_id}"):
                for line in screener().explain(res, r.cv_id)["reasons"]:
                    st.write(redact_text(line))
"""
from candidate_screener.inference.model import L1Model
from candidate_screener.inference.schema import CVProfile, JDProfile, Ranked, Screening
from candidate_screener.inference.screener import Screener

__all__ = ["CVProfile", "JDProfile", "L1Model", "Ranked", "Screener", "Screening"]
