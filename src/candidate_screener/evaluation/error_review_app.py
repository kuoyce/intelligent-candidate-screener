"""Thin Streamlit page over `error_review`: one mistake at a time, answer, save, resume."""
import json

import pandas as pd
import streamlit as st

from candidate_screener.evaluation import error_review as er

st.set_page_config(page_title="Error review", layout="wide")
if not er.CASES.exists():
    st.error("No cases yet. Run `uv run python -m candidate_screener.evaluation.error_review --build`.")
    st.stop()

cases = er.shuffled(pd.read_csv(er.CASES))
reviewer = st.sidebar.text_input("Reviewer", key="reviewer")
done = er.latest()
st.sidebar.write(f"{len(done)} of {len(cases)} answered")
if "pos" not in st.session_state:                       # resume at the first unanswered case
    unanswered = ~cases.case_id.isin(done.index)
    st.session_state.pos = int(unanswered.idxmax()) if unanswered.any() else len(cases)
pos = st.session_state.pos
if pos >= len(cases):
    st.success("All cases answered.")
    if st.button("Back"):
        st.session_state.pos = len(cases) - 1
        st.rerun()
    st.stop()
case = cases.iloc[pos]
d = json.loads(case.details)
prev = done.loc[case.case_id] if case.case_id in done.index else None   # an earlier answer, shown for editing

st.caption(f"Case {pos + 1} of {len(cases)}. The Fit and No Fit labels come from the final set; "
           "the score ordered this pair the wrong way. Why?")
st.subheader("Job description")
st.text_area("jd", case.jd_text, height=220, label_visibility="collapsed")
left, right = st.columns(2)
left.subheader("Fit CV"); left.text_area("fit", case.fit_cv, height=420, label_visibility="collapsed")
right.subheader("No Fit CV"); right.text_area("nofit", case.nofit_cv, height=420, label_visibility="collapsed")

if st.button("← Back", disabled=pos == 0):
    st.session_state.pos = pos - 1
    st.rerun()
with st.form(f"answer-{case.case_id}"):
    reason = st.radio("Reason", list(er.CHOICES), format_func=lambda k: f"{k}: {er.CHOICES[k]}",
                      index=list(er.CHOICES).index(prev.reason) if prev is not None else None)
    note = st.text_input("Note (needed for E, and for an override)",
                         value="" if prev is None or pd.isna(prev.note) else prev.note)
    override = st.radio("Override: I am very sure the final label is wrong (use sparingly; needs reason D)",
                        er.OVERRIDES, horizontal=True, index=er.OVERRIDES.index(prev.override) if prev is not None else 0,
                        format_func=lambda k: {
                            "none": "no override", "fit_side": "Fit CV is really No Fit",
                            "nofit_side": "No Fit CV is really Fit", "both": "both are wrong"}[k])
    if st.form_submit_button("Save and next"):
        if not reviewer:
            st.warning("Enter a reviewer name in the sidebar.")
        elif reason is None or (reason == "E" and not note):
            st.warning("Pick a reason (E needs a note).")
        elif override != "none" and (reason != "D" or not note):
            st.warning("An override needs reason D and a note saying why.")
        else:
            er.save(case.case_id, case.bucket, reason, note, reviewer, override)
            st.session_state.pos = pos + 1
            st.rerun()

with st.expander("Model details and who labelled each CV (open after answering)"):
    st.write(f"Score: Fit CV {d['score_fit']:.3f}, No Fit CV {d['score_nofit']:.3f}")
    a, b = st.columns(2)
    a.markdown("**Fit CV**\n\n" + er.describe(d["fit"])); b.markdown("**No Fit CV**\n\n" + er.describe(d["nofit"]))
    st.json({"features Fit": d["features_fit"], "features No Fit": d["features_nofit"]}, expanded=False)
