"""Thin Streamlit page over `chat_rerank`: copy a batch's prompt, paste the chat's reply, next.

    uv run --group app streamlit run src/candidate_screener/rerank/chat_rerank_app.py
"""
import pandas as pd
import streamlit as st

from candidate_screener.rerank import chat_rerank as cr

st.set_page_config(page_title="Chat rerank", layout="wide")
if not cr.PLAN.exists():
    st.error("No plan yet. Run `uv run python -m candidate_screener.rerank.chat_rerank --build`.")
    st.stop()


@st.cache_data
def plan_with_text() -> pd.DataFrame:
    return cr.texts(pd.read_csv(cr.PLAN))


plan = plan_with_text()
done = cr.latest()
done_batches = set(zip(done.jd_id, done.batch))
order = plan[["jd_order", "jd_id", "stratum"]].drop_duplicates().sort_values("jd_order")
batches = [(r.jd_order, r.jd_id, r.stratum, b) for r in order.itertuples()
           for b in sorted(plan[plan.jd_id == r.jd_id].batch.unique())]

# sidebar: progress and navigation
n_done = sum((j, b) in done_batches for _, j, _, b in batches)
st.sidebar.progress(n_done / len(batches), text=f"{n_done} of {len(batches)} batches")
jds_done = sum(all((j, b) in done_batches for _, j2, _, b in batches if j2 == j) for j in order.jd_id)
st.sidebar.write(f"{jds_done} of {len(order)} JDs complete")
first_open = next((i for i, (_, j, _, b) in enumerate(batches) if (j, b) not in done_batches), len(batches) - 1)
if "pos" not in st.session_state:
    st.session_state.pos = first_open
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("← Back", disabled=st.session_state.pos == 0):
    st.session_state.pos -= 1
if c2.button("Next →", disabled=st.session_state.pos >= len(batches) - 1):
    st.session_state.pos += 1
if c3.button("Open"):
    st.session_state.pos = first_open
pos = st.session_state.pos
jd_order, jd_id, stratum, b = batches[pos]
n_b = int(plan[plan.jd_id == jd_id].batch.max())

st.title(f"JD {jd_order} of {len(order)} · batch {b} of {n_b}")
status = "✅ saved (pasting again replaces it)" if (jd_id, b) in done_batches else "⏳ not done"
st.caption(f"{status} · stratum {stratum}")
if b == 1:
    st.warning("**Start a new chat** for this JD, then paste the prompt below.")
else:
    st.info("Paste into the **same chat** as this JD's batch 1.")

text, ids = cr.batch_prompt(plan, jd_id, b)
st.subheader("1 · Copy this prompt (copy icon, top right of the box)")
st.code(text, language=None, height=260, wrap_lines=True)
st.caption(f"{len(text):,} characters · CVs {ids[0]}–{ids[-1]}")

st.subheader("2 · Paste the chat's reply")
with st.form(f"reply-{jd_id}-{b}", clear_on_submit=True):
    raw = st.text_area("reply", height=200, label_visibility="collapsed",
                       placeholder='[{"cv":"C001","label":"Good Fit","score":80,"reason":"..."}, ...]')
    if st.form_submit_button("Check and save"):
        try:
            parsed = cr.parse_reply(raw, ids)
        except ValueError as e:
            st.error(f"Refused: {e}. Ask the chat to resend the JSON for exactly these ids.")
        else:
            cr.save_reply(jd_id, b, raw, parsed, plan)
            st.session_state.pos = min(pos + 1, len(batches) - 1)
            st.success(f"Saved {len(parsed)} CVs.")
            st.rerun()

if (jd_id, b) in done_batches:
    with st.expander("Saved scores for this batch"):
        mine = done[(done.jd_id == jd_id) & (done.batch == b)]
        st.dataframe(mine[["short_id", "label", "score", "reason"]].sort_values("short_id"), hide_index=True)
