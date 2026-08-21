"""
app.py -- single-file Streamlit frontend AND backend.

No FastAPI, no database, no files written to disk. Upload a CSV, ask a
question, the LangGraph agent pipeline (Supervisor -> tool agents ->
Critic -> Report, with a re-plan loop) runs in-process and renders
straight into the page. Everything lives only in Streamlit's session
state for the duration of the browser tab.

Run with: streamlit run app.py
(Ollama must be running locally: `ollama pull gemma3:1b`)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from graph.workflow import run_eda

st.set_page_config(page_title="Agentic EDA (Simple)", layout="wide")
st.title("Agentic EDA -- Simple Edition")
st.caption(
    "Upload a CSV, ask a question in plain English. A Supervisor agent plans which "
    "analyses to run, specialist agents run them, a Critic agent verifies the evidence "
    "before the report is written. Nothing is saved to disk."
)

if "df" not in st.session_state:
    st.session_state.df = None
if "result" not in st.session_state:
    st.session_state.result = None

st.header("1. Upload Dataset")
uploaded = st.file_uploader("CSV file", type=["csv"])
if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded)
    st.success(f"Loaded {st.session_state.df.shape[0]} rows x {st.session_state.df.shape[1]} columns.")
    st.dataframe(st.session_state.df.head(), use_container_width=True)

st.header("2. Ask a Question")
question = st.text_area(
    "Natural-language analytical question",
    placeholder="e.g. What factors are most associated with revenue, and are there any anomalies?",
)

run_disabled = st.session_state.df is None or not question.strip()
if st.button("Run Analysis", disabled=run_disabled):
    with st.spinner("Supervisor is planning and running the agent pipeline..."):
        st.session_state.result = run_eda(st.session_state.df, question)

result = st.session_state.result
if result:
    st.header("3. Agent Trace")
    planned = set(result.get("execution_plan", []))
    completed = set(result.get("completed_tasks", []))
    cols = st.columns(4)
    for i, task in enumerate(sorted(planned)):
        cols[i % 4].metric(task, "done" if task in completed else "pending")

    critic = result.get("critic_feedback", {})
    if critic.get("passed"):
        st.success("Critic Agent: verification passed.")
    elif critic:
        st.warning("Critic Agent flagged issues:\n" + "\n".join(f"- {i}" for i in critic.get("issues", [])))

    st.header("4. Report")
    report_text = result.get("final_report", "")
    st.markdown(report_text)
    st.download_button("Download Report (Markdown)", data=report_text, file_name="eda_report.md", mime="text/markdown")

    st.header("5. Charts")
    charts = result.get("visualizations", [])
    if charts:
        chart_cols = st.columns(2)
        for i, chart in enumerate(charts):
            with chart_cols[i % 2]:
                st.pyplot(chart["figure"])
                st.caption(chart.get("description", ""))
    else:
        st.write("No charts were generated for this question.")

    with st.expander("Raw insights (JSON)"):
        st.json(result.get("insights", []))
